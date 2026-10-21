# 🧠 Brain-Master: Worker de Inferencia IA (Python + CUDA)

Este microservicio constituye la **Capa 3** de la arquitectura híbrida de **Brain-Master**. Es un servicio _headless_ (sin interfaz web acoplada) responsable de ejecutar los modelos de difusión de video, generación de imágenes, transcripción de audio y orquestación de LLMs en hardware GPU.

> **Repositorio independiente** y hogar del worker desde que `brain-master`
> quedó retirado. Es la pieza que corre a la par del back y del front, con
> **un solo consumidor**: **back-synapta** (:8099, la API de producción), que
> despacha aquí los modelos `downloaded` por gRPC y re-hospeda los artefactos
> en `/outputs`. Corre donde haya GPU: tu workstation (costo eléctrico),
> una GPU dedicada alquilada, o gratis en Colab/Kaggle vía túnel.

## 🏗️ Arquitectura interna (engines)

```
server.py (gRPC :50051) ──► engines/registry.py ──► engine según MODEL_CATALOG
                                   │                    ├── mock            (sin pesos, determinista)
                                   │                    ├── diffusion_factory (genérico Diffusers, default)
                                   │                    ├── v1_bridge        (código V1 legado)
                                   │                    └── diffusers_engine (monolítico, deprecado)
artifact_server.py (:50052) ── publica los archivos generados (workers remotos)
vram_manager.py ─────────────── telemetría VRAM + torch.cuda.empty_cache()
```

- **Fuente única de modelos**: `engines/base.py` → `MODEL_CATALOG` (spec
  declarativa: repo HF, steps, guidance, dtype, VRAM estimada). `ListModels`
  y el catálogo de back-synapta salen de acá — **agregar un modelo = agregar
  una entrada** (más el pipeline en la factoría si no existe).
- **Flujo de un job**: `GenerateMedia` emite
  `LOADING_MODEL → GENERATING* → POSTPROCESSING → COMPLETED/FAILED` con
  cancelación cooperativa (`asyncio.Event`) y escribe el artefacto en
  `BM_OUTPUT_DIR` (default `outputs/` dentro del repo, ya en `.gitignore`).
- **Interruptores**: `BM_FORCE_MOCK=1` (fuerza mock, tests/diagnóstico),
  `BM_USE_DIFFUSERS_ENGINE=1` (motor legado).

---

## 🎯 ¿Qué hace este servicio?

- **Generación de Video:** Ejecuta pipelines de difusión como **Wan 2.1 (T2V/I2V 14B)**, **MiniMax H3**, **LTX-Video 2.5** y **Hunyuan**.
- **Generación de Imagen:** Pipeline de **Flux 2 Klein 9B**, Krea y Qwen Image.
- **Audio y Voz:** Transcripción/diarización con **Whisper**, generación musical con **YuE2** y clonación de voz con **Seed-VC**.
- **Gestión de VRAM ([`vram_manager.py`](vram_manager.py)):** Telemetría continua de memoria de GPU y liberación proactiva con `torch.cuda.empty_cache()` para prevenir errores de _Out of Memory (OOM)_.

---

## 🖥️ Requisitos de Hardware

- **GPU:** NVIDIA con arquitectura Ampere, Ada Lovelace o Blackwell (RTX 3060 12GB mínimo; **RTX 3090 / 4090 de 24 GB VRAM recomendada** para video 1080p y modelos 14B).
- **Drivers:** NVIDIA Driver 535+ (Linux) o 550+ (Windows) con soporte CUDA 12.1+ o CUDA 13.
- **Almacenamiento:** Disco SSD NVMe (los pesos de modelos de video rondan entre 10 GB y 40 GB).

---

## 🚀 ¿Dónde se PUEDE y DEBERÍA desplegar?

El worker de inferencia requiere hardware especializado con aceleración CUDA. A continuación se presentan las mejores alternativas recomendadas según el escenario de uso:

### 1. 🏆 Serverless GPU (La opción más recomendada para SaaS y Startups)

> **¿Dónde?** [Modal.com](https://modal.com), [RunPod Serverless](https://www.runpod.io/serverless-gpu), [Baseten](https://www.baseten.co) o [Beam.cloud](https://beam.cloud).

- **¿Por qué deberías usarlo?**
  - **Costo $0 cuando no hay usuarios activos:** La GPU se apaga automáticamente cuando la cola está vacía.
  - **Escalado automático:** Si 10 usuarios generan videos al mismo tiempo, la plataforma levanta 10 GPUs en paralelo.
  - **Sin mantenimiento:** Cero administración de drivers de CUDA, parches de sistema operativo o reinicios de máquinas.
- **Esquema de Conexión:**
  ```
  Angular (Vercel/Web) ➔ Go Gateway (VPS Hetzner $5/mes) ➔ Modal/RunPod Serverless (Pagas solo los segundos de GPU usados)
  ```

---

### 2. ⚡ Instancia Cloud GPU Dedicada (Recomendada para tráfico continuo y baja latencia)

> **¿Dónde?** [RunPod Dedicated](https://www.runpod.io), [Vast.ai](https://vast.ai), [Lambda Labs](https://lambdalabs.com) o [TensorDock](https://tensordock.com).

- **¿Por qué deberías usarlo?**
  - **Modelos siempre en VRAM:** Los pesos están precargados en memoria, ofreciendo **cero tiempo de inicio (_zero cold-start_)**.
  - **Precios económicos:** Una RTX 4090 de 24 GB cuesta ~$0.25 a $0.35 / hora (~$180 a $250 / mes continuo).
- **Cómo desplegarlo:**
  - Desplegar el worker mediante Docker con soporte para GPU:
    ```bash
    docker run -d --gpus all -p 50051:50051 -v /models:/app/models brain-master-worker:latest
    ```
  - Conectar de forma segura con el Gateway de Go utilizando una VPN privada como **Tailscale** o **WireGuard**.

---

### 3. 🏠 Servidor Local On-Premise / Edge (Recomendado para testing, estudio y privacidad total)

> **¿Dónde?** En tu propia Workstation o servidor local con GPU NVIDIA (RTX 3090 / 4080 / 4090).

- **¿Por qué deberías usarlo?**
  - **Costo recurrente $0** en infraestructura de nube.
  - **Privacidad absoluta:** Ningún video, audio o prompt sale de tu red local.
  - Ideal para creadores de contenido, productoras audiovisuales o entornos de desarrollo.
- **Acceso remoto seguro:** Puedes exponer el worker hacia un backend en la nube utilizando **Tailscale** o **Cloudflare Tunnels** sin abrir puertos en el router.

---

### 4. 🏢 Clúster Kubernetes + KEDA / Ray (Recomendado para Grandes Empresas)

> **¿Dónde?** Clúster privado de Kubernetes con **NVIDIA GPU Operator** y auto-escalado basado en eventos (**KEDA**).

- **¿Por qué deberías usarlo?**
  - Para plataformas con miles de usuarios concurrentes y políticas empresariales estrictas de SLA y almacenamiento compartido (NFS/Ceph).

---

### ❌ ¿Dónde NO se recomienda desplegar?

- **Hyperscalers tradicionales sin descuento (AWS EC2 g5, GCP A3, Azure NV):** Tienen costos de $1.50 a $4.50+ por hora (hasta **10 veces más caros** que RunPod, Vast.ai o Modal para la misma GPU). Solo se justifican si tu empresa cuenta con créditos promocionales o normativas de compliance bancario.

---

## 🛠️ Ejecución Local y Desarrollo

### 1. Instalación de dependencias

```powershell
cd inference-worker
uv pip install -r requirements.txt
```

### 2. Iniciar el worker

```powershell
python server.py
```

### 3. Salida esperada en consola:

```text
2026-09-19 00:00:00 [INFO] [InferenceEngine] Inicializado motor de IA.
🧠 [Brain-Master] Worker de Inferencia Python listo.
Telemetría: {'device_name': 'NVIDIA GeForce RTX 4090', 'total_vram_mb': 24564, 'free_vram_mb': 23100, ...}
```

---

## 🔌 Integración con Synapta (back-synapta)

back-synapta consume este worker **en vivo y sin caché** (cliente gRPC
stateless, `internal/worker`):

| Variable (en `.env` de back-synapta) | Efecto |
|---|---|
| `BM_WORKER_ADDR=host:port` | gRPC del worker; con prefijo `tls://` para túneles cloudflared |
| `BM_WORKER_TIMEOUT_SECONDS=6` | timeout por llamada (lecturas de catálogo son por llamada) |
| `BM_WORKER_ARTIFACT_BASE=https://…` | URL base para re-hospedar artefactos de workers remotos |

- E2E cross-language (worker real + back real):
  `BM_E2E_WORKER_ADDR=127.0.0.1:50051 BM_E2E_ARTIFACT_BASE=http://127.0.0.1:50052 go test ./internal/modules/agency/video/ -run TestE2EWorker -v` (desde back-synapta).
- Mock rápido sin GPU: `BM_ARTIFACT_PORT=50052 py -3 run_mock.py`.
- Modo de descarga (`BM_PROVIDER_MODE`, en el worker): `personal` (default)
  o vacío = nada se descarga solo; cada peso baja únicamente con el botón
  Descargar de /gpu, y generar sin pesos falla con un mensaje claro en vez
  de saturar el PC. `shared` = proveedor en la nube: al arrancar pre-descarga
  todo el catálogo y las generaciones pueden completar pesos al vuelo.
- Cambios de contrato: la fuente canónica es `proto/inference.proto` **de
  este repo**. Editala acá, regenerá `inference_pb2.py` y espejá el `.proto`
  más el código Go en `back-synapta/internal/worker/proto/`.

---

## ☁️ Conectar a Synapta (GPU como proveedor)

Si esta máquina tiene una GPU y querés que la app de Synapta la use para
generar (el back queda en la nube, la GPU en tu casa — **sin abrir puertos
ni tocar el router**):

1. **Instalá** (Windows): `scripts\setup.ps1` · Linux/macOS: `./scripts/setup.sh`
   — crea el entorno virtual, instala dependencias y corre el `doctor`.
2. **Conectá**: en la app, andá a **Tu GPU → Conectar GPU** y copiá el
   código de 6 dígitos que aparece. En esta máquina:

   ```powershell
   .venv\Scripts\python.exe pair.py --code 123456
   ```

   (pegá tu URL del back en `--back https://…` si no está en la misma red).
3. **Dejala corriendo**:

   ```powershell
   .venv\Scripts\python.exe run.py
   ```

   o instalalo como servicio: `scripts\setup.ps1 -Autostart` (Windows) o
   `./scripts/setup.sh --autostart` (systemd user scope en Linux).

Después de eso no hay nada más que hacer: el worker mantiene un heartbeat
con el back, reclama trabajos de la cola y sube cada video/imagen
terminado. Si la luz se corta a mitad de un job, el `journal` local
(`~/.config/synapta-worker/jobs/`) re-sube el artefacto pendiente al
volver — nunca se pierde trabajo por internet.

Troubleshooting:

| Síntoma | Causa probable |
|---|---|
| "pairing code not found/expired" | El código vence a los 10 min: generá uno nuevo en la app. |
| "invalid worker token" en los logs | El admin revocó la GPU desde la app: corré `pair.py` de nuevo. |
| La GPU no aparece online | Fijate que `run.py` siga corriendo; el back la marca online con <75 s de heartbeat. |
| `doctor` marca disco rojo | Los pesos pesan 10–40 GB: liberá espacio antes del primer job. |

## 🐳 Dockerfile de Producción (Ejemplo)

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# Instalar Python y dependencias de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 50051

CMD ["python3", "server.py"]
```
