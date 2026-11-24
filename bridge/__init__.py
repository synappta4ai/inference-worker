"""Bridge saliente de Synapta (Fase A del plan distribuido).

El worker se empareja con el back mediante un código corto y después
consulta trabajos por HTTP long-poll saliente: funciona detrás del
NAT/firewall del hogar, sin abrir puertos ni túneles.

Piezas:
    config    — archivo de configuración por-usuario (~/.config/…), cross-platform
    client    — cliente HTTP contra el back (register/heartbeat/claim/report/upload)
    journal   — journal de jobs en disco: sobrevive cortes de luz a mitad de job
    agent     — el loop: heartbeat + claim + ejecución vía gRPC local + upload
    pairing   — CLI de emparejamiento (el usuario pega el código de la app)
"""

# Versión del agente bridge; viaja en el heartbeat para que el back detecte
# workers desactualizados antes de un rollout (riesgo "versiones mezcladas"
# del plan).
#
# 0.2.6: el worker arregla el CA heredado del instalador (las descargas de
# pesos volvían a fallar con "Permission denied" y no arrancaban nunca) y la
# cancelación de descargas (corta también el camino Xet y descarta los restos
# a medio bajar, que inflaban el tamaño/porcentaje que ve el panel).
#
# 0.2.7: el catálogo (ListModels) reporta la ruta de los pesos en disco
# (download_path), así el back y el panel ubican los archivos sin adivinar el
# slug de la caché. Un worker 0.2.6 sigue funcionando: sólo omite el campo.
#
# 0.2.8: el snapshot del heartbeat también lleva `downloaded` y
# `download_path`, que es lo único que ve el back cuando la GPU está detrás de
# NAT: sin esto el panel admin no podía mostrar la ruta en equipos emparejados.
#
# 0.2.9: el artifact server lista (GET /artifacts) y borra
# (DELETE /artifacts/<nombre>) lo generado, que es lo que proxya el back en
# /api/v1/worker/artifacts. Antes sólo se podía pedir un archivo por nombre.
# 0.2.13: el instalador descargable fija las versiones de las librerías (diffusers
# 0.36 / transformers 4.57 / huggingface_hub 0.36) e instala torch CUDA con un
# constraint, así pip ya no puede dejar un torch CPU en una máquina con GPU.
# 0.2.14: el video respeta el ratio pedido (`_video_dims` deriva width/height del
# max_side del modelo en múltiplos de 32); antes un pedido 9:16 salía 832x480.
# 0.2.15: calidad del clip de Wan: bfloat16 + VAE fp32, negative prompt por
# defecto y `num_frames` en 4k+1 (el VAE temporal de Wan lo exige).
# 0.2.16: catálogo sin modelos de prueba (SD-Tiny y los TTS/música `mock`) y
# con SD-Turbo, pensado para GPUs de 8-12 GB e iGPU/CPU.
# 0.2.17: fuera Qwen-Image e HiDream-I1: la factoría solo carga sd15/sdxl/flux,
# así que esos dos fallaban al cargar y solo ensuciaban la lista.
# 0.2.18: el payload vuelve a incluir `installer/` (el auto-update del worker
# moría con ImportError) y el instalador suma ventana de modelos y actualización.
# 0.2.19: arreglado el auto-update: `stop(tree=False)` porque el updater es hijo
# del worker y el `taskkill /T` lo mataba a mitad de camino, y sus prints ya no
# revientan con UnicodeEncodeError en consolas cp1252.
# 0.2.20: el worker ya no asume CUDA: `device.py` elige CUDA/XPU/CPU con su
# VRAM y su precisión (fp16 en Arc, fp32 en CPU), y el instalador baja el torch
# XPU cuando detecta una Intel Arc.
# 0.2.21: `stop()` encuentra al worker por el puerto cuando no hay worker.pid.
# 0.2.22: el agente atiende `requested_deletions` del heartbeat (borrar pesos
# para liberar disco de una máquina emparejada) y confirma con
# `completed_deletions`. Un back sin la cola simplemente manda la lista vacía.
BRIDGE_VERSION = "0.2.22"
