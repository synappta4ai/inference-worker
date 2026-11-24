"""Doctor: verifica que la máquina pueda correr el worker de Synapta.

Chequeos (en orden):
  · versión de Python
  · GPU NVIDIA + driver (o CPU con aviso)
  · memoria RAM y espacio en disco (los pesos pesan 10–40 GB)
  · puertos libres (50051 gRPC local, 50052 artefactos)
  · estado de emparejamiento (config del bridge)
  · smoke test del catálogo con el motor mock

Sale con código 0 si todo está operable, 1 si hay bloqueos. Es el mismo
script que corren scripts/setup.ps1 y setup.sh tras instalar.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

# Consolas Windows (cp1252) revientan con los iconos del reporte.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # los tests importan desde la raíz del repo

OK, WARN, BAD = "OK", "WARN", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str) -> None:
    _results.append((name, status, detail))


def check_python() -> None:
    v = sys.version_info
    if (v.major, v.minor) >= (3, 11):
        check("Python", OK, f"{v.major}.{v.minor}.{v.micro}")
    else:
        check("Python", BAD, f"{v.major}.{v.minor} — necesitás 3.11+")


def check_device() -> None:
    """Qué dispositivo va a usar el worker: CUDA, XPU (Arc) o CPU.

    En una iGPU Ryzen el worker corre en CPU en fp32: SD-Turbo tarda ~1 min por
    imagen, pero funciona. En una Intel Arc, si falta el driver, `torch.xpu`
    queda en False y también cae a CPU: este chequeo lo dice antes de generar.
    """
    try:
        import device  # el worker está en cwd durante el doctor

        info = device.current()
    except Exception as exc:  # noqa: BLE001 — sin torch igual se instala
        check("dispositivo", WARN, f"no pude detectarlo: {exc}")
        return
    if info.is_gpu:
        check("dispositivo", OK, info.describe())
        return
    memoria = f"{info.total_mb / 1024:.0f} GB de RAM" if info.total_mb else "RAM ?"
    check("dispositivo", WARN,
          f"{info.name} → CPU en {info.pretty_precision} ({memoria}): "
          "SD-Turbo y SD-1.5 funcionan, video no es viable")


def check_gpu() -> None:
    if shutil.which("nvidia-smi") is None:
        check("GPU NVIDIA", WARN,
              "nvidia-smi no encontrado: el worker correrá en CPU (muy lento "
              "para video). ¿Es una máquina sin GPU?")
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            line = out.stdout.strip().splitlines()[0]
            check("GPU NVIDIA", OK, line)
        else:
            check("GPU NVIDIA", BAD, "nvidia-smi no devolvió GPUs")
    except (OSError, subprocess.TimeoutExpired) as exc:
        check("GPU NVIDIA", BAD, f"no pude consultar el driver: {exc}")
        return
    check_torch_cuda()


def check_torch_cuda() -> None:
    """torch y la GPU del sistema tienen que coincidir.

    Un torch CPU sobre una máquina con NVIDIA no falla: el worker arranca, el
    panel lo muestra online y las generaciones se arrastran (video 1.3B en CPU)
    o mueren en silencio. Pasó de verdad: el instalador terminó con
    torch 2.15.0+cpu porque pip resolvió `open_clip_torch` desde PyPI.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 — torch roto también es diagnóstico
        check("torch CUDA", BAD, f"no puedo importar torch: {exc}")
        return
    version = getattr(torch, "__version__", "?")
    if torch.version.cuda is None:
        check("torch CUDA", BAD,
              f"torch {version} es el build CPU y esta máquina tiene GPU: "
              "reinstalá el worker (el instalador baja el build CUDA)")
        return
    try:
        ok = torch.cuda.is_available()
    except Exception as exc:  # noqa: BLE001
        check("torch CUDA", BAD, f"torch {version} con CUDA falló: {exc}")
        return
    if not ok:
        check("torch CUDA", BAD,
              f"torch {version} (CUDA {torch.version.cuda}) no ve la GPU: "
              "actualizá el driver NVIDIA")
        return
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    check("torch CUDA", OK, f"{version} · {name} ({vram:.0f} GB)")


def check_ram() -> None:
    try:
        import psutil
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        # Umbrales en GiB con margen: una máquina "8 GB" reporta ~7,9 GiB
        # (y "16 GB" ~15,9) — sin margen, las de 8 GB caían a FAIL pese a
        # que el propio aviso ya decía que sirven para modelos chicos.
        if total_gb >= 15.5:
            check("RAM", OK, f"{total_gb:.1f} GB")
        elif total_gb >= 7.5:
            check("RAM", WARN,
                  f"{total_gb:.1f} GB (los modelos 14B necesitan ≥16 GB; "
                  "con estos alcanza para 1.3B/7B)")
        else:
            check("RAM", BAD, f"{total_gb:.1f} GB — insuficiente "
                              "(hacen falta 8 GB)")
    except ImportError:
        check("RAM", WARN, "psutil ausente (se instala con requirements.txt)")


def check_disk() -> None:
    try:
        import psutil
        free_gb = shutil.disk_usage(str(ROOT)).free / (1024 ** 3)
        if free_gb >= 50:
            check("Disco", OK, f"{free_gb:.0f} GB libres")
        elif free_gb >= 20:
            check("Disco", WARN, f"{free_gb:.0f} GB libres (pesos grandes: 10–40 GB c/u)")
        else:
            check("Disco", BAD, f"{free_gb:.0f} GB libres — no alcanza para los pesos")
    except ImportError:
        check("Disco", WARN, "psutil ausente")


def check_ports() -> None:
    busy = []
    for port in (50051, 50052):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                busy.append(port)
    if busy:
        check("Puertos", WARN,
              f"{busy} ocupados — ¿ya hay un worker corriendo? (no es un error "
              "si es tu worker activo)")
    else:
        check("Puertos", OK, "50051 y 50052 libres")


def check_pairing() -> None:
    from bridge import config
    cfg = config.load()
    if cfg.paired:
        check("Emparejamiento", OK,
              f"conectado a {cfg.back_url} (worker {cfg.worker_id[:8]}…)")
    else:
        check("Emparejamiento", WARN,
              "sin emparejar — corré `python pair.py` con el código de la app "
              "para conectar tu GPU a Synapta")


def check_catalog() -> None:
    """Smoke test: el catálogo y el motor mock responden sin GPU ni pesos."""
    os.environ.setdefault("BM_FORCE_MOCK", "1")
    try:
        from engines import registry
        models = registry.list_models()
        if models:
            check("Catálogo", OK, f"{len(models)} modelos declarados")
        else:
            check("Catálogo", BAD, "MODEL_CATALOG vacío")
    except Exception as exc:  # noqa: BLE001 — el doctor reporta, no decide
        check("Catálogo", BAD, f"no pude cargar engines: {exc}")


def main() -> int:
    print("Synapta Worker — diagnóstico\n" + "=" * 40)
    for fn in (check_python, check_gpu, check_device, check_ram, check_disk,
               check_ports, check_pairing, check_catalog):
        fn()
    icon = {OK: "✔", WARN: "⚠", BAD: "✘"}
    print()
    for name, status, detail in _results:
        print(f"{icon[status]} {name:<14} {detail}")
    failures = [r for r in _results if r[1] == BAD]
    warnings = [r for r in _results if r[1] == WARN]
    print()
    if failures:
        print(f"✘ {len(failures)} problema(s) deben resolverse antes de usar el worker.")
        return 1
    if warnings:
        print(f"⚠ {len(warnings)} aviso(s): el worker funciona, pero revisá lo indicado.")
        return 0
    print("✔ Todo listo. Arrancá con `python run.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
