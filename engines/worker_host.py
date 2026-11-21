"""Proceso hijo aislado que carga y ejecuta un pipeline diffusers.

Por qué existe: un modelo pesado agota la RAM y el sistema operativo mata el
proceso de golpe. Eso no es un `MemoryError` capturable: el proceso simplemente
desaparece y se lleva a todo el worker (gRPC incluido). Cargando los pesos acá,
ese final se queda en el hijo: el worker responde FAILED y sigue atendiendo.

Además, cuando el hijo termina el SO recupera TODA su memoria en el acto. Con
`gc.collect()` dentro del worker, Wan-1.3B (≈27 GB de safetensors en disco)
dejaba páginas que el proceso no devolvía, y la siguiente carga se quedaba sin
RAM a mitad de shard.

Protocolo: una línea JSON por mensaje.

  padre → hijo (stdin)
    {"cmd": "job", "steps": 20, "out_dir": "...", "request": {...}}
    {"cmd": "cancel"}
    {"cmd": "shutdown"}

  hijo → padre (stdout)
    {"type": "event", "pct": 42.0, "status": "GENERATING", "out": "", "err": ""}
    {"type": "log", "msg": "..."}

Si el padre cierra stdin (worker apagado o reiniciado), el hijo se va: nunca
queda un proceso huérfano ocupando RAM.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from types import SimpleNamespace

# Campos del proto que usan los runners de la factoría.
_FIELDS = ("job_id", "prompt", "negative_prompt", "width", "height",
           "num_frames", "fps", "seed")

_USE_GPU = False


def _emit(payload: dict) -> None:
    """Escribe un mensaje para el padre sin buffering (el padre lee por línea)."""
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    """Diagnóstico legible: stderr va al worker.log heredado del padre."""
    print(msg, file=sys.stderr, flush=True)


def _rss_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**3
    except Exception:  # noqa: BLE001 — telemetría best-effort
        return 0.0


def _reset_vram_peak() -> None:
    """Reinicia el contador de pico de VRAM (sólo GPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 — telemetría best-effort
        pass


def _vram_peak_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1024**3
    except Exception:  # noqa: BLE001
        return 0.0


def _stdin_loop(commands: "queue.Queue[dict]", state: dict) -> None:
    """Reparte los mensajes del padre: cancelar es inmediato, los jobs se encolan."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if not isinstance(msg, dict):
            continue
        if msg.get("cmd") == "cancel":
            cancel = state.get("cancel")
            if cancel is not None:
                cancel.set()
        elif msg.get("cmd") == "eof":
            break
        else:
            commands.put(msg)
    # stdin cerrado = el padre ya no existe: salir sin dejar huérfanos.
    os._exit(0)


def _build_request(job: dict) -> SimpleNamespace:
    raw = job.get("request") or {}
    values = {}
    for field in _FIELDS:
        default = "" if field in ("job_id", "prompt", "negative_prompt") else 0
        values[field] = raw.get(field, default)
    return SimpleNamespace(**values)


def _run_job(factory, spec, job: dict, cancel: threading.Event) -> None:
    """Carga (cacheado en este proceso) y genera; emite eventos al padre."""
    out_dir = str(job.get("out_dir") or "outputs")
    os.makedirs(out_dir, exist_ok=True)
    req = _build_request(job)
    steps = max(1, min(int(job.get("steps") or spec.steps), 60))

    phase = {"name": "LOADING_MODEL"}
    peaks: dict[str, float] = {}

    def cerrar_fase() -> None:
        """Guarda el pico exacto de VRAM de la fase en curso.

        `max_memory_allocated` da el pico real en vez de un muestreo cada
        0,2 s, y reiniciarlo al cambiar de fase separa cargar de denoisear de
        decodificar sin hilos de fondo.
        """
        pico = _vram_peak_gb()
        if pico:
            peaks[phase["name"]] = max(peaks.get(phase["name"], 0.0), pico)

    def report(pct: float, status: str, out: str = "", err: str = "") -> None:
        if status != phase["name"]:
            cerrar_fase()
            phase["name"] = status
            _reset_vram_peak()
        _emit({"type": "event", "pct": float(pct), "status": status,
               "out": out, "err": err})

    t0 = time.time()
    _reset_vram_peak()
    pipe = None
    try:
        report(0.0, "LOADING_MODEL")
        # Si el pipeline ya vive en este proceso sus pesos YA están en RAM:
        # volver a medir contra la RAM libre rechazaría el segundo job.
        cargado = factory._pipeline_loaded(spec, _USE_GPU)
        if not cargado:
            factory._check_memory_before_load(spec, _USE_GPU)
        pipe = factory._get_pipeline(spec, _USE_GPU)
        if not cargado:
            _log(f"[Host] {spec.repo} listo en {time.time() - t0:.1f}s "
                 f"(gpu={_USE_GPU}, pasos={steps}, RSS {_rss_gb():.1f} GB)")
        report(1.0, "GENERATING")

        if spec.mode == "video":
            out_name = factory._run_video(
                pipe, req, spec, out_dir, report, cancel, steps
            )
        else:
            image = factory._run_image(pipe, req, spec, report, cancel, steps)
            report(90.0, "POSTPROCESSING")
            out_name = f"{req.job_id}.png"
            image.save(os.path.join(out_dir, out_name), format="PNG")

        cerrar_fase()
        detalle = ", ".join(f"{k} {v:.2f} GB" for k, v in peaks.items() if v)
        pico = max(peaks.values(), default=0.0)
        _emit({"type": "log",
               "msg": f"job {req.job_id} completado en {time.time() - t0:.1f}s "
                      f"(RSS {_rss_gb():.1f} GB"
                      + (f", pico VRAM {pico:.2f} GB" if pico else "")
                      + (f" [{detalle}]" if detalle else "")
                      + ")"})
        report(100.0, "COMPLETED", out_name)
    except factory._CancelledError:
        _log(f"[Host] {req.job_id} cancelado")
        report(0.0, "CANCELLED")
    except MemoryError as exc:
        _log(f"[Host] {req.job_id} sin memoria: {exc}")
        report(0.0, "FAILED", "", str(exc))
    except Exception as exc:  # noqa: BLE001 — un job roto no es un worker roto
        _log(f"[Host] {req.job_id} falló: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        report(0.0, "FAILED", "", f"{spec.pipeline}: {exc}")
    finally:
        # El pipeline queda cacheado en RAM, pero devuelve la VRAM: entre jobs
        # el transformer y el VAE no tienen por qué estar cargados.
        if pipe is not None:
            factory.release_vram(pipe)


def main() -> int:
    global _USE_GPU

    # Los `logger.info` del factory (tiempos de encode y de paso) van a
    # stderr y terminan en worker.log; sin nivel configurado se descartan.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Consolas/pipes de Windows: sin esto, los acentos rompen el JSON.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("uso: python -m engines.worker_host '<spec json>' [auto|0|1]",
              file=sys.stderr)
        return 2

    spec_dict = json.loads(sys.argv[1])
    gpu_arg = sys.argv[2] if len(sys.argv) > 2 else "auto"

    from engines import diffusion_factory as factory  # tardío: importa torch
    from engines.base import ModelSpec

    # "auto" deja la decisión acá: así el worker padre nunca importa torch
    # (300 MB de RSS que el guardia de memoria terminaría contando).
    _USE_GPU = factory._has_cuda() if gpu_arg == "auto" else gpu_arg == "1"

    spec = ModelSpec(**spec_dict)
    commands: "queue.Queue[dict]" = queue.Queue()
    state: dict = {"cancel": None}
    threading.Thread(
        target=_stdin_loop, args=(commands, state),
        name="host-stdin", daemon=True,
    ).start()

    while True:
        msg = commands.get()
        if msg.get("cmd") == "shutdown":
            return 0
        if msg.get("cmd") != "job":
            continue
        cancel = threading.Event()
        state["cancel"] = cancel
        try:
            _run_job(factory, spec, msg, cancel)
        finally:
            state["cancel"] = None


if __name__ == "__main__":
    sys.exit(main())
