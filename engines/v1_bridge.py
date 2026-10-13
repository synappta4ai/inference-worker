"""Bridge al código V1 (app/models) para familias con handlers propietarios.

Los handlers de V1 (ltx25, minimax_h3, kandinsky5, etc.) dependen de mmgp,
WanGP y checkpoints con licencias específicas. Este bridge:
1. Localiza el directorio V1 (env BM_V1_MODELS_DIR o /models).
2. Importa el handler de la familia bajo demanda.
3. Si no existe el volumen o falla la importación → degrada a mock con un
   mensaje claro, de modo que el worker sigue operativo en cualquier entorno.

Cada familia expone un handler con interfaz `run(request, ctx)`; este bridge
normaliza el contrato V2 (eventos tupla) alrededor del resultado V1.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import threading
import time

from .base import EngineContext, ModelSpec

logger = logging.getLogger("V1Bridge")

# Ruta donde la plataforma monta el árbol app/models de la V1.
V1_MODELS_DIR = os.environ.get("BM_V1_MODELS_DIR", "/models")

# pipeline de la spec → (directorio V1, módulo handler)
FAMILY_MAP = {
    "ltx25": ("ltx25", "ltx25_handler"),
    "minimax_h3": ("minimax_h3", "minimax_h3_handler"),
    "kandinsky5": ("kandinsky5", "kandinsky_handler"),
    "krea2": ("krea2", "krea2_handler"),
    "longcat": ("longcat", "longcat_handler"),
    "ideogram4": ("ideogram4", "ideogram4_handler"),
    "z_image": ("z_image", "z_image_handler"),
    "yue2": ("TTS", "yue2_handler"),
}

_handlers: dict[str, object] = {}
_import_tried: set[str] = set()
_LOCK = threading.Lock()


def is_available() -> bool:
    """True si el árbol app/models de la V1 está montado en V1_MODELS_DIR."""
    return os.path.isdir(V1_MODELS_DIR)


def _try_import(pipeline_key: str):
    """Importa el handler V1 una sola vez por familia; None si no está."""
    with _LOCK:
        if pipeline_key in _handlers:
            return _handlers[pipeline_key]
        if pipeline_key in _import_tried:
            return None
        _import_tried.add(pipeline_key)

        if not os.path.isdir(V1_MODELS_DIR):
            logger.info(
                "[V1Bridge] %s: volumen V1 no presente en %s (modo mock)",
                pipeline_key, V1_MODELS_DIR,
            )
            return None

        if V1_MODELS_DIR not in __import__("sys").path:
            __import__("sys").path.insert(0, V1_MODELS_DIR)
            __import__("sys").path.insert(0, os.path.dirname(V1_MODELS_DIR))

        family, module = FAMILY_MAP.get(pipeline_key, (None, None))
        if not family:
            return None
        try:
            mod = importlib.import_module(f"{family}.{module}")
            _handlers[pipeline_key] = mod
            logger.info("[V1Bridge] Handler V1 cargado: %s.%s", family, module)
            return mod
        except Exception as exc:
            logger.warning(
                "[V1Bridge] %s: no se pudo importar %s.%s (%s) → mock",
                pipeline_key, family, module, exc,
            )
            return None


async def generate(req, spec: ModelSpec, ctx: EngineContext):
    """Ejecuta el handler V1 (en thread) o degrada a mock determinista."""
    handler = _try_import(spec.pipeline)

    if handler is None:
        # Degradación transparente: mismo contrato que el mock.
        from . import mock_engine

        async for event in mock_engine.generate(req, spec, ctx):
            yield event
        return

    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()

    def report(pct, status, out="", err=""):
        loop.call_soon_threadsafe(events.put_nowait, (pct, status, out, err))

    def worker():
        try:
            report(0.0, "LOADING_MODEL")
            out_path = handler.run(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width or spec.max_side,
                height=req.height or spec.max_side,
                num_frames=req.num_frames,
                fps=req.fps or 16,
                steps=req.steps or spec.steps,
                seed=req.seed,
                out_dir=ctx.out_dir,
                job_id=req.job_id,
                mode=req.mode,
                progress_cb=lambda pct, status: report(pct, status),
                cancel_check=ctx.cancel.is_set,
            )
            report(100.0, "COMPLETED", os.path.basename(out_path or ""))
        except Exception as exc:
            if ctx.cancel.is_set():
                report(0.0, "CANCELLED")
            else:
                logger.exception("[V1Bridge] %s falló: %s", req.job_id, exc)
                report(0.0, "FAILED", "", f"v1_bridge/{spec.pipeline}: {exc}")

    threading.Thread(target=worker, name=f"v1-{req.job_id}", daemon=True).start()

    while True:
        pct, status, out, err = await events.get()
        yield (pct, status, out, err)
        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            return
