"""Motor de difusión real basado en Hugging Face Diffusers.

Arquitectura:
- La inferencia (bloqueante) corre en un daemon thread; los eventos llegan al
  generador async vía asyncio.Queue + call_soon_threadsafe.
- Carga perezosa cacheada por (repo, gpu): los pesos se descargan una vez.
- Streaming real: cada paso del scheduler emite GENERATING vía callback.
- Cancelación cooperativa: el callback revisa ctx.cancel y aborta el loop.
- CPU-friendly: attention slicing + fp32 en CPU; fp16 + offload en GPU.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time

from .base import EngineContext, ModelSpec

logger = logging.getLogger("DiffusersEngine")

# Cache de pipelines por (repo, gpu). Cargar/descargar es costoso.
_PIPELINES: dict[tuple[str, bool], object] = {}
_PIPELINES_LOCK = threading.Lock()


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _load_pipeline(repo: str, use_gpu: bool):
    key = (repo, use_gpu)
    with _PIPELINES_LOCK:
        if key in _PIPELINES:
            return _PIPELINES[key]

        import torch
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

        dtype = torch.float16 if use_gpu else torch.float32
        cls = (
            StableDiffusionXLPipeline
            if "xl" in repo.lower()
            else StableDiffusionPipeline
        )
        logger.info("[Diffusers] Cargando %s (gpu=%s, dtype=%s)...", repo, use_gpu, dtype)
        pipe = cls.from_pretrained(
            repo,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        if use_gpu:
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_attention_slicing()
        _PIPELINES[key] = pipe
        return pipe


class _CancelledError(RuntimeError):
    """Señal interna para abortar el loop del scheduler."""


def _run_diffusion(pipe, req, spec: ModelSpec, report, steps: int, cancel):
    """Loop bloqueante del scheduler (corre en el worker thread)."""
    if cancel.is_set():
        raise _CancelledError()

    def on_step_end(_pipe, step_index, _timestep, callback_kwargs):
        if cancel.is_set():
            raise _CancelledError()
        # diffusers dispara un callback extra al final (step_index == steps):
        # clampeamos para no emitir >90% en GENERATING.
        done = min(step_index + 1, steps)
        report(done / steps * 90.0, "GENERATING")
        return {}

    result = pipe(
        prompt=req.prompt or "",
        negative_prompt=req.negative_prompt or None,
        width=spec.max_side,
        height=spec.max_side,
        num_inference_steps=steps,
        guidance_scale=spec.guidance_scale or 7.5,
        callback_on_step_end=on_step_end,
    )
    return result.images[0]


async def generate(req, spec: ModelSpec, ctx: EngineContext):
    """Genera una imagen real con diffusers emitiendo eventos de progreso."""
    loop = asyncio.get_running_loop()
    use_gpu = _has_cuda()
    steps = max(1, min(int(req.steps or spec.steps), 50))
    t0 = time.time()

    events: asyncio.Queue = asyncio.Queue()

    def report(pct: float, status: str, out: str = "", err: str = "") -> None:
        loop.call_soon_threadsafe(events.put_nowait, (pct, status, out, err))

    def worker() -> None:
        try:
            report(0.0, "LOADING_MODEL")
            pipe = _load_pipeline(spec.repo, use_gpu)
            logger.info(
                "[Diffusers] %s listo en %.1fs (gpu=%s, pasos=%d)",
                spec.repo, time.time() - t0, use_gpu, steps,
            )
            report(2.0, "GENERATING")

            image = _run_diffusion(pipe, req, spec, report, steps, ctx.cancel)

            report(94.0, "POSTPROCESSING")
            os.makedirs(ctx.out_dir, exist_ok=True)
            out_name = f"{req.job_id}.png"
            image.save(os.path.join(ctx.out_dir, out_name), format="PNG")
            logger.info("[Diffusers] %s completado en %.1fs", req.job_id, time.time() - t0)
            report(100.0, "COMPLETED", out_name)
        except _CancelledError:
            logger.info("[Diffusers] %s cancelado", req.job_id)
            report(0.0, "CANCELLED")
        except Exception as exc:  # sin red, OOM, repo inválido...
            logger.error("[Diffusers] %s falló: %s", req.job_id, exc)
            report(0.0, "FAILED", "", f"diffusers: {exc}")

    threading.Thread(target=worker, name=f"diff-{req.job_id}", daemon=True).start()

    while True:
        pct, status, out, err = await events.get()
        yield (pct, status, out, err)
        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            return
