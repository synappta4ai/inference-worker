"""Factoría genérica de difusión para familias de imagen y video.

Cubre todos los modelos públicos con pipeline diffusers estándar:
- Imagen: SD 1.5, SDXL/Turbo, Flux (schnell/dev), Qwen-Image, HiDream
- Video: LTX-Video, Wan 2.1, HunyuanVideo, CogVideoX

Un solo motor paramétrico por familia: cada ModelSpec declara `pipeline`
(clave de PIPELINE_FACTORIES) + repo + defaults, y esta factoría hace la
carga cacheada, el loop de progreso real, la cancelación y el guardado.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import threading
import time
from typing import Callable

from .base import EngineContext, ModelSpec

logger = logging.getLogger("DiffusionFactory")

# Cache global de pipelines (repo, gpu) — el peso está en GB: nunca recargar.
_PIPELINES: dict[tuple[str, str, bool], object] = {}
_LOCK = threading.Lock()


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _release_pipelines() -> int:
    """Suelta TODOS los pipelines cacheados y limpia cachés de torch.

    Los pipelines viejos quedan referenciados en _PIPELINES para siempre:
    empty_cache() solo no los libera. Devuelve cuántos se soltaron.
    """
    with _LOCK:
        n = len(_PIPELINES)
        _PIPELINES.clear()
    if n:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        logger.info("[Factory] %d pipeline(s) cacheados liberados", n)
    return n


def _check_memory_before_load(spec: ModelSpec, use_gpu: bool) -> None:
    """Rechaza el modelo ANTES de descargar pesos si no cabe en la máquina.

    Sin este guardia, un modelo 14B en una laptop revienta el proceso con un
    abort de Rust (OOM no capturable) y tumba el worker entero.

    Si no hay espacio, primero suelta pipelines cacheados (un SD-1.5 quedado
    de un job anterior puede ocupar los GB que este modelo necesita) y vuelve
    a medir; solo falla si aún así no cabe.
    """
    try:
        import psutil

        avail_gb = psutil.virtual_memory().available / 1024**3
    except Exception:
        return
    # CPU: el modelo vive entero en RAM (margen x1.5). GPU con cpu-offload:
    # los pesos residen en RAM y fluyen a VRAM por etapas (margen x1.1; el
    # x1.5 original hacia imposible cargar Wan-1.3B en Colab con 12.7 GB).
    need_gb = spec.vram_gb * (1.1 if use_gpu else 1.5)
    if avail_gb < need_gb:
        _release_pipelines()
        try:
            avail_gb = psutil.virtual_memory().available / 1024**3
        except Exception:
            return
        if avail_gb < need_gb:
            raise MemoryError(
                f"el modelo '{spec.label}' requiere ~{spec.vram_gb} GB y solo hay "
                f"{avail_gb:.1f} GB disponibles. Usa un modelo más pequeño o una "
                "máquina con más memoria."
            )


# ----------------------------------------------------------------------
# Carga por familia
# ----------------------------------------------------------------------

def _load_image_pipeline(repo: str, use_gpu: bool, dtype_str: str):
    import torch
    from diffusers import (
        FluxPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )

    dtype = getattr(torch, dtype_str)
    if "flux" in repo.lower():
        cls = FluxPipeline
    elif "sdxl" in repo.lower() or "xl-base" in repo.lower():
        cls = StableDiffusionXLPipeline
    else:
        cls = StableDiffusionPipeline

    logger.info("[Factory] Cargando imagen %s (gpu=%s, %s)...", repo, use_gpu, dtype_str)
    pipe = cls.from_pretrained(repo, torch_dtype=dtype)
    if use_gpu:
        pipe.enable_model_cpu_offload()
    else:
        pipe.enable_attention_slicing()
    return pipe


def _load_video_pipeline(repo: str, use_gpu: bool, dtype_str: str):
    """Carga un pipeline de texto-a-video según su familia diffusers."""
    import torch
    from diffusers import (
        CogVideoXPipeline,
        HunyuanVideoPipeline,
        LTXPipeline,
        WanPipeline,
    )

    dtype = getattr(torch, dtype_str)
    lower = repo.lower()
    if "ltx" in lower:
        cls = LTXPipeline
    elif "wan" in lower:
        cls = WanPipeline
    elif "hunyuan" in lower:
        cls = HunyuanVideoPipeline
    elif "cogvideo" in lower:
        cls = CogVideoXPipeline
    else:
        raise ValueError(f"familia de video no soportada para {repo}")

    logger.info("[Factory] Cargando video %s (gpu=%s, %s)...", repo, use_gpu, dtype_str)
    # Los repos Diffusers sirven los text encoders en fp32 (el UMT5-XXL de
    # Wan: 21 GB de los ~27 del repo). Cuantizarlo a 4-bit es lo que hace
    # que Wan-1.3B quepa (~13.5 GB → ~6 GB): sin esto, una T4/Colab (10 GB
    # de RAM) nunca puede cargarlo. Los pesos 4-bit viven en GPU, así que
    # el pipeline corre sin cpu-offload (además, más rápido).
    te_quant = False
    kwargs = {"torch_dtype": dtype}
    if use_gpu and "wan" in lower:
        try:
            import bitsandbytes  # noqa: F401

            from transformers import BitsAndBytesConfig

            kwargs["text_encoder_quant_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            te_quant = True
            logger.info("[Factory] text encoder a 4-bit (nf4): ahorra ~18 GB")
        except ImportError:
            raise RuntimeError(
                "Wan en GPU requiere 'bitsandbytes' (cuantiza el text encoder "
                "de 21 GB fp32 a ~2.5 GB); instalalo y reintenta"
            )
    pipe = cls.from_pretrained(repo, **kwargs)
    # Los videos son pesados: offload siempre que exista; en CPU es lo único
    # que cabe junto con attention slicing.
    if use_gpu:
        if te_quant:
            # bnb 4-bit no soporta moverse CPU<->GPU: el pipeline entero se
            # queda en VRAM (cabe: ese es justo el objetivo de cuantizar).
            logger.info("[Factory] %s en VRAM completa (sin offload)", repo)
        else:
            pipe.enable_model_cpu_offload()
    else:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    pipe.vae.enable_tiling()
    return pipe


LOADERS = {
    "image": _load_image_pipeline,
    "video": _load_video_pipeline,
}


def _get_pipeline(spec: ModelSpec, use_gpu: bool):
    key = (spec.pipeline, spec.repo, use_gpu)
    with _LOCK:
        if key in _PIPELINES:
            return _PIPELINES[key]
        pipe = LOADERS[spec.mode](spec.repo, use_gpu, spec.dtype)
        _PIPELINES[key] = pipe
        return pipe


# ----------------------------------------------------------------------
# Ejecución bloqueante (thread del executor)
# ----------------------------------------------------------------------

class _CancelledError(RuntimeError):
    pass


def _run_image(pipe, req, spec: ModelSpec, report: Callable, cancel, steps: int):
    from PIL import Image

    def on_step_end(_p, step_index, _t, cb_kwargs):
        if cancel.is_set():
            raise _CancelledError()
        report(min(step_index + 1, steps) / steps * 88.0, "GENERATING")
        return {}

    generator = None
    if req.seed:
        import torch

        generator = torch.Generator(
            device="cuda" if _has_cuda() else "cpu"
        ).manual_seed(req.seed)

    kwargs = dict(
        prompt=req.prompt or "",
        width=spec.max_side,
        height=spec.max_side,
        num_inference_steps=steps,
        callback_on_step_end=on_step_end,
    )
    if spec.guidance_scale:
        kwargs["guidance_scale"] = spec.guidance_scale
    if (req.negative_prompt or "").strip() and "flux" not in spec.repo.lower():
        kwargs["negative_prompt"] = req.negative_prompt
    if generator is not None:
        kwargs["generator"] = generator

    result = pipe(**kwargs)
    image = result.images[0]
    # Recorte/escala al aspect ratio solicitado (width/height del job).
    target_w = req.width or spec.max_side
    target_h = req.height or spec.max_side
    if (target_w, target_h) != image.size:
        image = image.resize((max(8, target_w), max(8, target_h)), Image.LANCZOS)
    return image


def _run_video(pipe, req, spec: ModelSpec, report: Callable, cancel, steps: int):
    """Genera video real: frames decodificados del VAE + MP4 con imageio."""
    import imageio
    import numpy as np

    frames = max(8, min(int(req.num_frames or 48), 257))
    fps = max(1, int(req.fps or 16))

    def on_step_end(_p, step_index, _t, cb_kwargs):
        if cancel.is_set():
            raise _CancelledError()
        report(min(step_index + 1, steps) / steps * 88.0, "GENERATING")
        return {}

    kwargs = dict(
        prompt=req.prompt or "",
        num_frames=frames,
        num_inference_steps=steps,
        callback_on_step_end=on_step_end,
    )
    if spec.guidance_scale:
        kwargs["guidance_scale"] = spec.guidance_scale
    if (req.negative_prompt or "").strip():
        kwargs["negative_prompt"] = req.negative_prompt

    result = pipe(**kwargs)
    video_frames = result.frames[0]  # lista/array de frames RGB

    out_name = f"{req.job_id}.mp4"
    report(90.0, "POSTPROCESSING")
    with imageio.get_writer(
        out_name, fps=fps, codec="libx264", quality=7, pixelformat="yuv420p"
    ) as writer:
        for i, frame in enumerate(video_frames):
            arr = frame if isinstance(frame, np.ndarray) else np.asarray(frame)
            writer.append_data(arr)
            if i % 24 == 0:
                report(90 + (i / len(video_frames)) * 8.0, "POSTPROCESSING")
    return out_name


# ----------------------------------------------------------------------
# Generador async (puente thread → asyncio con cola)
# ----------------------------------------------------------------------

async def generate(req, spec: ModelSpec, ctx: EngineContext):
    loop = asyncio.get_running_loop()
    use_gpu = _has_cuda()
    steps = max(1, min(int(req.steps or spec.steps), 60))
    t0 = time.time()

    events: asyncio.Queue = asyncio.Queue()

    def report(pct: float, status: str, out: str = "", err: str = "") -> None:
        loop.call_soon_threadsafe(events.put_nowait, (pct, status, out, err))

    def worker() -> None:
        try:
            report(0.0, "LOADING_MODEL")
            _check_memory_before_load(spec, use_gpu)
            pipe = _get_pipeline(spec, use_gpu)
            logger.info(
                "[Factory] %s listo en %.1fs (gpu=%s, pasos=%d)",
                spec.repo, time.time() - t0, use_gpu, steps,
            )
            report(1.0, "GENERATING")

            if spec.mode == "video":
                out_name = _run_video(pipe, req, spec, report, ctx.cancel, steps)
            else:
                image = _run_image(pipe, req, spec, report, ctx.cancel, steps)
                report(90.0, "POSTPROCESSING")
                out_name = f"{req.job_id}.png"
                image.save(os.path.join(ctx.out_dir, out_name), format="PNG")

            logger.info("[Factory] %s completado en %.1fs", req.job_id, time.time() - t0)
            report(100.0, "COMPLETED", out_name)
        except _CancelledError:
            logger.info("[Factory] %s cancelado", req.job_id)
            report(0.0, "CANCELLED")
        except Exception as exc:
            logger.exception("[Factory] %s falló: %s", req.job_id, exc)
            report(0.0, "FAILED", "", f"{spec.pipeline}: {exc}")

    threading.Thread(
        target=worker, name=f"diff-{req.job_id}", daemon=True
    ).start()

    while True:
        pct, status, out, err = await events.get()
        yield (pct, status, out, err)
        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            return
