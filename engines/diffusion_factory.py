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
import functools
import gc
import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
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


# RAM extra mientras se cargan los shards (intérprete, tensores temporales,
# buffers de activación). No es el modelo: es el costo fijo de cargarlo.
_RAM_OVERHEAD_GB = 1.5


def _load_need_gb(spec: ModelSpec, use_gpu: bool) -> tuple[float, str, float]:
    """Cuánta RAM necesita el modelo al cargar, medido con sus pesos REALES.

    El catálogo declara VRAM estimada (Wan-1.3B dice 6 GB), que en CPU no
    significa nada: sus safetensors pesan ~27 GB y el pipeline vive entero en
    RAM. Antes el guardia comparaba 6 x 1.5 = 9 GB contra 25 GB "libres" y
    dejaba cargar hasta que el sistema mataba el proceso a mitad de un shard.

    Devuelve (GB necesarios, de dónde sale el número, GB de pesos en disco).
    """
    try:
        from model_ops import weights_size_bytes  # tardío: evita ciclo de imports

        weights_gb = weights_size_bytes(spec.repo) / 1024**3
    except Exception:  # noqa: BLE001 — sin medida real se usa el catálogo
        weights_gb = 0.0
    if weights_gb <= 0:
        # Sin pesos en disco no hay nada que medir: vuelve la estimación
        # declarada (sólo sirve como orden de magnitud).
        return spec.vram_gb * (1.1 if use_gpu else 1.5), "catálogo", 0.0
    if use_gpu:
        # En GPU los pesos terminan en VRAM (y Wan cuantiza su text encoder a
        # 4-bit): en RAM queda el pico de carga, ~60 % del disco (fp32 → fp16).
        need = max(weights_gb * 0.6, spec.vram_gb * 1.1) + _RAM_OVERHEAD_GB
        return need, "disco", weights_gb
    # CPU: todo el pipeline se materializa en RAM.
    return weights_gb + _RAM_OVERHEAD_GB, "disco", weights_gb


def _check_memory_before_load(spec: ModelSpec, use_gpu: bool) -> None:
    """Rechaza el modelo ANTES de cargar si no entra en la máquina.

    Sin este guardia, un modelo 14B en una laptop revienta el proceso con un
    abort de Rust (OOM no capturable) y tumba el worker entero. Corre dentro
    del hijo: si aun así se quedara sin memoria, muere el hijo y no el worker.

    Si no hay espacio, primero suelta pipelines cacheados (un SD-1.5 quedado
    de un job anterior puede ocupar los GB que este modelo necesita) y vuelve
    a medir; solo falla si aún así no cabe.
    """
    try:
        import psutil

        avail_gb = psutil.virtual_memory().available / 1024**3
    except Exception:
        return
    need_gb, basis, weights_gb = _load_need_gb(spec, use_gpu)
    if avail_gb >= need_gb:
        return
    _release_pipelines()
    try:
        avail_gb = psutil.virtual_memory().available / 1024**3
    except Exception:
        return
    if avail_gb >= need_gb:
        return
    detalle = (
        f"{weights_gb:.1f} GB de pesos en disco"
        if basis == "disco"
        else f"estimación del catálogo: ~{spec.vram_gb} GB"
    )
    raise MemoryError(
        f"el modelo '{spec.label}' no entra en esta máquina: necesita "
        f"~{need_gb:.1f} GB de RAM para cargar ({detalle}) y solo hay "
        f"{avail_gb:.1f} GB libres. Probá un modelo más chico, liberá memoria "
        "o usá una máquina con más memoria."
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
    # Sin pesos en disco no se genera: la descarga se pide desde /gpu
    # (modo personal) en vez de saturar la máquina por sorpresa.
    from model_ops import require_weights  # tardío: evita ciclo de imports

    require_weights(repo)
    pipe = cls.from_pretrained(repo, torch_dtype=dtype)
    if use_gpu:
        pipe.enable_model_cpu_offload()
    else:
        pipe.enable_attention_slicing()
    return pipe


def _enable_vae_opt(pipe, *names: str) -> None:
    """Activa slicing/tiling del VAE, esté donde esté el método.

    No todos los pipelines exponen `enable_vae_slicing` (WanPipeline solo tiene
    el del VAE: `pipe.vae.enable_slicing`), y en CPU es la única forma de que
    quepa un video. Es una optimización: si no existe, se genera igual.
    """
    for target in (getattr(pipe, "vae", None), pipe):
        if target is None:
            continue
        for name in names:
            fn = getattr(target, name, None)
            if not callable(fn):
                continue
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 — optimización best-effort
                logger.warning("[Factory] no pude activar %s: %s", name, exc)
            return


def _offload_sequentially(pipe, dtype=None) -> None:
    """Uso secuencial de la VRAM: TE, transformer y VAE nunca conviven.

    Medido en la RTX 5070 (8 GB): con el pipeline entero residente el driver
    paginaba pesos a RAM del sistema (0 GB físicos libres) y un paso de
    denoising tardaba 141 s en vez de 0,8 s. El text encoder 4-bit solo ya pide
    5,5 GB, así que tampoco alcanza para tenerlo residente junto al resto.

    El transformer (2,65 GB en fp16) y el VAE entran en VRAM con hooks de
    accelerate y vuelven a CPU al terminar (`release_vram`); el text encoder se
    sube para codificar el prompt y baja dentro del mismo `encode_prompt`.
    """
    import torch
    from accelerate import cpu_offload_with_hook

    # El VAE se llama con `.decode()`, así que su hook va en el módulo entero
    # (diffusers aplica `apply_forward_hook` ahí).
    cpu_offload_with_hook(pipe.transformer, "cuda")
    cpu_offload_with_hook(pipe.vae, "cuda")

    te = getattr(pipe, "text_encoder", None)
    if te is None:
        return
    # bnb cuantiza 144 capas a 4-bit pero deja el embedding (1,96 GB) y las FFN
    # `wo` en fp32 (24 x 0,16 GB; UMT5 las marca `_keep_in_fp32_modules`).
    # `te.to(dtype)` lo bloquea bnb, así que se castea parámetro por parámetro.
    destino = dtype or torch.float16
    for prm in te.parameters():
        if prm.dtype == torch.float32:
            prm.data = prm.data.to(destino)

    _wrap_encode_prompt(pipe, te, torch)

    # El primer forward del TE paga ~40 s de kernels de bitsandbytes (el
    # segundo tarda 0,4 s). Se hace acá, mientras la carga ya está pagando la
    # cuantización, y se devuelve el TE a RAM para no dejar 5,5 GB ociosos.
    _warmup_text_encoder(pipe, te, torch)
    te.to("cpu")
    torch.cuda.empty_cache()


def _wrap_encode_prompt(pipe, te, torch) -> None:
    """Sube el TE para codificar y lo baja dentro del mismo `encode_prompt`.

    No alcanza con moverlo antes del `pipe(...)`: el encode ocurre dentro, y el
    TE (5,5 GB en 4-bit) no entra en VRAM junto al transformer. Por eso se
    envuelve el método del pipeline en vez de duplicar acá su firma, que cambia
    entre familias de video.
    """
    if getattr(pipe, "_bm_encode_wrapped", False):
        return
    original = pipe.encode_prompt

    @functools.wraps(original)
    def encode_then_offload(*args, **kwargs):
        t0 = time.time()
        te.to("cuda")
        t1 = time.time()
        try:
            out = original(*args, **kwargs)
        finally:
            t2 = time.time()
            te.to("cpu")
            torch.cuda.empty_cache()
        logger.info(
            "[Factory] encode del prompt en %.1f s (subir %.1f, forward %.1f, "
            "bajar %.1f)", time.time() - t0, t1 - t0, t2 - t1, time.time() - t2
        )
        return out

    pipe.encode_prompt = encode_then_offload
    pipe._bm_encode_wrapped = True


def _warmup_text_encoder(pipe, te, torch) -> None:
    """Adelanta el primer forward del text encoder durante la carga.

    Medido en la RTX 5070: ese forward cuesta ~40 s (init de kernels de
    bitsandbytes) y los siguientes 0,4 s. Hacerlo al cargar deja el primer job
    con el mismo tiempo que los demás. Es un adelanto, no un requisito: si el
    pipeline no expone tokenizer o el forward falla, se sigue igual.
    """
    tokenizer = getattr(pipe, "tokenizer", None)
    if not callable(tokenizer):
        return
    t0 = time.time()
    try:
        largos = [64]
        maximo = getattr(tokenizer, "model_max_length", 64)
        if isinstance(maximo, int) and 0 < maximo < 10**6:
            largos.append(maximo)
        ids = tokenizer(
            ["calentamiento"],
            padding="max_length",
            max_length=min(largos),
            truncation=True,
            return_tensors="pt",
        ).input_ids.to("cuda")
        with torch.no_grad():
            te(ids)
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.warning("[Factory] calentamiento del text encoder falló: %s", exc)
        return
    logger.info("[Factory] text encoder caliente en %.1f s", time.time() - t0)


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
    wan = "wan" in lower
    if use_gpu and wan and dtype_str != "bfloat16":
        # Wan en fp16 satura al decodificar clips largos (5 bits de exponente
        # contra los 8 de bf16) y el video sale lavado. En Blackwell bf16 corre
        # a la misma velocidad, así que no cuesta tiempo.
        dtype = torch.bfloat16
        logger.info("[Factory] Wan en GPU con bfloat16 (pedido %s)", dtype_str)
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
    kwargs = {"torch_dtype": dtype}
    if use_gpu and "wan" in lower:
        # El UMT5-XXL en fp16 pide ~10,6 GB: sin 4-bit Wan-1.3B no entra en 8.
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Wan en GPU requiere 'bitsandbytes' para cuantizar el text "
                "encoder (21 GB fp32 -> ~2,5 GB); instalalo y reintenta"
            ) from exc
        # diffusers >= 0.34 cuantiza por componente con
        # PipelineQuantizationConfig (`text_encoder_quant_config` no existe).
        from diffusers.quantizers import PipelineQuantizationConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": dtype,
            },
            components_to_quantize=["text_encoder"],
        )
        logger.info("[Factory] text encoder a 4-bit (nf4): ahorra ~18 GB")

    from model_ops import require_weights  # tardío: evita ciclo de imports

    require_weights(repo)
    pipe = cls.from_pretrained(repo, **kwargs)
    if use_gpu:
        # La cuantización deja ~5 GB de área de trabajo en el allocator: si no
        # se devuelve, el job arranca paginando pesos a RAM del sistema.
        torch.cuda.empty_cache()
        if wan:
            # El VAE en fp16 desborda al decodificar un clip largo: los latentes
            # de 81 frames tienen picos fuera de rango y el frame sale lavado.
            # En fp32 son ~0,5 GB y va offloadeado, así que no quita VRAM.
            vae = getattr(pipe, "vae", None)
            if vae is not None:
                vae.to(torch.float32)
                logger.info("[Factory] VAE de %s en fp32 para el decode", repo)
        if "quantization_config" in kwargs:
            _offload_sequentially(pipe, dtype)
            logger.info(
                "[Factory] %s por etapas: text encoder 4-bit, transformer y VAE "
                "entran y salen de VRAM",
                repo,
            )
        else:
            pipe.enable_model_cpu_offload()
    else:
        pipe.enable_attention_slicing()
        _enable_vae_opt(pipe, "enable_slicing", "enable_vae_slicing")
    # Tiling también en GPU: sin él el decode de 81 frames se va de los 8 GB y
    # el driver pagina (medido: 16 min sin devolver un solo frame, GPU al 100 %
    # pero a 33 W). Se pierde algo de detalle en las costuras, pero es lo único
    # que entra en una laptop de 8 GB.
    _enable_vae_opt(pipe, "enable_tiling", "enable_vae_tiling")
    return pipe


LOADERS = {
    "image": _load_image_pipeline,
    "video": _load_video_pipeline,
}


def _pipeline_loaded(spec: ModelSpec, use_gpu: bool) -> bool:
    """true si este proceso ya tiene el pipeline cargado y cacheado."""
    with _LOCK:
        return (spec.pipeline, spec.repo, use_gpu) in _PIPELINES


def release_vram(pipe) -> None:
    """Devuelve a RAM las componentes con hook de offload al terminar un job.

    Sin esto el transformer y el VAE quedan en VRAM entre jobs (el pipeline
    sigue cacheado en RAM, que es barato de volver a subir).
    """
    for module in list(getattr(pipe, "components", {}).values()):
        hook = getattr(module, "_hf_hook", None)
        if hook is None:
            continue
        try:
            # Hay dos tipos de hook: los de `cpu_offload_with_hook`
            # (`UserCpuOffloadHook`, con `offload()`) y el `CpuOffload` de
            # `enable_model_cpu_offload`, que sólo sabe volver a CPU por
            # `init_hook(module)`.
            apagar = getattr(hook, "offload", None)
            if callable(apagar):
                apagar()
            else:
                iniciar = getattr(hook, "init_hook", None)
                if callable(iniciar):
                    iniciar(module)
        except Exception:  # noqa: BLE001 — limpieza best-effort
            pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _get_pipeline(spec: ModelSpec, use_gpu: bool):
    # Un mode fuera de LOADERS es una spec mal declarada (engine diffusers con
    # un modo que no es de difusión): mejor decirlo acá, con modelo y modo, que
    # dejar que reviente el dict. Se valida antes de la caché para que el error
    # no dependa de si el pipeline ya estaba cargado.
    _validate_mode(spec)
    loader = LOADERS[spec.mode]
    key = (spec.pipeline, spec.repo, use_gpu)
    with _LOCK:
        if key in _PIPELINES:
            return _PIPELINES[key]
        pipe = loader(spec.repo, use_gpu, spec.dtype)
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


# Los pipelines de video exigen lados múltiplos de 32 (LTX) o 16 (Wan):
# redondear a 32 cumple las dos familias.
_VIDEO_DIM_STEP = 32


def _round_to_step(value: float, step: int = _VIDEO_DIM_STEP) -> int:
    return max(step, int(round(value / step)) * step)


# Negative prompt oficial de Wan. Sin él el sampler no tiene nada que empuje
# fuera de "imagen plana": los clips salen lavados (brillo medio 201/255) y
# casi estáticos. Solo se usa cuando el job no manda uno propio.
_WAN_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def _video_frames(value) -> int:
    """`num_frames` en 4k+1, como espera el VAE temporal de Wan.

    Comprime el tiempo 4:1 y agrega el frame inicial: con 80 (múltiplo de 16
    "redondo") el último grupo de latentes queda incompleto y se decodifican
    frames de relleno. 257 es el máximo que banca el modelo.
    """
    frames = max(8, min(int(value or 48), 257))
    return max(5, min(4 * int(round((frames - 1) / 4)) + 1, 257))


def _video_dims(req, spec: ModelSpec) -> tuple[int, int]:
    """Dimensiones del video a partir del ratio pedido (width/height del job).

    El lado corto cae en el `max_side` del modelo y el largo se deriva del
    aspecto pedido, ambos en múltiplos de 32. Sin esto el pipeline usa su
    default (832x480 en Wan) y un pedido 9:16 salía horizontal. Con max_side
    480: 9:16 -> 480x832, 16:9 -> 832x480, 1:1 -> 480x480, 4:3 -> 640x480.
    """
    short = _round_to_step(float(getattr(spec, "max_side", 0) or 512))
    target_w = int(getattr(req, "width", 0) or 0)
    target_h = int(getattr(req, "height", 0) or 0)
    if target_w <= 0 or target_h <= 0:
        # Sin ratio explícito: cuadrado del lado que el modelo banca.
        return short, short
    if target_w <= target_h:
        return short, _round_to_step(short * target_h / target_w)
    return _round_to_step(short * target_w / target_h), short


def _run_video(pipe, req, spec: ModelSpec, out_dir: str, report: Callable,
               cancel, steps: int):
    """Genera video real: frames decodificados del VAE + MP4 con imageio."""
    import imageio
    import numpy as np

    frames = _video_frames(req.num_frames)
    fps = max(1, int(req.fps or 16))

    marca = {"t": time.time()}

    def on_step_end(_p, step_index, _t, cb_kwargs):
        if cancel.is_set():
            raise _CancelledError()
        ahora = time.time()
        logger.info("[Factory] paso %s/%s en %.1f s", step_index + 1, steps,
                    ahora - marca["t"])
        marca["t"] = ahora
        report(min(step_index + 1, steps) / steps * 88.0, "GENERATING")
        return {}

    width, height = _video_dims(req, spec)
    logger.info(
        "[Factory] video %sx%s con %s frames (pedido %sx%s, max_side %s)",
        width, height, frames, getattr(req, "width", 0) or "?",
        getattr(req, "height", 0) or "?", getattr(spec, "max_side", 0) or "?",
    )

    kwargs = dict(
        prompt=req.prompt or "",
        width=width,
        height=height,
        num_frames=frames,
        num_inference_steps=steps,
        callback_on_step_end=on_step_end,
    )
    if spec.guidance_scale:
        kwargs["guidance_scale"] = spec.guidance_scale
    negative = (req.negative_prompt or "").strip()
    if not negative and "wan" in str(getattr(spec, "repo", "")).lower():
        negative = _WAN_NEGATIVE_PROMPT
        logger.info("[Factory] negative prompt por defecto de Wan")
    if negative:
        kwargs["negative_prompt"] = negative
    if req.seed:
        # Misma seed, mismo clip: sin esto cada job arranca de otros latentes.
        import torch

        kwargs["generator"] = torch.Generator(
            device="cuda" if _has_cuda() else "cpu"
        ).manual_seed(req.seed)

    t_pipe = time.time()
    result = pipe(**kwargs)
    logger.info("[Factory] pipeline completo en %.1f s", time.time() - t_pipe)
    video_frames = result.frames[0]  # lista/array de frames RGB

    out_name = f"{req.job_id}.mp4"
    report(90.0, "POSTPROCESSING")
    # Igual que la imagen: el archivo va al out_dir del motor (el que sirve el
    # artifact server) y el evento devuelve el nombre relativo.
    with imageio.get_writer(
        os.path.join(out_dir, out_name),
        fps=fps, codec="libx264", quality=7, pixelformat="yuv420p",
    ) as writer:
        for i, frame in enumerate(video_frames):
            arr = frame if isinstance(frame, np.ndarray) else np.asarray(frame)
            writer.append_data(arr)
            if i % 24 == 0:
                report(90 + (i / len(video_frames)) * 8.0, "POSTPROCESSING")
    return out_name


# ----------------------------------------------------------------------
# Supervisión: un proceso hijo por modelo (aislamiento de memoria)
# ----------------------------------------------------------------------
#
# Un pipeline grande vive en `engines/worker_host.py`, nunca acá. Motivo: si
# un modelo agota la RAM, el SO mata el proceso de golpe (no hay excepción que
# atrapar) y antes eso se llevaba al worker entero. En un hijo, ese final es un
# job FAILED y el servicio sigue vivo. Al morir el hijo, el SO recupera toda su
# memoria de inmediato: `gc.collect()` dentro del worker dejaba páginas que el
# proceso no devolvía y la siguiente carga de Wan se quedaba sin RAM.

_WORKER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Lista (no string) para que los tests puedan sustituir el intérprete por un
# hijo falso y ejercitar la supervisión sin cargar pesos reales.
_HOST_CMD = [sys.executable, "-u", "-m", "engines.worker_host"]

# repo → hijo vivo. Un solo modelo cargado a la vez: con dos de ~27 GB la
# máquina se queda sin RAM aunque "quepan" por separado.
_HOSTS: dict[str, "_Host"] = {}
_HOST_LOCK = threading.Lock()

_CRASH_ERR = (
    "el proceso de generación se cerró de golpe con '{label}' (lo más "
    "probable: se quedó sin RAM). El worker sigue funcionando: probá un "
    "modelo más liviano o liberá memoria y reintentá."
)


class _Host:
    """Proceso hijo que carga un modelo y ejecuta jobs de a uno.

    El hijo cachea su pipeline entre jobs (misma ventaja de antes), pero su
    memoria es suya: si se cae, se descarta el proceso y el worker sobrevive.
    """

    def __init__(self, key: str, payload: dict) -> None:
        self.key = key
        self.label = str(payload.get("label") or payload.get("key") or key)
        self.proc = subprocess.Popen(
            list(_HOST_CMD) + [json.dumps(payload), "auto"],
            cwd=_WORKER_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr heredado: los warnings de diffusers y el tqdm siguen
            # cayendo en worker.log, y no ensucian el canal JSON.
            stderr=None,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1",
                 "PYTHONIOENCODING": "utf-8"},
        )
        self._queue = None
        self._loop = None
        self._job_open = True   # reservado desde que existe
        self._terminal = False
        self._lock = None
        threading.Thread(
            target=self._read_stdout, name=f"host-{key}", daemon=True
        ).start()

    # -- estado -------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    @property
    def idle(self) -> bool:
        return not self._job_open

    @property
    def finished(self) -> bool:
        return self._terminal

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # -- E/S ----------------------------------------------------------
    def _deliver(self, evt) -> None:
        loop, queue = self._loop, self._queue
        if loop is None or queue is None:
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, evt)
        except RuntimeError:  # loop cerrado (apagando el worker)
            pass

    def _read_stdout(self) -> None:
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    logger.debug("[Host %s] salida no-JSON: %s", self.key, line[:200])
                    continue
                kind = msg.get("type")
                if kind == "log":
                    logger.info("[Host %s] %s", self.key, msg.get("msg", ""))
                    continue
                if kind != "event":
                    continue
                evt = (
                    float(msg.get("pct") or 0.0),
                    str(msg.get("status") or ""),
                    str(msg.get("out") or ""),
                    str(msg.get("err") or ""),
                )
                if evt[1] in ("COMPLETED", "FAILED", "CANCELLED"):
                    self._terminal = True
                self._deliver(evt)
        finally:
            # EOF: el hijo murió (OOM del sistema, crash nativo, kill).
            if self._job_open and not self._terminal:
                self._terminal = True
                self._deliver(
                    (0.0, "FAILED", "", _CRASH_ERR.format(label=self.label))
                )
            logger.info("[Host %s] proceso terminó (código %s)",
                        self.key, self.proc.poll())

    def send(self, msg: dict) -> None:
        if not self.alive or self.proc.stdin is None:
            raise OSError("el proceso de generación ya no está vivo")
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def cancel(self) -> None:
        try:
            self.send({"cmd": "cancel"})
        except OSError:
            pass

    def kill(self) -> None:
        if not self.alive:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass

    # -- ciclo de vida de un job --------------------------------------
    def attach(self, loop, queue) -> None:
        self._loop, self._queue = loop, queue
        self._terminal = False
        self._job_open = True

    def finish(self) -> None:
        self._job_open = False
        self._loop, self._queue = None, None


def _host_for(spec: ModelSpec) -> "_Host":
    """Hijo del modelo `spec`, matando a los ociosos de otros modelos."""
    key = spec.repo or spec.key
    with _HOST_LOCK:
        host = _HOSTS.get(key)
        if host is not None and host.alive and not host.idle:
            return host
        if host is not None and host.alive:
            return host
        _HOSTS.pop(key, None)
        for other_key, other in list(_HOSTS.items()):
            if other.idle:
                other.kill()
                _HOSTS.pop(other_key, None)
        host = _Host(key, asdict(spec))
        _HOSTS[key] = host
        return host


def active_hosts() -> list[str]:
    """Repos cargados ahora mismo (telemetría y tests)."""
    with _HOST_LOCK:
        return [key for key, host in _HOSTS.items() if host.alive]


def _validate_mode(spec: ModelSpec) -> None:
    """Un `mode` fuera de LOADERS es una spec mal declarada (engine diffusers
    con un modo que no es de difusión): se dice acá, con modelo y modo."""
    if spec.mode not in LOADERS:
        raise ValueError(
            f"modo sin loader diffusers: {spec.mode!r} (modelo {spec.key!r}); "
            f"modes soportados: {sorted(LOADERS)}"
        )


def _request_payload(req) -> dict:
    """Campos que necesitan los runners, serializables en JSON."""
    return {
        "job_id": str(getattr(req, "job_id", "") or ""),
        "prompt": str(getattr(req, "prompt", "") or ""),
        "negative_prompt": str(getattr(req, "negative_prompt", "") or ""),
        "width": int(getattr(req, "width", 0) or 0),
        "height": int(getattr(req, "height", 0) or 0),
        "num_frames": int(getattr(req, "num_frames", 0) or 0),
        "fps": int(getattr(req, "fps", 0) or 0),
        "seed": int(getattr(req, "seed", 0) or 0),
    }


# ----------------------------------------------------------------------
# Generador async: puente cola del hijo → evento gRPC
# Umbral (GB) a partir del cual un modelo pesa demasiado para convivir con
# otro: dos pipelines de ~27 GB (Wan, FLUX, SDXL) no entran juntos en una
# máquina de 32 GB y el SO terminaría matando a los dos hijos. Los livianos
# (SD-Tiny ~2 GB, SD-1.5 ~4 GB) sí pueden correr en paralelo.
_HEAVY_GB = 6

# Compuerta de un solo carril para los modelos pesados. Se ata al loop en uso
# (el worker tiene uno; los tests abren uno por caso).
_GATE: tuple[object, asyncio.Semaphore] | None = None


def _heavy(spec: ModelSpec) -> bool:
    """true si el modelo, cargado, ocupa RAM como para no convivir con otro."""
    return int(spec.vram_gb or 0) >= _HEAVY_GB


def _heavy_gate() -> asyncio.Semaphore:
    global _GATE
    loop = asyncio.get_running_loop()
    if _GATE is None or _GATE[0] is not loop:
        _GATE = (loop, asyncio.Semaphore(1))
    return _GATE[1]


# ----------------------------------------------------------------------
# Generador async: puente cola del hijo → evento gRPC
# ----------------------------------------------------------------------

async def _stream_from_host(req, spec: ModelSpec, ctx: EngineContext):
    """Pide el job al hijo del modelo y traduce su salida a eventos gRPC."""
    loop = asyncio.get_running_loop()
    steps = max(1, min(int(req.steps or spec.steps), 60))
    t0 = time.time()

    try:
        host = _host_for(spec)
    except OSError as exc:
        logger.exception("[Factory] no pude arrancar el hijo de %s", spec.repo)
        yield (0.0, "FAILED", "", f"no pude arrancar el proceso de generación: {exc}")
        return

    queue: asyncio.Queue = asyncio.Queue()
    async with host.lock:
        host.attach(loop, queue)
        cancel_sent = False
        try:
            try:
                host.send({
                    "cmd": "job",
                    "steps": steps,
                    "out_dir": ctx.out_dir,
                    "request": _request_payload(req),
                })
            except OSError:
                yield (0.0, "FAILED", "", _CRASH_ERR.format(label=spec.label))
                return

            while True:
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    # CancelGeneration del gateway: se traduce al hijo, que
                    # corta en el próximo callback de paso.
                    if ctx.cancel.is_set() and not cancel_sent:
                        host.cancel()
                        cancel_sent = True
                    continue
                yield evt
                if evt[1] in ("COMPLETED", "FAILED", "CANCELLED"):
                    logger.info(
                        "[Factory] %s %s en %.1fs", req.job_id,
                        "completado" if evt[1] == "COMPLETED" else evt[1].lower(),
                        time.time() - t0,
                    )
                    return
        finally:
            host.finish()
            # Cliente desconectado o consumo cortado: el hijo no puede seguir
            # generando solo (y ocupando RAM) sin que nadie escuche.
            if not host.finished and host.alive:
                host.cancel()


async def generate(req, spec: ModelSpec, ctx: EngineContext):
    _validate_mode(spec)  # falla antes de arrancar un proceso

    gate = _heavy_gate() if _heavy(spec) else None
    if gate is not None:
        if gate.locked():
            # Encolado detrás de otro modelo pesado: se emite LOADING_MODEL
            # para que el front no muestre un job mudo. Sólo espera ESTE job:
            # el loop sigue sirviendo ListModels, telemetría y cancelaciones.
            yield (0.0, "LOADING_MODEL")
        while True:
            if ctx.cancel.is_set():
                yield (0.0, "CANCELLED")
                return
            try:
                await asyncio.wait_for(gate.acquire(), timeout=0.2)
                break
            except asyncio.TimeoutError:
                continue
        if ctx.cancel.is_set():
            gate.release()
            yield (0.0, "CANCELLED")
            return

    try:
        async for evt in _stream_from_host(req, spec, ctx):
            yield evt
    finally:
        # También al cerrar el generador (cliente desconectado): el carril se
        # libera siempre, sin dejar pesados esperando para siempre.
        if gate is not None:
            gate.release()
