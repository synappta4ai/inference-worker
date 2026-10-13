"""Contrato común de motores y catálogo de modelos.

Un Engine convierte una MediaGenerationRequest en una secuencia de eventos
(LOADING_MODEL → GENERATING* → POSTPROCESSING → COMPLETED/FAILED) y escribe
el artefacto en `out_dir`. La cancelación es cooperativa vía asyncio.Event.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterator, Protocol

from inference_pb2 import MediaGenerationRequest  # type: ignore[import-not-found]


@dataclass
class EngineContext:
    """Entorno de ejecución pasado a cada motor."""

    out_dir: str
    cancel: asyncio.Event


@dataclass
class ModelSpec:
    """Definición declarativa de un modelo generativo.

    engine: "diffusers" (factoría genérica) | "v1_bridge" (código V1)
          | "mock" (sin pesos).
    pipeline: familia del pipeline diffusers (clave de LOADERS en la
          factoría) o punto de entrada del bridge V1.
    repo: repo de Hugging Face con los pesos.
    """

    key: str
    label: str
    engine: str
    repo: str | None
    mode: str = "image"
    pipeline: str = "sd15"
    dtype: str = "float16"
    steps: int = 20
    guidance_scale: float = 7.5
    max_side: int = 512
    # Requisitos de despliegue (catálogo dinámico / planificación)
    vram_gb: int = 8
    family: str = ""
    notes: str = ""


class Engine(Protocol):
    """Protocolo que implementan todos los motores."""

    async def generate(
        self, req: MediaGenerationRequest, ctx: EngineContext
    ) -> Iterator[tuple[float, str, str, str]]:
        """Produce eventos (pct, status, output_path, error) y escribe el output."""
        ...  # pragma: no cover


# ----------------------------------------------------------------------
# Catálogo de modelos (claves = request.model_name del gateway)
# ----------------------------------------------------------------------

MODEL_CATALOG: dict[str, ModelSpec] = {
    # ==================================================================
    # IMAGEN — factoría genérica diffusers (pesos públicos de HF)
    # ==================================================================
    # SD-Tiny: minúsculo (~100 MB) para smoke-tests sin descargas pesadas.
    "SD-Tiny-Test": ModelSpec(
        key="SD-Tiny-Test",
        label="SD Tiny (smoke-test del pipeline real)",
        engine="diffusers", repo="diffusers/tiny-stable-diffusion-torch",
        mode="image", pipeline="sd15", dtype="float32",
        steps=4, guidance_scale=7.5, max_side=64, vram_gb=2, family="sd",
    ),
    "SD-1.5": ModelSpec(
        key="SD-1.5",
        label="Stable Diffusion 1.5 (imagen real, ligera)",
        engine="diffusers", repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
        mode="image", pipeline="sd15",
        steps=20, guidance_scale=7.5, max_side=512, vram_gb=4, family="sd",
    ),
    "SDXL-Turbo": ModelSpec(
        key="SDXL-Turbo",
        label="SDXL Turbo (imagen real, 1-2 pasos)",
        engine="diffusers", repo="stabilityai/sdxl-turbo",
        mode="image", pipeline="sdxl", dtype="float16",
        steps=2, guidance_scale=0.0, max_side=512, vram_gb=8, family="sdxl",
    ),
    "SDXL-Base": ModelSpec(
        key="SDXL-Base",
        label="SDXL Base 1.0 (imagen HQ)",
        engine="diffusers", repo="stabilityai/stable-diffusion-xl-base-1.0",
        mode="image", pipeline="sdxl", dtype="float16",
        steps=30, guidance_scale=7.5, max_side=1024, vram_gb=12, family="sdxl",
    ),
    "FLUX.1-schnell": ModelSpec(
        key="FLUX.1-schnell",
        label="FLUX.1 schnell (imagen rápida, 4 pasos)",
        engine="diffusers", repo="black-forest-labs/FLUX.1-schnell",
        mode="image", pipeline="flux", dtype="bfloat16",
        steps=4, guidance_scale=0.0, max_side=1024, vram_gb=16, family="flux",
    ),
    "FLUX.1-dev": ModelSpec(
        key="FLUX.1-dev",
        label="FLUX.1 dev (imagen HQ)",
        engine="diffusers", repo="black-forest-labs/FLUX.1-dev",
        mode="image", pipeline="flux", dtype="bfloat16",
        steps=28, guidance_scale=3.5, max_side=1024, vram_gb=24, family="flux",
    ),
    "Qwen-Image": ModelSpec(
        key="Qwen-Image",
        label="Qwen-Image (imagen con texto legible)",
        engine="diffusers", repo="Qwen/Qwen-Image",
        mode="image", pipeline="qwen", dtype="bfloat16",
        steps=25, guidance_scale=4.0, max_side=1024, vram_gb=24, family="qwen",
        notes="requiere pipeline custom QwenImagePipeline",
    ),
    "HiDream-I1": ModelSpec(
        key="HiDream-I1",
        label="HiDream I1-Full (imagen HQ)",
        engine="diffusers", repo="HiDream-ai/HiDream-I1-Full",
        mode="image", pipeline="hidream", dtype="bfloat16",
        steps=28, guidance_scale=5.0, max_side=1024, vram_gb=24, family="hidream",
        notes="requiere transformers>=4.57 (qwen3_vl)",
    ),

    # ==================================================================
    # VIDEO — factoría genérica diffusers (pesos públicos de HF)
    # ==================================================================
    "LTX-Video": ModelSpec(
        key="LTX-Video",
        label="LTX-Video 0.9 (video rápido 2B)",
        engine="diffusers", repo="Lightricks/LTX-Video",
        mode="video", pipeline="ltx", dtype="bfloat16",
        steps=30, guidance_scale=3.0, max_side=704, vram_gb=12, family="ltx",
    ),
    "Wan2.1-T2V-1.3B": ModelSpec(
        key="Wan2.1-T2V-1.3B",
        label="Wan 2.1 T2V 1.3B (video ligero)",
        engine="diffusers", repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        mode="video", pipeline="wan", dtype="float16",
        # 10 era el spec fp16 del modelo entero; con el text encoder cuantizado
        # a 4-bit (pico real en VRAM) alcanza de sobra en la T4 de Colab.
        steps=30, guidance_scale=5.0, max_side=480, vram_gb=6, family="wan",
    ),
    "Wan2.1-T2V-14B": ModelSpec(
        key="Wan2.1-T2V-14B",
        label="Wan 2.1 Video (14B High Quality)",
        engine="diffusers", repo="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        mode="video", pipeline="wan", dtype="bfloat16",
        steps=40, guidance_scale=5.0, max_side=720, vram_gb=40, family="wan",
        notes="multi-GPU u offload agresivo",
    ),
    "HunyuanVideo": ModelSpec(
        key="HunyuanVideo",
        label="HunyuanVideo (video HQ)",
        engine="diffusers", repo="tencent/HunyuanVideo",
        mode="video", pipeline="hunyuan", dtype="bfloat16",
        steps=30, guidance_scale=6.0, max_side=720, vram_gb=45, family="hunyuan",
    ),
    "CogVideoX-5B": ModelSpec(
        key="CogVideoX-5B",
        label="CogVideoX 5B (video)",
        engine="diffusers", repo="THUDM/CogVideoX-5b",
        mode="video", pipeline="cogvideo", dtype="bfloat16",
        steps=6, guidance_scale=6.0, max_side=720, vram_gb=16, family="cogvideo",
    ),

    # ==================================================================
    # FAMILIAS V1 CON CÓDIGO PROPIETARIO → bridge (activar montando app/models)
    # ==================================================================
    "LTX-Video-2.5": ModelSpec(
        key="LTX-Video-2.5",
        label="LTX-2.5 22B (video+audio nativo, vía bridge V1)",
        engine="v1_bridge", repo="DeepBeepMeep/LTX-2",
        mode="video", pipeline="ltx25", dtype="bfloat16",
        steps=30, vram_gb=48, family="ltx2.5",
        notes="requiere mmgp==3.7.12 y volumen /models",
    ),
    "MiniMax-H3-High": ModelSpec(
        key="MiniMax-H3-High",
        label="MiniMax H3 (video con audio, vía bridge V1)",
        engine="v1_bridge", repo="MiniMax/H3",
        mode="video", pipeline="minimax_h3", dtype="bfloat16",
        steps=30, vram_gb=48, family="minimax",
        notes="SLA_LICENSE — código no redistribuible",
    ),
    "Kandinsky5": ModelSpec(
        key="Kandinsky5",
        label="Kandinsky 5 (vía bridge V1)",
        engine="v1_bridge", repo="ai-forever/Kandinsky5.0",
        mode="video", pipeline="kandinsky5", dtype="bfloat16",
        steps=30, vram_gb=40, family="kandinsky",
    ),
    "Krea2": ModelSpec(
        key="Krea2",
        label="Krea 2 (vía bridge V1)",
        engine="v1_bridge", repo="krea/krea-2",
        mode="image", pipeline="krea2", dtype="bfloat16",
        steps=28, vram_gb=32, family="krea",
    ),
    "LongCat-Video": ModelSpec(
        key="LongCat-Video",
        label="LongCat Video (vía bridge V1)",
        engine="v1_bridge", repo="meituan-longcat/LongCat-Video",
        mode="video", pipeline="longcat", dtype="bfloat16",
        steps=30, vram_gb=40, family="longcat",
    ),
    "Ideogram4": ModelSpec(
        key="Ideogram4",
        label="Ideogram 4 (vía bridge V1)",
        engine="v1_bridge", repo="ideogram-ai/ideogram-v4",
        mode="image", pipeline="ideogram4", dtype="bfloat16",
        steps=28, vram_gb=32, family="ideogram",
    ),
    "Z-Image": ModelSpec(
        key="Z-Image",
        label="Z-Image (vía bridge V1)",
        engine="v1_bridge", repo="tongyi/Z-Image",
        mode="image", pipeline="z_image", dtype="bfloat16",
        steps=20, vram_gb=16, family="z_image",
    ),

    # ==================================================================
    # TTS / MÚSICA — engines dedicados (fase siguiente del worker)
    # ==================================================================
    "Chatterbox-TTS": ModelSpec(
        key="Chatterbox-TTS",
        label="Chatterbox TTS (voz, clonación por referencia)",
        engine="mock", repo=None,
        mode="tts", pipeline="tts", vram_gb=6, family="tts",
        notes="engine dedicado en fase 2 del worker",
    ),
    "IndexTTS2": ModelSpec(
        key="IndexTTS2",
        label="IndexTTS2 (voz emocional)",
        engine="mock", repo=None,
        mode="tts", pipeline="tts", vram_gb=8, family="tts",
    ),
    "ACE-Step-Music": ModelSpec(
        key="ACE-Step-Music",
        label="ACE-Step (música por prompt/letra)",
        engine="mock", repo=None,
        mode="audio", pipeline="music", vram_gb=10, family="music",
    ),
    "YuE2-Music": ModelSpec(
        key="YuE2-Music",
        label="YuE2 (música completa con voz, vía bridge V1)",
        engine="v1_bridge", repo="microsoft/YuE2",
        mode="audio", pipeline="yue2", vram_gb=24, family="music",
    ),
}


def resolve(model_name: str, mode: str) -> ModelSpec:
    """Resuelve la spec de un modelo por nombre.

    Estrategia:
      1. Coincidencia exacta en el catálogo.
      2. Desconocido + modo image → el menor modelo real de imagen
         (evita descargar 57 GB por un typo en el nombre).
      3. Desconocido + resto → primer modelo real del modo pedido, o
         mock. Nunca un modelo de video 14B por accidente.
    """
    spec = MODEL_CATALOG.get(model_name or "")
    if spec is not None:
        return spec

    wanted = (mode or "").lower()
    candidates = [
        s for s in MODEL_CATALOG.values()
        if s.engine == "diffusers" and s.mode == wanted
    ]
    if not candidates:
        candidates = [s for s in MODEL_CATALOG.values() if s.engine == "diffusers"]
    if candidates:
        return min(candidates, key=lambda s: s.vram_gb)
    return MODEL_CATALOG["Wan2.1-T2V-14B"]
