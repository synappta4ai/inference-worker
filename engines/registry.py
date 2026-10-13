"""Registry de motores: resuelve la spec del modelo y delega al engine."""

from __future__ import annotations

import os

from . import diffusers_engine, diffusion_factory, mock_engine, v1_bridge
from .base import EngineContext, ModelSpec, resolve

# Interruptor de emergencia: BM_FORCE_MOCK=1 fuerza el motor mock (útil en
# tests y para diagnosticar sin descargar pesos).
FORCE_MOCK = os.environ.get("BM_FORCE_MOCK", "").lower() in ("1", "true", "yes")

# BM_USE_DIFFUSERS_ENGINE=1 usa el motor diffusers monolítico original
# (deprecado en favor de la factoría genérica).
LEGACY_ENGINE = os.environ.get("BM_USE_DIFFUSERS_ENGINE", "").lower() in (
    "1", "true", "yes",
)


def list_models() -> list[dict]:
    """Catálogo completo (fuente única de verdad: engines/base.py)."""
    from .base import MODEL_CATALOG
    return [
        {
            "id": spec.key,
            "name": spec.label,
            "family": spec.family,
            "mode": spec.mode,
            "engine": spec.engine,
            "repo": spec.repo or "",
            "steps": spec.steps,
            "guidance": spec.guidance_scale,
            "size": spec.max_side,
            "dtype": spec.dtype,
            "pipeline": spec.pipeline,
            "vram_gb": spec.vram_gb,
            "notes": spec.notes,
        }
        for spec in MODEL_CATALOG.values()
    ]


async def run_generation(req, ctx: EngineContext):
    """Ejecuta la generación con el motor correspondiente a la spec."""
    spec: ModelSpec = resolve(req.model_name, req.mode)

    if FORCE_MOCK:
        engine = mock_engine
    elif spec.engine == "v1":
        engine = v1_bridge
    elif LEGACY_ENGINE or (spec.engine == "diffusers" and spec.mode == "image"
                            and spec.pipeline == "sd15"):
        engine = diffusers_engine
    else:
        engine = diffusion_factory

    async for ev in engine.generate(req, spec, ctx):
        yield ev
