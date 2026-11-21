"""Registry de motores: resuelve la spec del modelo y delega al engine."""

from __future__ import annotations

import os
from types import ModuleType

from . import diffusion_factory, mock_engine, v1_bridge
from .base import EngineContext, ModelSpec, resolve

# Interruptor de emergencia: BM_FORCE_MOCK=1 fuerza el motor mock (útil en
# tests y para diagnosticar sin descargar pesos).
FORCE_MOCK = os.environ.get("BM_FORCE_MOCK", "").lower() in ("1", "true", "yes")

# Un handler por cada valor declarable en `spec.engine` (model_catalog.yaml y
# BUILTIN_CATALOG, validados contra _ENGINES en base.py). Las claves son los
# nombres reales del catálogo: el dispatch anterior comparaba contra "v1" y
# dejaba sin ruta a los ocho modelos "v1_bridge".
ENGINE_HANDLERS: dict[str, ModuleType] = {
    "diffusers": diffusion_factory,
    "v1_bridge": v1_bridge,
    "mock": mock_engine,
}


def engine_for(spec: ModelSpec) -> ModuleType:
    """Handler del engine declarado por la spec.

    Antes todo engine caía en `diffusion_factory`, donde los modelos mock de
    TTS/música morían con `KeyError: 'tts'` (su tabla LOADERS no tiene esos
    modes). Un engine sin handler falla explícito acá, con modelo y engine.
    """
    handler = ENGINE_HANDLERS.get(spec.engine)
    if handler is None:
        raise ValueError(
            f"engine sin handler: {spec.engine!r} (modelo {spec.key!r}); "
            f"handlers disponibles: {sorted(ENGINE_HANDLERS)}"
        )
    return handler


def list_models() -> list[dict]:
    """Catálogo completo (fuente única de verdad: model_catalog.yaml)."""
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

    engine = mock_engine if FORCE_MOCK else engine_for(spec)

    async for ev in engine.generate(req, spec, ctx):
        yield ev
