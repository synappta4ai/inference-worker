"""Test opt-in del motor diffusers REAL.

Descarga ~100 MB del modelo tiny de HF, por lo que solo corre cuando
BM_TEST_REAL=1:

    BM_TEST_REAL=1 py -3 -m pytest tests/test_diffusers_real.py -v
"""

import asyncio
import os

import inference_pb2
import pytest

from engines import diffusers_engine
from engines.base import EngineContext, MODEL_CATALOG

pytestmark = pytest.mark.skipif(
    os.environ.get("BM_TEST_REAL", "").lower() not in ("1", "true", "yes"),
    reason="requiere BM_TEST_REAL=1 (descarga pesos de Hugging Face)",
)


async def test_real_diffusion_pipeline(tmp_path):
    spec = MODEL_CATALOG["SD-Tiny-Test"]
    req = inference_pb2.MediaGenerationRequest(
        job_id="real1", mode="image", model_name="SD-Tiny-Test",
        prompt="a green tree", steps=2,
    )
    ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())

    events = []
    async for evt in diffusers_engine.generate(req, spec, ctx):
        events.append(evt)

    statuses = [e[1] for e in events]
    assert "LOADING_MODEL" in statuses
    assert "GENERATING" in statuses
    assert statuses[-1] == "COMPLETED"
    assert events[-1][2] == "real1.png"

    png = tmp_path / "real1.png"
    assert png.exists()
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    # Imagen real: bastante más grande que el placeholder de 73 bytes.
    assert len(data) > 500


async def test_real_pipeline_respects_cancellation(tmp_path):
    spec = MODEL_CATALOG["SD-Tiny-Test"]
    req = inference_pb2.MediaGenerationRequest(
        job_id="realc", mode="image", model_name="SD-Tiny-Test",
        prompt="a rock", steps=40,
    )
    ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())

    async def cancel_soon():
        await asyncio.sleep(0.2)
        ctx.cancel.set()

    task = asyncio.ensure_future(cancel_soon())
    events = []
    async for evt in diffusers_engine.generate(req, spec, ctx):
        events.append(evt)
    await task

    assert events[-1][1] == "CANCELLED"
    assert not (tmp_path / "realc.png").exists()
