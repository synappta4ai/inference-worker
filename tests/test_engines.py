"""Tests del registry de motores y del catálogo de modelos."""

import pytest

from engines import registry
from engines.base import MODEL_CATALOG, ModelSpec, resolve


class TestCatalog:

    def test_catalog_has_real_image_models(self):
        real = [
            s
            for s in MODEL_CATALOG.values()
            if s.engine == "diffusers" and s.mode == "image"
        ]
        assert len(real) >= 5
        assert all(s.repo for s in real)

    def test_catalog_has_real_video_models(self):
        video = [
            s for s in MODEL_CATALOG.values()
            if s.mode == "video" and s.engine == "diffusers"
        ]
        assert len(video) >= 3
        # Wan-1.3B queda en 6 tras cuantizar su text encoder a 4-bit (era 10);
        # los demás videos reales siguen pesando >= 12.
        assert all(s.vram_gb >= 6 for s in video)

    def test_every_spec_declares_pipeline_and_family(self):
        for key, spec in MODEL_CATALOG.items():
            assert spec.pipeline, f"{key} sin pipeline"
            assert spec.family, f"{key} sin familia"
            if spec.engine != "mock":
                assert spec.repo, f"{key} sin repo"

    def test_v1_bridge_families_declared(self):
        bridge = [s for s in MODEL_CATALOG.values() if s.engine == "v1_bridge"]
        assert len(bridge) >= 6  # ltx25, minimax, kandinsky, krea, longcat, z_image...

    def test_resolve_known_model(self):
        spec = resolve("SD-1.5", "image")
        assert spec.engine == "diffusers"
        assert "stable-diffusion" in spec.repo

    def test_resolve_unknown_falls_back_by_mode(self):
        spec = resolve("modelo-inexistente", "image")
        assert spec.engine == "diffusers"  # fallback a un motor real de imagen

        spec_v = resolve("modelo-inexistente", "video")
        assert spec_v.engine == "diffusers"  # el video ya tiene motores reales

    def test_resolve_empty_name(self):
        spec = resolve("", "video")
        assert spec.mode == "video"


class TestRegistry:

    async def test_mock_engine_produces_completed_event(self, tmp_path):
        import asyncio

        import inference_pb2

        req = inference_pb2.MediaGenerationRequest(
            job_id="reg1", mode="video", model_name="Wan2.1-T2V-14B",
            prompt="x", steps=3,
        )
        from engines.base import EngineContext

        ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())
        events = []
        async for evt in registry.run_generation(req, ctx):
            events.append(evt)

        statuses = [e[1] for e in events]
        assert statuses[-1] == "COMPLETED"
        assert events[-1][2] == "reg1.mp4"
        assert (tmp_path / "reg1.mp4").exists()

    async def test_force_mock_env_overrides_diffusers(self, tmp_path, monkeypatch):
        import asyncio

        import inference_pb2
        from engines.base import EngineContext

        monkeypatch.setattr(registry, "FORCE_MOCK", True)
        req = inference_pb2.MediaGenerationRequest(
            job_id="reg2", mode="image", model_name="SD-1.5",
            prompt="x", steps=2,
        )
        ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())
        events = []
        async for evt in registry.run_generation(req, ctx):
            events.append(evt)

        # Con mock forzado, el output es un PNG placeholder (no pasa por diffusers).
        assert events[-1][1] == "COMPLETED"
        assert events[-1][2] == "reg2.png"

    async def test_mock_image_writes_valid_png(self, tmp_path):
        import asyncio

        import inference_pb2
        from engines.base import EngineContext

        req = inference_pb2.MediaGenerationRequest(
            job_id="reg3", mode="image", model_name="desconocido",
            prompt="x", steps=2,
        )
        ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())
        async for _ in registry.run_generation(req, ctx):
            pass

        png = tmp_path / "reg3.png"
        assert png.exists()
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
