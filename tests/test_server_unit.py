"""Tests unitarios del servicer gRPC del worker de inferencia.

Ejecutan las corrutinas directamente (sin red), con el directorio de
outputs aislado por el fixture autouse de conftest.py.
"""

import asyncio
import os

import inference_pb2
import pytest

import server as server_mod
from server import InferenceServicer


def make_request(**overrides):
    defaults = dict(
        job_id="t1",
        mode="image",
        model_name="Flux.1-dev",
        prompt="a red apple on a table",
        negative_prompt="blurry",
        width=512,
        height=512,
        num_frames=0,
        fps=0,
        steps=4,
        seed=7,
    )
    defaults.update(overrides)
    return inference_pb2.MediaGenerationRequest(**defaults)


async def collect(agen):
    return [item async for item in agen]


# ----------------------------------------------------------------------
# GenerateMedia
# ----------------------------------------------------------------------

class TestGenerateMedia:

    async def test_image_completes_and_writes_png(self, servicer):
        events = await collect(servicer.GenerateMedia(make_request(), None))
        statuses = [e.status for e in events]

        assert statuses[0] == "LOADING_MODEL"
        assert statuses[-1] == "COMPLETED"
        assert "GENERATING" in statuses
        assert "POSTPROCESSING" in statuses

        final = events[-1]
        assert final.percentage == 100.0
        assert final.output_file_path == "t1.png"

        # Archivo real y válido (PNG 1x1 byte-estable).
        path = os.path.join(server_mod.OUTPUT_DIR, final.output_file_path)
        assert os.path.isfile(path)
        with open(path, "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n"

    async def test_video_writes_mp4_placeholder(self, servicer):
        req = make_request(job_id="v1", mode="video", num_frames=80, fps=16)
        events = await collect(servicer.GenerateMedia(req, None))

        assert events[-1].status == "COMPLETED"
        assert events[-1].output_file_path == "v1.mp4"
        path = os.path.join(server_mod.OUTPUT_DIR, "v1.mp4")
        assert os.path.isfile(path)
        with open(path, "rb") as fh:
            head = fh.readline()
        assert head.startswith(b"BM_MOCK_VIDEO")
        assert b"job=v1" in head

    async def test_progress_is_monotonic_and_covers_all_steps(self, servicer):
        events = await collect(
            servicer.GenerateMedia(make_request(steps=6), None)
        )
        generating = [e for e in events if e.status == "GENERATING"]
        assert len(generating) == 6
        pcts = [e.percentage for e in generating]
        assert pcts == sorted(pcts)
        # El mock reserva 97-100% para POSTPROCESSING/COMPLETED.
        assert pcts[-1] == pytest.approx(95.0)
        final = events[-1]
        assert final.status == "COMPLETED"
        assert final.percentage == 100.0
        # current_step / total_steps consistentes.
        for i, e in enumerate(generating, start=1):
            assert e.current_step == i
            assert e.total_steps == 6

    async def test_zero_steps_clamps_to_one(self, servicer):
        events = await collect(
            servicer.GenerateMedia(make_request(steps=0), None)
        )
        assert events[-1].status == "COMPLETED"

    async def test_job_id_generated_when_missing(self, servicer):
        events = await collect(
            servicer.GenerateMedia(make_request(job_id=""), None)
        )
        job_id = events[0].job_id
        assert job_id.startswith("job_")
        assert events[-1].output_file_path == f"{job_id}.png"

    async def test_active_jobs_cleaned_up_after_completion(self, servicer):
        await collect(servicer.GenerateMedia(make_request(), None))
        assert "t1" not in servicer._active
        assert servicer._cancel_done.get("t1") in (None, False)

    async def test_cancel_event_aborts_stream(self, servicer):
        req = make_request(job_id="cx", steps=50)

        async def cancel_soon():
            await asyncio.sleep(0.05)
            servicer._active["cx"].set()

        task = asyncio.ensure_future(
            collect(servicer.GenerateMedia(req, None))
        )
        asyncio.ensure_future(cancel_soon())
        events = await task

        assert events[-1].status == "CANCELLED"
        # No debe existir archivo de salida.
        assert not os.path.exists(os.path.join(server_mod.OUTPUT_DIR, "cx.png"))


# ----------------------------------------------------------------------
# CancelGeneration
# ----------------------------------------------------------------------

class TestCancelGeneration:

    async def test_cancel_unknown_job_not_accepted(self, servicer):
        resp = await servicer.CancelGeneration(
            inference_pb2.CancelRequest(job_id="ghost"), None
        )
        assert resp.accepted is False

    async def test_cancel_active_job_waits_for_confirmation(self, servicer):
        req = make_request(job_id="cy", steps=200)

        async def run():
            return await collect(servicer.GenerateMedia(req, None))

        async def canceler():
            await asyncio.sleep(0.05)
            return await servicer.CancelGeneration(
                inference_pb2.CancelRequest(job_id="cy"), None
            )

        gen_task = asyncio.ensure_future(run())
        await asyncio.sleep(0.02)  # deja registrar el job
        cancel_task = asyncio.ensure_future(canceler())

        events = await gen_task
        resp = await cancel_task

        assert events[-1].status == "CANCELLED"
        assert resp.accepted is True


# ----------------------------------------------------------------------
# EnhancePrompt
# ----------------------------------------------------------------------

class TestEnhancePrompt:

    async def test_cinematic_video_adds_motion_terms(self, servicer):
        resp = await servicer.EnhancePrompt(
            inference_pb2.PromptEnhanceRequest(
                raw_prompt="a dragon flying", target_model="Wan 2.1", style="cinematic"
            ),
            None,
        )
        low = resp.enhanced_prompt.lower()
        assert "a dragon flying" in low
        assert "cinematic lighting" in low
        assert "smooth motion" in low
        assert resp.rationale  # no vacío

    async def test_image_model_adds_focus_terms(self, servicer):
        resp = await servicer.EnhancePrompt(
            inference_pb2.PromptEnhanceRequest(
                raw_prompt="a castle", target_model="Flux.1-dev", style="photoreal"
            ),
            None,
        )
        low = resp.enhanced_prompt.lower()
        assert "photorealistic" in low
        assert "sharp focus" in low

    async def test_no_duplicate_lighting_hint(self, servicer):
        resp = await servicer.EnhancePrompt(
            inference_pb2.PromptEnhanceRequest(
                raw_prompt="moody scene with dramatic lighting",
                target_model="generic",
                style="cinematic",
            ),
            None,
        )
        # El prompt ya contiene "lighting": el hint de estilo se omite
        # (cero duplicados) y la palabra aparece solo por el prompt original.
        assert resp.enhanced_prompt.lower().count("lighting") == 1

    async def test_unknown_style_falls_back_to_cinematic(self, servicer):
        resp = await servicer.EnhancePrompt(
            inference_pb2.PromptEnhanceRequest(
                raw_prompt="test", target_model="generic", style="vaporwave"
            ),
            None,
        )
        assert "cinematic lighting" in resp.enhanced_prompt.lower()

    async def test_empty_prompt_still_responds(self, servicer):
        resp = await servicer.EnhancePrompt(
            inference_pb2.PromptEnhanceRequest(raw_prompt=""), None
        )
        assert resp.enhanced_prompt  # hint de estilo al menos


# ----------------------------------------------------------------------
# GetGpuTelemetry
# ----------------------------------------------------------------------

class TestGetGpuTelemetry:

    async def test_telemetry_without_cuda(self, servicer):
        resp = await servicer.GetGpuTelemetry(
            inference_pb2.GpuTelemetryRequest(), None
        )
        # En el entorno de CI/test no hay GPU: el mock reporta CPU.
        assert resp.device_name == "CPU / No CUDA"
        assert resp.total_vram_mb == 0


# ----------------------------------------------------------------------
# Rutas de salida
# ----------------------------------------------------------------------

class TestOutputPath:

    def test_extension_by_mode(self):
        from engines.mock_engine import _output_path

        assert _output_path("j", "image") == "j.png"
        assert _output_path("j", "video") == "j.mp4"
        assert _output_path("j", "") == "j.mp4"
        assert _output_path("j", "IMAGE") == "j.png"
