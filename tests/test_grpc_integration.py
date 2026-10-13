"""Integración gRPC in-process: servidor aio real + canal + stub.

Ejercita la pila gRPC completa (serialización, streaming HTTP/2) sin salir
del proceso, igual que hará el gateway Go en producción.
"""

import asyncio
import os

import grpc
import pytest

import inference_pb2
import inference_pb2_grpc
import server as server_mod
from server import InferenceServicer


@pytest.fixture
async def grpc_env(_isolate_output_dir):
    """Levanta un servidor gRPC real en un puerto efímero con el servicer."""
    from grpc import aio

    servicer = InferenceServicer()
    server = aio.server()
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    async with aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        yield servicer, stub

    await server.stop(grace=0.5)


async def test_full_stream_over_real_grpc(grpc_env):
    servicer, stub = grpc_env
    req = inference_pb2.MediaGenerationRequest(
        job_id="int1", mode="image", model_name="Flux.1-dev",
        prompt="a lighthouse at dawn", steps=5,
    )
    events = [e async for e in stub.GenerateMedia(req)]

    assert events[0].status == "LOADING_MODEL"
    assert events[-1].status == "COMPLETED"
    assert events[-1].percentage == 100.0
    assert events[-1].output_file_path == "int1.png"
    # Archivo físico en el output aislado del test.
    assert os.path.isfile(os.path.join(server_mod.OUTPUT_DIR, "int1.png"))


async def test_enhance_and_telemetry_over_grpc(grpc_env):
    _, stub = grpc_env

    enh = await stub.EnhancePrompt(
        inference_pb2.PromptEnhanceRequest(
            raw_prompt="a samurai", target_model="Wan 2.1", style="cinematic"
        )
    )
    assert "a samurai" in enh.enhanced_prompt
    assert enh.rationale

    tel = await stub.GetGpuTelemetry(inference_pb2.GpuTelemetryRequest())
    assert tel.device_name == "CPU / No CUDA"


async def test_remote_cancellation_via_grpc(grpc_env):
    servicer, stub = grpc_env
    req = inference_pb2.MediaGenerationRequest(
        job_id="rc1", mode="video", model_name="wan", prompt="x", steps=500
    )

    async def collect():
        return [e async for e in stub.GenerateMedia(req)]

    async def cancel():
        await asyncio.sleep(0.1)
        return await stub.CancelGeneration(inference_pb2.CancelRequest(job_id="rc1"))

    gen = asyncio.ensure_future(collect())
    await asyncio.sleep(0.03)
    cancel_task = asyncio.ensure_future(cancel())

    events = await gen
    resp = await cancel_task

    assert events[-1].status == "CANCELLED"
    assert resp.accepted is True
