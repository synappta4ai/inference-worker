"""Servidor gRPC del motor de inferencia (capa 3 de la arquitectura híbrida).

Implementa maestro.inference.v1.InferenceService:
  - GenerateMedia: streaming de progreso y generación de un archivo real.
  - CancelGeneration: cancelación cooperativa de generaciones en curso.
  - EnhancePrompt: mejora de prompts con heurística local de estudio.
  - GetGpuTelemetry: estado de GPU/VRAM vía VRAMManager.

El motor es un mock determinista listo para enchufar los pipelines reales
(WanGP / Diffusers): cuando exista GPU, `run_pipeline` es el punto de sustitución.
"""

import asyncio
import os
import sys
import time
import logging
import uuid

# Compatibilidad: forzar UTF-8 en consolas Windows (cp1252 no soporta emojis)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import grpc
from grpc import aio

import inference_pb2
import inference_pb2_grpc
from vram_manager import VRAMManager
from engines.base import EngineContext, resolve
from engines import registry
from model_ops import ModelOps

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BrainMasterInferenceWorker")

OUTPUT_DIR = os.environ.get("BM_OUTPUT_DIR", "outputs")


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """Implementación gRPC del servicio de inferencia."""

    def __init__(self) -> None:
        self.vram = VRAMManager()
        # job_id -> asyncio.Event para cancelación cooperativa
        self._active: dict[str, asyncio.Event] = {}
        self._cancel_done: dict[str, bool] = {}
        self.ops = ModelOps()

    # ------------------------------------------------------------------
    # GenerateMedia: streaming de progreso + archivo de salida real
    # ------------------------------------------------------------------
    async def GenerateMedia(self, request, context):
        if not request.job_id:
            request.job_id = f"job_{uuid.uuid4().hex[:12]}"
        job_id = request.job_id
        steps = max(1, request.steps or 20)
        cancel_event = asyncio.Event()
        self._active[job_id] = cancel_event
        self._cancel_done[job_id] = False

        telemetry = self.vram.get_telemetry()
        logger.info(
            "[GenerateMedia] Job %s | modelo=%s modo=%s | GPU: %s (libre %s MB)",
            job_id, request.model_name, request.mode,
            telemetry["device_name"], telemetry["free_vram_mb"],
        )

        # El registry resuelve el motor (mock o diffusers real) según el modelo.
        spec = resolve(request.model_name, request.mode)
        logger.info("[GenerateMedia] Motor=%s repo=%s", spec.engine, spec.repo or "-")

        def to_response(pct: float, status: str, out: str, err: str):
            if status in ("CANCELLED", "FAILED"):
                self._cancel_done[job_id] = status == "CANCELLED"
            if status == "GENERATING":
                # El mock emite (step/steps)*95; el motor real (step/steps)*90.
                gen_step = max(1, round(pct * steps / 95.0))
            else:
                gen_step = steps if pct >= 100 else 0
            return inference_pb2.GenerationProgressResponse(
                job_id=job_id,
                current_step=gen_step,
                total_steps=steps,
                percentage=pct,
                status=status,
                output_file_path=out,
                error_message=err,
            )

        ctx = EngineContext(out_dir=OUTPUT_DIR, cancel=cancel_event)
        try:
            async for pct, status, out, err in registry.run_generation(request, ctx):
                yield to_response(pct, status, out, err)
                if status in ("COMPLETED", "FAILED", "CANCELLED"):
                    return
            # El motor terminó sin evento final: tratar como completado.
            yield inference_pb2.GenerationProgressResponse(
                job_id=job_id, current_step=steps, total_steps=steps,
                percentage=100.0, status="COMPLETED",
            )
        except MemoryError:
            logger.error("[GenerateMedia] Job %s: sin memoria para %s", job_id, request.model_name)
            yield to_response(0.0, "FAILED", "",
                              "sin memoria para cargar el modelo (prueba un modelo más pequeño o una máquina con más VRAM/RAM)")
        except Exception as exc:  # un motor no debe tumbar el worker
            logger.exception("[GenerateMedia] Job %s falló: %s", job_id, exc)
            yield to_response(0.0, "FAILED", "", f"{type(exc).__name__}: {exc}")
        finally:
            self._active.pop(job_id, None)
            self._cancel_done.pop(job_id, None)
            self.vram.clear_vram()

    # ------------------------------------------------------------------
    # CancelGeneration
    # ------------------------------------------------------------------
    async def CancelGeneration(self, request, context):
        event = self._active.get(request.job_id)
        if event is None:
            return inference_pb2.CancelResponse(job_id=request.job_id, accepted=False)
        event.set()
        # Espera breve a que el pipeline confirme la cancelación.
        for _ in range(50):
            if self._cancel_done.get(request.job_id):
                break
            await asyncio.sleep(0.02)
        logger.info("[CancelGeneration] Job %s cancelado", request.job_id)
        return inference_pb2.CancelResponse(job_id=request.job_id, accepted=True)

    # ------------------------------------------------------------------
    # ListModels: catálogo dinámico (fuente única: engines/base.py)
    # ------------------------------------------------------------------
    async def ListModels(self, request, context):
        from engines.v1_bridge import is_available

        tel = self.vram.get_telemetry()
        cuda = bool(tel.get("cuda_available")) or "cpu" not in tel.get("device_name", "cpu").lower()
        resp = inference_pb2.ListModelsResponse(
            device_name=tel["device_name"],
            cuda_available=cuda,
        )
        for m in registry.list_models():
            state = self.ops.snapshot(m["id"])
            if m["engine"] == "v1_bridge":
                available = is_available()
            elif m["engine"] == "diffusers":
                # Listo para generar YA: los pesos completos ya están en disco.
                available = state.downloaded
            else:
                available = True
            info = inference_pb2.ModelInfo(
                key=m["id"],
                label=m["name"],
                mode=m["mode"],
                engine=m["engine"],
                pipeline=m["pipeline"],
                repo=m["repo"],
                steps=int(m["steps"]),
                vram_gb=int(m.get("vram_gb") or 0),
                family=m["family"],
                notes=m.get("notes") or "",
                available=available,
                downloaded=state.downloaded,
                downloading=state.downloading,
                download_percent=float(state.progress),
                size_bytes=int(state.size_bytes),
                download_error=state.error,
            )
            resp.models.append(info)
        logger.info("[ListModels] %d modelos | device=%s", len(resp.models), tel["device_name"])
        return resp

    # ------------------------------------------------------------------
    # Gestión de pesos del panel "Tu GPU"
    # ------------------------------------------------------------------
    @staticmethod
    def _operation_response(state) -> "inference_pb2.ModelOperationResponse":
        return inference_pb2.ModelOperationResponse(
            key=state.key,
            state=state.state,
            progress=float(state.progress),
            bytes_downloaded=int(state.bytes_downloaded),
            total_bytes=int(state.total_bytes),
            error=state.error,
        )

    @staticmethod
    def _model_key(raw_key: str | None) -> str:
        key = (raw_key or "").strip()
        return key

    async def DownloadModel(self, request, context):
        key = self._model_key(request.key)
        if not key:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "key requerido")
        try:
            state = self.ops.cancel(key) if request.cancel else self.ops.start_download(key)
        except KeyError:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"modelo desconocido: {key}")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        logger.info("[DownloadModel] %s cancel=%s estado=%s", key, bool(request.cancel), state.state)
        return self._operation_response(state)

    async def DeleteModel(self, request, context):
        key = self._model_key(request.key)
        if not key:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "key requerido")
        try:
            state = self.ops.delete(key)
        except KeyError:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"modelo desconocido: {key}")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        logger.info("[DeleteModel] %s estado=%s", key, state.state)
        return self._operation_response(state)

    async def GetModelState(self, request, context):
        key = self._model_key(request.key)
        try:
            states = self.ops.states(key)
        except KeyError:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"modelo desconocido: {key}")
        resp = inference_pb2.GetModelStateResponse(
            bytes_on_disk=int(self.ops.disk_total()),
        )
        for state in states:
            resp.states.append(self._operation_response(state))
        logger.info("[GetModelState] key=%s estados=%d", key or "todos", len(resp.states))
        return resp

    # ------------------------------------------------------------------
    # EnhancePrompt: heurística local de estudio (sin LLM externo)
    # ------------------------------------------------------------------
    STYLE_HINTS = {
        "cinematic": "cinematic lighting, 35mm film grain, shallow depth of field, dramatic composition",
        "photoreal": "photorealistic, ultra detailed, natural lighting, 8k",
        "anime": "anime style, cel shading, vibrant colors, studio quality",
        "documentary": "documentary style, handheld camera, natural colors, realistic motion",
    }

    async def EnhancePrompt(self, request, context):
        raw = (request.raw_prompt or "").strip().rstrip(".,")
        style = (request.style or "cinematic").lower()
        hints = self.STYLE_HINTS.get(style, self.STYLE_HINTS["cinematic"])
        model = request.target_model or "generic"

        parts = [raw]
        if "lighting" not in raw.lower():
            parts.append(hints)
        if "video" in model.lower() or "wan" in model.lower() or "ltx" in model.lower():
            parts.append("smooth motion, stable camera trajectory, 24fps film cadence")
        else:
            parts.append("sharp focus, high dynamic range")
        enhanced = ", ".join(p for p in parts if p)

        rationale = (
            f"Prompt reescrito para '{model}' con estilo '{style}': se añaden "
            "términos de iluminación/composición y directrices de movimiento "
            "según el dominio del modelo destino."
        )
        logger.info("[EnhancePrompt] '%s...' → '%s...'", raw[:40], enhanced[:60])
        return inference_pb2.PromptEnhanceResponse(
            enhanced_prompt=enhanced, rationale=rationale,
        )

    # ------------------------------------------------------------------
    # GetGpuTelemetry
    # ------------------------------------------------------------------
    async def GetGpuTelemetry(self, request, context):
        tel = self.vram.get_telemetry()
        return inference_pb2.GpuTelemetryResponse(
            device_name=tel["device_name"],
            total_vram_mb=tel["total_vram_mb"],
            used_vram_mb=tel["used_vram_mb"],
            free_vram_mb=tel["free_vram_mb"],
            gpu_utilization=tel["gpu_utilization"],
            temperature_c=tel["temperature_c"],
        )


async def serve() -> None:
    port = int(os.environ.get("GRPC_PORT", "50051"))

    # Servidor de artefactos HTTP (despliegues divididos: worker en la nube,
    # gateway en otra máquina). BM_ARTIFACT_PORT=0 lo desactiva.
    if os.environ.get("BM_ARTIFACT_PORT", "50052") != "0":
        from artifact_server import start_background
        start_background(int(os.environ.get("BM_ARTIFACT_PORT", "50052")))

    server = aio.server()
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    logger.info("🧠 [Brain-Master] Worker de inferencia gRPC escuchando en :%s", port)
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Apagando worker...")
        await server.stop(grace=2.0)


if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
