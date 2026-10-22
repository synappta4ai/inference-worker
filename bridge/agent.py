"""Agente bridge: conecta el worker con el back de Synapta por HTTP saliente.

Loop de vida (una vez emparejado):

    heartbeat (20 s)  → estado vivo + cancelaciones entrantes
    claim (long-poll) → ¿hay trabajo para mi GPU?
         │
         ▼
    ejecuta vía gRPC local (127.0.0.1:50051) — el mismo servidor de siempre,
    con todos los engines, el VRAM manager y la cancelación cooperativa
         │
         ▼
    reporta progreso al back → sube el artefacto terminado (multipart)

El agente NO reimplementa inferencia: solo transporta trabajos entre el back
y el gRPC local. Si el back cae, el loop espera con backoff; si la máquina
se apaga a mitad de job, el journal recupera el artefacto pendiente de
subida al arrancar (nunca se pierde trabajo por internet).
"""

from __future__ import annotations

import logging
import platform
import socket
import threading
import time
from typing import Optional

import grpc

import inference_pb2
import inference_pb2_grpc
from vram_manager import VRAMManager

from . import BRIDGE_VERSION, journal
from .client import BridgeClient, BridgeError, Job, with_retries

logger = logging.getLogger("bridge.agent")

# Cadencia del heartbeat: el back considera "online" a un worker con menos
# de 75 s de silencio, así que 20 s deja margen holgado para dos reintentos.
HEARTBEAT_INTERVAL = 20.0

# Puerto del servidor gRPC local (server.py). El bridge siempre habla con
# loopback: el back nunca necesita alcanzar esta máquina.
LOCAL_GRPC = "127.0.0.1:50051"


def _local_model_states() -> dict:
    """Estado real de pesos vía gRPC local (disponible/descargando/%).

    El agente y el servidor viven en el mismo proceso (run.py): el snapshot
    que viaja en el heartbeat refleja disco de verdad, no un
    `available=True` fijo. Si el gRPC aún no levantó, {} (fallback abajo).
    """
    try:
        with grpc.insecure_channel(LOCAL_GRPC) as channel:
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            resp = stub.ListModels(inference_pb2.ListModelsRequest(), timeout=8)
    except Exception as exc:  # noqa: BLE001 — snapshot best-effort, nunca bloquea
        # Sin este log, un gRPC local que no responde se veía en el panel
        # como "todos los modelos disponibles" y no había forma de saber por qué.
        logger.warning("[bridge] el gRPC local (%s) no devolvió el estado de "
                       "pesos: %s. Los modelos viajan como no disponibles "
                       "hasta que responda.", LOCAL_GRPC, exc)
        return {}
    out = {}
    try:
        for m in resp.models:
            out[m.key] = {
                "available": bool(m.available),
                "downloaded": bool(m.downloaded),
                "downloading": bool(m.downloading),
                "download_percent": float(m.download_percent or 0.0),
                "size_bytes": int(m.size_bytes or 0),
                "download_error": str(m.download_error or ""),
                "download_path": str(m.download_path or ""),
            }
    except Exception as exc:  # noqa: BLE001 — proto parcial: lo que haya sirve
        logger.warning("[bridge] snapshot de pesos incompleto: %s", exc)
    return out


def gpu_snapshot() -> dict:
    """Telemetría de GPU + catálogo, en el formato que espera el back.

    El estado de pesos (available/downloading/%) sale del gRPC local. Si el
    gRPC no responde, el modelo viaja como NO disponible: inventar
    disponibilidad deja el panel sin botón "Descargar" para un modelo que no
    está en disco (el usuario no tiene cómo bajarlo) y manda a generar con
    pesos que no existen.
    """
    tel = VRAMManager.get_telemetry()
    from engines.registry import list_models
    catalog = list_models()
    states = _local_model_states()
    if catalog and not states:
        logger.warning("[bridge] snapshot sin estado de pesos: los %d modelos "
                       "del catálogo se reportan como no disponibles",
                       len(catalog))
    models = []
    for m in catalog:
        st = states.get(m["id"], {})
        available = bool(st.get("available", False))
        models.append({
            "key": m["id"],
            "label": m["name"],
            "mode": m["mode"],
            "engine": m["engine"],
            "pipeline": m["pipeline"],
            "repo": m["repo"],
            "steps": m["steps"],
            "vram_gb": m["vram_gb"],
            "family": m["family"],
            "available": available,
            "downloaded": bool(st.get("downloaded", False)),
            "downloading": bool(st.get("downloading", False)),
            "download_percent": float(st.get("download_percent",
                                          100.0 if available else 0.0)),
            "size_bytes": int(st.get("size_bytes", 0)),
            "download_error": str(st.get("download_error", "")),
            "download_path": str(st.get("download_path", "")),
            "notes": m.get("notes") or "",
        })
    return {
        "gpu_name": tel.get("device_name", "CPU / No CUDA"),
        "vram_total_mb": int(tel.get("total_vram_mb", 0)),
        "vram_free_mb": int(tel.get("free_vram_mb", 0)),
        "models": models,
    }


def machine_name() -> str:
    """Nombre legible para el panel admin: host + OS."""
    return f"{socket.gethostname()} ({platform.system()})"


class Agent:
    """Bucle principal del worker emparejado."""

    def __init__(self, cfg, local_grpc: str = LOCAL_GRPC):
        self.cfg = cfg
        self.client = BridgeClient(cfg.back_url, cfg.token)
        self.local_grpc = local_grpc
        self._stop = threading.Event()
        self._jobs_active = 0
        # task_id → grpc call handle para cancelaciones que llegan por heartbeat.
        self._cancel_lock = threading.Lock()
        self._active_calls: dict = {}

    # ─── ciclo de vida ──────────────────────────────────────────

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        """Arranca heartbeat + claim loop. Bloquea hasta stop()."""
        self._recover_journal()
        heartbeat = threading.Thread(
            target=self._heartbeat_loop, name="bridge-heartbeat", daemon=True
        )
        heartbeat.start()
        logger.info("[bridge] agente arrancado → %s (worker=%s)",
                    self.cfg.back_url, self.cfg.worker_id)
        while not self._stop.is_set():
            try:
                job = self.client.claim(wait_seconds=25)
            except BridgeError as exc:
                logger.warning("[bridge] claim falló: %s", exc)
                self._stop.wait(5.0)
                continue
            if job is None:
                continue  # el long-poll ya esperó; loop inmediato
            self._execute_job(job)

    # ─── recuperación post-arranque ─────────────────────────────

    def _recover_journal(self) -> None:
        """Re-sube artefactos que quedaron listos sin confirmar (corte de
        luz a mitad de upload) y descarta jobs viejos cuyo proceso murió."""
        for entry in journal.pending_uploads():
            logger.info("[bridge] re-subiendo artefacto pendiente %s (%s)",
                        entry.task_id, entry.artifact_path)
            try:
                with_retries(
                    lambda e=entry: self.client.upload_artifact(e.task_id, e.artifact_path),
                    attempts=8, base_delay=2.0, what=f"re-upload {entry.task_id}",
                )
                journal.discard(entry.task_id)
                logger.info("[bridge] artefacto %s recuperado", entry.task_id)
            except BridgeError as exc:
                # Se queda en el journal: el próximo arranque reintenta.
                logger.error("[bridge] no pude recuperar %s: %s", entry.task_id, exc)
        for entry in journal.stale_unfinished():
            logger.info(descartar_msg(entry))
            journal.discard(entry.task_id)

    # ─── heartbeat ──────────────────────────────────────────────

    def _heartbeat_loop(self) -> None:
        # Descargas pedidas desde el panel ya tomadas: se confirman (ack)
        # en el próximo heartbeat para sacarlas de la cola del back.
        pending_acks: list[str] = []
        while not self._stop.is_set():
            snap = gpu_snapshot()
            try:
                res = self.client.heartbeat(
                    gpu_name=snap["gpu_name"],
                    version=BRIDGE_VERSION,
                    vram_free_mb=snap["vram_free_mb"],
                    jobs_active=self._jobs_active,
                    models=snap["models"],
                    completed_downloads=pending_acks,
                )
                pending_acks = []
                cancels = res.get("cancel_task_ids", [])
                if isinstance(cancels, list):
                    for task_id in cancels:
                        self._cancel_local(task_id)
                requested = res.get("requested_models", [])
                if isinstance(requested, list):
                    for key in requested:
                        if self._start_remote_download(str(key)):
                            pending_acks.append(str(key))
            except BridgeError as exc:
                # Token revocado → el admin desemparejó la GPU: parar.
                if exc.status == 401:
                    logger.error("[bridge] token rechazado: la GPU fue "
                                 "desemparejada desde la app. Deteniendo.")
                    self._stop.set()
                    return
                logger.warning("[bridge] heartbeat falló: %s", exc)
            self._stop.wait(HEARTBEAT_INTERVAL)

    def _start_remote_download(self, key: str) -> bool:
        """Dispara DownloadModel en el gRPC local (pedido del panel).

        Devuelve True si el back puede darlo por tomado (descarga lanzada
        o ya en curso/completa): el progreso real viaja en el snapshot.
        False solo si la clave es desconocida o el gRPC no responde — en
        ese caso NO se hace ack y el back lo reintenta en el próximo loop.
        """
        key = (key or "").strip()
        if not key:
            return False
        try:
            with grpc.insecure_channel(self.local_grpc) as channel:
                stub = inference_pb2_grpc.InferenceServiceStub(channel)
                stub.DownloadModel(
                    inference_pb2.DownloadModelRequest(key=key), timeout=30
                )
            logger.info("[bridge] descarga pedida desde el panel: %s", key)
            return True
        except grpc.RpcError as exc:
            code = exc.code() if hasattr(exc, "code") else None
            # Clave desconocida o precondición (mock/bridge V1): no reintentar.
            name = str(getattr(code, "name", code))
            if name in ("NOT_FOUND", "FAILED_PRECONDITION", "INVALID_ARGUMENT"):
                logger.warning("[bridge] descarga %s rechazada (%s): %s",
                               key, name, exc.details() if hasattr(exc, "details") else exc)
                return True
            logger.warning("[bridge] descarga %s no lanzada, reintento luego: %s", key, exc)
            return False

    def _cancel_local(self, task_id: str) -> None:
        """Pide al gRPC local abortar el job en curso (si es este agente)."""
        with self._cancel_lock:
            call = self._active_calls.get(task_id)
        if call is None:
            return
        logger.info("[bridge] cancelación remota recibida para %s", task_id)
        try:
            with grpc.insecure_channel(self.local_grpc) as channel:
                stub = inference_pb2_grpc.InferenceServiceStub(channel)
                stub.CancelGeneration(
                    inference_pb2.CancelRequest(job_id=task_id), timeout=10
                )
        except grpc.RpcError as exc:
            logger.warning("[bridge] cancelación local falló: %s", exc)

    # ─── ejecución ──────────────────────────────────────────────

    def _execute_job(self, job: Job) -> None:
        logger.info("[bridge] job %s reclamado (modelo=%s)",
                    job.task_id, job.payload.get("model_name", "?"))
        entry = journal.JobEntry(
            task_id=job.task_id,
            state=journal.STATE_CLAIMED,
            model_name=str(job.payload.get("model_name", "")),
        )
        journal.record(entry)

        with grpc.insecure_channel(self.local_grpc) as channel:
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            req = _request_from_payload(job)
            self._jobs_active += 1
            try:
                with self._cancel_lock:
                    self._active_calls[job.task_id] = True
                self._run_stream(stub, req, job, entry)
            finally:
                self._jobs_active -= 1
                with self._cancel_lock:
                    self._active_calls.pop(job.task_id, None)

    def _run_stream(self, stub, req, job: Job, entry) -> None:
        """Consume el stream de GenerateMedia y espeja el progreso al back."""
        entry.state = journal.STATE_RUNNING
        journal.record(entry)
        try:
            last_report = 0.0
            # Hitos locales para ver el avance en worker.log sin espiar el
            # stream: 3 líneas por job en vez de cientos de eventos.
            milestones = [25.0, 50.0, 75.0]
            for event in stub.GenerateMedia(req, timeout=6 * 3600):
                pct = float(event.percentage or 0.0)
                while milestones and pct >= milestones[0]:
                    done = milestones.pop(0)
                    logger.info("[bridge] job %s: %d%% (%s)",
                                job.task_id, int(done), event.status)
                # Reporto como mucho 1 vez/2 s: el back solo dibuja una barra.
                if event.status in ("GENERATING",) and time.time() - last_report < 2.0:
                    continue
                last_report = time.time()
                if event.status == "COMPLETED":
                    self._complete(job, entry, event.output_file_path)
                    return
                if event.status in ("FAILED", "CANCELLED"):
                    self._fail(job, event.status, event.error_message)
                    return
                self._safe_report(job.task_id, "running", event.percentage,
                                  stream_status=event.status)
        except grpc.RpcError as exc:
            # El stream local murió (worker se reinició a mitad de job):
            # el back lo reconcila por heartbeat; acá solo registramos.
            logger.error("[bridge] stream de %s murió: %s", job.task_id, exc)
            self._fail(job, "FAILED", f"stream lost: {exc}")

    def _complete(self, job: Job, entry, artifact_path: str) -> None:
        if not artifact_path:
            self._fail(job, "FAILED", "COMPLETED sin artefacto")
            return
        entry.state = journal.STATE_ARTIFACT_READY
        entry.artifact_path = artifact_path
        journal.record(entry)
        try:
            url = with_retries(
                lambda: self.client.upload_artifact(job.task_id, artifact_path),
                attempts=8, base_delay=2.0, what=f"upload {job.task_id}",
            )
        except BridgeError as exc:
            # El artefacto queda en el journal: se sube en el próximo arranque.
            logger.error("[bridge] upload de %s falló (quedó en journal): %s",
                         job.task_id, exc)
            self._safe_report(job.task_id, "running", 99.0,
                              stream_status="UPLOAD_PENDING")
            return
        journal.discard(job.task_id)
        logger.info("[bridge] job %s entregado: %s", job.task_id, url)

    def _fail(self, job: Job, status: str, message: str) -> None:
        # Terminal sin artefacto pendiente: la entrada del journal se borra;
        # el back ya tiene el estado vía report (y su reconciler cubre cortes).
        journal.discard(job.task_id)
        self._safe_report(job.task_id, "failed" if status == "FAILED" else "cancelled",
                          0.0, stream_status=status, error=message)
        logger.info("[bridge] job %s terminó %s: %s", job.task_id, status, message)

    def _safe_report(self, task_id: str, status: str, progress: float,
                     stream_status: str = "", error: str = "") -> None:
        """Report best-effort: un reporte perdido no mata el job (el próximo
        heartbeat del back ya ve el progress por el claim)."""
        try:
            self.client.report(task_id, status, progress, stream_status, error)
        except BridgeError as exc:
            logger.warning("[bridge] reporte de %s falló: %s", task_id, exc)


def _request_from_payload(job: Job) -> inference_pb2.MediaGenerationRequest:
    """Reconstruye el request gRPC desde el payload JSON del back.

    El back serializa MediaGenerationRequest con protojson (camelCase), y
    este parser acepta tanto camelCase como snake_case del proto.
    """
    from google.protobuf import json_format
    req = inference_pb2.MediaGenerationRequest()
    json_format.ParseDict(job.payload, req, ignore_unknown_fields=True)
    # El job_id del journal/back manda: el worker no genera ids propios.
    req.job_id = job.task_id
    return req


def descartar_msg(entry) -> str:
    return (f"[bridge] job viejo {entry.task_id} ({entry.model_name}) descartado "
            f"del journal: su proceso ya no existe")
