"""Cliente HTTP del bridge contra el back de Synapta.

Solo stdlib (urllib): el instalador no arrastra dependencias nuevas y el
mismo código corre en Windows, Linux y macOS. Los endpoints son el contrato
de Fase A (back: internal/modules/workerbridge):

    POST /api/v1/internal/workers/register    (code → token, una sola vez)
    POST /api/v1/internal/workers/heartbeat   (X-Worker-Token) → cancelaciones
    POST /api/v1/internal/workers/claim?wait= (long-poll de la cola)
    POST /api/v1/internal/workers/report      (progreso / failed / cancelled)
    POST /api/v1/internal/workers/artifacts   (multipart del output)

Todo request lleva timeout: una red doméstica mala no debe colgar al agente.
"""

from __future__ import annotations

import json
import logging
import socket
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("bridge.client")

# Timeout de red por defecto. El claim es un long-poll: su timeout lo pide
# el caller (wait + margen).
DEFAULT_TIMEOUT = 15.0


class BridgeError(Exception):
    """Error del bridge con contexto suficiente para el usuario."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


@dataclass
class Job:
    """Un trabajo reclamado del back (fila de worker_jobs)."""

    task_id: str
    payload: dict

    @classmethod
    def from_api(cls, raw: dict) -> "Job":
        payload = raw.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = {}
        return cls(task_id=str(raw.get("task_id", "")), payload=payload)


class BridgeClient:
    """Un cliente por worker emparejado. No threads: el agente secuencia."""

    def __init__(self, back_url: str, token: str = "", timeout: float = DEFAULT_TIMEOUT):
        self.back_url = back_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    # ─── capa HTTP ──────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return self.back_url + path

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["X-Worker-Token"] = self.token
        return h

    def _request(self, method: str, path: str, body: Any = None,
                 timeout: Optional[float] = None) -> dict:
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self._url(path), data=data, headers=self._headers(), method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except OSError:
                pass
            raise BridgeError(
                f"{method} {path} → HTTP {exc.code}: {detail}", status=exc.code
            ) from exc
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            raise BridgeError(f"{method} {path} sin conexión: {exc}") from exc
        try:
            parsed = json.loads(raw) if raw else {}
        except ValueError:
            parsed = {}
        # El back envuelve todo en {"success": bool, "data": ..., "message": ...}
        if isinstance(parsed, dict) and parsed.get("success") is False:
            raise BridgeError(
                f"{method} {path} rechazado: {parsed.get('message', 'sin detalle')}",
                status=400,
            )
        if isinstance(parsed, dict) and "data" in parsed:
            return parsed["data"] if parsed["data"] is not None else {}
        return parsed if isinstance(parsed, dict) else {}

    # ─── endpoints ──────────────────────────────────────────────

    def register(self, code: str, name: str, gpu_name: str, os_name: str,
                 version: str, vram_total_mb: int, vram_free_mb: int,
                 models: list) -> dict:
        """Canjea el código de emparejamiento por el token de larga duración."""
        return self._request("POST", "/api/v1/internal/workers/register", {
            "code": code,
            "name": name,
            "gpu_name": gpu_name,
            "os": os_name,
            "version": version,
            "vram_total_mb": vram_total_mb,
            "vram_free_mb": vram_free_mb,
            "models": models,
        })

    def heartbeat(self, gpu_name: str, version: str, vram_free_mb: int,
                  jobs_active: int, models: list) -> list:
        """Refresh del estado vivo. Devuelve los task_id a cancelar."""
        data = self._request("POST", "/api/v1/internal/workers/heartbeat", {
            "gpu_name": gpu_name,
            "version": version,
            "vram_free_mb": vram_free_mb,
            "jobs_active": jobs_active,
            "models": models,
        })
        ids = data.get("cancel_task_ids") if isinstance(data, dict) else None
        return list(ids) if isinstance(ids, list) else []

    def claim(self, wait_seconds: int = 25) -> Optional[Job]:
        """Long-poll: espera un trabajo hasta wait_seconds; None = cola vacía."""
        # El timeout del socket debe superar al del long-poll del back.
        data = self._request(
            "POST", f"/api/v1/internal/workers/claim?wait={wait_seconds}",
            {}, timeout=wait_seconds + 10.0,
        )
        raw = data.get("job") if isinstance(data, dict) else None
        if not raw or not isinstance(raw, dict):
            return None
        job = Job.from_api(raw)
        return job if job.task_id else None

    def report(self, task_id: str, status: str, progress: float,
               stream_status: str = "", error_message: str = "") -> None:
        """Progreso o estado terminal (failed/cancelled)."""
        self._request("POST", "/api/v1/internal/workers/report", {
            "task_id": task_id,
            "status": status,
            "progress": round(float(progress), 2),
            "stream_status": stream_status,
            "error_message": error_message,
        })

    def upload_artifact(self, task_id: str, file_path: str) -> str:
        """Sube el output terminado (multipart). Devuelve la URL en el back.

        Idempotente del lado del back: si la conexión se corta a mitad de
        subida, reintentar no duplica el artefacto.
        """
        boundary = f"----synapta{uuid.uuid4().hex}"
        filename = file_path.replace("\\", "/").rsplit("/", 1)[-1]
        with open(file_path, "rb") as fh:
            file_bytes = fh.read()
        if not file_bytes:
            raise BridgeError(f"artifact vacío: {file_path}")

        body = bytearray()
        body += f"--{boundary}\r\n".encode()
        body += (f'Content-Disposition: form-data; name="task_id"\r\n\r\n'
                 f"{task_id}\r\n").encode()
        body += f"--{boundary}\r\n".encode()
        body += (f'Content-Disposition: form-data; name="file"; '
                 f'filename="{filename}"\r\n'
                 f"Content-Type: application/octet-stream\r\n\r\n").encode()
        body += file_bytes
        body += b"\r\n"
        body += f"--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            self._url("/api/v1/internal/workers/artifacts"),
            data=bytes(body),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "X-Worker-Token": self.token,
            },
            method="POST",
        )
        # Videos de minutos: timeout generoso pero acotado.
        timeout = max(self.timeout, 120.0 + len(file_bytes) / (256 * 1024))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except OSError:
                pass
            raise BridgeError(
                f"upload → HTTP {exc.code}: {detail}", status=exc.code
            ) from exc
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            raise BridgeError(f"upload sin conexión: {exc}") from exc
        try:
            parsed = json.loads(raw) if raw else {}
        except ValueError:
            parsed = {}
        data = parsed.get("data", parsed) if isinstance(parsed, dict) else {}
        url = data.get("artifact_url", "") if isinstance(data, dict) else ""
        return str(url)


def with_retries(fn, *, attempts: int = 5, base_delay: float = 1.0,
                 what: str = "operación"):
    """Reintenta con backoff exponencial + jitter.

    Pensado para la red doméstica: un bajón de 10 s no debe matar el job.
    Último error propagado si se agotan los intentos.
    """
    last: Optional[BridgeError] = None
    for attempt in range(attempts):
        try:
            return fn()
        except BridgeError as exc:
            last = exc
            # Errores 4xx (contrato roto / token inválido): reintentar no ayuda.
            if 400 <= exc.status < 500 and exc.status != 429:
                raise
            delay = base_delay * (2 ** attempt) * (0.5 + 0.5 * __import__("random").random())
            logger.warning("%s falló (intento %d/%d): %s; reintento en %.1fs",
                           what, attempt + 1, attempts, exc, delay)
            time.sleep(delay)
    assert last is not None
    raise last
