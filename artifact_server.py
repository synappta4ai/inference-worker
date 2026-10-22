"""Servidor HTTP de artefactos y gestión de modelos del worker.

Cuando el worker corre en la nube (Colab/Runpod/Vast) y el gateway en otra
máquina, el PNG/MP4 generado vive en el disco del worker. Este servidor
expone BM_OUTPUT_DIR por HTTP para que el gateway pueda servir `artifact_url`
al front.

  GET    /artifacts                              → lista de archivos con tamaño
  GET    /artifacts/<nombre>[?token=...]         → el archivo
  DELETE /artifacts/<nombre>[?token=...]         → borrar el archivo

  GET    /models                                 → listado de modelos con
                                                   estado de pesos (dispo-
                                                   nible, descargado, %)
  GET    /models/<clave>[?token=...]             → estado de un modelo
  POST   /models/<clave>/download[?token=...]    → iniciar descarga de pesos
  POST   /models/<clave>/cancel[?token=...]      → cancelar descarga
  DELETE /models/<clave>[?token=...]             → borrar pesos del disco

Header alternativo: X-Artifact-Token
GET    /healthz                                → {"status":"ok"}

Solo nombres de archivo (sin rutas): el handler rechaza cualquier path
con separadores, `..` o prefijo de punto (incluye .token). El token se
configura con BM_ARTIFACT_TOKEN; sin esa variable el servidor queda abierto
(ensayado para túneles efímeros de Colab/Kaggle).

Se usa de dos formas:
  - independiente:  python artifact_server.py          (bloquea)
  - dentro de server.py: se arranca en un daemon thread automáticamente
    cuando BM_ARTIFACT_PORT está definido (default 50052, 0 = desactivado),
    pasándole el mismo ModelOps del servidor gRPC para compartir el estado
    de descarga en tiempo real.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from model_ops import ModelOps, repo_cache_dir

logger = logging.getLogger("ArtifactServer")

OUTPUT_DIR = os.environ.get("BM_OUTPUT_DIR", "outputs")
ARTIFACT_TOKEN = os.environ.get("BM_ARTIFACT_TOKEN", "")

# ModelOps compartido con el servidor gRPC (se setea desde server.py). Si
# nadie lo setea (modo standalone), cada lectura crea uno nuevo: los estados
# de disco siguen siendo consistentes porque se leen de la caché de HF.
_OPS = None


def _get_ops() -> ModelOps:
    global _OPS
    if _OPS is None:
        _OPS = ModelOps()
    return _OPS


def _safe_name(name: str) -> bool:
    """True solo para nombres planos de archivo (sin traversal ni ocultos)."""
    if not name or name.startswith("."):
        return False
    if "/" in name or "\\" in name or name in (".", ".."):
        return False
    return ".." not in name.split(os.sep)


class ArtifactHandler(BaseHTTPRequestHandler):
    server_version = "BrainMasterArtifacts/1.0"

    # ------------------------------------------------------------------
    # Modelos: listado y estado de pesos
    # ------------------------------------------------------------------
    def _model_dict(self, key: str) -> dict | None:
        ops = _get_ops()
        try:
            spec = ops.spec(key)
        except KeyError:
            return None
        state = ops.snapshot(key)
        available = state.downloaded
        if spec.engine == "v1_bridge":
            from engines.v1_bridge import is_available  # tardío: sin import ciclo
            available = is_available()
        elif spec.engine == "mock":
            available = True
        return {
            "key": key,
            "label": spec.label,
            "mode": spec.mode,
            "engine": spec.engine,
            "pipeline": spec.pipeline,
            "repo": spec.repo,
            "steps": spec.steps,
            "vram_gb": spec.vram_gb,
            "family": spec.family,
            "notes": spec.notes or "",
            "available": available,
            "downloaded": state.downloaded,
            "downloading": state.downloading,
            "size_bytes": state.size_bytes,
            "progress": round(state.progress, 2),
            "error": state.error or "",
            "download_path": repo_cache_dir(spec.repo) or "",
            "bytes_on_disk": state.bytes_downloaded,
        }

    def _op_dict(self, state) -> dict:
        return {
            "key": state.key,
            "state": state.state,
            "progress": round(state.progress, 2),
            "bytes_downloaded": state.bytes_downloaded,
            "total_bytes": state.total_bytes,
            "error": state.error or "",
        }

    def _token_ok_req(self, parsed) -> bool:
        if not ARTIFACT_TOKEN:
            return True
        provided = parse_qs(parsed.query).get("token", [""])[0]
        return self._token_ok(provided)

    # ------------------------------------------------------------------
    # HTTP verbs
    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 (API del stdlib)
        self._serve(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 (API del stdlib)
        # HEAD: mismos headers que GET sin el cuerpo (probes del ps1, etc.)
        self._serve(include_body=False)

    def do_POST(self) -> None:  # noqa: N802 (API del stdlib)
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/models/"):
            self._send_json(404, {"error": "not found"})
            return
        if not self._token_ok_req(parsed):
            self._send_json(403, {"error": "token inválido"})
            return
        rest = parsed.path[len("/models/"):]
        if not rest or "/" in rest:
            self._send_json(404, {"error": "not found"})
            return
        key = rest
        if parsed.path == "/models/"+key+"/download":
            ops = _get_ops()
            try:
                state = ops.start_download(key)
            except KeyError:
                self._send_json(404, {"error": "modelo desconocido: "+key})
                return
            except ValueError as exc:
                self._send_json(409, {"error": str(exc)})
                return
            self._send_json(200, self._op_dict(state))
            return
        if parsed.path == "/models/"+key+"/cancel":
            ops = _get_ops()
            try:
                state = ops.cancel(key)
            except KeyError:
                self._send_json(404, {"error": "modelo desconocido: "+key})
                return
            self._send_json(200, self._op_dict(state))
            return
        self._send_json(404, {"error": "not found"})

    def do_DELETE(self) -> None:  # noqa: N802 (API del stdlib)
        parsed = urlparse(self.path)
        if parsed.path.startswith("/models/"):
            if not self._token_ok_req(parsed):
                self._send_json(403, {"error": "token inválido"})
                return
            rest = parsed.path[len("/models/"):]
            if not rest or rest == "/" or "/" in rest:
                self._send_json(400, {"error": "clave de modelo requerida"})
                return
            key = rest
            if not _safe_name(key):
                self._send_json(400, {"error": "clave de modelo inválida"})
                return
            ops = _get_ops()
            try:
                state = ops.delete(key)
            except KeyError:
                self._send_json(404, {"error": "modelo desconocido: "+key})
                return
            except ValueError as exc:
                self._send_json(409, {"error": str(exc)})
                return
            self._send_json(200, self._op_dict(state))
            return
        if not parsed.path.startswith("/artifacts/"):
            self._send_json(404, {"error": "not found"})
            return
        if ARTIFACT_TOKEN and not self._token_ok(parse_qs(parsed.query).get("token", [""])[0]):
            self._send_json(403, {"error": "token inválido"})
            return
        name = parsed.path[len("/artifacts/"):]
        if not name or name == "/":
            self._send_json(400, {"error": "nombre de archivo requerido"})
            return
        if not _safe_name(name):
            self._send_json(400, {"error": "nombre de archivo inválido"})
            return
        if not _delete_artifact(name):
            self._send_json(404, {"error": "artefacto no encontrado"})
            return
        self._send_json(200, {"deleted": name, "ok": True})

    def _serve(self, include_body: bool) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self._send_json(200, {"status": "ok"})
            return
        if parsed.path == "/models":
            if not self._token_ok_req(parsed):
                self._send_json(403, {"error": "token inválido"})
                return
            ops = _get_ops()
            models = []
            for key in ops.keys():
                entry = self._model_dict(key)
                if entry is not None:
                    models.append(entry)
            self._send_json(200, {"models": models, "count": len(models)})
            return
        if parsed.path.startswith("/models/"):
            if not self._token_ok_req(parsed):
                self._send_json(403, {"error": "token inválido"})
                return
            rest = parsed.path[len("/models/"):]
            if not rest or rest == "/" or "/" in rest:
                self._send_json(404, {"error": "not found"})
                return
            entry = self._model_dict(rest)
            if entry is None:
                self._send_json(404, {"error": "modelo desconocido: "+rest})
                return
            self._send_json(200, entry)
            return
        if not parsed.path.startswith("/artifacts/"):
            self._send_json(404, {"error": "not found"})
            return
        if ARTIFACT_TOKEN and not self._token_ok(parse_qs(parsed.query).get("token", [""])[0]):
            self._send_json(403, {"error": "token inválido"})
            return
        name = parsed.path[len("/artifacts/"):]
        if not _safe_name(name):
            self._send_json(400, {"error": "nombre de archivo inválido"})
            return
        path = os.path.join(os.path.abspath(OUTPUT_DIR), name)
        if not os.path.isfile(path):
            self._send_json(404, {"error": "artefacto no encontrado"})
            return
        ctype = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".mp4": "video/mp4", ".webm": "video/webm", ".wav": "audio/wav",
            ".mp3": "audio/mpeg", ".json": "application/json",
        }.get(os.path.splitext(name)[1].lower(), "application/octet-stream")
        try:
            with open(path, "rb") as fh:
                payload = fh.read()
        except OSError as exc:
            logger.warning("No se pudo leer %s: %s", name, exc)
            self._send_json(500, {"error": "error leyendo artefacto"})
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        if include_body:
            self.wfile.write(payload)

    # ------------------------------------------------------------------
    def _token_ok(self, provided: str) -> bool:
        if provided == ARTIFACT_TOKEN:
            return True
        return self.headers.get("X-Artifact-Token", "") == ARTIFACT_TOKEN

    def _send_json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:  # silencia el log por request
        logger.debug("artifact: " + fmt, *args)


def _list_artifacts() -> list[dict]:
    """Archivos planos de OUTPUT_DIR con su tamaño (los ocultos no se listan).

    Mismo criterio que el GET por nombre: sólo nombres planos, así que las
    subcarpetas (y lo que haya adentro) quedan fuera del listado.
    """
    out = []
    base = os.path.abspath(OUTPUT_DIR)
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isfile(path) or name.startswith("."):
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        out.append({"name": name, "size_bytes": size})
    return out


def _delete_artifact(name: str) -> bool:
    """Borra un artefacto por nombre plano. True si existía y se borró."""
    if not _safe_name(name):
        return False
    path = os.path.join(os.path.abspath(OUTPUT_DIR), name)
    if not os.path.isfile(path):
        return False
    try:
        os.remove(path)
        logger.info("Artefacto borrado: %s", name)
        return True
    except OSError as exc:
        logger.warning("No se pudo borrar %s: %s", name, exc)
        return False


def start_server(port: int, ops: ModelOps | None = None) -> ThreadingHTTPServer:
    global _OPS
    if ops is not None:
        _OPS = ops
    httpd = ThreadingHTTPServer(("0.0.0.0", port), ArtifactHandler)
    logger.info("🎨 Artifact server escuchando en :%s (dir=%s)", port, OUTPUT_DIR)
    return httpd


def start_background(port: int, ops: ModelOps | None = None) -> threading.Thread | None:
    """Arranca el servidor en daemon thread; None si port <= 0.

    `ops` es el ModelOps del servidor gRPC, para compartir el estado de
    descarga en tiempo real entre gRPC y HTTP.
    """
    if port <= 0:
        return None
    try:
        httpd = start_server(port, ops)
    except OSError as exc:
        logger.warning("Artifact server no disponible en :%s (%s)", port, exc)
        return None
    t = threading.Thread(target=httpd.serve_forever, name="artifact-server", daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    port = int(os.environ.get("BM_ARTIFACT_PORT", "50052"))
    logging.basicConfig(level=logging.INFO)
    httpd = start_server(port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
