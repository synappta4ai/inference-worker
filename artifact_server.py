"""Servidor HTTP de artefactos para despliegues divididos.

Cuando el worker corre en la nube (Colab/Runpod/Vast) y el gateway en otra
máquina, el PNG/MP4 generado vive en el disco del worker. Este servidor
expone BM_OUTPUT_DIR por HTTP para que el gateway pueda servir `artifact_url`
al front.

  GET /artifacts/<nombre_de_archivo>[?token=...]   → el archivo
  Header alternativo: X-Artifact-Token
  GET /healthz                                     → {"status":"ok"}

Solo nombres de archivo (sin rutas): el handler rechaza cualquier path
con separadores, `..` o prefijo de punto (incluye .token). El token se
configura con BM_ARTIFACT_TOKEN; sin esa variable el servidor queda abierto
(pensado para túneles efímeros de Colab/Kaggle).

Se puede usar de dos formas:
  - independiente:  python artifact_server.py          (bloquea)
  - dentro de server.py: se arranca en un daemon thread automáticamente
    cuando BM_ARTIFACT_PORT está definido (default 50052, 0 = desactivado).
"""

from __future__ import annotations

import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("ArtifactServer")

OUTPUT_DIR = os.environ.get("BM_OUTPUT_DIR", "outputs")
ARTIFACT_TOKEN = os.environ.get("BM_ARTIFACT_TOKEN", "")


def _safe_name(name: str) -> bool:
    """True solo para nombres planos de archivo (sin traversal ni ocultos)."""
    if not name or name.startswith("."):
        return False
    if "/" in name or "\\" in name or name in (".", ".."):
        return False
    return ".." not in name.split(os.sep)


class ArtifactHandler(BaseHTTPRequestHandler):
    server_version = "BrainMasterArtifacts/1.0"

    def do_GET(self) -> None:  # noqa: N802 (API del stdlib)
        self._serve(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 (API del stdlib)
        # HEAD: mismos headers que GET sin el cuerpo (probes del ps1, etc.)
        self._serve(include_body=False)

    def _serve(self, include_body: bool) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self._send_json(200, {"status": "ok"})
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
        import json
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:  # silencia el log por request
        logger.debug("artifact: " + fmt, *args)


def start_server(port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("0.0.0.0", port), ArtifactHandler)
    logger.info("🎨 Artifact server escuchando en :%s (dir=%s)", port, OUTPUT_DIR)
    return httpd


def start_background(port: int) -> threading.Thread | None:
    """Arranca el servidor en daemon thread; None si port <= 0."""
    if port <= 0:
        return None
    try:
        httpd = start_server(port)
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
