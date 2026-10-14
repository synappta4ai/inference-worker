"""Punto de entrada único del worker, en cualquier sistema operativo.

Arranca en un solo proceso:

  1. El servidor gRPC local (server.py) — la cola de inferencia de siempre,
     con todos los engines y el VRAM manager.
  2. Si la máquina está emparejada (bridge/config.json), el agente bridge:
     heartbeat + claim de trabajos + subida de artefactos al back.

Uso:

    python run.py                 # arranca todo lo que esté configurado
    BM_FORCE_MOCK=1 python run.py # modo mock (sin GPU, sin pesos)

Si no está emparejado, el worker sirve igual en modo local (el back de la
misma máquina puede hablarle por gRPC directo como siempre); el bridge solo
agrega el modo "GPU en la nube". Ctrl+C detiene todo limpiamente.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

# Consolas Windows (cp1252) revientan con los emojis de los logs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Cargar .env si existe (desarrollo local; el servicio de producción usa env real).
def _load_dotenv() -> None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"'))
    except OSError:
        pass


# Variables que apuntan al CA store para las librerías que las leen solas.
_CA_ENV_VARS = ("SSL_CERT_FILE", "PIP_CERT", "REQUESTS_CA_BUNDLE",
                "CURL_CA_BUNDLE")


def _own_cacert() -> Optional[Path]:
    """cacert.pem del python del worker (ruta estable, dentro de la app)."""
    try:
        import certifi
    except Exception:  # noqa: BLE001 — sin certifi se cae a quitar la var
        return None
    path = Path(certifi.where())
    return path if path.is_file() else None


def _inherited_ca_is_usable(value: str) -> bool:
    """true si el CA heredado sirve de verdad para TLS.

    Tiene que ser un ARCHIVO legible y fuera de la carpeta temporal del
    instalador: PyInstaller extrae `_MEIxxxx` al arrancar el .exe y la borra
    al cerrarse, así que un SSL_CERT_FILE que apunte ahí (o a una carpeta
    con ese nombre) deja TODA descarga de modelos muerta con
    "PermissionError: [Errno 13] Permission denied" — sin traza en el panel,
    que sólo ve "No disponible" para siempre.
    """
    if not value:
        return False
    path = Path(value)
    if not path.is_file() or not os.access(path, os.R_OK):
        return False
    try:
        parts = path.resolve().parts
    except OSError:
        return False
    return not any(part.upper().startswith("_MEI") for part in parts)


def _normalize_ca_env() -> None:
    """Sanea el CA heredado del proceso que arrancó al worker.

    httpx (huggingface_hub) arma su contexto con
    `ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])`: con una
    carpeta o una ruta ya inexistente falla y la descarga de pesos nunca
    empieza. Se reemplaza por el certifi del propio worker (que sí es
    estable) y, si no está, se quita la variable — mejor contexto del
    sistema que una ruta rota.
    """
    own = _own_cacert()
    for var in _CA_ENV_VARS:
        value = os.environ.get(var)
        if not value or _inherited_ca_is_usable(value):
            continue
        if own is not None:
            os.environ[var] = str(own)
        else:
            os.environ.pop(var, None)
        logger.info("[run] %s heredado inválido (%s) → usando el CA del "
                    "worker: %s", var, value,
                    own if own is not None else "el del sistema")


_load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run")

# Después del logging: el saneo avisa por log cuándo reemplaza una CA heredada.
_normalize_ca_env()


def _start_bridge_agent() -> "threading.Thread | None":
    """Arranca el agente bridge si hay config; devuelve el thread (o None)."""
    from bridge import config
    from bridge.agent import Agent

    cfg = config.load()
    if not cfg.paired:
        logger.info("[run] sin emparejar: worker solo local. "
                    "Para conectarlo a Synapta: python pair.py")
        return None
    agent = Agent(cfg)
    thread = threading.Thread(target=agent.run, name="bridge-agent", daemon=True)
    thread.start()
    logger.info("[run] agente bridge arrancado (worker=%s → %s)",
                cfg.worker_id, cfg.back_url)
    return thread


def main() -> int:
    import server  # después de .env para que OUTPUT_DIR lea el env correcto

    agent_thread = _start_bridge_agent()

    try:
        # serve() bloquea hasta SIGINT/SIGTERM (ya los maneja con KeyboardInterrupt).
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        pass
    finally:
        if agent_thread is not None:
            # El agente es daemon: el proceso puede salir; el journal ya
            # persistió cualquier artefacto pendiente de subida.
            logger.info("[run] apagando… los artefactos pendientes quedan "
                        "en el journal para el próximo arranque.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
