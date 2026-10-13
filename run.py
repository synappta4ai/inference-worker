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


_load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run")


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
