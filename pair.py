"""Wrapper de conveniencia: `python pair.py` desde la raíz del worker.

La implementación vive en bridge/pair.py; este archivo existe para que el
usuario final no tenga que saber de paquetes Python.
"""

import sys

# Consolas Windows (cp1252) revientan con los emojis de la salida.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from bridge.pair import main  # noqa: E402  (tras reconfigurar la consola)

if __name__ == "__main__":
    raise SystemExit(main())
