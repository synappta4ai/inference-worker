"""Genera el instalador gráfico: SynaptaWorker-Setup.exe (PyInstaller).

    py -3 scripts/build_installer.py [--into-front]

Qué empaqueta:
  * installer/app.py + installer/core.py (GUI tkinter, stdlib solamente)
  * _payload.zip   → código del worker (lo que se extrae en el cliente)
  * _cacert.pem    → raíces CA (PyInstaller y el Python embeddable no
                     traen store propio; sin esto pip/HTTPS fallarían)

Salida:
  dist/SynaptaWorker-Setup.exe                      (onefile, sin consola)
  ../front-synapta/src/assets/worker/SynaptaWorker-Setup.exe   (--into-front)
  ../front-synapta/src/assets/worker/synapta-worker.zip        (--into-front)

El zip que se regenera acá es la descarga para máquinas sin Windows
(mismo contenido que instala el .exe), así ambos siempre coinciden.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Windows + stdout en pipe = cp1252: los "→" de estos prints reventarían.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

WORKER_ROOT = Path(__file__).resolve().parent.parent
FRONT_ASSETS = WORKER_ROOT.parent / "front-synapta" / "src" / "assets" / "worker"

sys.path.insert(0, str(WORKER_ROOT))
from installer import core  # noqa: E402


def build(into_front: bool = False) -> Path:
    import certifi

    tmp = Path(tempfile.mkdtemp(prefix="synapta-build-"))
    payload = tmp / "_payload.zip"
    files = core.build_payload(WORKER_ROOT, payload)
    cacert = Path(certifi.where())
    print(f"payload: {files} archivos · CA: {cacert.name}")

    sep = ";" if sys.platform == "win32" else ":"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onefile", "--windowed",
        "--name", "SynaptaWorker-Setup",
        "--paths", str(WORKER_ROOT),
        "--add-data", f"{payload}{sep}_payload.zip",
        "--add-data", f"{cacert}{sep}_cacert.pem",
        "--distpath", str(WORKER_ROOT / "dist"),
        "--workpath", str(tmp / "build"),
        "--specpath", str(tmp),
        str(WORKER_ROOT / "installer" / "app.py"),
    ]
    print("$", " ".join(cmd[1:]))
    rc = subprocess.run(cmd, cwd=str(WORKER_ROOT)).returncode
    if rc != 0:
        raise SystemExit(f"PyInstaller falló ({rc})")

    exe = WORKER_ROOT / "dist" / (
        "SynaptaWorker-Setup.exe" if sys.platform == "win32"
        else "SynaptaWorker-Setup")
    if not exe.exists():
        raise SystemExit(f"no apareció {exe}")

    if into_front:
        FRONT_ASSETS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exe, FRONT_ASSETS / "SynaptaWorker-Setup.exe")
        zip_dest = FRONT_ASSETS / "synapta-worker.zip"
        core.build_payload(WORKER_ROOT, zip_dest)  # siempre fresco
        print(f"→ copiado al front: {(FRONT_ASSETS / 'SynaptaWorker-Setup.exe').name}"
              f" ({(FRONT_ASSETS / 'SynaptaWorker-Setup.exe').stat().st_size // 1024} KB)")
        print(f"→ zip no-Windows regenerado: {zip_dest.name}"
              f" ({zip_dest.stat().st_size // 1024} KB)")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"OK: {exe} ({exe.stat().st_size // 1024 // 1024} MB)")
    return exe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--into-front", action="store_true",
                        help="copia el .exe y regenera el zip en assets/worker")
    args = parser.parse_args()
    build(into_front=args.into_front)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
