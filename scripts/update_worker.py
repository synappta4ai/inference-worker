"""Auto-update del Synapta Worker.

python scripts/update_worker.py --url <url_zip> [--sha256 <hex>] [--dir <workdir>]

Se lanza como subproceso desacoplado desde POST /worker/update del artifact
server. Descarga el payload zip, verifica hash, detiene el worker viejo,
reemplaza archivos y relanza el worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))

from core import (  # noqa: E402
    default_dir,
    download_payload,
    ensure_pip,
    ensure_python,
    extract_payload,
    install_deps,
    run_doctor,
    start,
    stop,
    worker_dir,
)

STATE_FILE = "update-state.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_state(workdir: Path, data: dict) -> None:
    (workdir / STATE_FILE).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser(prog="update_worker")
    ap.add_argument("--url", required=True)
    ap.add_argument("--sha256", default="")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()

    workdir = (Path(args.dir) if args.dir else default_dir()).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    def fail(msg: str) -> int:
        write_state(workdir, {"status": "failed", "error": msg})
        print(f"✘ {msg}")
        return 1

    tmp = workdir / "_update-download.zip"
    try:
        download_payload(args.url, tmp)
    except Exception as exc:
        return fail(f"descarga falló: {exc}")

    if args.sha256:
        actual = sha256_file(tmp)
        if actual.lower() != args.sha256.lower():
            return fail(
                f"hash no coincide: esperado={args.sha256} obtenido={actual}"
            )

    try:
        stop(workdir)
    except Exception as exc:
        print(f"⚠ stop falló (ignorando): {exc}")

    try:
        extract_payload(tmp, worker_dir(workdir))
    except Exception as exc:
        return fail(f"extracción falló: {exc}")

    try:
        py = ensure_python(workdir)
        ensure_pip(py, workdir)
        install_deps(py, workdir)
    except Exception as exc:
        return fail(f"deps falló: {exc}")

    try:
        run_doctor(py, workdir)
    except Exception as exc:
        return fail(f"doctor falló: {exc}")

    try:
        start(workdir)
    except Exception as exc:
        return fail(f"arranque falló: {exc}")

    write_state(workdir, {"status": "ok"})
    print("✓ worker actualizado y corriendo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
