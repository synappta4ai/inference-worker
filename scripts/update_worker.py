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
    PAYLOAD_URL,
    run_doctor,
    start,
    stop_for_update,
    worker_dir,
)

STATE_FILE = "update-state.json"
# Fallo que el agente manda al back en su próximo heartbeat (id + motivo), para
# que el panel muestre por qué no se aplicó en vez de esperar «Actualizando…»
# hasta que se le acabe el reloj.
ERROR_FILE = "update-error.json"


def say(msg: str) -> None:
    """print que no rompe el updater.

    Corre desacoplado, sin consola: sys.stdout puede ser None (pythonw) y en
    Windows cp1252 revienta con los "✓" (UnicodeEncodeError). Perder una
    línea de log jamás debe abortar la actualización.
    """
    try:
        print(msg, flush=True)
    except Exception:  # noqa: BLE001 — sólo es diagnóstico
        pass


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


def write_error(workdir: Path, request_id: str, url: str, msg: str) -> None:
    """Deja el motivo del fallo para que el agente lo reporte al back."""
    (workdir / ERROR_FILE).write_text(
        json.dumps({"id": request_id, "url": url, "error": msg},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    try:  # consola cp1252 de Windows: los acentos y "✓" no deben romper nada
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — stdout puede no existir
        pass
    ap = argparse.ArgumentParser(prog="update_worker")
    # Sin --url se usa el payload publicado: así lo puede lanzar el agente
    # cuando el panel no eligió una URL explícita.
    ap.add_argument("--url", default="")
    ap.add_argument("--sha256", default="")
    # Id del pedido que encoló el panel: viaja en el reporte del fallo para que
    # el back lo asocie a ESA actualización (y no a un reintento posterior).
    ap.add_argument("--id", default="")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()

    workdir = (Path(args.dir) if args.dir else default_dir()).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    # Un intento nuevo supersede cualquier fallo viejo que no se haya reportado.
    (workdir / ERROR_FILE).unlink(missing_ok=True)
    stopped = False

    def fail(msg: str) -> int:
        write_state(workdir, {"status": "failed", "error": msg})
        write_error(workdir, args.id, args.url or PAYLOAD_URL, msg)
        say(f"[x] {msg}")
        # El worker es quien reporta al back: si ya lo habíamos detenido (falló
        # la extracción, las deps o el arranque) lo relanzamos para que el
        # próximo heartbeat lleve el motivo. Si ni así arranca, el back lo da
        # por fallido con el watchdog del reinicio.
        if stopped:
            try:
                start(workdir, require_clean=False)
                say("worker relanzado tras el fallo")
            except Exception as exc:  # noqa: BLE001 — best-effort
                say(f"[x] no pude relanzar el worker tras el fallo: {exc}")
        return 1

    tmp = workdir / "_update-download.zip"
    try:
        download_payload(args.url or PAYLOAD_URL, tmp)
    except Exception as exc:
        return fail(f"descarga falló: {exc}")

    if args.sha256:
        actual = sha256_file(tmp)
        if actual.lower() != args.sha256.lower():
            return fail(
                f"hash no coincide: esperado={args.sha256} obtenido={actual}"
            )

    try:
        # Sin árbol: este proceso es hijo del worker y `/T` lo mataría a él.
        # Si el puerto no queda libre se aborta ANTES de extraer, en vez de
        # dejar archivos nuevos sobre un proceso que sigue con el código viejo.
        stop_for_update(workdir, tree=False)
        stopped = True
    except Exception as exc:
        return fail(f"no pude detener el worker: {exc}")

    try:
        extract_payload(tmp, worker_dir(workdir))
        tmp.unlink(missing_ok=True)
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
        start(workdir, require_clean=True)
    except Exception as exc:
        return fail(f"arranque falló: {exc}")

    write_state(workdir, {"status": "ok"})
    say("worker actualizado y corriendo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
