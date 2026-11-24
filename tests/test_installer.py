"""Tests del instalador (installer/core.py y la GUI).

Cubren la lógica sin red ni descargas: construcción de payload, parche
del ._pth del Python embeddable, validación del código de emparejamiento,
CLI (status/errores) y construcción de la GUI (una instancia Tk real,
destruida al terminar — no necesita pantalla interactiva).
"""

from __future__ import annotations

import io
import json
import logging
import os
import queue as queue_mod
import sys
import zipfile
from pathlib import Path

import pytest

from installer import core


# ── payload ───────────────────────────────────────────────────────
def _make_repo(root, files: dict) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_build_payload_excludes_junk(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo, {
        "run.py": "print('hi')\n",
        "requirements.txt": "grpcio\n",
        "bridge/agent.py": "x = 1\n",
        "tests/test_x.py": "def test_x(): pass\n",
        "__pycache__/run.cpython-312.pyc": "junk",
        ".git/HEAD": "ref: refs/heads/main\n",
        "outputs/video.mp4": "junk",
        "worker-err.log": "traceback",
        ".env": "SECRET=1",
        "models/weights.pt": "junk",
        "installer/app.py": "self = 1\n",
    })
    dest = tmp_path / "payload.zip"
    count = core.build_payload(repo, dest)

    with zipfile.ZipFile(dest) as zf:
        names = set(zf.namelist())
    # lo que el worker necesita para correr
    assert "run.py" in names
    assert "bridge/agent.py" in names
    assert "requirements.txt" in names
    # basura, tests y secretos quedan afuera
    assert not any(n.startswith((".git/", "__pycache__/", "outputs/",
                                 "models/", "tests/"))
                   for n in names)
    assert "worker-err.log" not in names
    assert ".env" not in names
    # `installer/` SÍ viaja: update_worker.py importa core de ahí y sin eso el
    # auto-update del worker moría con ImportError.
    assert "installer/app.py" in names
    assert count == len(names) >= 3


def test_extract_payload_roundtrip(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo, {"run.py": "print('ok')\n"})
    zip_path = tmp_path / "p.zip"
    core.build_payload(repo, zip_path)

    target = tmp_path / "install" / "worker"
    core.extract_payload(zip_path, target)
    assert (target / "run.py").read_text(encoding="utf-8") == "print('ok')\n"


def test_extract_payload_retries_av_lock(tmp_path, monkeypatch):
    """El antivirus bloquea el zip recién extraído: reintenta y sale."""
    repo = tmp_path / "repo"
    _make_repo(repo, {"run.py": "print('ok')\n"})
    zip_path = tmp_path / "p.zip"
    core.build_payload(repo, zip_path)

    real_zipfile = zipfile.ZipFile
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(13, "Permission denied", str(args[0]))
        return real_zipfile(*args, **kwargs)

    monkeypatch.setattr(zipfile, "ZipFile", flaky)
    monkeypatch.setattr(core.time, "sleep", lambda *a: None)

    target = tmp_path / "install" / "worker"
    core.extract_payload(zip_path, target)
    assert (target / "run.py").read_text(encoding="utf-8") == "print('ok')\n"
    assert calls["n"] == 3


def test_extract_payload_gives_up_with_hint(tmp_path, monkeypatch):
    """Si el bloqueo persiste, el error dice qué hacer (antivirus)."""
    repo = tmp_path / "repo"
    _make_repo(repo, {"run.py": "print('ok')\n"})
    zip_path = tmp_path / "p.zip"
    core.build_payload(repo, zip_path)

    def locked(*args, **kwargs):
        raise PermissionError(13, "Permission denied", str(args[0]))

    monkeypatch.setattr(zipfile, "ZipFile", locked)
    monkeypatch.setattr(core.time, "sleep", lambda *a: None)
    with pytest.raises(RuntimeError, match="antivirus"):
        core.extract_payload(tmp_path / "p.zip", tmp_path / "install")


# ── python embeddable ─────────────────────────────────────────────
def test_patch_pth_enables_site(tmp_path):
    pth = tmp_path / "python312._pth"
    pth.write_text("python312.zip\n.\n#import site\n", encoding="utf-8")
    core._patch_pth(pth)
    text = pth.read_text(encoding="utf-8")
    assert "import site" in text
    assert "#import site" not in text


def test_patch_pth_idempotent(tmp_path):
    pth = tmp_path / "python312._pth"
    pth.write_text("python312.zip\n.\n#import site\n", encoding="utf-8")
    core._patch_pth(pth)
    first = pth.read_text(encoding="utf-8")
    core._patch_pth(pth)
    assert pth.read_text(encoding="utf-8") == first


def test_patch_pth_without_marker_still_adds_site(tmp_path):
    pth = tmp_path / "python312._pth"
    pth.write_text("python312.zip\n.\n", encoding="utf-8")
    core._patch_pth(pth)
    assert pth.read_text(encoding="utf-8").rstrip().endswith("import site")


# ── paths ─────────────────────────────────────────────────────────
def test_default_dir_uses_localappdata(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert core.default_dir() == tmp_path / "SynaptaWorker"


def test_python_and_worker_paths(tmp_path):
    assert core.python_path(tmp_path).name in ("python.exe", "python")
    assert core.python_path(tmp_path).parent.name == "python"
    assert core.worker_dir(tmp_path) == tmp_path / "worker"


# ── CA store (SSL_CERT_FILE / PIP_CERT) ──────────────────────────
def test_cacert_path_accepts_bundled_pem_file(tmp_path, monkeypatch):
    bundled = tmp_path / "_cacert.pem"
    bundled.write_text("PEM", encoding="utf-8")
    monkeypatch.setattr(core.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert core.cacert_path() == bundled


def test_cacert_path_unwraps_bundled_directory(tmp_path, monkeypatch):
    """Regresión: PyInstaller extrajo `_cacert.pem` como CARPETA.

    Devolverla como CA rompe `ssl.create_default_context(cafile=...)` con
    "PermissionError: [Errno 13] Permission denied", y el worker heredaba ese
    SSL_CERT_FILE: ninguna descarga de pesos llegaba a arrancar.
    """
    folder = tmp_path / "_cacert.pem"
    folder.mkdir()
    (folder / "cacert.pem").write_text("PEM", encoding="utf-8")
    monkeypatch.setattr(core.sys, "_MEIPASS", str(tmp_path), raising=False)
    resolved = core.cacert_path()
    assert resolved is not None and resolved.is_file()
    assert resolved.name == "cacert.pem"


def test_net_env_never_points_ca_at_a_directory(tmp_path, monkeypatch):
    """El env que hereda el worker (y pip) debe apuntar a un archivo."""
    folder = tmp_path / "_cacert.pem"
    folder.mkdir()
    (folder / "cacert.pem").write_text("PEM", encoding="utf-8")
    monkeypatch.setattr(core.sys, "_MEIPASS", str(tmp_path), raising=False)
    env = core.net_env()
    for var in ("SSL_CERT_FILE", "PIP_CERT"):
        value = env.get(var)
        if value is not None:
            assert Path(value).is_file(), f"{var} → {value} no es un archivo"


# ── payload embebido en el .exe ─────────────────────────────────
def test_payload_source_unwraps_bundled_directory(tmp_path, monkeypatch):
    """Regresión: PyInstaller guardó `_payload.zip` como CARPETA.

    Devolverla tal cual hacía que `ZipFile(<carpeta>)` reventara con
    IsADirectoryError y el instalador bajara el payload de la red siempre
    (y sin internet no instalaba), aunque el zip embebido estaba al lado.
    """
    folder = tmp_path / "_payload.zip"
    folder.mkdir()
    inner = folder / "_payload.zip"
    inner.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    monkeypatch.setattr(core.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert core.payload_source() == inner


def test_payload_source_accepts_flat_zip(tmp_path, monkeypatch):
    bundled = tmp_path / "_payload.zip"
    bundled.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    monkeypatch.setattr(core.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert core.payload_source() == bundled


def test_payload_source_frozen_without_bundle_never_zips_the_bundle(tmp_path, monkeypatch):
    """Sin payload embebido el instalador debe caer al zip publicado (OSError).

    Armar "desde el repo" congelado tomaría el propio _MEIPASS (DLLs y todo)
    como payload del worker.
    """
    (tmp_path / "_payload.zip").mkdir()
    monkeypatch.setattr(core.sys, "_MEIPASS", str(tmp_path), raising=False)
    with pytest.raises(OSError, match="payload embebido"):
        core.payload_source()


def test_payload_source_falls_back_to_repo(tmp_path, monkeypatch):
    """Sin embebido (corriendo desde el repo) arma el zip al vuelo."""
    repo = tmp_path / "repo"
    _make_repo(repo, {"run.py": "print('ok')\n"})
    monkeypatch.setattr(core, "__file__", str(repo / "installer" / "core.py"))
    monkeypatch.delattr(core.sys, "_MEIPASS", raising=False)
    with zipfile.ZipFile(core.payload_source()) as zf:
        assert "run.py" in zf.namelist()


def test_bundled_payload_is_a_real_zip_after_build():
    """El .exe de dist/ existe: su payload embebido se extrae sin tocar la red.

    Es el caso exacto que rompía: si el layout del bundle cambia (otro
    PyInstaller, otro `--add-data`), este test lo caza antes de publicarlo.
    """
    exe = Path(__file__).resolve().parent.parent / "dist" / (
        "SynaptaWorker-Setup.exe" if os.name == "nt" else "SynaptaWorker-Setup")
    if not exe.is_file():
        pytest.skip("no hay dist/ construido en esta máquina")
    from PyInstaller.archive.readers import CArchiveReader

    archive = CArchiveReader(str(exe))
    entries = archive.toc
    name = next(n for n in entries if n.rstrip("\\").endswith("_payload.zip"))
    assert entries[name][-1] == "b", f"{name} no es un archivo embebido"
    blob = archive.extract(name)
    blob = blob if isinstance(blob, bytes) else blob.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        assert {"run.py", "pair.py", "requirements.txt"} <= set(zf.namelist())
        assert "_normalize_ca_env" in zf.read("run.py").decode("utf-8")


# ── validación de emparejamiento (sin red: falla antes) ──────────
def test_pair_rejects_bad_code(tmp_path):
    with pytest.raises(RuntimeError, match="6 dígitos"):
        core.pair(tmp_path, back="http://localhost:9999", code="12ab56")


def test_pair_rejects_bad_back(tmp_path):
    with pytest.raises(RuntimeError, match="URL"):
        core.pair(tmp_path, back="ftp://x", code="123456")


def test_pair_requires_install(tmp_path):
    with pytest.raises(RuntimeError, match="instalado"):
        core.pair(tmp_path, back="http://localhost:9999", code="123456")


# ── status ────────────────────────────────────────────────────────
def test_worker_status_uninstalled(tmp_path, monkeypatch):
    # el puerto real puede tener un worker de desarrollo: lo mockeamos
    monkeypatch.setattr(core, "port_open", lambda *a, **k: False)
    st = core.worker_status(tmp_path)
    assert st["installed"] is False
    assert st["running"] is False
    json.dumps(st)  # el CLI lo imprime como JSON


def test_worker_status_installed_not_running(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "port_open", lambda *a, **k: False)
    py = core.python_path(tmp_path)
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    run_py = core.worker_dir(tmp_path) / "run.py"
    run_py.parent.mkdir(parents=True, exist_ok=True)
    run_py.write_text("print('hi')\n", encoding="utf-8")

    st = core.worker_status(tmp_path)
    assert st["installed"] is True
    assert st["running"] is False


def test_port_open_closed_port():
    # 1 nunca tiene listener: no debe colgarse ni lanzar
    assert core.port_open(1) is False


def test_autostart_cmd_shape(tmp_path):
    cmd = core.autostart_cmd(tmp_path, enable=True)
    assert cmd[0] == "powershell"
    joined = " ".join(cmd)
    assert "SynaptaWorker.lnk" in joined
    assert str(tmp_path.resolve()) in joined
    assert "Startup" in joined


def test_autostart_cmd_disable(tmp_path):
    cmd = core.autostart_cmd(tmp_path, enable=False)
    assert "Remove-Item" in " ".join(cmd)


@pytest.mark.skipif(os.name != "nt", reason="autostart aplica a Windows")
def test_autostart_enabled_reflects_link(tmp_path, monkeypatch):
    """El estado que muestra la GUI debe ser el del atajo real."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    link = core.startup_link()
    assert link.name == "SynaptaWorker.lnk"
    assert link.parent.name == "Startup"
    assert core.autostart_enabled() is False
    link.parent.mkdir(parents=True)
    link.write_bytes(b"lnk")
    assert core.autostart_enabled() is True


def test_autostart_cmd_targets_startup_link(tmp_path):
    cmd = " ".join(core.autostart_cmd(tmp_path, enable=True))
    assert core.startup_link().name in cmd


def test_autostart_cmd_sin_comillas_anidadas(tmp_path):
    """Regresión: `-c \"import …\"` en Arguments rompía powershell -Command
    (y el --off salía con rc=1 aunque no hubiera atajo)."""
    for enable in (True, False):
        cmd = " ".join(core.autostart_cmd(tmp_path, enable=enable))
        assert '"' not in cmd, cmd
        assert "exit 0" in cmd, cmd
    assert "start-worker.cmd" in \
        " ".join(core.autostart_cmd(tmp_path, enable=True))


def test_write_launcher(tmp_path):
    script = core._write_launcher(tmp_path)
    assert script == core.launcher_script(tmp_path)
    body = script.read_text(encoding="utf-8")
    assert str(core.python_path(tmp_path)) in body
    assert "run.py" in body
    assert "autostart.log" in body


# ── CLI ───────────────────────────────────────────────────────────
def test_cli_status_json(tmp_path, capsys):
    rc = core.main(["--dir", str(tmp_path), "status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"installed"' in out


def test_cli_pair_invalid_code_exits_1(tmp_path, capsys):
    rc = core.main(["--dir", str(tmp_path), "pair", "--code", "abc"])
    assert rc == 1
    assert "6 dígitos" in capsys.readouterr().out


def test_cli_unknown_command(capsys):
    with pytest.raises(SystemExit) as exc:
        core.main(["no-existe"])
    assert exc.value.code == 2


# ── clasificación de errores de pairing ─────────────────────────
def test_classify_expired_code():
    msg = core.classify_pair_failure(1, ["❌ No pude emparejar: POST /register → HTTP 401: expired"])
    assert "venció" in msg and "nuevo" in msg


def test_classify_connection_refused():
    msg = core.classify_pair_failure(
        1, ["❌ No pude emparejar: POST /register sin conexión: [Errno 111] Connection refused"])
    assert "no llegué al back" in msg


def test_classify_tls_failure():
    msg = core.classify_pair_failure(1, ["❌ No pude emparejar: SSL: CERTIFICATE_VERIFY_FAILED"])
    assert "TLS" in msg


def test_classify_404_wrong_url():
    msg = core.classify_pair_failure(1, ["POST /register → HTTP 404: not found"])
    assert "/api/v1" in msg


def test_classify_unknown_keeps_last_line():
    msg = core.classify_pair_failure(3, ["Conectando con http://x…", "algo raro pasó"])
    assert "algo raro pasó" in msg


def test_classify_empty_output_never_generic_only():
    msg = core.classify_pair_failure(1, [])
    assert "Diagnosticar" in msg


def test_pair_surfaces_bridge_error(tmp_path, monkeypatch):
    py = core.python_path(tmp_path)
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("x", encoding="utf-8")
    (core.worker_dir(tmp_path)).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        core, "run_capture",
        lambda *a, **k: (1, ["❌ No pude emparejar: HTTP 401: code expired"]))
    with pytest.raises(RuntimeError, match="venció"):
        core.pair(tmp_path, back="http://localhost:9999", code="123456")


def test_pair_ok_swallows_nothing(tmp_path, monkeypatch):
    py = core.python_path(tmp_path)
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("x", encoding="utf-8")
    (core.worker_dir(tmp_path)).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(core, "run_capture", lambda *a, **k: (0, ["ok"]))
    core.pair(tmp_path, back="http://localhost:9999", code="123456")  # no lanza


# ── descarga del payload (fallback sin intervención) ─────────────
def _serve_zip(zip_bytes, wrong_type=False):
    import http.server
    import threading

    class Zip(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"<html>cautivo</html>" if wrong_type else zip_bytes
            ctype = "text/html" if wrong_type else "application/zip"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Zip)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _worker_zip(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo, {"run.py": "print('ok')\n",
                       "pair.py": "# pair\n",
                       "requirements.txt": "x\n"})
    dest = tmp_path / "worker.zip"
    core.build_payload(repo, dest)
    return dest.read_bytes()


def test_download_payload_roundtrip(tmp_path):
    blob = _worker_zip(tmp_path)
    srv = _serve_zip(blob)
    try:
        out = core.download_payload(
            f"http://127.0.0.1:{srv.server_port}/w.zip", tmp_path / "d.zip")
    finally:
        srv.shutdown()
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        assert {"run.py", "pair.py", "requirements.txt"} <= set(zf.namelist())


def test_download_payload_rejects_non_worker(tmp_path):
    srv = _serve_zip(b"", wrong_type=True)
    try:
        with pytest.raises(RuntimeError):
            core.download_payload(
                f"http://127.0.0.1:{srv.server_port}/w.zip", tmp_path / "d.zip")
    finally:
        srv.shutdown()


def test_extract_fallback_downloads_when_blocked(tmp_path, monkeypatch):
    """Embebido bloqueado → descarga y extrae sin pedir nada."""
    blob = _worker_zip(tmp_path)
    srv = _serve_zip(blob)
    monkeypatch.setattr(core.time, "sleep", lambda *a: None)
    try:
        target = tmp_path / "install" / "worker"
        core._extract_with_fallback(
            tmp_path / "no-existe.zip", target,
            payload_url=f"http://127.0.0.1:{srv.server_port}/w.zip",
            workdir=tmp_path / "install")
    finally:
        srv.shutdown()
    assert (target / "run.py").read_text(encoding="utf-8") == "print('ok')\n"


def test_ensure_python_retries_av_lock(tmp_path, monkeypatch):
    """El zip del embeddable también lo traba el AV: reintenta y sale."""
    import io
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as zf:
        zf.writestr("python.exe", "fake")
        zf.writestr("python312._pth", "python312.zip\n.\n#import site\n")
    def fake_download(url, dest, **kwargs):
        dest.write_bytes(blob.getvalue())
        return dest

    monkeypatch.setattr(core, "download", fake_download)
    real_zipfile = zipfile.ZipFile
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(13, "Permission denied", str(args[0]))
        return real_zipfile(*args, **kwargs)

    monkeypatch.setattr(zipfile, "ZipFile", flaky)
    monkeypatch.setattr(core.time, "sleep", lambda *a: None)
    py = core.ensure_python(tmp_path)
    assert py.exists()
    assert calls["n"] >= 3


def test_install_stops_worker_first(tmp_path, monkeypatch):
    """Reinstalar con el worker corriendo no debe chocar por DLLs bloqueadas."""
    order: list = []
    monkeypatch.setattr(core, "stop",
                        lambda wd=None: order.append("stop") or None)
    monkeypatch.setattr(core, "_extract_with_fallback",
                        lambda *a, **k: order.append("extract"))
    monkeypatch.setattr(core, "ensure_python",
                        lambda wd: order.append("python"))
    monkeypatch.setattr(core, "ensure_pip", lambda *a: None)
    monkeypatch.setattr(core, "install_deps", lambda *a: None)
    monkeypatch.setattr(core, "run_doctor", lambda *a: None)
    monkeypatch.setattr(core, "worker_status", lambda wd=None: {})
    core.install(tmp_path)
    assert order[:3] == ["stop", "extract", "python"]


# ── diagnose_back (servidor HTTP local, sin internet) ────────────
def _serve_health(tmp_path):
    import functools
    import http.server
    import threading

    # /healthz con 200: el back real expone ese endpoint
    class Health(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/healthz":
                body = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                super().do_GET()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(
        ("127.0.0.1", 0),
        functools.partial(Health, directory=str(tmp_path)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_diagnose_all_green_against_local_server(tmp_path):
    srv = _serve_health(tmp_path)
    try:
        steps = core.diagnose_back(f"http://127.0.0.1:{srv.server_port}")
    finally:
        srv.shutdown()
    assert [s["step"] for s in steps] == ["URL", "DNS", "TCP", "HTTP"]
    assert all(s["ok"] for s in steps)


def test_diagnose_dead_port_fails_tcp(tmp_path):
    steps = core.diagnose_back("http://127.0.0.1:9")  # discard: casi seguro cerrado
    by_step = {s["step"]: s for s in steps}
    assert by_step["URL"]["ok"] is True
    assert by_step["DNS"]["ok"] is True
    assert by_step["TCP"]["ok"] is False
    assert len(steps) == 3  # sin TCP no hay HTTP


def test_diagnose_bad_url_fails_first_step():
    steps = core.diagnose_back("ftp://x")
    assert steps == [{"step": "URL", "ok": False, "detail": steps[0]["detail"]}]
    assert "http" in steps[0]["detail"]


def test_cli_diagnose_reports_steps(tmp_path, capsys):
    srv = _serve_health(tmp_path)
    try:
        rc = core.main(["--dir", str(tmp_path), "diagnose",
                        "--back", f"http://127.0.0.1:{srv.server_port}"])
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert rc == 0
    assert "✓ HTTP" in out


# ── GUI (una instancia real, sin mainloop) ────────────────────────
def test_gui_builds_and_reports_status(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    monkeypatch.setattr(core, "default_dir", lambda: tmp_path)
    monkeypatch.setattr(core, "port_open", lambda *a, **k: False)

    from installer import app as app_mod

    try:
        gui = app_mod.InstallerApp()
    except tk.TclError:  # entorno sin display
        pytest.skip("sin pantalla interactiva")

    try:
        # las tres pantallas existen
        assert set(gui.screens) == {"install", "pair", "done"}
        assert gui.install_btn.cget("text")
        # validación de código: solo dígitos, hasta 6
        assert gui._digit_code("123456")
        assert not gui._digit_code("1234567")
        assert not gui._digit_code("12ab56")
        assert gui._digit_code("")  # borrar todo siempre se puede
        # el logger de core está cableado a la cola de la GUI
        logging.getLogger("synapta.installer").info("hola-gui")
        gui.update_idletasks()
        kind, text = gui.msgs.get_nowait()
        assert kind == "log" and "hola-gui" in text
        # refresh sin instalar no explota y refleja el estado
        gui.refresh_done()
        assert "No instalado" in gui.done_status.get()
        gui.update_idletasks()
    finally:
        gui.destroy()


def test_queue_log_handler_moves_records():
    sink: queue_mod.Queue = queue_mod.Queue()
    from installer import app as app_mod

    handler = app_mod.QueueLogHandler(sink)
    record = logging.LogRecord(
        name="synapta.installer", level=logging.INFO, pathname=__file__,
        lineno=1, msg="hola %s", args=("mundo",), exc_info=None)
    handler.emit(record)
    kind, text = sink.get_nowait()
    assert kind == "log"
    assert "hola mundo" in text


def test_gui_destroy_detaches_log_handler(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    monkeypatch.setattr(core, "default_dir", lambda: tmp_path)
    from installer import app as app_mod
    root_logger = logging.getLogger("synapta.installer")
    try:
        gui = app_mod.InstallerApp()
    except tk.TclError:
        pytest.skip("sin pantalla interactiva")
    try:
        handlers = list(root_logger.handlers)
        assert any(isinstance(h, app_mod.QueueLogHandler) for h in handlers)
    finally:
        gui.destroy()
    assert not any(isinstance(h, app_mod.QueueLogHandler)
                   for h in root_logger.handlers)


# ── modelos y actualización (GUI del instalador) ─────────────────
def _init_with_version(version: str) -> str:
    return f'"""doc"""\n\nBRIDGE_VERSION = "{version}"\n'


def test_version_from_init_y_payload():
    text = _init_with_version("0.2.17")
    assert core.version_from_init(text) == "0.2.17"
    assert core.version_from_init("sin version") == ""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("bridge/__init__.py", text)
        zf.writestr("run.py", "print('hi')\n")
    assert core.version_from_payload(buffer.getvalue()) == "0.2.17"
    assert core.version_from_payload(b"no soy un zip") == ""


def test_local_version_lee_el_worker_instalado(tmp_path):
    init = tmp_path / "worker" / "bridge" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text(_init_with_version("0.2.16"), encoding="utf-8")

    assert core.local_version(tmp_path) == "0.2.16"
    assert core.local_version(tmp_path / "no-instalado") == ""


def test_check_update_sin_red_no_rompe(tmp_path, monkeypatch):
    def boom(url=None, timeout=0):
        raise OSError("sin red")

    monkeypatch.setattr(core, "published_version", boom)
    info = core.check_update(tmp_path)

    assert info["available"] is False
    assert "sin red" in info["error"]


def test_check_update_detecta_version_nueva(tmp_path, monkeypatch):
    init = tmp_path / "worker" / "bridge" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text(_init_with_version("0.2.16"), encoding="utf-8")
    monkeypatch.setattr(core, "published_version",
                        lambda url=None, timeout=0: "0.2.17")

    info = core.check_update(tmp_path)

    assert info == {"current": "0.2.16", "latest": "0.2.17",
                    "available": True, "error": ""}


def test_check_update_sin_novedad(tmp_path, monkeypatch):
    init = tmp_path / "worker" / "bridge" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text(_init_with_version("0.2.17"), encoding="utf-8")
    monkeypatch.setattr(core, "published_version",
                        lambda url=None, timeout=0: "0.2.17")

    assert core.check_update(tmp_path)["available"] is False


def test_modelos_pegan_en_los_paths_del_artifact_server(monkeypatch):
    """Listar/descargar/cancelar/borrar contra la API local del worker."""
    calls = []

    class _Resp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=0):
        calls.append((req.get_method(), req.full_url))
        return _Resp(b'{"models": [{"key": "SD-1.5"}], "count": 1}')

    monkeypatch.setattr(core.urllib.request, "urlopen", fake_urlopen)

    assert core.list_models() == [{"key": "SD-1.5"}]
    core.download_model("SD 1.5")
    core.cancel_model_download("SD-1.5")
    core.delete_model("SD-1.5")

    base = core.art_url()
    assert calls == [
        ("GET", f"{base}/models"),
        ("POST", f"{base}/models/SD%201.5/download"),
        ("POST", f"{base}/models/SD-1.5/cancel"),
        ("DELETE", f"{base}/models/SD-1.5"),
    ]


def test_update_reemplaza_codigo_sin_tocar_python_ni_pesos(tmp_path, monkeypatch):
    """La actualización baja el zip, para, reemplaza y vuelve a arrancar."""
    (tmp_path / "worker").mkdir()
    (tmp_path / "worker" / "run.py").write_text("print('hi')\n", encoding="utf-8")
    core.python_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    core.python_path(tmp_path).write_bytes(b"")
    events = []

    def fake_download(url, dest, timeout=60.0):
        events.append(("download", url))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("run.py", "print('nuevo')\n")
            zf.writestr("bridge/__init__.py", _init_with_version("0.2.18"))
            zf.writestr("requirements.txt", "grpcio\n")
        Path(dest).write_bytes(buffer.getvalue())
        return Path(dest)

    def fake_extract(zip_path, target, retries=6):
        events.append(("extract", str(target)))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)

    monkeypatch.setattr(core, "download_payload", fake_download)
    monkeypatch.setattr(core, "extract_payload", fake_extract)
    monkeypatch.setattr(core, "stop", lambda *a, **k: events.append(("stop", None)))
    monkeypatch.setattr(core, "_wait_port_closed", lambda *a, **k: True)
    monkeypatch.setattr(core, "start",
                        lambda *a, **k: events.append(("start", k.get("require_clean"))))
    monkeypatch.setattr(core, "install_deps",
                        lambda *a, **k: events.append(("deps", None)))

    info = core.update(tmp_path, url="http://example/worker.zip")

    assert info["version"] == "0.2.18"
    assert (tmp_path / "worker" / "run.py").read_text(encoding="utf-8") == "print('nuevo')\n"
    # python/ y el resto del workdir quedan intactos; el zip temporal se borra
    assert (tmp_path / "python").is_dir()
    assert not (tmp_path / "update-payload.zip").exists()
    # orden: bajar → detener → reemplazar → arrancar
    assert [e[0] for e in events][:3] == ["download", "stop", "extract"]
    # require_clean: arrancar de verdad, no dar por bueno un puerto ocupado
    assert events[-1] == ("start", True)
    # sin requirements.txt previo se reinstalan y queda la marca
    assert ("deps", None) in events
    assert (tmp_path / core.REQ_MARKER).exists()


def test_update_saltea_las_dependencias_si_no_cambiaron(tmp_path, monkeypatch):
    (tmp_path / "worker").mkdir()
    (tmp_path / "worker" / "run.py").write_text("print('hi')\n", encoding="utf-8")
    # bytes y no write_text: en Windows el modo texto traduce \n a \r\n y el
    # hash cambiaria respecto del zip (que guarda \n tal cual).
    (tmp_path / "worker" / "requirements.txt").write_bytes(b"grpcio\n")
    core.python_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    core.python_path(tmp_path).write_bytes(b"")
    core._mark_requirements(tmp_path)
    events = []

    def fake_download(url, dest, timeout=60.0):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("bridge/__init__.py", _init_with_version("0.2.18"))
            zf.writestr("requirements.txt", "grpcio\n")
        Path(dest).write_bytes(buffer.getvalue())
        return Path(dest)

    monkeypatch.setattr(core, "download_payload", fake_download)
    monkeypatch.setattr(core, "extract_payload",
                        lambda zip_path, target, retries=6: zipfile.ZipFile(zip_path).extractall(target))
    monkeypatch.setattr(core, "stop", lambda *a, **k: None)
    monkeypatch.setattr(core, "_wait_port_closed", lambda *a, **k: True)
    monkeypatch.setattr(core, "start", lambda *a, **k: None)
    monkeypatch.setattr(core, "install_deps",
                        lambda *a, **k: events.append("deps"))

    core.update(tmp_path)

    assert events == []  # requirements igual: pip no corre


def test_update_sin_worker_instalado_falla_claro(tmp_path):
    with pytest.raises(RuntimeError, match="no está instalado"):
        core.update(tmp_path)


def test_stop_encuentra_el_worker_por_puerto(tmp_path, monkeypatch):
    """Sin `worker.pid` (arrancado fuera del instalador) igual lo detiene."""
    if os.name != "nt":
        pytest.skip("netstat/taskkill son sólo Windows")
    llamadas = []
    monkeypatch.setattr(core, "port_pid", lambda port=core.GRPC_PORT: 777)
    monkeypatch.setattr(core, "process_name", lambda pid: "python.exe")
    monkeypatch.setattr(core, "_alive", lambda pid: True)
    monkeypatch.setattr(core.subprocess, "run",
                        lambda cmd, **kwargs: llamadas.append(cmd))

    core.stop(tmp_path)

    assert len(llamadas) == 1
    assert "777" in llamadas[0]


def test_stop_no_mata_un_proceso_ajeno_en_el_puerto(tmp_path, monkeypatch):
    if os.name != "nt":
        pytest.skip("netstat/taskkill son sólo Windows")
    llamadas = []
    monkeypatch.setattr(core, "port_pid", lambda port=core.GRPC_PORT: 999)
    monkeypatch.setattr(core, "process_name", lambda pid: "chrome.exe")
    monkeypatch.setattr(core.subprocess, "run",
                        lambda cmd, **kwargs: llamadas.append(cmd))

    core.stop(tmp_path)

    assert llamadas == []


def test_detecta_intel_arc_para_el_torch_xpu(monkeypatch):
    """Sólo Arc / Iris Xe: las Intel UHD viejas no tienen backend XPU."""
    monkeypatch.setattr(
        core, "video_controllers",
        lambda: ["Intel(R) Arc(TM) A770 Graphics", "Intel(R) UHD Graphics 770"],
    )
    assert core.has_intel_xpu() is True

    monkeypatch.setattr(
        core, "video_controllers",
        lambda: ["NVIDIA GeForce RTX 4060 Laptop GPU", "Intel(R) UHD Graphics"],
    )
    assert core.has_intel_xpu() is False

    monkeypatch.setattr(core, "video_controllers",
                        lambda: ["Intel(R) Iris(R) Xe Graphics"])
    assert core.has_intel_xpu() is True


def test_install_torch_xpu_usa_el_indice_xpu(tmp_path, monkeypatch):
    # El worker ya tiene un torch cpu: hay que forzar la reinstalación, porque
    # pip considera "satisfecha" una versión más nueva del wheel de PyPI.
    llamadas = []
    monkeypatch.setattr(core, "index_torch_version", lambda py, wd, idx: "2.13.0")
    monkeypatch.setattr(core, "torch_version", lambda py, wd: "2.13.0")
    monkeypatch.setattr(core, "run", lambda cmd, **kw: llamadas.append(cmd) or 0)
    monkeypatch.setattr(core, "run_capture", lambda cmd, **kw: (0, ["True"]))

    version = core.install_torch_xpu(tmp_path / "python", tmp_path)

    assert version == "2.13.0"
    pip = [c for c in llamadas if "install" in c]
    assert pip, llamadas
    texto = " ".join(str(x) for x in pip[0])
    assert "download.pytorch.org/whl/xpu" in texto
    assert "--force-reinstall" in texto
    assert "torch==2.13.0" in texto


def test_install_torch_xpu_falla_si_no_ve_la_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "index_torch_version", lambda py, wd, idx: "2.13.0")
    monkeypatch.setattr(core, "torch_version", lambda py, wd: "2.13.0")
    monkeypatch.setattr(core, "run", lambda cmd, **kw: 0)
    monkeypatch.setattr(core, "run_capture", lambda cmd, **kw: (0, ["False"]))

    with pytest.raises(RuntimeError, match="no ve la GPU Intel"):
        core.install_torch_xpu(tmp_path / "python", tmp_path)


def test_install_deps_prefiere_xpu_cuando_no_hay_nvidia(tmp_path, monkeypatch):
    worker = tmp_path / "worker"
    worker.mkdir()
    (worker / "requirements.txt").write_text("grpcio\n", encoding="utf-8")
    pedido = {}
    monkeypatch.setattr(core, "has_nvidia", lambda: False)
    monkeypatch.setattr(core, "has_intel_xpu", lambda: True)
    monkeypatch.setattr(
        core, "install_torch_xpu",
        lambda py, wd: pedido.setdefault("xpu", "2.13.0"),
    )
    monkeypatch.setattr(core, "run", lambda cmd, **kw: 0)

    core.install_deps(tmp_path / "python", tmp_path)

    assert pedido["xpu"] == "2.13.0"


def test_stop_sin_arbol_no_mata_al_llamador(tmp_path, monkeypatch):
    """`tree=False` evita el `/T` que mataba al updater, hijo del worker."""
    if os.name != "nt":
        pytest.skip("taskkill es sólo Windows")
    calls = []
    monkeypatch.setattr(core, "_alive", lambda pid: True)
    monkeypatch.setattr(core.subprocess, "run",
                        lambda cmd, **kwargs: calls.append(cmd))
    pidf = core._pid_file(tmp_path)
    pidf.parent.mkdir(parents=True, exist_ok=True)
    pidf.write_text("4321", encoding="utf-8")

    core.stop(tmp_path, tree=False)

    assert len(calls) == 1
    assert "/T" not in calls[0] and "/F" in calls[0]
    assert "4321" in calls[0]
    assert not pidf.exists()


def test_updater_sobrevive_a_consola_cp1252(tmp_path, monkeypatch):
    """El updater no debe morir imprimiendo "✓" en una consola Windows."""
    import importlib.util

    repo = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "synapta_update_worker", repo / "scripts" / "update_worker.py")
    uw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(uw)

    monkeypatch.setattr(uw, "download_payload",
                        lambda url, dest, timeout=60.0: Path(dest).write_bytes(b"PK\x05\x06"))
    monkeypatch.setattr(uw, "stop_for_update", lambda *a, **k: None)
    monkeypatch.setattr(uw, "extract_payload", lambda *a, **k: None)
    monkeypatch.setattr(uw, "ensure_python", lambda *a, **k: tmp_path / "python.exe")
    monkeypatch.setattr(uw, "ensure_pip", lambda *a, **k: None)
    monkeypatch.setattr(uw, "install_deps", lambda *a, **k: None)
    monkeypatch.setattr(uw, "run_doctor", lambda *a, **k: None)
    monkeypatch.setattr(uw, "start", lambda *a, **k: None)
    # consola cp1252 estricta: "✓" no se puede escribir
    monkeypatch.setattr(sys, "stdout",
                        io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    monkeypatch.setattr(sys, "argv", ["update_worker", "--url",
                                      "http://x/worker.zip", "--dir", str(tmp_path)])

    assert uw.main() == 0
    state = json.loads((tmp_path / "update-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "ok"
    # el zip temporal no queda tirado
    assert not (tmp_path / "_update-download.zip").exists()


# ── update con el worker arrancado fuera del instalador ──────────
# Sin `worker.pid` (autostart, script suelto) el update reemplazaba los
# archivos y avisaba "actualizado a vX" mientras el proceso viejo seguía
# respondiendo en :50051 con el código anterior.

def _worker_instalado(tmp_path):
    (tmp_path / "worker").mkdir()
    (tmp_path / "worker" / "run.py").write_text("print('hi')\n", encoding="utf-8")
    core.python_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    core.python_path(tmp_path).write_bytes(b"")
    return tmp_path


def _payload(version: str):
    """fake de download_payload: escribe un zip con esa versión."""
    def _download(url, dest, timeout=60.0):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("run.py", "print('nuevo')\n")
            zf.writestr("bridge/__init__.py", _init_with_version(version))
        Path(dest).write_bytes(buffer.getvalue())
        return Path(dest)
    return _download


def test_stop_for_update_reintenta_contra_el_dueno_del_puerto(tmp_path, monkeypatch):
    """El pid no estaba (o estaba viejo) y el puerto seguía tomado."""
    llamadas = []
    monkeypatch.setattr(core, "stop", lambda *a, **k: llamadas.append(("stop", None)))
    esperas = iter([False, True])
    monkeypatch.setattr(core, "_wait_port_closed", lambda *a, **k: next(esperas))
    monkeypatch.setattr(core, "port_pid", lambda *a, **k: 7777)
    monkeypatch.setattr(core, "process_name", lambda pid: "python.exe")
    monkeypatch.setattr(core, "_kill_pid",
                        lambda pid, tree=True: llamadas.append(("kill", pid, tree)))

    core.stop_for_update(tmp_path)

    assert llamadas == [("stop", None), ("kill", 7777, True)]


def test_stop_for_update_no_mata_un_proceso_ajeno(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "stop", lambda *a, **k: None)
    monkeypatch.setattr(core, "_wait_port_closed", lambda *a, **k: False)
    monkeypatch.setattr(core, "port_pid", lambda *a, **k: 5555)
    monkeypatch.setattr(core, "process_name", lambda pid: "chrome.exe")
    monkeypatch.setattr(core, "_kill_pid",
                        lambda *a, **k: pytest.fail("mató un proceso ajeno"))

    with pytest.raises(RuntimeError, match="5555"):
        core.stop_for_update(tmp_path)


def test_update_no_da_por_buena_la_actualizacion_si_el_puerto_sigue_ocupado(
        tmp_path, monkeypatch):
    """El bug reportado: sin worker.pid, el proceso viejo queda vivo."""
    _worker_instalado(tmp_path)
    extraido = []
    monkeypatch.setattr(core, "download_payload", _payload("0.2.18"))
    monkeypatch.setattr(core, "extract_payload", lambda *a, **k: extraido.append(a))
    monkeypatch.setattr(core, "stop", lambda *a, **k: None)
    monkeypatch.setattr(core, "_wait_port_closed", lambda *a, **k: False)
    monkeypatch.setattr(core, "port_pid", lambda *a, **k: 4242)
    monkeypatch.setattr(core, "process_name", lambda pid: "python.exe")
    monkeypatch.setattr(core, "_kill_pid", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="No se tocaron los archivos"):
        core.update(tmp_path)

    # aborta ANTES de reemplazar: el worker viejo sigue entero y sin zip tirado
    assert extraido == []
    assert (tmp_path / "worker" / "run.py").read_text(encoding="utf-8") == \
        "print('hi')\n"
    assert not (tmp_path / "update-payload.zip").exists()


def test_start_require_clean_falla_si_el_puerto_sigue_ocupado(tmp_path, monkeypatch):
    _worker_instalado(tmp_path)
    monkeypatch.setattr(core, "port_open", lambda *a, **k: True)
    monkeypatch.setattr(core, "port_pid", lambda *a, **k: 4321)
    monkeypatch.setattr(core, "process_name", lambda pid: "python.exe")

    with pytest.raises(RuntimeError, match="sigue ocupado"):
        core.start(tmp_path, require_clean=True)


def test_start_sigue_siendo_idempotente_en_la_instalacion(tmp_path, monkeypatch):
    """La instalación no cambia: con el worker ya arriba, no hace nada."""
    _worker_instalado(tmp_path)
    monkeypatch.setattr(core, "port_open", lambda *a, **k: True)

    assert core.start(tmp_path) is None


def test_start_require_clean_exige_que_escuche_el_proceso_nuevo(tmp_path, monkeypatch):
    """El puerto aparece pero lo tiene otro pid: es el worker viejo."""
    _worker_instalado(tmp_path)
    estado = {"arrancado": False}

    class Proc:
        pid = 111

        def poll(self):
            return None

    def fake_popen(*a, **k):
        estado["arrancado"] = True
        return Proc()

    monkeypatch.setattr(core, "port_open", lambda *a, **k: estado["arrancado"])
    monkeypatch.setattr(core.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(core, "listener_is", lambda *a, **k: False)
    monkeypatch.setattr(core, "port_pid", lambda *a, **k: 999)

    with pytest.raises(RuntimeError, match="no el worker nuevo"):
        core.start(tmp_path, require_clean=True)


def test_gui_modelos_lista_y_habilita_acciones(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    monkeypatch.setattr(core, "default_dir", lambda: tmp_path)
    monkeypatch.setattr(core, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(core, "list_models", lambda: [
        {"key": "SD-1.5", "label": "SD 1.5", "mode": "image", "vram_gb": 4,
         "available": True, "downloading": False, "size_bytes": 4 * 1024**3,
         "progress": 100.0, "error": ""},
        {"key": "FLUX.1-dev", "label": "FLUX dev", "mode": "image",
         "vram_gb": 24, "available": False, "downloading": True,
         "size_bytes": 1024**3, "progress": 12.0, "error": ""},
    ])

    from installer import app as app_mod

    try:
        gui = app_mod.InstallerApp()
    except tk.TclError:
        pytest.skip("sin pantalla interactiva")

    window = None
    try:
        window = app_mod.ModelsWindow(gui)
        window.refresh()  # el auto-refresh puede no haber corrido aún
        rows = window.tree.get_children()
        assert list(rows) == ["SD-1.5", "FLUX.1-dev"]
        assert "4.0 GB" in window.tree.item("SD-1.5", "values")
        assert "Descargando 12%" in window.tree.item("FLUX.1-dev", "values")

        window.tree.selection_set("SD-1.5")
        window._sync_buttons()
        assert window.delete_btn.instate(["!disabled"])
        assert window.download_btn.instate(["disabled"])

        window.tree.selection_set("FLUX.1-dev")
        window._sync_buttons()
        assert window.cancel_btn.instate(["!disabled"])
        assert window.download_btn.instate(["disabled"])
    finally:
        if window is not None:
            window.close()
        gui.destroy()
