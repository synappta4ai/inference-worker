"""Tests del instalador (installer/core.py y la GUI).

Cubren la lógica sin red ni descargas: construcción de payload, parche
del ._pth del Python embeddable, validación del código de emparejamiento,
CLI (status/errores) y construcción de la GUI (una instancia Tk real,
destruida al terminar — no necesita pantalla interactiva).
"""

from __future__ import annotations

import json
import logging
import os
import queue as queue_mod
import zipfile

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
                                 "models/", "tests/", "installer/"))
                   for n in names)
    assert "worker-err.log" not in names
    assert ".env" not in names
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
