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
