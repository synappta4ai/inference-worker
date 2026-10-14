"""Tests del saneo de CA heredado en el arranque del worker (run.py).

El instalador arranca el worker con SSL_CERT_FILE/PIP_CERT apuntando a su
carpeta temporal (`_MEIxxxx`), que además puede venir como CARPETA en vez de
archivo. httpx (huggingface_hub) arma su contexto con
`ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])`: con eso
falla con "PermissionError: [Errno 13] Permission denied" antes de tocar la
red, así que NINGUNA descarga de pesos arrancaba y el panel sólo mostraba
"No disponible" para siempre.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import run

WORKER_ROOT = Path(__file__).resolve().parent.parent


def _reset(monkeypatch, **values: str) -> None:
    """Deja el entorno sin CA heredado y setea sólo lo que pide el test."""
    for var in run._CA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for var, value in values.items():
        monkeypatch.setenv(var, value)


def test_replaces_ca_pointing_at_a_directory(monkeypatch, tmp_path):
    _reset(monkeypatch, SSL_CERT_FILE=str(tmp_path))  # carpeta: como el bundle
    run._normalize_ca_env()
    assert Path(os.environ["SSL_CERT_FILE"]).is_file()


def test_replaces_ca_inside_installer_temp_dir(monkeypatch, tmp_path):
    meipass = tmp_path / "_MEI00001cb42"
    meipass.mkdir()
    pem = meipass / "_cacert.pem"
    pem.write_text("PEM", encoding="utf-8")
    _reset(monkeypatch, SSL_CERT_FILE=str(pem))
    run._normalize_ca_env()
    assert "_MEI" not in os.environ["SSL_CERT_FILE"]


def test_keeps_durable_ca(monkeypatch, tmp_path):
    """Un CA propio (proxy corporativo, etc.) no se pisa."""
    pem = tmp_path / "corp-ca.pem"
    pem.write_text("PEM", encoding="utf-8")
    _reset(monkeypatch, SSL_CERT_FILE=str(pem), REQUESTS_CA_BUNDLE=str(pem))
    run._normalize_ca_env()
    assert os.environ["SSL_CERT_FILE"] == str(pem)
    assert os.environ["REQUESTS_CA_BUNDLE"] == str(pem)


def test_drops_var_when_no_own_certifi(monkeypatch, tmp_path):
    """Sin certifi propio, mejor contexto del sistema que una ruta rota."""
    _reset(monkeypatch, CURL_CA_BUNDLE=str(tmp_path / "no-existe.pem"))
    monkeypatch.setattr(run, "_own_cacert", lambda: None)
    run._normalize_ca_env()
    assert "CURL_CA_BUNDLE" not in os.environ


def test_import_left_a_usable_ca():
    """Al importarse, run.py ya dejó un CA que existe (o ninguno)."""
    value = os.environ.get("SSL_CERT_FILE")
    assert value is None or Path(value).is_file()


def test_import_with_poisoned_ca_does_not_crash(tmp_path):
    """Regresión: el saneo corre al importar y loguea (el logger debe existir).

    Un `logger` todavía sin definir tiraba NameError y el worker NO arrancaba
    — justo en el caso que el saneo viene a arreglar.
    """
    bad = tmp_path / "_cacert.pem"
    bad.mkdir()  # carpeta: el layout que extrae PyInstaller
    env = {**os.environ, "SSL_CERT_FILE": str(bad), "PIP_CERT": str(bad)}
    proc = subprocess.run([sys.executable, "-c", "import run"],
                          cwd=str(WORKER_ROOT), env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "SSL_CERT_FILE" in proc.stderr
