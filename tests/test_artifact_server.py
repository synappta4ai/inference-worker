"""Tests del servidor HTTP de artefactos (despliegues divididos)."""

import os
import threading
import time
import urllib.request

import pytest

import artifact_server


@pytest.fixture()
def httpd(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_server, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(artifact_server, "ARTIFACT_TOKEN", "")
    srv = artifact_server.start_server(0)  # puerto efímero
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _get(url: str, token: str | None = None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("X-Artifact-Token", token)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as err:  # type: ignore[attr-defined]
        return err.code, dict(err.headers), err.read()


def test_healthz(httpd):
    _, base = httpd
    code, _, body = _get(f"{base}/healthz")
    assert code == 200
    assert b"ok" in body


def test_serves_artifact_with_content_type(httpd, tmp_path):
    _, base = httpd
    (tmp_path / "job_x.png").write_bytes(b"\x89PNG fake")
    code, headers, body = _get(f"{base}/artifacts/job_x.png")
    assert code == 200
    assert headers.get("Content-Type") == "image/png"
    assert body == b"\x89PNG fake"


def test_token_required_when_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_server, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(artifact_server, "ARTIFACT_TOKEN", "s3cr3t")
    srv = artifact_server.start_server(0)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.05)
    try:
        (tmp_path / "a.png").write_bytes(b"x")
        assert _get(f"{base}/artifacts/a.png")[0] == 403
        assert _get(f"{base}/artifacts/a.png", token="s3cr3t")[0] == 200
        # token por query string también válido
        assert _get(f"{base}/artifacts/a.png?token=s3cr3t")[0] == 200
    finally:
        srv.shutdown()


def test_rejects_path_traversal(httpd):
    _, base = httpd
    assert _get(f"{base}/artifacts/..%2Fserver.py")[0] == 400
    assert _get(f"{base}/artifacts/..%5Cserver.py")[0] == 400


def test_rejects_hidden_and_dirs(httpd):
    _, base = httpd
    assert _get(f"{base}/artifacts/.token")[0] == 400
    assert _get(f"{base}/artifacts/missing.png")[0] == 404


def test_safe_name():
    assert artifact_server._safe_name("job_1.png")
    assert not artifact_server._safe_name("a/b.png")
    assert not artifact_server._safe_name("..")
    assert not artifact_server._safe_name(".env")
    assert not artifact_server._safe_name("")


def test_start_background_disabled(monkeypatch):
    assert artifact_server.start_background(0) is None
