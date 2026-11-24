"""Tests del bridge saliente: config, journal y cliente HTTP.

Todo corre sin GPU, sin red y sin el back real: el cliente se enfrenta a un
fake server stdlib (http.server) que reproduce el contrato
(internal/modules/workerbridge). Los tests del agente usan el motor mock.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from bridge import config, journal
from bridge.client import BridgeClient, BridgeError, Job, with_retries


# ─── Fake back ─────────────────────────────────────────────────

class FakeBack(BaseHTTPRequestHandler):
    """Contrato mínimo del back: register/heartbeat/claim/report/artifacts."""

    # Estado compartido por clase (los tests lo resetean).
    registered_codes = {"123456"}
    heartbeats: list = []
    cancel_ids: list = []
    requested_models: list = []
    requested_deletions: list = []
    requested_download_cancels: list = []
    claimed_job: dict | None = None
    # True = el claim devuelve el job pelado en `data` (forma de backs viejos).
    bare_job = False
    reports: list = []
    uploads: list = []
    revoked = False

    def log_message(self, *args):  # silencia el ruido del server
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        if self.path.startswith("/api/v1/internal/workers/register"):
            data = json.loads(raw)
            if data.get("code") not in self.registered_codes:
                return self._json(401, {"success": False, "message": "pairing code not found"})
            return self._json(201, {"success": True, "data": {
                "worker_id": "w-123", "token": "tok-secret"}})

        token = self.headers.get("X-Worker-Token", "")
        if self.revoked or token != "tok-secret":
            return self._json(401, {"success": False, "message": "invalid worker token"})

        if self.path.startswith("/api/v1/internal/workers/heartbeat"):
            self.heartbeats.append(json.loads(raw or b"{}"))
            return self._json(200, {"success": True, "data": {
                "cancel_task_ids": self.cancel_ids,
                "requested_models": self.requested_models,
                "requested_deletions": self.requested_deletions,
                "requested_download_cancels": self.requested_download_cancels}})
        if self.path.startswith("/api/v1/internal/workers/claim"):
            job = self.claimed_job
            self.claimed_job = None  # solo una entrega
            if self.bare_job:
                # Forma que devolvía el back real (`data` = job pelado).
                return self._json(200, {"success": True, "data": job})
            return self._json(200, {"success": True, "data": {"job": job}})
        if self.path.startswith("/api/v1/internal/workers/report"):
            self.reports.append(json.loads(raw))
            return self._json(200, {"success": True, "message": "reported"})
        if self.path.startswith("/api/v1/internal/workers/artifacts"):
            body = raw
            ctype = self.headers.get("Content-Type", "")
            boundary = ctype.split("boundary=")[-1]
            # Extrae el filename y el task_id del multipart (parseo mínimo).
            text = body.decode("latin-1")
            task_id = ""
            for part in text.split(f"--{boundary}"):
                if 'name="task_id"' in part:
                    task_id = part.split("\r\n\r\n", 1)[-1].split("\r\n")[0].strip()
                if 'name="file"' in part:
                    filename = part.split('filename="', 1)[1].split('"', 1)[0]
            self.uploads.append({"task_id": task_id, "filename": locals().get("filename", "")})
            return self._json(200, {"success": True, "data": {
                "artifact_url": f"/outputs/video/{locals().get('filename', 'x')}"}})
        return self._json(404, {"success": False, "message": "unknown path"})


@pytest.fixture()
def fake_back():
    FakeBack.registered_codes = {"123456"}
    FakeBack.heartbeats, FakeBack.reports, FakeBack.uploads = [], [], []
    FakeBack.cancel_ids, FakeBack.claimed_job = [], None
    FakeBack.bare_job = False
    FakeBack.requested_models = []
    FakeBack.requested_deletions = []
    FakeBack.requested_download_cancels = []
    FakeBack.revoked = False
    server = HTTPServer(("127.0.0.1", 0), FakeBack)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join(timeout=2)


# ─── config ────────────────────────────────────────────────────

def test_config_roundtrip_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path / "SynaptaWorker")
    cfg = config.BridgeConfig(back_url="https://back.example", worker_id="w-1",
                              token="sekrit", worker_name="PC-Gaming")
    path = config.save(cfg)
    assert path.exists()
    assert config.load() == cfg
    # El token no debe ser legible por otros usuarios (POSIX; en Windows el
    # profile del usuario ya es privado y chmod no tiene el mismo significado).
    if os.name == "posix":
        assert oct(path.stat().st_mode)[-3:] == "600"


def test_config_missing_or_corrupt_is_unpaired(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    assert not config.load().paired
    (tmp_path / "config.json").write_text("{{{no-json", encoding="utf-8")
    assert not config.load().paired


def test_config_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    config.save(config.BridgeConfig(back_url="https://x", worker_id="1", token="t"))
    assert config.load().paired
    config.clear()
    assert not config.load().paired


# ─── journal ───────────────────────────────────────────────────

def _patch_journal_dir(monkeypatch, tmp_path):
    """journal.py bindea config_dir al importar: hay que parchear su propio
    nombre de módulo, no el de config."""
    import bridge.journal as jmod
    monkeypatch.setattr(jmod, "config_dir", lambda: tmp_path)


def test_journal_survives_and_lists_pending(tmp_path, monkeypatch):
    _patch_journal_dir(monkeypatch, tmp_path)
    journal.record(journal.JobEntry(task_id="syn_1", state=journal.STATE_ARTIFACT_READY,
                                    artifact_path="/tmp/a.mp4", model_name="Wan"))
    journal.record(journal.JobEntry(task_id="syn_2", state=journal.STATE_RUNNING))
    pending = journal.pending_uploads()
    assert [p.task_id for p in pending] == ["syn_1"]  # running sin artefacto no cuenta
    loaded = journal.load("syn_1")
    assert loaded.artifact_path == "/tmp/a.mp4"
    journal.discard("syn_1")
    assert journal.pending_uploads() == []


def test_journal_record_is_atomic(tmp_path, monkeypatch):
    """Un crash entre write y rename no deja JSON a medio escribir."""
    _patch_journal_dir(monkeypatch, tmp_path)
    journal.record(journal.JobEntry(task_id="syn_9", state=journal.STATE_RUNNING))
    files = list((tmp_path / "jobs").glob("*"))
    assert len(files) == 1 and files[0].suffix == ".json"
    assert not list((tmp_path / "jobs").glob("*.tmp"))


def test_journal_stale_unfinished(tmp_path, monkeypatch):
    _patch_journal_dir(monkeypatch, tmp_path)
    entry = journal.JobEntry(task_id="syn_old", state=journal.STATE_RUNNING)
    journal.record(entry)
    # Envejece la entrada manualmente (proceso que ya no existe).
    path = tmp_path / "jobs" / "syn_old.json"
    data = json.loads(path.read_text())
    data["updated_at"] = time.time() - 13 * 3600
    path.write_text(json.dumps(data))
    stale = journal.stale_unfinished(max_age_hours=12)
    assert [s.task_id for s in stale] == ["syn_old"]


def test_journal_task_id_is_sanitized(tmp_path, monkeypatch):
    _patch_journal_dir(monkeypatch, tmp_path)
    journal.record(journal.JobEntry(task_id="../../evil", state=journal.STATE_RUNNING))
    # El path trazado no sale del dir de jobs.
    assert list((tmp_path / "jobs").glob("*.json"))
    assert not (tmp_path / "evil.json").exists()


# ─── client ────────────────────────────────────────────────────

def test_client_register_ok(fake_back):
    client = BridgeClient(fake_back)
    resp = client.register(code="123456", name="PC", gpu_name="RTX 4090",
                           os_name="win32", version="0.1.0",
                           vram_total_mb=24564, vram_free_mb=23000, models=[])
    assert resp["token"] == "tok-secret"


def test_client_register_bad_code(fake_back):
    client = BridgeClient(fake_back)
    with pytest.raises(BridgeError) as exc:
        client.register(code="000000", name="PC", gpu_name="", os_name="win32",
                        version="0.1.0", vram_total_mb=0, vram_free_mb=0, models=[])
    assert exc.value.status == 401


def test_client_heartbeat_returns_cancels(fake_back):
    FakeBack.cancel_ids = ["syn_77"]
    client = BridgeClient(fake_back, token="tok-secret")
    res = client.heartbeat(gpu_name="RTX", version="0.1.0",
                           vram_free_mb=100, jobs_active=0, models=[])
    assert res["cancel_task_ids"] == ["syn_77"]
    assert res["requested_models"] == []
    assert FakeBack.heartbeats[-1]["gpu_name"] == "RTX"


def test_client_heartbeat_reports_total_vram(fake_back):
    """«Memoria» y la columna «En tu GPU» del panel salen de la VRAM total: el
    agente la manda en cada heartbeat, no sólo al emparejar."""
    client = BridgeClient(fake_back, token="tok-secret")
    client.heartbeat(gpu_name="RTX 4090", version="0.2.25", vram_total_mb=24564,
                     vram_free_mb=23000, jobs_active=0, models=[])
    assert FakeBack.heartbeats[-1]["vram_total_mb"] == 24564


def test_client_heartbeat_download_queue(fake_back):
    """El panel encola descargas: viajan como requested_models y el
    agente las confirma con completed_downloads en el próximo latido."""
    FakeBack.requested_models = ["FLUX.1-dev"]
    client = BridgeClient(fake_back, token="tok-secret")
    res = client.heartbeat(gpu_name="RTX", version="0.2.5",
                           vram_free_mb=100, jobs_active=0, models=[])
    assert res["requested_models"] == ["FLUX.1-dev"]
    assert FakeBack.heartbeats[-1].get("completed_downloads", []) == []
    # Segundo latido: el agente ya tomó la descarga y hace ack.
    FakeBack.requested_models = []
    res2 = client.heartbeat(gpu_name="RTX", version="0.2.5",
                            vram_free_mb=100, jobs_active=0, models=[],
                            completed_downloads=["FLUX.1-dev"])
    assert res2["requested_models"] == []
    assert FakeBack.heartbeats[-1]["completed_downloads"] == ["FLUX.1-dev"]


def test_client_heartbeat_deletion_queue(fake_back):
    """El panel pide liberar disco: viaja como requested_deletions y el agente
    lo confirma con completed_deletions en el latido siguiente."""
    FakeBack.requested_deletions = ["SDXL-Turbo"]
    client = BridgeClient(fake_back, token="tok-secret")
    res = client.heartbeat(gpu_name="RTX", version="0.2.22",
                           vram_free_mb=100, jobs_active=0, models=[])
    assert res["requested_deletions"] == ["SDXL-Turbo"]
    assert FakeBack.heartbeats[-1].get("completed_deletions", []) == []

    FakeBack.requested_deletions = []
    res2 = client.heartbeat(gpu_name="RTX", version="0.2.22",
                            vram_free_mb=100, jobs_active=0, models=[],
                            completed_deletions=["SDXL-Turbo"])
    assert res2["requested_deletions"] == []
    assert FakeBack.heartbeats[-1]["completed_deletions"] == ["SDXL-Turbo"]


def test_client_heartbeat_download_cancel_queue(fake_back):
    """El panel cancela una descarga: viaja como requested_download_cancels y
    el agente lo confirma con completed_download_cancels en el latido siguiente."""
    FakeBack.requested_download_cancels = ["FLUX.1-dev"]
    client = BridgeClient(fake_back, token="tok-secret")
    res = client.heartbeat(gpu_name="RTX", version="0.2.23",
                           vram_free_mb=100, jobs_active=0, models=[])
    assert res["requested_download_cancels"] == ["FLUX.1-dev"]
    assert FakeBack.heartbeats[-1].get("completed_download_cancels", []) == []

    FakeBack.requested_download_cancels = []
    res2 = client.heartbeat(gpu_name="RTX", version="0.2.23",
                            vram_free_mb=100, jobs_active=0, models=[],
                            completed_download_cancels=["FLUX.1-dev"])
    assert res2["requested_download_cancels"] == []
    assert FakeBack.heartbeats[-1]["completed_download_cancels"] == ["FLUX.1-dev"]


# ─── Borrado de pesos en el agente ─────────────────────────────

def _agent(jobs_active: int = 0):
    from bridge.agent import Agent  # importa grpc/pb: sólo cuando hace falta

    cfg = config.BridgeConfig(back_url="http://127.0.0.1:1", worker_id="w-1",
                              token="t", worker_name="pc")
    ag = Agent(cfg, local_grpc="127.0.0.1:1")
    ag._jobs_active = jobs_active
    return ag


def _stub_delete(monkeypatch, *, raises=None, registro=None):
    """Reemplaza el canal gRPC local y el stub que usa el agente."""
    import contextlib

    from bridge import agent as agent_mod

    class Stub:
        def __init__(self, _channel):
            pass

        def DeleteModel(self, req, timeout=None):  # noqa: N802 — nombre del proto
            if registro is not None:
                registro.append(req.key)
            if raises is not None:
                raise raises
            return object()

    monkeypatch.setattr(agent_mod.grpc, "insecure_channel",
                        lambda target: contextlib.nullcontext(object()))
    monkeypatch.setattr(agent_mod.inference_pb2_grpc, "InferenceServiceStub", Stub)
    return agent_mod


def test_agent_delete_calls_local_grpc(monkeypatch):
    llamadas: list[str] = []
    _stub_delete(monkeypatch, registro=llamadas)
    ag = _agent()

    assert ag._start_remote_delete("SDXL-Turbo") is True
    assert llamadas == ["SDXL-Turbo"]


def test_agent_delete_waits_while_generating(monkeypatch):
    """Borrar los pesos de un pipeline en uso puede fallar y no libera disco:
    el pedido espera (sin ack) a que termine la generación."""
    agent_mod = _stub_delete(monkeypatch)

    def no_llamar(_target):
        raise AssertionError("no debe tocar el gRPC con un job activo")

    monkeypatch.setattr(agent_mod.grpc, "insecure_channel", no_llamar)
    ag = _agent(jobs_active=1)

    assert ag._start_remote_delete("SDXL-Turbo") is False


def test_agent_delete_retries_when_download_is_running(monkeypatch):
    """Una descarga en curso rechaza el borrado (FAILED_PRECONDITION): no se
    hace ack, el back reintenta y el pedido no se pierde."""
    agent_mod = _stub_delete(monkeypatch)

    class Condicion(agent_mod.grpc.RpcError):
        def code(self):
            return agent_mod.grpc.StatusCode.FAILED_PRECONDITION

        def details(self):
            return "cancelá la descarga antes de borrar el modelo"

    _stub_delete(monkeypatch, raises=Condicion())
    ag = _agent()

    assert ag._start_remote_delete("SDXL-Turbo") is False
    # El mismo aviso no se repite en cada latido (20 s).
    assert ag._start_remote_delete("SDXL-Turbo") is False
    assert "SDXL-Turbo" in ag._delete_warned


def test_agent_delete_acks_unknown_key(monkeypatch):
    """Una clave que este worker no conoce no se reintenta para siempre."""
    agent_mod = _stub_delete(monkeypatch)

    class NoExiste(agent_mod.grpc.RpcError):
        def code(self):
            return agent_mod.grpc.StatusCode.NOT_FOUND

        def details(self):
            return "modelo desconocido"

    _stub_delete(monkeypatch, raises=NoExiste())
    ag = _agent()

    assert ag._start_remote_delete("Fantasma") is True


def _stub_download(monkeypatch, *, raises=None, registro=None):
    """Reemplaza el canal gRPC local y captura (key, cancel) de DownloadModel."""
    import contextlib

    from bridge import agent as agent_mod

    class Stub:
        def __init__(self, _channel):
            pass

        def DownloadModel(self, req, timeout=None):  # noqa: N802 — nombre del proto
            if registro is not None:
                registro.append((req.key, bool(req.cancel)))
            if raises is not None:
                raise raises
            return object()

    monkeypatch.setattr(agent_mod.grpc, "insecure_channel",
                        lambda target: contextlib.nullcontext(object()))
    monkeypatch.setattr(agent_mod.inference_pb2_grpc, "InferenceServiceStub", Stub)
    return agent_mod


def test_agent_heartbeat_sends_total_vram(monkeypatch):
    """El loop de heartbeat reenvía la VRAM total del snapshot de la GPU."""
    from bridge import agent as agent_mod

    ag = _agent()
    capturado: dict = {}

    def snapshot():
        return {"gpu_name": "RTX 4090", "vram_total_mb": 24576,
                "vram_free_mb": 21000, "disk_free_mb": 90000, "models": []}

    class FakeClient:
        def heartbeat(self, **kwargs):
            capturado.update(kwargs)
            ag._stop.set()  # una sola vuelta del loop
            return {"cancel_task_ids": [], "requested_models": [],
                    "requested_deletions": [], "requested_download_cancels": []}

    monkeypatch.setattr(agent_mod, "gpu_snapshot", snapshot)
    ag.client = FakeClient()
    ag._heartbeat_loop()

    assert capturado["vram_total_mb"] == 24576
    assert capturado["vram_free_mb"] == 21000


def test_agent_cancel_download_calls_local_grpc(monkeypatch):
    """Cancelar descarga pide DownloadModel(cancel=True): el gRPC local aborta
    la transferencia y descarta los bytes parciales."""
    llamadas: list[tuple[str, bool]] = []
    _stub_download(monkeypatch, registro=llamadas)
    ag = _agent()

    assert ag._cancel_remote_download("SDXL-Turbo") is True
    assert llamadas == [("SDXL-Turbo", True)]


def test_agent_cancel_download_acks_unknown_key(monkeypatch):
    """Una clave que este worker no conoce no se reintenta para siempre."""
    agent_mod = _stub_download(monkeypatch)

    class NoExiste(agent_mod.grpc.RpcError):
        def code(self):
            return agent_mod.grpc.StatusCode.NOT_FOUND

        def details(self):
            return "modelo desconocido"

    _stub_download(monkeypatch, raises=NoExiste())
    ag = _agent()

    assert ag._cancel_remote_download("Fantasma") is True


def test_agent_cancel_download_retries_when_grpc_is_down(monkeypatch):
    """Si el gRPC local no responde no se hace ack: el back reintenta."""
    agent_mod = _stub_download(monkeypatch)

    class Caido(agent_mod.grpc.RpcError):
        def code(self):
            return agent_mod.grpc.StatusCode.UNAVAILABLE

        def details(self):
            return "connection refused"

    _stub_download(monkeypatch, raises=Caido())
    ag = _agent()

    assert ag._cancel_remote_download("SDXL-Turbo") is False


def test_client_heartbeat_rejects_bad_token(fake_back):
    client = BridgeClient(fake_back, token="wrong")
    with pytest.raises(BridgeError) as exc:
        client.heartbeat(gpu_name="", version="", vram_free_mb=0, jobs_active=0, models=[])
    assert exc.value.status == 401


def test_client_claim_empty_and_delivered(fake_back):
    client = BridgeClient(fake_back, token="tok-secret")
    assert client.claim(wait_seconds=1) is None
    FakeBack.claimed_job = {"task_id": "syn_5",
                            "payload": {"model_name": "Wan2.1", "jobId": "syn_5"}}
    job = client.claim(wait_seconds=1)
    assert job is not None and job.task_id == "syn_5"
    assert job.payload["model_name"] == "Wan2.1"


def test_client_claim_accepts_bare_job_payload(fake_back):
    """Backs que devuelven el job pelado en `data` (sin la clave `job`).

    Si el cliente lo descarta, el back ya marcó la fila 'running' y el trabajo
    no lo ejecuta nadie: la toma se queda en 0 % para siempre, sin error.
    """
    client = BridgeClient(fake_back, token="tok-secret")
    FakeBack.bare_job = True
    FakeBack.claimed_job = {"task_id": "syn_7",
                            "payload": {"model_name": "SD-Tiny-Test"}}
    job = client.claim(wait_seconds=1)
    assert job is not None and job.task_id == "syn_7"


def test_client_report_and_upload(fake_back, tmp_path):
    client = BridgeClient(fake_back, token="tok-secret")
    client.report("syn_5", "running", 42.5, stream_status="GENERATING")
    assert FakeBack.reports[-1]["progress"] == 42.5

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-video-bytes")
    url = client.upload_artifact("syn_5", str(clip))
    assert url.endswith("clip.mp4")
    assert FakeBack.uploads[-1] == {"task_id": "syn_5", "filename": "clip.mp4"}


def test_upload_empty_artifact_raises(tmp_path, fake_back):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    client = BridgeClient(fake_back, token="tok-secret")
    with pytest.raises(BridgeError):
        client.upload_artifact("syn_5", str(empty))


# ─── reintentos ────────────────────────────────────────────────

def test_with_retries_recovers():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise BridgeError("temporal", status=503)
        return "ok"

    assert with_retries(flaky, attempts=5, base_delay=0.01) == "ok"
    assert calls["n"] == 3


def test_with_retries_does_not_retry_4xx():
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise BridgeError("contrato roto", status=400)

    with pytest.raises(BridgeError):
        with_retries(broken, attempts=3, base_delay=0.01)
    assert calls["n"] == 1  # un 4xx no mejora reintentando


def test_job_payload_accepts_stringified_json():
    job = Job.from_api({"task_id": "t1", "payload": '{"model_name": "X"}'})
    assert job.payload == {"model_name": "X"}


# ─── agente: reconstrucción del request gRPC ───────────────────

def test_request_from_payload_protojson():
    """El payload protojson del back (camelCase) debe parsear al proto."""
    from bridge.agent import _request_from_payload
    job = Job.from_api({"task_id": "syn_1", "payload": {
        "modelName": "Wan2.1-T2V-1.3B", "mode": "video", "prompt": "un perro",
        "width": 832, "height": 480, "numFrames": 32, "cfgScale": 5.0, "seed": 7,
    }})
    req = _request_from_payload(job)
    assert req.model_name == "Wan2.1-T2V-1.3B"
    assert req.num_frames == 32
    assert req.job_id == "syn_1"  # el job_id del back manda


def test_artifact_file_path_resolves_worker_output_dir(tmp_path, monkeypatch):
    """El artefacto llega como nombre pelado y el upload abre outputs/<nombre>.

    Sin esta resolución el bridge buscaba `syn_1.mp4` en la raíz del worker,
    fallaba los 8 reintentos de subida y el job quedaba en 90 % para siempre.
    """
    from bridge import agent

    out = tmp_path / "outputs"
    out.mkdir(exist_ok=True)
    target = out / "syn_1.mp4"
    target.write_bytes(b"video")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BM_OUTPUT_DIR", raising=False)

    assert agent.artifact_file_path("syn_1.mp4") == os.path.join("outputs", "syn_1.mp4")
    assert agent.artifact_file_path(str(target)) == str(target)
    # Nombre desconocido: se devuelve igual para que el error sea del upload.
    assert agent.artifact_file_path("nope.mp4") == os.path.join("outputs", "nope.mp4")


def test_gpu_snapshot_shape():
    """Sin torch/CUDA el snapshot igual tiene la forma que espera el back."""
    os.environ["BM_FORCE_MOCK"] = "1"
    from bridge.agent import gpu_snapshot
    snap = gpu_snapshot()
    assert {"gpu_name", "vram_total_mb", "vram_free_mb", "models"} <= set(snap)
    assert isinstance(snap["models"], list) and snap["models"]
    first = snap["models"][0]
    assert {"key", "label", "mode", "engine", "vram_gb", "available"} <= set(first)


def test_gpu_snapshot_fails_closed_without_local_state(monkeypatch):
    """Sin estado del gRPC local los modelos viajan como NO disponibles.

    El default viejo (`available=True`) dejaba el panel mostrando todo
    "Disponible" en una máquina sin pesos: el botón Descargar solo aparece
    para los no disponibles, así que no había forma de bajarlos.
    """
    import bridge.agent as agent

    monkeypatch.setattr(agent, "_local_model_states", lambda: {})
    snap = agent.gpu_snapshot()
    assert snap["models"]
    assert all(m["available"] is False for m in snap["models"])
    assert all(m["download_percent"] == 0.0 for m in snap["models"])


def test_gpu_snapshot_reports_local_weight_state(monkeypatch):
    """El estado real de pesos (disponible / descargando / %) viaja tal cual."""
    import bridge.agent as agent

    monkeypatch.setattr(agent, "_local_model_states", lambda: {
        "SDXL-Turbo": {"available": True, "downloaded": True,
                       "downloading": False,
                         "download_percent": 100.0, "size_bytes": 8_832_013,
                         "download_error": "",
                         "download_path": r"C:\Users\x\.cache\huggingface\hub\models--org--demo"},
        "SD-1.5": {"available": False, "downloaded": False,
                   "downloading": True,
                   "download_percent": 17.8, "size_bytes": 9_014_119_167,
                   "download_error": "", "download_path": ""},
    })
    snap = agent.gpu_snapshot()
    by_key = {m["key"]: m for m in snap["models"]}
    assert by_key["SDXL-Turbo"]["available"] is True
    assert by_key["SDXL-Turbo"]["size_bytes"] == 8_832_013
    # La ruta de los pesos viaja al back: sin esto el panel no puede mostrarla.
    assert by_key["SDXL-Turbo"]["downloaded"] is True
    assert by_key["SDXL-Turbo"]["download_path"].endswith("models--org--demo")
    assert by_key["SD-1.5"]["downloading"] is True
    assert by_key["SD-1.5"]["download_percent"] == 17.8
    assert by_key["SD-1.5"]["download_path"] == ""
    # Clave sin estado: nunca se inventa disponibilidad.
    assert by_key["Wan2.1-T2V-14B"]["available"] is False
