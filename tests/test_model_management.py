"""Gestión de pesos del worker: inspección local y RPC nuevas.

Todas las pruebas son herméticas: usan una caché HF temporal y sustituyen
`snapshot_download`, de modo que nunca tocan la red ni descargan pesos.
"""

import threading
import time

import grpc
import inference_pb2
import pytest

import model_ops
from engines.base import ModelSpec


DEMO_SPEC = ModelSpec(
    key="demo",
    label="Modelo demo",
    engine="diffusers",
    repo="org/demo",
    mode="image",
    pipeline="sd15",
    vram_gb=4,
    family="demo",
)


def use_demo_cache(tmp_path, monkeypatch):
    """Aísla la caché HF y el catálogo en el directorio temporal del test."""
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    monkeypatch.setattr(model_ops, "HF_HUB_CACHE", str(cache))
    monkeypatch.setitem(model_ops.MODEL_CATALOG, "demo", DEMO_SPEC)
    return cache


def use_demo_catalog_only(monkeypatch):
    """Reduce el catálogo visible a la demo: los tests bulk (predownload,
    delete_all) NUNCA deben recorrer ni tocar la caché HF real del equipo
    (50 GB: lento y destructivo)."""
    monkeypatch.setattr(model_ops, "MODEL_CATALOG", {"demo": DEMO_SPEC})


def write_cache_file(cache, *parts, content=b"x" * 16):
    path = cache.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def write_repo_tree(cache, files):
    tree = {
        name: {"size": size, "blob_id": f"{index:064x}"}
        for index, (name, size) in enumerate(files.items())
    }
    write_cache_file(
        cache,
        "models--org--demo",
        "trees",
        "commit.json",
        content=__import__("json").dumps({"files": tree}).encode(),
    )


def wait_for(op_key_done, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if op_key_done():
            return True
        time.sleep(0.02)
    return False


class TestInspectRepo:
    def test_complete_repo_reports_ready(self, tmp_path, monkeypatch):
        cache = use_demo_cache(tmp_path, monkeypatch)
        write_repo_tree(cache, {"model.safetensors": 32, "config.json": 4})
        write_cache_file(cache, "models--org--demo", "snapshots", "rev", "model.safetensors")
        write_cache_file(cache, "models--org--demo", "snapshots", "rev", "config.json")

        ops = model_ops.ModelOps()
        state = ops.snapshot("demo")

        assert state.state == model_ops.READY
        assert state.downloaded is True
        assert state.progress == 100.0
        assert state.total_bytes == 36
        assert state.size_bytes > 0

    def test_partial_blob_is_not_ready(self, tmp_path, monkeypatch):
        cache = use_demo_cache(tmp_path, monkeypatch)
        write_repo_tree(cache, {"model.safetensors": 32, "config.json": 4})
        write_cache_file(cache, "models--org--demo", "snapshots", "rev", "config.json")
        write_cache_file(
            cache,
            "models--org--demo",
            "blobs",
            f"{0:064x}.x.incomplete",
            content=b"x" * 8,
        )

        state = model_ops.ModelOps().snapshot("demo")

        assert state.state == model_ops.IDLE
        assert state.downloaded is False
        assert 0.0 < state.progress < 100.0

    def test_unknown_key_raises(self, tmp_path, monkeypatch):
        use_demo_cache(tmp_path, monkeypatch)
        with pytest.raises(KeyError):
            model_ops.ModelOps().snapshot("missing")


class TestDownloadLifecycle:
    def test_start_download_completes_without_network(self, tmp_path, monkeypatch):
        use_demo_cache(tmp_path, monkeypatch)
        monkeypatch.setattr(
            model_ops,
            "snapshot_download",
            lambda repo, tqdm_class=None, max_workers=None: repo,
        )

        ops = model_ops.ModelOps()
        ops.start_download("demo")

        assert ops._ops["demo"].thread is not None
        assert wait_for(lambda: not ops._running(ops._ops["demo"]))
        finished = ops.snapshot("demo")

        # Sin árbol ni snapshot, el hilo termina sin marcar listo ni fallar.
        assert finished.state == model_ops.IDLE
        assert finished.error == ""

    def test_failed_download_reports_error(self, tmp_path, monkeypatch):
        use_demo_cache(tmp_path, monkeypatch)

        def boom(repo, tqdm_class=None, max_workers=None):
            raise RuntimeError("red caída")

        monkeypatch.setattr(model_ops, "snapshot_download", boom)

        ops = model_ops.ModelOps()
        ops.start_download("demo")
        assert wait_for(lambda: not ops._running(ops._ops["demo"]))
        state = ops.snapshot("demo")

        assert state.state == model_ops.FAILED
        assert "red caída" in state.error

    def test_cancel_aborts_progress_bar(self, tmp_path, monkeypatch):
        use_demo_cache(tmp_path, monkeypatch)
        entered = threading.Event()

        def slow_download(repo, tqdm_class=None, max_workers=None):
            bar = tqdm_class(total=1, desc="cancel-test")
            entered.set()
            deadline = time.time() + 5
            while time.time() < deadline:
                bar.update(1)
                time.sleep(0.01)
            raise AssertionError("la cancelación no abortó la descarga")

        monkeypatch.setattr(model_ops, "snapshot_download", slow_download)

        ops = model_ops.ModelOps()
        ops.start_download("demo")
        assert entered.wait(timeout=5)

        cancelled = ops.cancel("demo")

        assert cancelled.error == ""
        assert cancelled.state in {model_ops.IDLE, model_ops.DOWNLOADING}
        assert wait_for(lambda: not ops._running(ops._ops["demo"]))
        assert ops.snapshot("demo").state == model_ops.IDLE

    def test_rejects_mock_bridge_and_busy_models(self, tmp_path, monkeypatch):
        use_demo_cache(tmp_path, monkeypatch)
        monkeypatch.setitem(
            model_ops.MODEL_CATALOG,
            "mock-demo",
            ModelSpec(key="mock-demo", label="Mock", engine="mock", repo=None),
        )
        monkeypatch.setitem(
            model_ops.MODEL_CATALOG,
            "bridge-demo",
            ModelSpec(
                key="bridge-demo",
                label="Bridge",
                engine="v1_bridge",
                repo="org/bridge",
            ),
        )

        ops = model_ops.ModelOps()
        with pytest.raises(ValueError):
            ops.start_download("mock-demo")
        with pytest.raises(ValueError):
            ops.start_download("bridge-demo")

    def test_delete_removes_repo_cache(self, tmp_path, monkeypatch):
        cache = use_demo_cache(tmp_path, monkeypatch)
        write_repo_tree(cache, {"model.safetensors": 8})
        write_cache_file(cache, "models--org--demo", "snapshots", "rev", "model.safetensors")

        ops = model_ops.ModelOps()
        deleted = ops.delete("demo")

        assert deleted.state == model_ops.DELETED
        assert deleted.bytes_downloaded == 0
        assert not (cache / "models--org--demo").exists()


class FakeOps:
    def __init__(self, state):
        self._state = state
        self.calls = []

    def snapshot(self, key):
        self.calls.append(("snapshot", key))
        return self._state

    def states(self, key=""):
        self.calls.append(("states", key))
        return [self._state]

    def disk_total(self):
        return 123

    def start_download(self, key):
        self.calls.append(("start", key))
        return self._state

    def cancel(self, key):
        self.calls.append(("cancel", key))
        return self._state

    def delete(self, key):
        self.calls.append(("delete", key))
        return self._state


class AbortError(Exception):
    def __init__(self, code, details):
        super().__init__(details)
        self.code = code
        self.details = details


class FakeContext:
    async def abort(self, code, details):
        raise AbortError(code, details)


def ready_state(**overrides):
    state = model_ops.ModelState(key="demo", state=model_ops.READY)
    state.downloaded = True
    state.progress = 100.0
    state.total_bytes = 100
    state.bytes_downloaded = 100
    state.size_bytes = 120
    for name, value in overrides.items():
        setattr(state, name, value)
    return state


class TestServerModelRPCs:
    async def test_list_models_reports_weight_state(self, servicer, monkeypatch):
        state = ready_state()
        monkeypatch.setattr(servicer, "ops", FakeOps(state))
        monkeypatch.setattr(
            "server.registry.list_models",
            lambda: [
                {
                    "id": "demo",
                    "name": "Demo",
                    "mode": "image",
                    "engine": "diffusers",
                    "pipeline": "sd15",
                    "repo": "org/demo",
                    "steps": 4,
                    "vram_gb": 4,
                    "family": "demo",
                    "notes": "",
                },
                {
                    "id": "bridge",
                    "name": "Bridge",
                    "mode": "video",
                    "engine": "v1_bridge",
                    "pipeline": "ltx25",
                    "repo": "org/bridge",
                    "steps": 30,
                    "vram_gb": 48,
                    "family": "bridge",
                    "notes": "",
                },
            ],
        )
        monkeypatch.setattr("engines.v1_bridge.is_available", lambda: False)

        resp = await servicer.ListModels(inference_pb2.ListModelsRequest(), None)

        assert [m.key for m in resp.models] == ["demo", "bridge"]
        assert resp.models[0].available is True
        assert resp.models[0].downloaded is True
        assert resp.models[0].download_percent == 100.0
        assert resp.models[0].size_bytes == 120
        assert resp.models[1].available is False

    async def test_download_cancel_and_delete_map_state(self, servicer, monkeypatch):
        state = ready_state(state=model_ops.DOWNLOADING, downloading=True, progress=12.5)
        monkeypatch.setattr(servicer, "ops", FakeOps(state))

        downloaded = await servicer.DownloadModel(
            inference_pb2.DownloadModelRequest(key="demo"), None
        )
        cancelled = await servicer.DownloadModel(
            inference_pb2.DownloadModelRequest(key="demo", cancel=True), None
        )
        deleted = await servicer.DeleteModel(
            inference_pb2.DeleteModelRequest(key="demo"), None
        )
        states = await servicer.GetModelState(
            inference_pb2.GetModelStateRequest(key="demo"), None
        )

        assert (downloaded.state, downloaded.progress) == ("downloading", 12.5)
        assert cancelled.bytes_downloaded == 100
        assert deleted.total_bytes == 100
        assert [s.key for s in states.states] == ["demo"]
        assert states.bytes_on_disk == 123

    async def test_unknown_key_aborts_not_found(self, servicer):
        with pytest.raises(AbortError) as exc:
            await servicer.DownloadModel(
                inference_pb2.DownloadModelRequest(key="missing"), FakeContext()
            )

        assert exc.value.code == grpc.StatusCode.NOT_FOUND

    async def test_empty_key_aborts_invalid_argument(self, servicer):
        with pytest.raises(AbortError) as exc:
            await servicer.DeleteModel(
                inference_pb2.DeleteModelRequest(key="  "), FakeContext()
            )

        assert exc.value.code == grpc.StatusCode.INVALID_ARGUMENT


class TestLazyDownloads:
    """Modo personal: nada se descarga solo; el modo compartido sí."""

    def test_personal_is_default(self, monkeypatch):
        monkeypatch.delenv("BM_PROVIDER_MODE", raising=False)
        assert model_ops.provider_mode() == model_ops.PROVIDER_MODE_PERSONAL
        assert model_ops.provider_shared() is False

    def test_shared_mode_flag(self, monkeypatch):
        monkeypatch.setenv("BM_PROVIDER_MODE", "shared")
        assert model_ops.provider_shared() is True

    def test_require_weights_fails_fast_when_missing(self, tmp_path, monkeypatch):
        use_demo_cache(tmp_path, monkeypatch)
        monkeypatch.delenv("BM_PROVIDER_MODE", raising=False)
        with pytest.raises(RuntimeError, match="Tu GPU"):
            model_ops.require_weights("org/demo")

    def test_require_weights_passes_when_complete(self, tmp_path, monkeypatch):
        cache = use_demo_cache(tmp_path, monkeypatch)
        monkeypatch.delenv("BM_PROVIDER_MODE", raising=False)
        write_repo_tree(cache, {"model.safetensors": 32})
        write_cache_file(cache, "models--org--demo", "snapshots", "rev", "model.safetensors")
        model_ops.require_weights("org/demo")  # no debe lanzar

    def test_require_weights_is_noop_in_shared_mode(self, tmp_path, monkeypatch):
        use_demo_cache(tmp_path, monkeypatch)
        monkeypatch.setenv("BM_PROVIDER_MODE", "shared")
        model_ops.require_weights("org/demo")  # sin pesos, pero permitido

    def test_require_weights_ignores_repoless_models(self, monkeypatch):
        monkeypatch.delenv("BM_PROVIDER_MODE", raising=False)
        model_ops.require_weights(None)  # mock/bridge: nada que esperar

    def test_predownload_all_skips_complete(self, tmp_path, monkeypatch):
        use_demo_catalog_only(monkeypatch)
        cache = use_demo_cache(tmp_path, monkeypatch)
        write_repo_tree(cache, {"model.safetensors": 32})
        write_cache_file(cache, "models--org--demo", "snapshots", "rev", "model.safetensors")
        launched = model_ops.predownload_all(model_ops.ModelOps())
        assert "demo" not in launched  # ya completo: no se relanza

    def test_predownload_all_launches_missing(self, tmp_path, monkeypatch):
        use_demo_catalog_only(monkeypatch)
        use_demo_cache(tmp_path, monkeypatch)
        started = []
        monkeypatch.setattr(
            model_ops.ModelOps,
            "start_download",
            lambda self, key: started.append(key) or model_ops.ModelState(key=key),
        )
        launched = model_ops.predownload_all(model_ops.ModelOps())
        assert "demo" in launched
        assert "demo" in started


class TestDeleteAll:
    def test_delete_all_removes_every_repo(self, tmp_path, monkeypatch):
        use_demo_catalog_only(monkeypatch)
        cache = use_demo_cache(tmp_path, monkeypatch)
        write_repo_tree(cache, {"model.safetensors": 32})
        write_cache_file(cache, "models--org--demo", "snapshots", "rev", "model.safetensors")
        ops = model_ops.ModelOps()

        deleted, failed, freed = ops.delete_all()

        assert "demo" in deleted
        assert failed == {}
        assert freed > 0
        assert not (cache / "models--org--demo").exists()
        assert ops.snapshot("demo").downloaded is False

    def test_delete_all_reports_in_progress(self, tmp_path, monkeypatch):
        use_demo_catalog_only(monkeypatch)
        use_demo_cache(tmp_path, monkeypatch)
        ops = model_ops.ModelOps()

        def slow_download(repo, tqdm_class=None, max_workers=None):
            time.sleep(3.0)
            return repo

        monkeypatch.setattr(model_ops, "snapshot_download", slow_download)
        ops.start_download("demo")
        try:
            deleted, failed, freed = ops.delete_all()
        finally:
            ops.cancel("demo")

        assert "demo" not in deleted
        assert "demo" in failed  # con descarga en curso: se pide cancelar primero
        assert freed == 0
