"""El pipeline corre en un proceso hijo: un job pesado no tumba el worker.

Regresión real: al cargar Wan-1.3B por segunda vez, el sistema se quedó sin
RAM a mitad de un shard y mató el proceso del worker entero (no hay
`MemoryError` que atrapar). Ahora los pesos viven en `engines.worker_host.py`:
si ese hijo muere, el job termina en FAILED y el worker sigue atendiendo.
"""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from engines import diffusion_factory
from engines.base import EngineContext, ModelSpec

# Hijo falso: habla el mismo protocolo por líneas JSON que worker_host.py,
# sin descargar ni cargar nada. "crash" simula el OOM (muerte del proceso).
FAKE_HOST = r"""
import json, os, queue, sys, threading, time

def emit(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()

cmds = queue.Queue()

def reader():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg.get("cmd") == "cancel":
            emit({"type": "event", "pct": 0.0, "status": "CANCELLED"})
        else:
            cmds.put(msg)

threading.Thread(target=reader, daemon=True).start()

while True:
    job = cmds.get()
    prompt = (job.get("request") or {}).get("prompt", "")
    emit({"type": "event", "pct": 1.0, "status": "GENERATING"})
    if prompt == "crash":
        os._exit(3)          # como un OOM del sistema: sin traceback
    if prompt == "slow":
        while True:          # generando hasta que llegue el cancel
            time.sleep(0.05)
    if prompt == "wait":
        time.sleep(0.6)      # job "largo": permite ver solapamientos
    emit({"type": "log", "msg": "job completado"})
    emit({"type": "event", "pct": 100.0, "status": "COMPLETED", "out": "x.png"})
"""


@pytest.fixture(autouse=True)
def _fake_host(monkeypatch):
    monkeypatch.setattr(
        diffusion_factory, "_HOST_CMD",
        [sys.executable, "-u", "-c", FAKE_HOST],
    )
    yield
    with diffusion_factory._HOST_LOCK:
        hosts = list(diffusion_factory._HOSTS.values())
        diffusion_factory._HOSTS.clear()
    for host in hosts:
        host.kill()


def _spec(repo: str = "fake/a", mode: str = "image", vram_gb: int = 8) -> ModelSpec:
    return ModelSpec(
        key="fake", label="Modelo Falso", engine="diffusers",
        repo=repo, mode=mode, pipeline="sd15", vram_gb=vram_gb,
    )


def _req(prompt: str = "hola"):
    return SimpleNamespace(
        job_id="job1", prompt=prompt, negative_prompt="", width=0, height=0,
        num_frames=0, fps=0, seed=0, steps=1,
    )


def _run(spec, prompt: str = "hola", tmp_path=None) -> list:
    async def runner():
        ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())
        events = []
        async for evt in diffusion_factory.generate(_req(prompt), spec, ctx):
            events.append(evt)
        return events

    return asyncio.run(asyncio.wait_for(runner(), timeout=20))


class TestHostIsolation:

    def test_events_flow_from_host(self, tmp_path):
        events = _run(_spec(), tmp_path=tmp_path)

        assert [e[1] for e in events][-1] == "COMPLETED"
        assert events[-1][2] == "x.png"
        assert events[0][1] == "GENERATING"

    def test_host_crash_does_not_raise_and_marks_failed(self, tmp_path):
        events = _run(_spec(), prompt="crash", tmp_path=tmp_path)

        status, err = events[-1][1], events[-1][3]
        assert status == "FAILED"
        assert "se cerró de golpe" in err
        # El worker sigue en pie y no le queda ningún hijo vivo colgado.
        assert diffusion_factory.active_hosts() == []

    def test_next_job_respawns_host_after_crash(self, tmp_path):
        _run(_spec(), prompt="crash", tmp_path=tmp_path)
        events = _run(_spec(), tmp_path=tmp_path)

        assert events[-1][1] == "COMPLETED"
        assert diffusion_factory.active_hosts() == ["fake/a"]

    def test_switching_model_kills_idle_host(self, tmp_path):
        _run(_spec("fake/a"), tmp_path=tmp_path)
        first = diffusion_factory._HOSTS["fake/a"]

        _run(_spec("fake/b"), tmp_path=tmp_path)

        # Un modelo cargado a la vez: al matar al anterior el SO recupera su RAM.
        assert first.proc.poll() is not None
        assert diffusion_factory.active_hosts() == ["fake/b"]

    def test_cancel_reaches_the_host(self, tmp_path):
        async def runner():
            ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())
            stream = diffusion_factory.generate(_req("slow"), _spec(), ctx)
            first = await stream.__anext__()
            ctx.cancel.set()
            rest = [evt async for evt in stream]
            return first, rest

        first, rest = asyncio.run(asyncio.wait_for(runner(), timeout=20))

        assert first[1] == "GENERATING"
        assert rest[-1][1] == "CANCELLED"


class TestHeavySerialization:
    """Un solo modelo pesado cargado a la vez; los livianos no se encolan."""

    def _recorder(self, monkeypatch):
        """Cuenta cuántos hijos viven a la vez (máximo observado)."""
        live = {"now": 0, "max": 0}
        real = diffusion_factory._Host

        class Recording(real):
            def __init__(self, key, payload):
                live["now"] += 1
                live["max"] = max(live["max"], live["now"])
                super().__init__(key, payload)

            def kill(self):
                if self.alive:
                    live["now"] -= 1
                super().kill()

        monkeypatch.setattr(diffusion_factory, "_Host", Recording)
        return live

    def _parallel(self, repos, tmp_path, prompt="wait"):
        async def run():
            async def job(repo):
                ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())
                out = []
                async for evt in diffusion_factory.generate(
                    _req(prompt), _spec(repo), ctx
                ):
                    out.append(evt)
                return out

            return await asyncio.gather(*(job(r) for r in repos))

        return asyncio.run(asyncio.wait_for(run(), timeout=40))

    def test_two_heavy_models_do_not_load_together(self, tmp_path, monkeypatch):
        live = self._recorder(monkeypatch)

        results = self._parallel(["fake/heavy-a", "fake/heavy-b"], tmp_path)

        assert [r[-1][1] for r in results] == ["COMPLETED", "COMPLETED"]
        # El segundo hijo aparece recién cuando el primero terminó.
        assert live["max"] == 1

    def test_queued_job_reports_loading_state(self, tmp_path, monkeypatch):
        self._recorder(monkeypatch)

        results = self._parallel(["fake/heavy-c", "fake/heavy-d"], tmp_path)

        # El encolado avisa LOADING_MODEL antes de esperar (no queda mudo).
        assert [r[0][1] for r in results].count("LOADING_MODEL") == 1

    def test_light_models_still_run_in_parallel(self, tmp_path, monkeypatch):
        live = self._recorder(monkeypatch)

        async def run():
            async def job(repo):
                ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())
                spec = _spec(repo, vram_gb=2)
                out = []
                async for evt in diffusion_factory.generate(_req("wait"), spec, ctx):
                    out.append(evt)
                return out

            return await asyncio.gather(job("fake/light-a"), job("fake/light-b"))

        asyncio.run(asyncio.wait_for(run(), timeout=40))

        assert live["max"] == 2

    def test_cancel_while_queued_never_spawns_a_host(self, tmp_path, monkeypatch):
        live = self._recorder(monkeypatch)

        async def run():
            holder_ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())
            queued_ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())
            queued_ctx.cancel.set()  # se cancela antes de conseguir el carril

            async def holder():
                async for _ in diffusion_factory.generate(
                    _req("wait"), _spec("fake/heavy-e"), holder_ctx
                ):
                    pass

            async def queued():
                out = []
                async for evt in diffusion_factory.generate(
                    _req("wait"), _spec("fake/heavy-f"), queued_ctx
                ):
                    out.append(evt)
                return out

            return await asyncio.gather(holder(), queued())

        _holder_done, queued_events = asyncio.run(
            asyncio.wait_for(run(), timeout=40)
        )

        assert queued_events[-1][1] == "CANCELLED"
        assert "fake/heavy-f" not in diffusion_factory._HOSTS
        assert live["max"] == 1
