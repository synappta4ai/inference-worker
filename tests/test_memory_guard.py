"""Tests del guardia de memoria previo a cargar modelos (anti-OOM)."""

import pytest

from engines.diffusion_factory import _check_memory_before_load
from engines.base import ModelSpec


def _spec(vram_gb: int) -> ModelSpec:
    return ModelSpec(
        key="test", label="Test", engine="diffusers", repo="x/y",
        mode="image", vram_gb=vram_gb,
    )


def test_rejects_model_that_does_not_fit(monkeypatch):
    class FakeMem:
        available = 2 * 1024**3  # 2 GB libres

    monkeypatch.setattr("psutil.virtual_memory", lambda: FakeMem())

    with pytest.raises(MemoryError, match="más pequeños|más memoria"):
        _check_memory_before_load(_spec(8), use_gpu=False)  # necesita 12 GB


def test_allows_model_that_fits(monkeypatch):
    class FakeMem:
        available = 32 * 1024**3  # 32 GB libres

    monkeypatch.setattr("psutil.virtual_memory", lambda: FakeMem())

    # 8 GB * 1.5 = 12 < 32 → pasa sin excepción
    _check_memory_before_load(_spec(8), use_gpu=False)
    _check_memory_before_load(_spec(20), use_gpu=True)


def _fake_weights(monkeypatch, gb: float) -> None:
    import model_ops

    monkeypatch.setattr(model_ops, "weights_size_bytes", lambda repo: int(gb * 1024**3))


def _fake_available(monkeypatch, gb: float) -> None:
    class FakeMem:
        available = gb * 1024**3

    monkeypatch.setattr("psutil.virtual_memory", lambda: FakeMem())


def test_cpu_uses_real_weights_size(monkeypatch):
    """Wan declara 6 GB de VRAM pero en disco pesa ~27: el número sale de ahí."""
    _fake_available(monkeypatch, 22)
    _fake_weights(monkeypatch, 26.95)

    with pytest.raises(MemoryError) as exc:
        _check_memory_before_load(_spec(6), use_gpu=False)

    msg = str(exc.value)
    # 26.95 x 2 (en CPU se corre en fp32) + 1.5 de trabajo
    assert "necesita ~55.4 GB de RAM" in msg
    assert "26.9 GB de pesos en disco" in msg
    assert "22.0 GB libres" in msg


def test_allows_weights_that_really_fit(monkeypatch):
    _fake_available(monkeypatch, 8)
    # 3 GB en disco → 6 GB en fp32 + 1.5 de trabajo = 7.5 GB: entra en 8.
    _fake_weights(monkeypatch, 3.0)

    _check_memory_before_load(_spec(4), use_gpu=False)


def test_gpu_needs_less_ram_than_cpu(monkeypatch):
    """En GPU los pesos van a VRAM: sólo queda el pico de carga en RAM."""
    _fake_available(monkeypatch, 18)
    _fake_weights(monkeypatch, 26.95)

    with pytest.raises(MemoryError):
        _check_memory_before_load(_spec(6), use_gpu=False)

    # 26.95 x 0.6 + 1.5 = 17.7 GB < 18 GB
    _check_memory_before_load(_spec(6), use_gpu=True)


def test_falls_back_to_catalog_without_local_weights(monkeypatch):
    _fake_available(monkeypatch, 2)
    _fake_weights(monkeypatch, 0)

    with pytest.raises(MemoryError, match="estimación del catálogo"):
        _check_memory_before_load(_spec(8), use_gpu=False)


def test_continues_without_psutil(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("sin psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Sin psutil no puede medir → no bloquea (fail-open)
    _check_memory_before_load(_spec(48), use_gpu=False)
