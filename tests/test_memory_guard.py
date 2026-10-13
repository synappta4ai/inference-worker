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
