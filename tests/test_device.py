"""Tests de `device.py`: CUDA, XPU (Intel Arc) y CPU.

Sin hardware: se reemplaza `device._torch()` por un torch falso con las
mismas superficies que usa la detección (`torch.cuda`, `torch.xpu`).
"""

from __future__ import annotations

import types

import pytest

import device as device_mod

GiB = 1024**3


class _Props:
    def __init__(self, nombre: str, total: int) -> None:
        self.name = nombre
        self.total_memory = total


def _fake_torch(*, cuda=False, xpu=False, capability=(8, 6), total=8 * GiB,
                libre=6 * GiB):
    """Torch mínimo con `cuda` y/o `xpu` disponibles."""
    torch = types.SimpleNamespace()

    if cuda:
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda i: _Props("NVIDIA GeForce RTX 5070", total),
            get_device_name=lambda i: "NVIDIA GeForce RTX 5070",
            mem_get_info=lambda i: (libre, total),
            # default como el torch real: dtype_name lo llama sin índice
            get_device_capability=lambda i=0: capability,
        )
    else:
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    if xpu:
        torch.xpu = types.SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda i: _Props("Intel(R) Arc(TM) A770", total),
            mem_get_info=lambda i: (libre, total),
        )
    # `hasattr(torch, "xpu")` es la señal de "torch con soporte XPU".
    return torch


@pytest.fixture(autouse=True)
def limpio(monkeypatch):
    monkeypatch.setattr(device_mod, "_OVERRIDE", None)
    monkeypatch.setattr(device_mod, "_CACHE", {})
    monkeypatch.delenv(device_mod.ENV_DEVICE, raising=False)
    monkeypatch.delenv(device_mod.ENV_CPU_DTYPE, raising=False)


def _con_torch(monkeypatch, torch):
    monkeypatch.setattr(device_mod, "_torch", lambda: torch)


# ── sin acelerador ───────────────────────────────────────────────────
def test_sin_acelerador_usa_cpu_en_fp32(monkeypatch):
    _con_torch(monkeypatch, _fake_torch())

    dev = device_mod.detect()

    assert dev.backend == "cpu"
    assert dev.torch_device == "cpu"
    assert dev.precision == "float32"
    assert dev.is_gpu is False
    assert "sin acelerador" in dev.reason
    # La CPU reporta RAM del equipo, no 0 MB como antes.
    assert dev.total_mb > 0


def test_cpu_explica_que_falta_torch_xpu(monkeypatch):
    """En una Arc sin build XPU el mensaje tiene que decir qué instalar."""
    _con_torch(monkeypatch, _fake_torch())

    dev = device_mod.detect()

    assert "Intel Arc" in dev.reason
    assert "torch.xpu" in dev.reason


def test_cpu_en_bf16_optativo(monkeypatch):
    monkeypatch.setenv(device_mod.ENV_CPU_DTYPE, "bfloat16")
    _con_torch(monkeypatch, _fake_torch())

    dev = device_mod.detect()

    assert dev.precision == "bfloat16"
    assert device_mod.dtype_name("float16", dev) == "bfloat16"


def test_bm_device_cpu_fuerza_cpu(monkeypatch):
    monkeypatch.setenv(device_mod.ENV_DEVICE, "cpu")
    _con_torch(monkeypatch, _fake_torch(cuda=True))

    dev = device_mod.detect()

    assert dev.backend == "cpu"
    assert "BM_DEVICE=cpu" in dev.reason


# ── CUDA ─────────────────────────────────────────────────────────────
def test_cuda_reporta_vram_y_fp16(monkeypatch):
    _con_torch(monkeypatch, _fake_torch(cuda=True))

    dev = device_mod.detect()

    assert dev.backend == "cuda"
    assert dev.torch_device == "cuda:0"
    assert dev.vendor == "NVIDIA"
    assert dev.total_mb == 8192
    assert dev.free_mb == 6144
    assert dev.precision == "float16"
    assert device_mod.dtype_name("bfloat16", dev) == "bfloat16"


def test_cuda_pre_ampere_baja_bf16_a_fp16(monkeypatch):
    """bf16 llegó con Ampere: en Turing/Ampere-viejo se usa fp16."""
    _con_torch(monkeypatch, _fake_torch(cuda=True, capability=(7, 5)))

    dev = device_mod.detect()

    assert dev.precision == "float16"
    assert device_mod.dtype_name("bfloat16", dev) == "float16"


# ── XPU (Intel Arc) ──────────────────────────────────────────────────
def test_detecta_intel_arc_como_xpu(monkeypatch):
    _con_torch(monkeypatch, _fake_torch(xpu=True, total=16 * GiB, libre=12 * GiB))

    dev = device_mod.detect()

    assert dev.backend == "xpu"
    assert dev.torch_device == "xpu:0"
    assert dev.vendor == "Intel"
    assert dev.is_gpu is True
    assert dev.total_mb == 16384
    assert dev.free_mb == 12288
    # Arc (Xe-HPG) tiene XMX para fp16 y bf16.
    assert device_mod.dtype_name("float16", dev) == "float16"
    assert device_mod.dtype_name("bfloat16", dev) == "bfloat16"


def test_cuda_gana_sobre_xpu_si_estan_los_dos(monkeypatch):
    _con_torch(monkeypatch, _fake_torch(cuda=True, xpu=True))

    assert device_mod.detect().backend == "cuda"


def test_pedir_xpu_sin_soporte_cae_a_cpu(monkeypatch):
    _con_torch(monkeypatch, _fake_torch())

    dev = device_mod.detect("xpu")

    assert dev.backend == "cpu"
    assert "XPU" in dev.reason


# ── memoria, precisión y textos ──────────────────────────────────────
def test_memory_mb_usa_la_medicion_fija_en_tests():
    dev = device_mod.DeviceInfo(
        backend="xpu", torch_device="xpu:0", name="Intel Arc A770",
        vendor="Intel", total_mb=16384, free_mb=9000,
        precision="float16", fixed=True,
    )

    assert device_mod.memory_mb(dev) == (9000, 16384)


def test_describe_incluye_backend_precision_y_vram(monkeypatch):
    _con_torch(monkeypatch, _fake_torch(xpu=True, total=16 * GiB, libre=12 * GiB))

    texto = device_mod.detect().describe()

    assert "Intel(R) Arc(TM) A770" in texto
    assert "XPU, fp16" in texto
    assert "16.0 GB de VRAM" in texto
    assert "12.0 GB libres" in texto


def test_generator_en_cpu_funciona():
    """La ruta de CPU no debe devolver None: la seed tiene que servir."""
    cpu = device_mod.current(use_accel=False)

    gen = device_mod.generator(1234, cpu)

    assert gen is not None


def test_override_manda_en_auto():
    fijo = device_mod.DeviceInfo(
        backend="xpu", torch_device="xpu:0", name="Arc de prueba",
        vendor="Intel", total_mb=8192, free_mb=7000,
        precision="float16", fixed=True,
    )
    device_mod.set_override(fijo)

    try:
        assert device_mod.detect() is fijo
        # `use_accel=False` sigue forzando CPU aunque haya override.
        assert device_mod.current(use_accel=False).backend == "cpu"
    finally:
        device_mod.clear_override()
