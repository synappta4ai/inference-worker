"""Selección del dispositivo de cómputo: CUDA, XPU o CPU.

El worker asumía NVIDIA en todos lados (`torch.cuda.is_available()`,
`device="cuda"`, `enable_model_cpu_offload()` sin device, `float16` fijo). En
una Intel Arc eso terminaba siempre en CPU, y en las iGPU Radeon (Ryzen) el
worker intentaba CUDA y moría al cargar el pipeline.

Acá se centraliza la decisión:

    CUDA (NVIDIA, o ROCm en Linux)  → fp16/bf16 del catálogo   + offload real
    XPU  (Intel Arc)                → fp16/bf16 (XMX)          + offload real
    CPU  (iGPU Ryzen/Intel, o nada)  → fp32 (fp16 no es fiable) + slicing

El detalle de cada caso (backend, VRAM total/libre y precisión efectiva) se
expone en `DeviceInfo.describe()` —va al log y a la telemetría— para que se
vea por qué se eligió cada cosa.

Variables de entorno:
    BM_DEVICE=cuda|xpu|cpu|auto   fuerza el backend (default auto)
    BM_CPU_DTYPE=bfloat16        CPU en bf16 en vez de fp32 (Zen 4 / AMX)
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass, replace

logger = logging.getLogger("Device")

ENV_DEVICE = "BM_DEVICE"
ENV_CPU_DTYPE = "BM_CPU_DTYPE"

# Cuánta VRAM necesita cada backend para considerar el modelo "residente".
# Por debajo de esto se usa offload por etapas (el pipeline entra y sale).
BACKENDS = ("cuda", "xpu", "cpu")

# Precisión efectiva: la pedida por el catálogo se respeta en acelerador y se
# degrada a fp32 en CPU (half en CPU usa kernels emulados: lentísimo y con
# errores de `addmm_impl_cpu_` en varias capas de diffusers).
_PRECISION = {
    "cuda": {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"},
    "xpu": {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"},
    "cpu": {"float16": "float32", "bfloat16": "float32", "float32": "float32"},
}

_PRETTY = {
    "float16": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
}


@dataclass(frozen=True)
class DeviceInfo:
    """Dispositivo elegido y el porqué.

    `total_mb`/`free_mb` son VRAM en acelerador y RAM en CPU: es la memoria
    contra la que hay que decidir si un pipeline entra entero.
    """

    backend: str            # "cuda" | "xpu" | "cpu"
    torch_device: str       # "cuda", "cuda:0", "xpu", "cpu"
    name: str               # "NVIDIA GeForce RTX 5070", "Intel(R) Arc(TM) A770", ...
    vendor: str             # "NVIDIA" | "Intel" | "AMD" | ...
    total_mb: int
    free_mb: int
    precision: str          # "float16" | "bfloat16" | "float32"
    reason: str = ""
    # true en dispositivos inyectados (tests): `memory_mb()` devuelve los
    # valores de acá en vez de medir el equipo real.
    fixed: bool = False

    @property
    def is_gpu(self) -> bool:
        return self.backend != "cpu"

    @property
    def pretty_precision(self) -> str:
        return _PRETTY.get(self.precision, self.precision)

    @property
    def display(self) -> str:
        """Nombre para telemetría: `Intel Arc A770 (XPU, fp16)`."""
        return f"{self.name} ({self.backend.upper()}, {self.pretty_precision})"

    def describe(self) -> str:
        """Detalle humano: incluye VRAM total/libre y el motivo."""
        memoria = "RAM" if self.backend == "cpu" else "VRAM"
        detalle = (
            f"{self.display} · {self.total_mb / 1024:.1f} GB de {memoria} "
            f"({self.free_mb / 1024:.1f} GB libres)"
        )
        return f"{detalle} · {self.reason}" if self.reason else detalle

    def summary(self) -> dict:
        return {
            "backend": self.backend,
            "vendor": self.vendor,
            "device_name": self.display,
            "precision": self.precision,
            "is_gpu": self.is_gpu,
            "total_mb": self.total_mb,
            "free_mb": self.free_mb,
            "reason": self.reason,
        }


# Permite a los tests (y a un futuro flag) fijar el dispositivo sin hardware.
_OVERRIDE: DeviceInfo | None = None
_CACHE: dict[str, DeviceInfo] = {}


def set_override(device: DeviceInfo | None) -> None:
    """Fija un dispositivo (tests o `BM_DEVICE` resuelto a mano)."""
    global _OVERRIDE
    _OVERRIDE = device
    _CACHE.clear()


def clear_override() -> None:
    set_override(None)


# ── detección ────────────────────────────────────────────────────────
def _torch():
    try:
        import torch  # noqa: PLC0415 — import pesado y opcional

        return torch
    except Exception:  # noqa: BLE001 — sin torch se cae a CPU
        return None


def _cpu_name() -> str:
    """Nombre legible del CPU (`AMD Ryzen 7 7840U ...` en Windows)."""
    if os.name == "nt":
        try:
            import winreg  # noqa: PLC0415 — sólo Windows

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                nombre = str(winreg.QueryValueEx(key, "ProcessorNameString")[0])
                if nombre.strip():
                    return " ".join(nombre.split())
        except Exception:  # noqa: BLE001 — best-effort
            pass
    return (
        os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.processor()
        or platform.machine()
        or "CPU"
    )


def _vendor_of(name: str) -> str:
    low = name.lower()
    if "nvidia" in low or "geforce" in low or "quadro" in low or "rtx" in low:
        return "NVIDIA"
    if "intel" in low or "arc" in low:
        return "Intel"
    if "amd" in low or "radeon" in low or "ryzen" in low or "instinct" in low:
        return "AMD"
    return "desconocido"


def _cuda_info(torch, precision: str) -> DeviceInfo | None:
    try:
        if not torch.cuda.is_available():
            return None
        indice = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(indice)
        nombre = torch.cuda.get_device_name(indice)
        total = int(props.total_memory) // 1024**2
        # mem_get_info consulta al driver: descuenta lo que otros procesos ya
        # comprometieron (total - reserved mentía con la VRAM ocupada).
        try:
            libre, _ = torch.cuda.mem_get_info(indice)
            libre = int(libre) // 1024**2
        except Exception:  # noqa: BLE001 — driver viejo
            reservado = torch.cuda.memory_reserved(indice) // 1024**2
            libre = max(0, total - int(reservado))
        return DeviceInfo(
            backend="cuda",
            torch_device=f"cuda:{indice}",
            name=nombre,
            vendor=_vendor_of(nombre),
            total_mb=total,
            free_mb=libre,
            precision=precision,
            reason="acelerador CUDA disponible",
        )
    except Exception as exc:  # noqa: BLE001 — un driver roto no es fatal
        logger.warning("[Device] CUDA no disponible (%s)", exc)
        return None


def _xpu_info(torch, precision: str) -> DeviceInfo | None:
    """Intel Arc vía `torch.xpu` (build XPU de PyTorch / IPEX)."""
    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            return None
        indice = xpu.current_device()
        props = xpu.get_device_properties(indice)
        nombre = str(getattr(props, "name", "") or "Intel XPU")
        total = int(getattr(props, "total_memory", 0)) // 1024**2
        libre = 0
        try:
            libre_bytes, _ = xpu.mem_get_info(indice)
            libre = int(libre_bytes) // 1024**2
        except Exception:  # noqa: BLE001 — mem_get_info no siempre existe
            try:
                reservado = int(xpu.memory_reserved(indice)) // 1024**2
                libre = max(0, total - reservado)
            except Exception:  # noqa: BLE001
                libre = total
        # Arc (Xe-HPG) tiene XMX para fp16 y bf16: se respeta lo pedido.
        return DeviceInfo(
            backend="xpu",
            torch_device=f"xpu:{indice}",
            name=nombre,
            vendor="Intel",
            total_mb=total,
            free_mb=libre,
            precision=precision,
            reason="Intel XPU disponible",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Device] XPU no disponible (%s)", exc)
        return None


def _cpu_info(reason: str, precision: str) -> DeviceInfo:
    total_mb = libre_mb = 0
    try:
        import psutil  # noqa: PLC0415 — opcional

        memoria = psutil.virtual_memory()
        total_mb = int(memoria.total) // 1024**2
        libre_mb = int(memoria.available) // 1024**2
    except Exception:  # noqa: BLE001 — sin psutil queda 0
        pass
    nombre = _cpu_name()
    return DeviceInfo(
        backend="cpu",
        torch_device="cpu",
        name=nombre,
        vendor=_vendor_of(nombre),
        total_mb=total_mb,
        free_mb=libre_mb,
        precision=precision,
        reason=reason,
    )


def _hint_sin_acelerador(torch) -> str:
    """Explica por qué no hay XPU/CUDA: casi siempre falta el build de torch."""
    pistas = ["sin acelerador: torch no reporta CUDA ni XPU"]
    if torch is None:
        pistas.append("torch no está instalado")
    elif getattr(torch, "xpu", None) is None:
        pistas.append(
            "para Intel Arc hace falta un torch con XPU "
            "(torch.xpu); el build CPU/PyPI no lo incluye"
        )
    return "; ".join(pistas)


def detect(force: str | None = None) -> DeviceInfo:
    """Resuelve el dispositivo. `force` acepta cuda|xpu|cpu|auto."""
    pedido = (force or os.environ.get(ENV_DEVICE, "") or "auto").strip().lower()
    if pedido in ("", "auto"):
        pedido = "auto"
    if pedido not in (*BACKENDS, "auto"):
        logger.warning("[Device] %s=%r desconocido: uso auto", ENV_DEVICE, pedido)
        pedido = "auto"

    if _OVERRIDE is not None and pedido == "auto":
        return _OVERRIDE
    if pedido in _CACHE:
        return _CACHE[pedido]

    cpu_dtype = (os.environ.get(ENV_CPU_DTYPE, "") or "").strip().lower()
    cpu_precision = "bfloat16" if cpu_dtype in ("bfloat16", "bf16") else "float32"
    torch = _torch()

    if pedido == "cpu":
        info = _cpu_info("forzado por BM_DEVICE=cpu", cpu_precision)
    elif pedido == "cuda":
        info = (_cuda_info(torch, "float16") if torch is not None else None) or \
            _cpu_info("se pidió CUDA pero no está disponible", cpu_precision)
    elif pedido == "xpu":
        info = (_xpu_info(torch, "float16") if torch is not None else None) or \
            _cpu_info("se pidió XPU pero no está disponible", cpu_precision)
    else:
        info = None
        if torch is not None:
            info = _cuda_info(torch, "float16") or _xpu_info(torch, "float16")
        if info is None:
            info = _cpu_info(_hint_sin_acelerador(torch), cpu_precision)

    _CACHE[pedido] = info
    return info


def current(use_accel: bool = True) -> DeviceInfo:
    """Dispositivo para cargar un modelo.

    `use_accel=False` fuerza CPU sin tocar torch (rutas de CPU y tests).
    """
    if not use_accel:
        return detect("cpu")
    return detect()


def memory_mb(dev: DeviceInfo | None = None) -> tuple[int, int]:
    """(libre, total) en MB del dispositivo: VRAM en GPU, RAM en CPU.

    Se vuelve a medir en cada llamada: entre la detección y la carga pueden
    pasar minutos (descarga de pesos) y la VRAM libre cambia.
    """
    dev = dev or current()
    if dev.fixed:
        return dev.free_mb, dev.total_mb
    if dev.backend == "cpu":
        info = _cpu_info("", dev.precision)
        return info.free_mb, info.total_mb
    torch = _torch()
    if torch is None:
        return dev.free_mb, dev.total_mb
    try:
        if dev.backend == "cuda":
            indice = torch.cuda.current_device()
            libre, total = torch.cuda.mem_get_info(indice)
            return int(libre) // 1024**2, int(total) // 1024**2
        xpu = torch.xpu
        indice = xpu.current_device()
        libre, total = xpu.mem_get_info(indice)
        return int(libre) // 1024**2, int(total) // 1024**2
    except Exception:  # noqa: BLE001 — se usa la última medición conocida
        return dev.free_mb, dev.total_mb


def dtype_name(requested: str, dev: DeviceInfo | None = None) -> str:
    """Precisión efectiva para `requested` en `dev` (string de torch)."""
    dev = dev or current()
    pedido = (requested or "float16").strip().lower()
    pedido = pedido.replace("fp16", "float16").replace("bf16", "bfloat16")
    pedido = pedido.replace("fp32", "float32").replace("half", "float16")
    if dev.backend == "cpu":
        # En CPU manda el dtype del equipo (BM_CPU_DTYPE): fp32 por defecto y
        # bf16 opt-in en Zen 4 / AMX. Pedir fp16 en CPU nunca gana.
        return dev.precision if dev.precision in ("bfloat16", "float32") else "float32"
    if pedido == "bfloat16" and dev.backend == "cuda":
        # bf16 llega con Ampere (sm_80); en Turing y anteriores el resultado
        # es ruido, así que se baja a fp16.
        try:
            torch = _torch()
            if tuple(torch.cuda.get_device_capability()) < (8, 0):
                return "float16"
        except Exception:  # noqa: BLE001 — sin datos se respeta lo pedido
            pass
    tabla = _PRECISION.get(dev.backend, _PRECISION["cuda"])
    return tabla.get(pedido, pedido)


def torch_dtype(requested: str, dev: DeviceInfo | None = None):
    """Igual que `dtype_name` pero devuelve el dtype de torch."""
    torch = _torch()
    if torch is None:
        raise RuntimeError("torch no está instalado")
    return getattr(torch, dtype_name(requested, dev))


def with_precision(dev: DeviceInfo, requested: str) -> DeviceInfo:
    """Copia de `dev` con la precisión ya resuelta para un modelo."""
    return replace(dev, precision=dtype_name(requested, dev))


def generator(seed: int, dev: DeviceInfo | None = None):
    """`torch.Generator` en el dispositivo, con fallback a CPU.

    XPU acepta Generator desde torch 2.5; con builds viejos tirar TypeError,
    y ahí se usa uno de CPU (el pipeline igual corre en la GPU).
    """
    torch = _torch()
    if torch is None or not seed:
        return None
    dev = dev or current()
    for destino in (dev.torch_device, "cpu"):
        try:
            return torch.Generator(device=destino).manual_seed(int(seed))
        except Exception:  # noqa: BLE001 — probar el siguiente
            continue
    return None


def empty_cache(dev: DeviceInfo | None = None) -> None:
    """Devuelve al driver la memoria no referenciada del dispositivo."""
    torch = _torch()
    if torch is None:
        return
    dev = dev or current()
    try:
        if dev.backend == "cuda":
            torch.cuda.empty_cache()
        elif dev.backend == "xpu":
            torch.xpu.empty_cache()
    except Exception:  # noqa: BLE001 — limpieza best-effort
        pass


def synchronize(dev: DeviceInfo | None = None) -> None:
    torch = _torch()
    if torch is None:
        return
    dev = dev or current()
    try:
        if dev.backend == "cuda":
            torch.cuda.synchronize()
        elif dev.backend == "xpu":
            torch.xpu.synchronize()
    except Exception:  # noqa: BLE001
        pass


def reset_peak(dev: DeviceInfo | None = None) -> None:
    torch = _torch()
    if torch is None:
        return
    dev = dev or current()
    try:
        if dev.backend == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elif dev.backend == "xpu":
            torch.xpu.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def peak_mb(dev: DeviceInfo | None = None) -> int:
    """Pico de memoria del dispositivo desde el último `reset_peak`."""
    torch = _torch()
    if torch is None:
        return 0
    dev = dev or current()
    try:
        if dev.backend == "cuda":
            return int(torch.cuda.max_memory_allocated()) // 1024**2
        if dev.backend == "xpu":
            return int(torch.xpu.max_memory_allocated()) // 1024**2
    except Exception:  # noqa: BLE001
        pass
    return 0
