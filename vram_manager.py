"""Telemetría y liberación de memoria del dispositivo (CUDA, XPU o CPU).

Antes reportaba `torch.cuda.*` a secas: en una Intel Arc o en una iGPU Ryzen
devolvía "CPU / No CUDA" con 0 GB aunque la máquina tuviera memoria de sobra.
Ahora la fuente es `device.py`, que sabe qué backend hay y cuánta memoria le
queda en cada caso.
"""

import gc
import logging

import device as device_mod

logger = logging.getLogger("VRAMManager")


class VRAMManager:
    """Gestiona la telemetría y la liberación de memoria del dispositivo."""

    def __init__(self) -> None:
        # Una línea al arrancar con el detalle completo: backend, VRAM y la
        # precisión efectiva que va a usar cada modelo.
        logger.info("[VRAMManager] dispositivo: %s", device_mod.current().describe())

    @staticmethod
    def get_telemetry():
        dev = device_mod.current()
        libre, total = device_mod.memory_mb(dev)
        usado = max(0, total - libre)
        return {
            "device_name": dev.display,
            "backend": dev.backend,
            "vendor": dev.vendor,
            "precision": dev.precision,
            "is_gpu": dev.is_gpu,
            # `cuda_available` se mantiene por compatibilidad con el proto:
            # ahora significa "hay acelerador" (CUDA o XPU).
            "cuda_available": dev.is_gpu,
            "reason": dev.reason,
            "total_vram_mb": total,
            "used_vram_mb": usado,
            "free_vram_mb": libre,
            "gpu_utilization": (usado / total) * 100.0 if total else 0.0,
            "temperature_c": 0.0,
        }

    @staticmethod
    def clear_vram():
        """Limpia la memoria residual no referenciada del dispositivo."""
        gc.collect()
        dispositivo = device_mod.current()
        device_mod.empty_cache(dispositivo)
        logger.info(
            "[VRAMManager] memoria liberada (%s)", dispositivo.backend.upper()
        )
