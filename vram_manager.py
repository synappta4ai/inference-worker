import gc
import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger("VRAMManager")

class VRAMManager:
    """Gestiona la telemetría y liberación de memoria VRAM de la GPU."""
    
    @staticmethod
    def get_telemetry():
        if not HAS_TORCH or not torch.cuda.is_available():
            return {
                "device_name": "CPU / No CUDA",
                "total_vram_mb": 0,
                "used_vram_mb": 0,
                "free_vram_mb": 0,
                "gpu_utilization": 0.0,
                "temperature_c": 0.0,
            }
        
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)
        allocated = torch.cuda.memory_allocated(device) // (1024 * 1024)
        reserved = torch.cuda.memory_reserved(device) // (1024 * 1024)
        # Libre REAL visible para nuevos modelos: mem_get_info consulta al
        # driver (descuenta lo reservado por otros procesos). total-reserved
        # mentía: reportaba 14.9 GB libres con la VM recién iniciada mientras
        # el driver ya tenía comprometidos ~7.5 GB.
        try:
            free_driver, _total_driver = torch.cuda.mem_get_info(device)
            free = free_driver // (1024 * 1024)
        except Exception:
            free = total - reserved
        
        return {
            "device_name": torch.cuda.get_device_name(device),
            "total_vram_mb": total,
            "used_vram_mb": reserved,
            "free_vram_mb": free,
            "gpu_utilization": (reserved / total) * 100.0 if total > 0 else 0.0,
            "temperature_c": 0.0,
        }

    @staticmethod
    def clear_vram():
        """Limpia la memoria residual no referenciada en la GPU."""
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("[VRAMManager] Memoria VRAM liberada exitosamente.")
