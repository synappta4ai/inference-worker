"""Motor mock: artefactos deterministas sin pesos ni descargas.

Se usa para los modelos de video (hasta integrar WanGP/LTX) y como fallback
seguro. Es un generador async que produce tuplas de eventos
(pct, status, output, error) y escribe el artefacto al final.
"""

from __future__ import annotations

import asyncio
import os
import struct

from .base import EngineContext, ModelSpec

# Modo → extensión del artefacto. `tts`/`audio` producen WAV: con el "mp4 el
# resto" anterior completaban con un archivo que decía ser video.
_EXT_BY_MODE = {"image": "png", "tts": "wav", "audio": "wav"}

_PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd40000000049454e44ae426082"
)


def _output_path(job_id: str, mode: str) -> str:
    """Artefacto por modo: png (imagen), wav (tts/audio), mp4 el resto."""
    return f"{job_id}.{_EXT_BY_MODE.get((mode or '').lower(), 'mp4')}"


def _silent_wav(samples: int = 800, rate: int = 8000) -> bytes:
    """WAV PCM 16-bit mono en silencio (0,1 s): archivo válido, no un renombre."""
    data = b"\x00\x00" * samples
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data" + struct.pack("<I", len(data))
    )
    return header + data


def _write_output(path: str, req) -> None:
    """Artefacto determinista válido (placeholder del encoder final).

    El contenido sale de la extensión que fijó `_output_path`, así el mapa
    modo → artefacto vive en un solo lugar.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        payload = _PNG_1PX
    elif ext == ".wav":
        payload = _silent_wav()
    else:
        payload = f"BM_MOCK_VIDEO job={req.job_id} frames={req.num_frames}\n".encode()
    with open(path, "wb") as fh:
        fh.write(payload)


async def generate(req, spec: ModelSpec, ctx: EngineContext):
    """Pipeline simulado con el mismo contrato de eventos que el motor real."""
    steps = max(1, req.steps or 20)

    yield (0.0, "LOADING_MODEL", "", "")
    await asyncio.sleep(0.3)

    for step in range(1, steps + 1):
        if ctx.cancel.is_set():
            yield (step / steps * 100.0, "CANCELLED", "", "")
            return
        await asyncio.sleep(0.06)
        yield (step / steps * 95.0, "GENERATING", "", "")

    yield (97.0, "POSTPROCESSING", "", "")
    await asyncio.sleep(0.15)

    out_name = _output_path(req.job_id, req.mode)
    out_path = os.path.join(ctx.out_dir, os.path.basename(out_name))
    try:
        _write_output(out_path, req)
    except OSError as exc:
        yield (0.0, "FAILED", "", str(exc))
        return

    yield (100.0, "COMPLETED", out_name, "")
