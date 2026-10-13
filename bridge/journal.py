"""Journal de jobs en disco: nada se pierde si se corta la luz.

Un archivo JSON por task_id bajo ``<config_dir>/jobs/``. El agente escribe
una entrada al reclamar el job y la actualiza en cada transición:

    claimed → running → artifact_ready → uploaded

Si la máquina se apaga a mitad de un job, al arrancar el journal permite:
  * artifact_ready pendiente → re-subir el output al back (el upload es
    idempotente por task_id, así que un corte a mitad de subida también
    se recupera reintentando).
  * claimed/running sin artefacto → el job se perdió con el proceso; el
    back lo marca failed vía reconciler (heartbeat vencido) y el usuario
    lo reintenta. Nunca queda "running" eterno de ambos lados.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import config_dir

logger = logging.getLogger("bridge.journal")

# Estados del ciclo de vida de un job local.
STATE_CLAIMED = "claimed"
STATE_RUNNING = "running"
STATE_ARTIFACT_READY = "artifact_ready"
STATE_UPLOADED = "uploaded"


def jobs_dir() -> Path:
    return config_dir() / "jobs"


@dataclass
class JobEntry:
    task_id: str
    state: str = STATE_CLAIMED
    model_name: str = ""
    artifact_path: str = ""
    error: str = ""
    updated_at: float = field(default_factory=time.time)
    # Copia mínima del payload para diagnóstico post-mortem (sin secretos).
    payload_digest: str = ""


def _path_for(task_id: str) -> Path:
    # task_id viene del back (syn_<n>): alfanumérico + guiones bajos.
    safe = "".join(ch for ch in task_id if ch.isalnum() or ch in "-_")
    return jobs_dir() / f"{safe}.json"


def record(entry: JobEntry) -> None:
    """Escribe/actualiza la entrada del job (fsync: el journal debe sobrevivir)."""
    d = jobs_dir()
    d.mkdir(parents=True, exist_ok=True)
    entry.updated_at = time.time()
    path = _path_for(entry.task_id)
    tmp = path.with_suffix(".json.tmp")
    raw = json.dumps(asdict(entry), indent=2)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(raw)
        fh.flush()
    tmp.replace(path)  # rename atómico: nunca un JSON a medio escribir


def load(task_id: str) -> Optional[JobEntry]:
    try:
        raw = json.loads(_path_for(task_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    return JobEntry(
        task_id=str(raw.get("task_id", task_id)),
        state=str(raw.get("state", STATE_CLAIMED)),
        model_name=str(raw.get("model_name", "")),
        artifact_path=str(raw.get("artifact_path", "")),
        error=str(raw.get("error", "")),
        updated_at=float(raw.get("updated_at", 0.0)),
        payload_digest=str(raw.get("payload_digest", "")),
    )


def pending_uploads() -> List[JobEntry]:
    """Jobs con artefacto listo que todavía no se confirmaron subidos."""
    out: List[JobEntry] = []
    d = jobs_dir()
    if not d.is_dir():
        return out
    for path in sorted(d.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        if raw.get("state") == STATE_ARTIFACT_READY and raw.get("artifact_path"):
            out.append(JobEntry(
                task_id=str(raw.get("task_id", "")),
                state=STATE_ARTIFACT_READY,
                model_name=str(raw.get("model_name", "")),
                artifact_path=str(raw.get("artifact_path", "")),
                updated_at=float(raw.get("updated_at", 0.0)),
            ))
    return out


def discard(task_id: str) -> None:
    """Borra la entrada (tras confirmar la subida, o para limpiar un job muerto)."""
    try:
        _path_for(task_id).unlink()
    except OSError:
        pass


def stale_unfinished(max_age_hours: float = 12.0) -> List[JobEntry]:
    """Jobs claimed/running viejos: su proceso ya no existe (se reinició la
    máquina). Se descartan del journal local; el back los reconcila solo."""
    out: List[JobEntry] = []
    cutoff = time.time() - max_age_hours * 3600
    d = jobs_dir()
    if not d.is_dir():
        return out
    for path in d.glob("*.json"):
        entry = load(path.stem)
        if entry and entry.state in (STATE_CLAIMED, STATE_RUNNING) \
                and entry.updated_at < cutoff:
            out.append(entry)
    return out
