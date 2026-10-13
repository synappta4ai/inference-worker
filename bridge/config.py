"""Configuración del bridge, por usuario, cross-platform.

El archivo vive en el directorio de configuración del usuario (no en el
directorio del programa, que puede ser de solo lectura tras el instalador):

    Windows : %APPDATA%\\SynaptaWorker\\config.json
    macOS   : ~/Library/Application Support/SynaptaWorker/config.json
    Linux   : $XDG_CONFIG_HOME/synapta-worker/config.json  (~/.config/…)

Guarda la URL del back y el token de larga duración que devolvió el
emparejamiento. El token es un secreto: el archivo se crea con permisos
0600 y nunca se loguea.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

APP_DIR_NAME = "SynaptaWorker"
CONFIG_FILE_NAME = "config.json"


def config_dir() -> Path:
    """Directorio de configuración del usuario para el worker."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    # En Linux el nombre en minúscula sigue la convención XDG.
    return Path(base) / "synapta-worker"


def config_path() -> Path:
    return config_dir() / CONFIG_FILE_NAME


@dataclass
class BridgeConfig:
    """Estado persistente del emparejamiento."""

    back_url: str = ""
    worker_id: str = ""
    token: str = ""
    worker_name: str = ""
    # Campos de diagnóstico que el agente completa solo.
    extra: dict = field(default_factory=dict)

    @property
    def paired(self) -> bool:
        return bool(self.back_url and self.worker_id and self.token)

    def api_base(self) -> str:
        """URL base de la API del back (los caminos son fijos del contrato)."""
        return self.back_url.rstrip("/")


def load() -> BridgeConfig:
    """Lee la configuración; un archivo ausente o corrupto = sin emparejar."""
    path = config_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return BridgeConfig()
    if not isinstance(raw, dict):
        return BridgeConfig()
    return BridgeConfig(
        back_url=str(raw.get("back_url", "")),
        worker_id=str(raw.get("worker_id", "")),
        token=str(raw.get("token", "")),
        worker_name=str(raw.get("worker_name", "")),
        extra={k: v for k, v in raw.items()
               if k not in ("back_url", "worker_id", "token", "worker_name")},
    )


def save(cfg: BridgeConfig) -> Path:
    """Persiste la configuración con permisos de solo el usuario."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(cfg)
    payload.update(payload.pop("extra", {}) or {})
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)  # el token no debe ser legible por otros users
    except OSError:
        pass  # Windows sin ACL equivalente: el profile ya es privado
    return path


def clear() -> None:
    """Borra la configuración (desemparejar)."""
    try:
        config_path().unlink()
    except OSError:
        pass
