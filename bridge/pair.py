"""Emparejamiento en 3 pasos para usuarios sin conocimiento técnicos.

El usuario abre la app Synapta → "Tu GPU" → "Conectar GPU": la app muestra
un código de 6 dígitos. Acá solo pega ese código:

    python pair.py --back https://sintapta.example.com --code 123456

o sin argumentos (modo interactivo, con preguntas en lenguaje humano):

    python pair.py

Guarda el token en el archivo de configuración del usuario y sale. Después
`run.py` (o el servicio instalado) arranca el worker ya conectado: no hay
que pegar tokens ni tocar puertos nunca más.
"""

from __future__ import annotations

import argparse
import sys

from . import BRIDGE_VERSION, config
from .agent import gpu_snapshot, machine_name
from .client import BridgeClient, BridgeError


def pair(back_url: str, code: str, name: str = ""):
    """Canjea el código por un token y lo persiste.

    Devuelve (config, ruta del archivo). Lanza BridgeError."""
    client = BridgeClient(back_url)
    snap = gpu_snapshot()
    resp = client.register(
        code=code,
        name=name or machine_name(),
        gpu_name=snap["gpu_name"],
        os_name=sys.platform,
        version=BRIDGE_VERSION,
        vram_total_mb=snap["vram_total_mb"],
        vram_free_mb=snap["vram_free_mb"],
        models=snap["models"],
    )
    worker_id = str(resp.get("worker_id", ""))
    token = str(resp.get("token", ""))
    if not worker_id or not token:
        raise BridgeError("el back no devolvió worker_id/token")
    cfg = config.BridgeConfig(
        back_url=back_url.rstrip("/"),
        worker_id=worker_id,
        token=token,
        worker_name=name or machine_name(),
    )
    path = config.save(cfg)
    return cfg, path


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Conecta esta máquina con tu cuenta de Synapta usando el "
                    "código de 6 dígitos que muestra la app.")
    parser.add_argument("--back", help="URL del back de Synapta (ej. https://…)")
    parser.add_argument("--code", help="Código de emparejamiento de 6 dígitos")
    parser.add_argument("--name", help="Nombre de esta GPU en la app "
                                       "(default: hostname + SO)")
    parser.add_argument("--yes", action="store_true",
                        help="No preguntar al re-emparejar (para el "
                             "instalador y scripts no interactivos)")
    args = parser.parse_args(argv)

    existing = config.load()
    if existing.paired and not args.yes:
        print("⚠️  Esta máquina ya está emparejada con "
              f"{existing.back_url} (worker {existing.worker_id[:8]}…).")
        if input("¿Re-emparejar y reemplazar? (s/N): ").strip().lower() not in ("s", "si", "sí", "y"):
            print("Cancelado. Nada cambió.")
            return 0

    back = args.back or _prompt("URL de Synapta (ej. https://sintapta.example.com)",
                                existing.back_url)
    if not back.startswith(("http://", "https://")):
        print(f"❌ La URL debe empezar con http:// o https:// (recibido: {back!r})")
        return 2

    code = args.code or _prompt("Código de 6 dígitos que muestra la app")
    if len(code) != 6 or not code.isdigit():
        print(f"❌ El código debe tener 6 dígitos (recibido: {code!r})")
        return 2

    if args.name:
        name = args.name
    elif args.yes or not sys.stdin.isatty():
        # Sin terminal (instalador/GUI): no hay a quién preguntar.
        name = existing.worker_name or machine_name()
    else:
        name = _prompt("Nombre de esta GPU (opcional)",
                       existing.worker_name or machine_name())

    print(f"\nConectando con {back}…")
    try:
        cfg, path = pair(back, code, name)
    except BridgeError as exc:
        print(f"❌ No pude emparejar: {exc}")
        if "401" in str(exc):
            print("   · El código venció (vive 10 minutos): generá uno nuevo "
                  "en la app.")
        elif "sin conexión" in str(exc):
            print("   · Revisá tu conexión a internet y la URL del back.")
        return 1

    print("\n✅ ¡GPU conectada!")
    print(f"   · ID de worker : {cfg.worker_id}")
    print(f"   · Nombre       : {cfg.worker_name}")
    print(f"   · Config       : {path}")
    print("\nAhora mantené corriendo `python run.py` (o el servicio que lo "
          "inicia con el sistema) y la app verá tu GPU en línea.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
