#!/usr/bin/env bash
# Instalación del Synapta Worker en Linux y macOS.
#
# Uso:
#   ./scripts/setup.sh              # instala todo
#   BM_FORCE_MOCK=1 ./scripts/setup.sh   # verifica sin GPU ni pesos
#   ./scripts/setup.sh --autostart  # además lo inicia con la sesión (systemd)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
VENV_PY="$VENV/bin/python"
AUTOSTART=0
[ "${1:-}" = "--autostart" ] && AUTOSTART=1

say()  { printf '\033[36m→ %s\033[0m\n' "$1"; }
ok()   { printf '  \033[32m✔ %s\033[0m\n' "$1"; }
fail() { printf '  \033[31m✘ %s\033[0m\n' "$1"; }

# ── 1. Python 3.11+ ────────────────────────────────────────────
say "Buscando Python 3.11+…"
PY=""
for cand in python3.12 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        minor="$("$cand" -c 'import sys; print(sys.version_info.minor)')"
        if [ "$minor" -ge 11 ]; then PY="$cand"; break; fi
    fi
done
if [ -z "$PY" ]; then
    fail "No encontré Python 3.11+."
    fail "Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    fail "macOS        : brew install python@3.12"
    exit 1
fi
ok "Python ($( "$PY" --version ))"

# ── 2. Entorno virtual ─────────────────────────────────────────
if [ ! -x "$VENV_PY" ]; then
    say "Creando entorno virtual…"
    "$PY" -m venv "$VENV"
fi
ok "Entorno virtual listo"

# ── 3. Dependencias ────────────────────────────────────────────
say "Instalando dependencias (puede tardar la primera vez)…"
"$VENV_PY" -m pip install --quiet --upgrade pip
"$VENV_PY" -m pip install --quiet -r "$ROOT/requirements.txt"
ok "Dependencias instaladas"

# ── 4. Smoke test ──────────────────────────────────────────────
say "Verificando la instalación…"
"$VENV_PY" "$ROOT/scripts/doctor.py"

# ── 5. Autostart opcional (systemd, user scope) ────────────────
if [ "$AUTOSTART" = "1" ] && command -v systemctl >/dev/null 2>&1; then
    say "Registrando el servicio de usuario synapta-worker…"
    UNIT_DIR="$HOME/.config/systemd/user"
    mkdir -p "$UNIT_DIR"
    cat > "$UNIT_DIR/synapta-worker.service" <<EOF
[Unit]
Description=Synapta Worker (GPU provider)
After=network-online.target

[Service]
ExecStart=$VENV_PY $ROOT/run.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now synapta-worker.service
    ok "Servicio activo (logs: journalctl --user -u synapta-worker -f)"
elif [ "$AUTOSTART" = "1" ]; then
    fail "systemd no disponible: agregá '$VENV_PY $ROOT/run.py' al inicio de tu sesión."
fi

echo
ok "Instalación completa."
echo
echo "Siguiente paso (solo la primera vez): conectá la GPU con el código"
echo "que muestra la app Synapta (Tu GPU → Conectar GPU):"
echo
echo "    $VENV_PY $ROOT/pair.py"
echo
echo "Y para dejarla corriendo:"
echo "    $VENV_PY $ROOT/run.py"
