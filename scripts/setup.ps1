# Instalación del Synapta Worker en Windows — un solo clic en la terminal.
#
# Uso (PowerShell):
#   .\scripts\setup.ps1                # instala todo
#   .\scripts\setup.ps1 -Mock          # instala y verifica sin GPU ni pesos
#   .\scripts\setup.ps1 -Autostart     # además lo inicia con Windows
#
# No requiere Python previo: si no hay Python en el PATH, avisa y sale con
# el link oficial (el instalador de la app futura lo embeberá).

param(
    [switch]$Mock,
    [switch]$Autostart
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

function Say($msg)  { Write-Host "→ $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  ✔ $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "  ✘ $msg" -ForegroundColor Red }

# ── 1. Python 3.11+ ────────────────────────────────────────────
Say "Buscando Python 3.11+…"
$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) { $pyArgs = @("-3") } else {
    $py = Get-Command python -ErrorAction SilentlyContinue
    $pyArgs = @()
}
if (-not $py) {
    Fail "No encontré Python. Instalá 3.11 desde https://www.python.org/downloads/"
    Fail "¡Importante! Marcá 'Add python.exe to PATH' en el instalador."
    exit 1
}
$ver = & $py.Source @pyArgs -c "import sys; print(sys.version_info.minor)"
if ([int]$ver -lt 11) {
    Fail "Python demasiado viejo (3.$ver). Necesitás 3.11 o superior."
    exit 1
}
Ok "Python 3.$ver"

# ── 2. Entorno virtual ─────────────────────────────────────────
$venv = Join-Path $root ".venv"
$venvPy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Say "Creando entorno virtual…"
    & $py.Source @pyArgs -m venv $venv
    if ($LASTEXITCODE -ne 0) { Fail "No pude crear el venv"; exit 1 }
}
Ok "Entorno virtual listo"

# ── 3. Dependencias ────────────────────────────────────────────
Say "Instalando dependencias (puede tardar la primera vez)…"
& $venvPy -m pip install --quiet --upgrade pip
& $venvPy -m pip install --quiet -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) { Fail "Falló la instalación de dependencias"; exit 1 }
Ok "Dependencias instaladas"

# ── 4. Smoke test ──────────────────────────────────────────────
Say "Verificando la instalación…"
$env:BM_FORCE_MOCK = if ($Mock) { "1" } else { $env:BM_FORCE_MOCK }
& $venvPy (Join-Path $root "scripts\doctor.py")
if ($LASTEXITCODE -ne 0) { Fail "El doctor encontró problemas (arriba)"; exit 1 }

# ── 5. Autostart opcional ──────────────────────────────────────
if ($Autostart) {
    Say "Registrando el worker para que arranque con Windows…"
    $runPy = Join-Path $root "run.py"
    schtasks /Create /TN "SynaptaWorker" /SC ONLOGON /RL HIGHEST `
        /TR "`"$venvPy`" `"$runPy`"" /F | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Ok "Tarea 'SynaptaWorker' creada (arranca al iniciar sesión)"
    } else {
        Fail "No pude crear la tarea programada (¿permisos de admin?)"
    }
}

Write-Host ""
Ok "Instalación completa."
Write-Host ""
Write-Host "Siguiente paso (solo la primera vez): conectá la GPU con el código" -ForegroundColor Yellow
Write-Host "que muestra la app Synapta (Tu GPU → Conectar GPU):" -ForegroundColor Yellow
Write-Host ""
$pairCmd = "`"$venvPy`" `"$(Join-Path $root 'pair.py')`""
Write-Host "    $pairCmd" -ForegroundColor White
Write-Host ""
Write-Host "Y para dejarla corriendo:" -ForegroundColor Yellow
$runCmd = "`"$venvPy`" `"$(Join-Path $root 'run.py')`""
Write-Host "    $runCmd" -ForegroundColor White
