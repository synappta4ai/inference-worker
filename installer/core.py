"""Núcleo del instalador del Synapta Worker — headless y testeable.

Diseño (Fase D del plan distribuido, versión sin Inno Setup):

  1. extrae el código del worker (payload zip embebido en el .exe) a
     %LOCALAPPDATA%\\SynaptaWorker\\worker
  2. bajа Python embeddable 3.12 (~11 MB) si la máquina no trae uno ya
     gestionado por nosotros — el cliente NO necesita Python preinstalado
  3. bootstrap de pip (get-pip.py) + torch (índice CUDA si hay NVIDIA,
     CPU si no) + requirements.txt
  4. corre scripts/doctor.py (0 = operable; 1 = bloqueos → falla)
  5. empareja: `pair.py --back … --code NNNNNN --yes` (el código de 6
     dígitos lo pega el usuario desde la app; nada de tokens a mano)
  6. arranca run.py desacoplado (log en worker.log) y ofrece atajo de
     inicio con Windows

Todo con clicks o copiar-pegar: la GUI (app.py) y este CLI comparten la
misma lógica; la GUI solo encola logs y llama estas funciones.

Certificados: los .exe embebidos (PyInstaller y el Python embeddable) no
traen CA store, así que TODA invocación lleva SSL_CERT_FILE/PIP_CERT al
cacert.pem que empaqueta el instalador — pip, get-pip, el bridge (HTTPS
contra el back) y las descargas de python.org comparten la misma raíz.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger("synapta.installer")

# ── constantes de red/versión ─────────────────────────────────────
EMBED_VERSION = "3.12.10"
EMBED_URL = (
    f"https://www.python.org/ftp/python/{EMBED_VERSION}/"
    f"python-{EMBED_VERSION}-embed-amd64.zip"
)
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
TORCH_GPU_INDEX = "https://download.pytorch.org/whl/cu124"   # = Dockerfile
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# URL raíz del back (el cliente la pega solo la primera vez; la app la
# muestra en el diálogo de emparejamiento). SIN /api/v1: el bridge agrega
# los paths completos del contrato.
DEFAULT_BACK = os.environ.get("SYNAPTA_BACK", "https://back-dev.synappta.cloud")

GRPC_PORT = 50051

# ── payload ───────────────────────────────────────────────────────
EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".venv", "venv",
    "dist", "build", "outputs", "models", ".cache", ".angular",
    "installer", "node_modules", "huggingface", "checkpoints", "temp",
    "tests",
}
EXCLUDE_EXTS = {".log", ".pyc", ".pyo", ".safetensors", ".ckpt", ".pth",
                ".pt", ".bin", ".onnx", ".gguf"}
EXCLUDE_FILES = {".env", ".env.local", "worker-err.log"}


def build_payload(repo_root: Path, dest: Path) -> int:
    """Zip del código del worker (sin basura) → dest. Devuelve #archivos."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(repo_root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for fn in filenames:
                if fn in EXCLUDE_FILES or any(fn.endswith(e) for e in EXCLUDE_EXTS):
                    continue
                full = Path(dirpath) / fn
                rel = full.relative_to(repo_root).as_posix()
                zf.write(full, rel)
                count += 1
    logger.info("payload: %d archivos → %s", count, dest)
    return count


def payload_source(explicit: Optional[Path] = None) -> Path:
    """Zip del payload: --payload, el embebido en el .exe, o el del repo."""
    if explicit is not None:
        return explicit
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bundled = Path(meipass) / "_payload.zip"
        if bundled.exists():
            return bundled
    # desde el repo: lo generamos al vuelo en un tmp
    import tempfile
    tmp = Path(tempfile.gettempdir()) / "synapta-worker-payload.zip"
    build_payload(Path(__file__).resolve().parent.parent, tmp)
    return tmp


def extract_payload(zip_path: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    logger.info("código extraído en %s", target)


# ── paths locales ─────────────────────────────────────────────────
def default_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "SynaptaWorker"


def python_path(workdir: Path) -> Path:
    exe = "python.exe" if os.name == "nt" else "python"
    return workdir / "python" / exe


def worker_dir(workdir: Path) -> Path:
    return workdir / "worker"


def cacert_path() -> Optional[Path]:
    """cacert.pem: embebido en el .exe, o el de certifi si está instalado."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bundled = Path(meipass) / "_cacert.pem"
        if bundled.exists():
            return bundled
    try:
        import certifi
        return Path(certifi.where())
    except Exception:  # noqa: BLE001 — sin certifi seguimos sin TLS verificado
        return None


def net_env(extra: Optional[dict] = None) -> dict:
    """Env para todo subprocess con red: CA + utf8 + pip sin ruido."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cert = cacert_path()
    if cert is not None:
        env["SSL_CERT_FILE"] = str(cert)
        env["PIP_CERT"] = str(cert)
    if extra:
        env.update(extra)
    return env


# ── utilidades ────────────────────────────────────────────────────
def download(url: str, dest: Path, timeout: float = 60.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context(cafile=str(cacert_path() or "") or None)
    logger.info("bajando %s …", url)
    req = urllib.request.Request(url, headers={"User-Agent": "SynaptaWorker-Setup"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp, \
            open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)
    logger.info("  → %s (%d KB)", dest.name, dest.stat().st_size // 1024)
    return dest


def run_capture(cmd: list, cwd: Optional[Path] = None, env: Optional[dict] = None,
                timeout: Optional[float] = None) -> tuple:
    """Corre un comando y devuelve (rc, líneas de salida).

    La salida se loguea igual que en `run()`, pero además vuelve al caller
    para clasificar el error (el pairing necesita el motivo real, no solo
    el código de salida).
    """
    logger.info("$ %s", " ".join(str(c) for c in cmd))
    try:
        proc = subprocess.run(
            [str(c) for c in cmd], cwd=str(cwd) if cwd else None,
            env=env or net_env(), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("  (timeout de %.0f s)", timeout or 0)
        return 124, [f"se agotó el tiempo de espera ({timeout:.0f} s)"]
    lines: list = []
    for stream in (proc.stdout, proc.stderr):
        if stream:
            for line in stream.splitlines():
                if line.strip():
                    logger.info("  %s", line)
                    lines.append(line.strip())
    return proc.returncode, lines


def run(cmd: list, cwd: Optional[Path] = None, env: Optional[dict] = None,
        timeout: Optional[float] = None) -> int:
    """Corre un comando logueando su salida línea a línea."""
    rc, _ = run_capture(cmd, cwd=cwd, env=env, timeout=timeout)
    return rc


def has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def port_open(port: int = GRPC_PORT, host: str = "127.0.0.1",
              timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ── python embeddable ─────────────────────────────────────────────
def _patch_pth(pth: Path, extra: Optional[Iterable[Path]] = None) -> None:
    """Habilita `import site` y agrega rutas fijas al python embeddable.

    El embeddable NO arma sys.path con el directorio del script ni con
    PYTHONPATH (solo lo que lista el ._pth): sin la línea del directorio
    del worker, `pair.py`/`run.py` revientan con `No module named
    'bridge'`. Idempotente.
    """
    lines = [ln.strip() for ln in pth.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    # 1) import site (venía comentado)
    if "#import site" in lines:
        lines = ["import site" if ln == "#import site" else ln
                 for ln in lines]
    elif "import site" not in lines:
        lines.append("import site")
    # 2) rutas extra, antes de `import site` (convención del formato)
    site_at = lines.index("import site")
    for path in extra or ():
        resolved = str(Path(path).resolve())
        if resolved not in lines:
            lines.insert(site_at, resolved)
            site_at += 1
    pth.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("._pth actualizado: import site + worker en sys.path")


def ensure_python(workdir: Path) -> Path:
    """Python embeddable propio bajo workdir/python. Idempotente."""
    py = python_path(workdir)
    pydir = workdir / "python"
    if not py.exists():
        logger.info("preparando Python %s (embeddable) …", EMBED_VERSION)
        zip_path = download(EMBED_URL, workdir / "_python.zip")
        pydir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(pydir)
        zip_path.unlink(missing_ok=True)

    # El parche corre SIEMPRE: mantiene el ._pth sincronizado aunque el
    # .exe se ejecute de nuevo sobre una instalación existente.
    pth_files = list(pydir.glob("python*._pth"))
    if not pth_files:
        raise RuntimeError(f"python ._pth no encontrado en {pydir}")
    _patch_pth(pth_files[0], extra=[worker_dir(workdir)])

    if not py.exists():
        raise RuntimeError(f"python.exe no quedó en {py}")
    return py


def ensure_pip(py: Path, workdir: Path) -> None:
    """Bootstrap de pip dentro del embeddable (get-pip.py). Idempotente."""
    check = run([py, "-m", "pip", "--version"], cwd=workdir)
    if check == 0:
        return
    get_pip = download(GET_PIP_URL, workdir / "_get-pip.py")
    rc = run([py, get_pip, "--no-warn-script-location"], cwd=workdir,
             timeout=300)
    get_pip.unlink(missing_ok=True)
    if rc != 0:
        raise RuntimeError("no pude instalar pip (mirá el log arriba)")


# ── dependencias ──────────────────────────────────────────────────
def install_deps(py: Path, workdir: Path) -> None:
    """torch (índice según GPU) + requirements.txt. Idempotente."""
    index = TORCH_GPU_INDEX if has_nvidia() else TORCH_CPU_INDEX
    logger.info("GPU NVIDIA: %s", "sí → torch cu124" if has_nvidia()
                else "no → torch cpu")
    rc = run([py, "-m", "pip", "install", "--no-warn-script-location",
              "torch", "--index-url", index], cwd=workdir, timeout=None)
    if rc != 0:
        raise RuntimeError("falló la instalación de torch")

    req = worker_dir(workdir) / "requirements.txt"
    if not req.exists():
        raise RuntimeError(f"no encuentro {req} (payload incompleto)")
    rc = run([py, "-m", "pip", "install", "--no-warn-script-location",
              "-r", req], cwd=workdir, timeout=None)
    if rc != 0:
        raise RuntimeError("falló pip install -r requirements.txt")


def run_doctor(py: Path, workdir: Path) -> None:
    """doctor.py: 0 = operable (avisos ok), 1 = bloqueos reales → fatal."""
    script = worker_dir(workdir) / "scripts" / "doctor.py"
    rc = run([py, script], cwd=worker_dir(workdir), timeout=120)
    if rc != 0:
        raise RuntimeError(
            "el doctor encontró bloqueos (mirá el reporte arriba)")


# ── orquestación ──────────────────────────────────────────────────
def install(workdir: Optional[Path] = None,
            payload: Optional[Path] = None,
            skip_doctor: bool = False) -> Path:
    """Instalación completa. Lanza RuntimeError con mensaje user-facing."""
    workdir = (workdir or default_dir()).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    logger.info("instalando en %s", workdir)

    extract_payload(payload_source(payload), worker_dir(workdir))
    py = ensure_python(workdir)
    ensure_pip(py, workdir)
    install_deps(py, workdir)
    if not skip_doctor:
        run_doctor(py, workdir)

    status = worker_status(workdir)
    logger.info("estado: %s", json.dumps(status, ensure_ascii=False))
    return workdir


def _name_flag(name: str = "") -> str:
    if not name:
        host = platform.node() or socket.gethostname() or "GPU"
        name = f"{host} ({platform.system()})"
    return name


def _last_meaningful(lines: list) -> str:
    """Última línea útil de un subprocess (saltea prompts y vacías)."""
    for line in reversed(lines):
        text = line.lstrip("❌⚠️✅· ").strip()
        if text and not text.startswith("Conectando"):
            return text
    return ""


def classify_pair_failure(rc: int, lines: list) -> str:
    """Motivo user-facing de un pairing fallido (rc + salida de pair.py).

    Nunca devuelve el genérico "revisá el código y la conexión" a secas:
    cada causa conocida trae su pista de arreglo.
    """
    blob = "\n".join(lines)
    low = blob.lower()
    last = _last_meaningful(lines)

    if rc == 2 or "debe tener 6 dígitos" in blob or "debe empezar con http" in blob:
        return last or "parámetros inválidos (URL o código)"
    if "401" in blob or "venc" in low or "ya fue usad" in low or "usado" in low:
        return ("el código venció o ya se usó (vive 10 minutos): generá uno "
                "nuevo en la app (Tu GPU → Conectar GPU)")
    if "404" in blob:
        return ("el back no reconoce esa ruta (404): revisá que la URL sea la "
                "del back de Synapta, sin /api/v1 al final")
    if any(k in low for k in ("sin conexión", "connection refused", "rechaz",
                              "timeout", "timed out", "tiempo de espera",
                              "nodename nor servname", "getaddrinfo",
                              "name resolution", "no such host",
                              "network is unreachable", "unreachable")):
        return ("no llegué al back: revisá la URL y tu conexión a internet "
                f"({last})" if last else
                "no llegué al back: revisá la URL y tu conexión a internet")
    if any(k in low for k in ("ssl", "certificate", "certificado",
                              "tls", "handshake")):
        return ("falló la conexión segura (TLS): revisá la fecha/hora de esta "
                f"PC y si hay un proxy/antivirus interceptando HTTPS ({last})"
                if last else
                "falló la conexión segura (TLS): revisá la fecha/hora de "
                "esta PC y si hay un proxy/antivirus interceptando HTTPS")
    if last:
        return f"no pude emparejar: {last}"
    return (f"no pude emparejar (código {rc} sin detalle): probá de nuevo "
            "y si se repite usá «Diagnosticar conexión»")


def pair(workdir: Optional[Path] = None, back: str = DEFAULT_BACK,
         code: str = "", name: str = "") -> None:
    """Canjea el código de 6 dígitos por el token (pair.py --yes).

    Si falla, el error trae el motivo real que imprimió pair.py (código
    vencido, sin red, TLS, …), no el genérico de antes.
    """
    workdir = (workdir or default_dir()).expanduser().resolve()
    code = code.strip()
    if len(code) != 6 or not code.isdigit():
        raise RuntimeError(f"el código debe tener 6 dígitos (recibido: {code!r})")
    if not back.startswith(("http://", "https://")):
        raise RuntimeError(f"URL inválida: {back!r}")
    py = python_path(workdir)
    if not py.exists():
        raise RuntimeError("worker no instalado todavía")
    script = worker_dir(workdir) / "pair.py"
    rc, lines = run_capture(
        [py, script, "--back", back, "--code", code,
         "--name", _name_flag(name), "--yes"],
        cwd=worker_dir(workdir), timeout=60)
    if rc != 0:
        raise RuntimeError(classify_pair_failure(rc, lines))
    logger.info("¡GPU conectada con %s!", back)


def diagnose_back(back: str, timeout: float = 5.0) -> list:
    """Diagnóstico por pasos contra el back (para "Diagnosticar conexión").

    Devuelve [{step, ok, detail}]: URL → DNS → TCP → HTTP. Nunca lanza:
    cada paso fallido explica qué revisar. Lo usa la GUI y el CLI.
    """
    from urllib.parse import urlparse

    steps: list = []

    def add(step: str, ok: bool, detail: str = "") -> None:
        steps.append({"step": step, "ok": bool(ok), "detail": detail})

    parsed = urlparse(back.strip())
    if parsed.scheme not in ("http", "https"):
        add("URL", False, f"debe empezar con http:// o https:// ({back!r})")
        return steps
    add("URL", True, f"{parsed.scheme}://{parsed.hostname or '?'}")
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        ip = socket.gethostbyname(host)
    except OSError as exc:
        add("DNS", False, f"no resolví {host}: {exc}")
        return steps
    add("DNS", True, f"{host} → {ip}")

    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as exc:
        add("TCP", False,
            f"no llegué a {host}:{port} ({exc}): firewall, VPN o URL mal copiada")
        return steps
    add("TCP", True, f"{host}:{port} responde")

    url = back.rstrip("/") + "/healthz"
    try:
        ctx = ssl.create_default_context(cafile=str(cacert_path() or "") or None)
        req = urllib.request.Request(url, headers={"User-Agent": "SynaptaWorker-Setup"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            code = getattr(resp, "status", 200)
    except urllib.error.HTTPError as exc:
        code = exc.code
    except Exception as exc:  # noqa: BLE001 — el detalle ES el diagnóstico
        add("HTTP", False, f"{url} falló: {exc}")
        return steps
    if code is not None and code < 400:
        add("HTTP", True, f"{url} → {code}: el back está vivo")
    else:
        add("HTTP", False,
            f"{url} → {code}: responde pero no es el back de Synapta")
    return steps


# ── proceso del worker ────────────────────────────────────────────
def _pid_file(workdir: Path) -> Path:
    return workdir / "worker.pid"


def _alive(pid: int) -> bool:
    if os.name == "nt":
        cmd = ["tasklist", "/FI", f"PID eq {pid}"]
        out = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
        return str(pid) in (out.stdout or "")
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start(workdir: Optional[Path] = None, wait: float = 20.0) -> None:
    """Arranca run.py desacoplado (sobrevive al cierre de la GUI)."""
    workdir = (workdir or default_dir()).expanduser().resolve()
    if port_open():
        logger.info("worker ya escuchando en :%d", GRPC_PORT)
        return
    py = python_path(workdir)
    run_py = worker_dir(workdir) / "run.py"
    if not py.exists() or not run_py.exists():
        raise RuntimeError("worker no instalado todavía")

    log_fh = open(workdir / "worker.log", "a", encoding="utf-8",
                  errors="replace")
    env = net_env({"PYTHONUNBUFFERED": "1"})
    kwargs: dict = dict(cwd=str(worker_dir(workdir)), env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log_fh, stderr=subprocess.STDOUT)
    if os.name == "nt":
        kwargs["creationflags"] = (subprocess.DETACHED_PROCESS
                                   | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen([str(py), str(run_py)], **kwargs)
    log_fh.close()
    _pid_file(workdir).write_text(str(proc.pid), encoding="utf-8")
    logger.info("worker arrancado (pid %d), log en worker.log", proc.pid)

    deadline = time.time() + wait
    while time.time() < deadline:
        if port_open():
            logger.info("gRPC local activo en :%d ✓", GRPC_PORT)
            return
        if proc.poll() is not None:
            tail = _tail(workdir / "worker.log")
            raise RuntimeError(f"el worker murió al arrancar:\n{tail}")
        time.sleep(0.5)
    logger.warning("el puerto :%d no respondió en %.0f s (mirá worker.log)",
                   GRPC_PORT, wait)


def stop(workdir: Optional[Path] = None) -> None:
    workdir = (workdir or default_dir()).expanduser().resolve()
    pidf = _pid_file(workdir)
    if not pidf.exists():
        logger.info("sin worker registrado")
        return
    try:
        pid = int(pidf.read_text(encoding="utf-8").strip())
    except ValueError:
        pid = 0
    if pid and _alive(pid):
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True)
        else:
            os.kill(pid, 15)
        logger.info("worker (pid %d) detenido", pid)
    else:
        logger.info("worker no estaba corriendo")
    pidf.unlink(missing_ok=True)


def _tail(path: Path, lines: int = 12) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(content.splitlines()[-lines:])
    except OSError:
        return "(sin log)"


def _config_path() -> Path:
    """Dónde bridge/config.py guarda el emparejamiento (mismas reglas).

    Se lee el JSON a mano en vez de importar `bridge`: el .exe del
    instalador no empaqueta el código del worker (va dentro del payload).
    """
    if os.name == "nt":
        base = os.environ.get("APPDATA") or \
            str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "SynaptaWorker" / "config.json"
    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Application Support" /
                "SynaptaWorker" / "config.json")
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "synapta-worker" / "config.json"


def _pairing() -> tuple:
    """(paired, back_url, worker_id) — sin token por pantalla ni por log."""
    try:
        raw = json.loads(_config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "", ""
    if not isinstance(raw, dict):
        return False, "", ""
    back_url = str(raw.get("back_url", ""))
    worker_id = str(raw.get("worker_id", ""))
    token = str(raw.get("token", ""))
    return bool(back_url and worker_id and token), back_url, worker_id


def worker_status(workdir: Optional[Path] = None) -> dict:
    """Estado local: instalado / emparejado / corriendo."""
    workdir = (workdir or default_dir()).expanduser().resolve()
    installed = python_path(workdir).exists() and \
        (worker_dir(workdir) / "run.py").exists()
    paired, back_url, worker_id = _pairing()
    return {
        "installed": bool(installed),
        "paired": bool(paired),
        "back_url": back_url,
        "worker_id": worker_id,
        "running": port_open(GRPC_PORT),
    }


# ── autostart (carpeta de inicio: sin admin) ──────────────────────
def startup_link() -> Path:
    """Ruta del atajo en la carpeta de inicio del usuario (Windows)."""
    base = os.environ.get("APPDATA") or \
        str(Path.home() / "AppData" / "Roaming")
    return (Path(base) / "Microsoft" / "Windows" / "Start Menu" /
            "Programs" / "Startup" / "SynaptaWorker.lnk")


def launcher_script(workdir: Path) -> Path:
    return workdir / "start-worker.cmd"


def _write_launcher(workdir: Path) -> Path:
    """start-worker.cmd: el atajo apunta a este archivo y no a una línea de
    comandos Python, así no hay comillas anidadas que se rompan en PowerShell
    y la salida queda en autostart.log para soporte.
    """
    script = launcher_script(workdir)
    body = ("@echo off\r\n"
            f'cd /d "{worker_dir(workdir)}"\r\n'
            f'"{python_path(workdir)}" run.py >> '
            f'"{workdir / "autostart.log"}" 2>&1\r\n')
    script.write_text(body, encoding="utf-8")
    return script


def autostart_enabled() -> bool:
    """¿El atajo de inicio existe ya? (lo que debe mostrar la GUI)."""
    return os.name == "nt" and startup_link().exists()


def autostart_cmd(workdir: Path, enable: bool) -> list:
    """Comando PowerShell que crea/borra el atajo en Startup.

    Sin /RL HIGHEST ni schtasks: un .lnk en la carpeta de inicio del
    usuario no requiere privilegios de administrador.

    Todo se arma con comillas simples de PowerShell y sin comillas
    dobles anidadas (un `-c "import …"` hace fallar el parseo de
    powershell -Command); `exit 0` explícito evita falsos negativos al
    borrar un atajo que no existe.
    """
    workdir = workdir.resolve()
    link = str(startup_link())
    if not enable:
        return ["powershell", "-NoProfile", "-Command",
                f"if (Test-Path -LiteralPath '{link}') "
                f"{{ Remove-Item -LiteralPath '{link}' -Force }}; exit 0"]
    cmd = ("$s=New-Object -ComObject WScript.Shell;"
           f"$l=$s.CreateShortcut('{link}');"
           f"$l.TargetPath='{launcher_script(workdir)}';"
           f"$l.WorkingDirectory='{workdir}';"
           "$l.WindowStyle=7;$l.Save();exit 0")
    return ["powershell", "-NoProfile", "-Command", cmd]


def set_autostart(workdir: Optional[Path] = None,
                  enable: bool = True) -> None:
    workdir = (workdir or default_dir()).expanduser().resolve()
    if os.name != "nt":
        logger.info("autostart solo aplica en Windows (omitido)")
        return
    if enable:
        workdir.mkdir(parents=True, exist_ok=True)
        _write_launcher(workdir)
    rc = run(autostart_cmd(workdir, enable), timeout=60)
    if rc != 0:
        raise RuntimeError("no pude configurar el inicio automático")
    # Self-check: PowerShell puede salir en 0 igual, así que el
    # estado real manda.
    if autostart_enabled() != enable:
        raise RuntimeError("no pude configurar el inicio automático")
    logger.info("inicio con Windows: %s", "activado" if enable else "quitado")


# ── CLI ───────────────────────────────────────────────────────────
def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="synapta-worker",
        description="Instala, empareja y controla el Synapta Worker "
                    "(GUI: doble click en SynaptaWorker-Setup.exe).")
    parser.add_argument("--dir", type=Path, default=None,
                        help="directorio de instalación")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ins = sub.add_parser("install", help="instala Python + dependencias")
    p_ins.add_argument("--payload", type=Path, default=None)
    p_ins.add_argument("--skip-doctor", action="store_true")

    p_pair = sub.add_parser("pair", help="empareja con el código de la app")
    p_pair.add_argument("--back", default=DEFAULT_BACK)
    p_pair.add_argument("--code", required=True)
    p_pair.add_argument("--name", default="")

    p_diag = sub.add_parser("diagnose", help="diagnóstico por pasos contra el back")
    p_diag.add_argument("--back", default=DEFAULT_BACK)

    sub.add_parser("start", help="arranca el worker")
    sub.add_parser("stop", help="detiene el worker")
    sub.add_parser("status", help="estado en JSON")

    p_auto = sub.add_parser("autostart", help="inicio con Windows")
    p_auto.add_argument("--off", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)
    # Windows + stdout redirigido a archivo = cp1252: los "→" de los logs
    # reventarían. Forzamos UTF-8 como hace server.py.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout, force=True)
    workdir: Optional[Path] = args.dir
    try:
        if args.cmd == "install":
            install(workdir, payload=args.payload,
                    skip_doctor=args.skip_doctor)
        elif args.cmd == "pair":
            pair(workdir, back=args.back, code=args.code, name=args.name)
        elif args.cmd == "diagnose":
            failed = False
            for step in diagnose_back(args.back):
                mark = "✓" if step["ok"] else "✘"
                print(f"{mark} {step['step']}: {step['detail']}")
                failed = failed or not step["ok"]
            return 0 if not failed else 1
        elif args.cmd == "start":
            start(workdir)
        elif args.cmd == "stop":
            stop(workdir)
        elif args.cmd == "status":
            print(json.dumps(worker_status(workdir), ensure_ascii=False))
        elif args.cmd == "autostart":
            set_autostart(workdir, enable=not args.off)
        return 0
    except RuntimeError as exc:
        logger.error("✘ %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI: reporta y sale
        logger.error("✘ error inesperado: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
