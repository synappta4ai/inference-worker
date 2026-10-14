"""Gestión de pesos de modelos: descarga, borrado y estado.

El panel "Tu GPU" administra desde acá la caché de Hugging Face
(HF_HUB_CACHE), la misma que usan `snapshot_download` y `from_pretrained`:

  • start_download(key) → descarga en un hilo daemon, con progreso y
    cancelación (la barra de progreso de huggingface aborta la descarga).
  • cancel(key)         → pide abortar y espera hasta 2 s a que el hilo pare.
  • delete(key)         → borra la caché del repo y libera el disco.
  • snapshot(key)       → estado por clave: idle | downloading | ready |
    failed. `deleted` sólo aparece como respuesta inmediata de un borrado.

El estado "completo" se valida contra el árbol del repo que guarda
huggingface en `trees/*.json`: sólo se exigen los ficheros que el motor
necesita (pesos, configs, código). Un README o una imagen que falte NO deja
un modelo "indisponible"; un peso a medio bajar (`.incomplete`) sí.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
from dataclasses import dataclass, field

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from engines.base import MODEL_CATALOG, ModelSpec

logger = logging.getLogger("BrainMasterInferenceWorker")

# Estados compartidos con el proto (ModelOperationResponse.state).
IDLE = "idle"
DOWNLOADING = "downloading"
READY = "ready"
FAILED = "failed"
DELETED = "deleted"

# Espera máxima de `cancel()` a que el hilo de descarga confirme la parada.
_CANCEL_GRACE_S = 2.0

# Hilos concurrentes de `snapshot_download` (1 hilo = 1 fichero a la vez).
_DOWNLOAD_WORKERS = max(1, int(os.environ.get("BM_DOWNLOAD_WORKERS", "4")))

# Extensiones que nunca hacen falta para ejecutar un modelo (documentación y
# media del repo). Los .txt quedan fuera: vocab.txt/tokenizer.txt sí importan.
_DOC_EXTS = {".md", ".markdown", ".rst", ".png", ".jpg", ".jpeg", ".gif",
             ".webp", ".svg", ".bmp"}
_DOC_NAMES = {
    ".gitattributes", ".gitignore", "license", "license.md", "license.txt",
    "license_mit.md", "license_apache2.txt", "notice", "notice.txt",
    "code_of_conduct.md", "contributing.md", "authors.md",
}


class DownloadCancelled(BaseException):
    """Abort de descarga pedido por el usuario.

    Deriva de `BaseException` a propósito: debe atravesar los
    `except Exception` internos de huggingface_hub (que reintentan) y llegar
    entero hasta `snapshot_download`.
    """


@dataclass
class ModelState:
    """Estado de un modelo (paralelo a ModelInfo / ModelOperationResponse)."""

    key: str
    state: str = IDLE
    progress: float = 0.0          # 0..100
    bytes_downloaded: int = 0      # bytes de pesos ya en disco
    total_bytes: int = 0           # bytes totales (0 si se desconocen)
    error: str = ""                # último error de descarga/borrado
    downloaded: bool = False       # pesos completos en disco
    downloading: bool = False       # descarga en curso ahora mismo
    size_bytes: int = 0            # bytes ocupados en disco


@dataclass
class RepoStats:
    """Foto de la caché local de un repo."""

    on_disk: int = 0               # bytes ocupados en disco (directorio entero)
    downloaded: int = 0            # bytes de pesos requeridos ya en disco
    total: int = 0                 # bytes de pesos requeridos
    complete: bool = False
    has_files: bool = False        # hay algún snapshot en disco
    partial: bool = False          # hay ficheros a medio bajar


@dataclass
class _Op:
    """Registro interno de la última operación sobre una clave."""

    state: str = IDLE
    error: str = ""
    total_bytes: int = 0
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


def _is_doc(name: str) -> bool:
    """true si el fichero del repo es documentación (no requerido para correr)."""
    base = name.rsplit("/", 1)[-1].lower()
    if base in _DOC_NAMES:
        return True
    return os.path.splitext(base)[1] in _DOC_EXTS


def repo_cache_dir(repo: str | None) -> str | None:
    """Directorio de caché HF del repo (`models--org--name`), si tiene repo."""
    if not repo:
        return None
    return os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"))


def _read_tree(d: str) -> dict | None:
    """Árbol del repo guardado por huggingface (`trees/<commit>.json`)."""
    tree_dir = os.path.join(d, "trees")
    if not os.path.isdir(tree_dir):
        return None
    for name in sorted(os.listdir(tree_dir)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(tree_dir, name), encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        files = data.get("files") if isinstance(data, dict) else None
        if isinstance(files, dict):
            return files
    return None


def inspect_repo(repo: str | None) -> RepoStats:
    """Lee la caché local del repo: tamaño, progreso y completitud."""
    d = repo_cache_dir(repo)
    stats = RepoStats()
    if d is None or not os.path.isdir(d):
        return stats

    tree = _read_tree(d)
    required: dict[str, dict] = {}
    if tree:
        required = {
            name: meta for name, meta in tree.items()
            if isinstance(meta, dict) and not _is_doc(name)
        }

    # Snapshot: rutas relativas presentes (unión de todas las revisiones).
    present: dict[str, int] = {}
    snap_dir = os.path.join(d, "snapshots")
    if os.path.isdir(snap_dir):
        for rev in os.listdir(snap_dir):
            rev_dir = os.path.join(snap_dir, rev)
            if not os.path.isdir(rev_dir):
                continue
            for root, _dirs, files in os.walk(rev_dir):
                for f in files:
                    path = os.path.join(root, f)
                    rel = os.path.relpath(path, rev_dir).replace(os.sep, "/")
                    try:
                        present[rel] = os.path.getsize(path)
                    except OSError:
                        continue
    stats.has_files = bool(present)
    stats.on_disk = _dir_size(d)

    # Descargas a medio terminar: blobs/<blob_id>.<x>.incomplete
    partial: dict[str, int] = {}
    blobs_dir = os.path.join(d, "blobs")
    if os.path.isdir(blobs_dir):
        for f in os.listdir(blobs_dir):
            if not f.endswith(".incomplete"):
                continue
            try:
                size = os.path.getsize(os.path.join(blobs_dir, f))
            except OSError:
                continue
            partial[f.split(".", 1)[0]] = size
            stats.partial = True

    if required:
        stats.total = sum(int(m.get("size") or 0) for m in required.values())
        missing = False
        for name, meta in required.items():
            if name in present:
                stats.downloaded += present[name]
                continue
            # ¿Está bajándose todavía (blob a medio terminar)?
            blob = str(meta.get("blob_id") or "")
            if blob in partial:
                stats.downloaded += partial[blob]
            else:
                missing = True
        stats.complete = stats.has_files and not missing and not stats.partial
    else:
        # Sin árbol (caché de versiones viejas): heurística — hay snapshot,
        # no hay .incomplete y hay algún peso. El total se desconoce.
        stats.downloaded = stats.on_disk
        stats.complete = (
            stats.has_files
            and not stats.partial
            and any(
                os.path.splitext(name)[1].lower() in
                {".safetensors", ".bin", ".pt", ".ckpt", ".gguf", ".onnx", ".npz"}
                for name in present
            )
        )
    return stats


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
    return total


def _bars_disabled() -> bool:
    """respeta HF_HUB_DISABLE_PROGRESS_BARS y la ausencia de TTY."""
    if os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS", "").lower() in (
        "1", "true", "yes",
    ):
        return True
    try:
        return not sys.stderr.isatty()
    except Exception:  # stderr sin isatty: mejor no imprimir barras
        return True


def progress_tqdm(cancel: threading.Event):
    """Clase tqdm que aborta la descarga en cuanto se pide cancelar.

    huggingface la recibe como `tqdm_class` de snapshot_download: cada update
    de bytes (y cada refresco/apertura de barra) revisa el evento. Al no
    venir del paquete huggingface_hub, `_create_progress_bar` no le inyecta
    `disable`/`name`: la clase decide sus propios defaults.
    """
    from tqdm import tqdm  # import tardío: sólo hace falta al descargar

    class _CancelTqdm(tqdm):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("disable", _bars_disabled())
            kwargs.pop("name", None)  # por si huggingface cambia el contrato
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            if cancel.is_set():
                raise DownloadCancelled()
            return super().update(n)

        def refresh(self, *args, **kwargs):
            if cancel.is_set():
                raise DownloadCancelled()
            return super().refresh(*args, **kwargs)

        def set_postfix_str(self, *args, **kwargs):
            if cancel.is_set():
                raise DownloadCancelled()
            return super().set_postfix_str(*args, **kwargs)

    return _CancelTqdm


class ModelOps:
    """Descarga/borrado de pesos por clave, en hilos aparte del gRPC."""

    def __init__(self) -> None:
        # RLock: los métodos internos se anidan (p.ej. snapshot llama a _op).
        self._lock = threading.RLock()
        self._ops: dict[str, _Op] = {}

    # ── catálogo ───────────────────────────────────────────────────────
    @staticmethod
    def spec(key: str) -> ModelSpec:
        """Spec del modelo; KeyError si la clave no existe en el catálogo."""
        spec = MODEL_CATALOG.get(key)
        if spec is None:
            raise KeyError(key)
        return spec

    @staticmethod
    def keys() -> list[str]:
        return list(MODEL_CATALOG)

    # ── lectura ────────────────────────────────────────────────────────
    def snapshot(self, key: str) -> ModelState:
        """Estado actual de una clave (lectura barata de disco + memoria)."""
        spec = self.spec(key)
        stats = inspect_repo(spec.repo)

        with self._lock:
            op = self._op(key)
            state, error, total = op.state, op.error, op.total_bytes
            running = self._running(op)

        model = ModelState(key=key)
        model.downloaded = stats.complete
        model.downloading = running
        model.bytes_downloaded = stats.downloaded
        model.total_bytes = total or stats.total
        model.size_bytes = stats.on_disk

        if running:
            model.state = DOWNLOADING
        elif state == FAILED and error:
            model.state = FAILED
            model.error = error
        elif stats.complete:
            model.state = READY
        else:
            model.state = IDLE

        if model.state == READY:
            model.progress = 100.0
        elif model.total_bytes > 0:
            model.progress = min(
                99.9, 100.0 * stats.downloaded / model.total_bytes
            )
        return model

    def states(self, key: str = "") -> list[ModelState]:
        """Estado de una clave o de todo el catálogo (clave vacía)."""
        keys = [key] if key else self.keys()
        return [self.snapshot(k) for k in keys]

    def disk_total(self) -> int:
        """Bytes de pesos en disco de todo el catálogo (GetModelState)."""
        total = 0
        for key in self.keys():
            total += inspect_repo(self.spec(key).repo).on_disk
        return total

    # ── descarga ───────────────────────────────────────────────────────
    def start_download(self, key: str) -> ModelState:
        """Lanza la descarga de los pesos en un hilo daemon."""
        spec = self.spec(key)
        if not spec.repo:
            raise ValueError("este modelo no tiene pesos descargables (motor mock)")
        if spec.engine == "v1_bridge":
            raise ValueError(
                "este modelo corre por el bridge V1: sus pesos se sirven de un "
                "volumen montado, no se descargan de Hugging Face"
            )

        with self._lock:
            op = self._op(key, create=True)
            if self._running(op):
                raise ValueError(
                    "ya hay una descarga en curso para este modelo"
                )
            op.state = DOWNLOADING
            op.error = ""
            op.cancel = threading.Event()
            thread = threading.Thread(
                target=self._download,
                args=(key, spec, op),
                name=f"model-download-{key}",
                daemon=True,
            )
            op.thread = thread

        thread.start()
        logger.info("[model_ops] descarga iniciada: %s (%s)", key, spec.repo)
        return self.snapshot(key)

    def cancel(self, key: str) -> ModelState:
        """Aborta la descarga en curso y espera (máx. 2 s) a que pare."""
        self.spec(key)
        with self._lock:
            op = self._op(key, create=True)
            running = self._running(op)
            if running:
                op.cancel.set()
            thread = op.thread

        if not running:
            state = self.snapshot(key)
            state.error = "no hay ninguna descarga en curso para este modelo"
            return state

        # La parada se dispara en el siguiente update de la barra: casi
        # siempre es instantánea; si tarda, devolvemos el estado actual.
        if thread is not None:
            thread.join(_CANCEL_GRACE_S)
        return self.snapshot(key)

    def _download(self, key: str, spec: ModelSpec, op: _Op) -> None:
        def finish(state: str, error: str = "") -> None:
            with self._lock:
                op.state, op.error = state, error

        # El total sale del árbol local (huggingface lo escribe al arrancar
        # la descarga); si todavía no existe, total=0 y el progreso sólo se
        # conoce al terminar.
        with self._lock:
            op.total_bytes = inspect_repo(spec.repo).total

        try:
            snapshot_download(
                spec.repo,
                tqdm_class=progress_tqdm(op.cancel),
                max_workers=_DOWNLOAD_WORKERS,
            )
        except DownloadCancelled:
            finish(IDLE)
            logger.info("[model_ops] descarga cancelada: %s", key)
            return
        except Exception as exc:  # un fallo de red no debe tumbar el worker
            finish(FAILED, f"{type(exc).__name__}: {exc}")
            logger.warning("[model_ops] %s no se pudo descargar: %s", key, exc)
            return

        finish(IDLE)
        stats = inspect_repo(spec.repo)
        logger.info("[model_ops] %s descargado (%d bytes)", key, stats.on_disk)

    # ── borrado ────────────────────────────────────────────────────────
    def delete(self, key: str) -> ModelState:
        """Borra la caché del repo y libera el disco."""
        spec = self.spec(key)
        if not spec.repo:
            raise ValueError("este modelo no tiene pesos en disco que borrar")

        with self._lock:
            op = self._op(key, create=True)
            if self._running(op):
                raise ValueError(
                    "cancelá la descarga antes de borrar el modelo"
                )

        path = repo_cache_dir(spec.repo)
        freed = inspect_repo(spec.repo).on_disk
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        with self._lock:
            op.state, op.error = IDLE, ""
            op.total_bytes = 0

        state = self.snapshot(key)
        state.state = DELETED
        state.bytes_downloaded = 0
        logger.info("[model_ops] %s borrado (%d bytes liberados)", key, freed)
        return state

    # ── interno ───────────────────────────────────────────────────────
    def _op(self, key: str, create: bool = False) -> _Op:
        """Registro interno de la clave (siempre bajo self._lock)."""
        op = self._ops.get(key)
        if op is None:
            op = _Op()
            if create:
                self._ops[key] = op
        return op

    @staticmethod
    def _running(op: _Op) -> bool:
        """true si hay una descarga viva para este registro."""
        return bool(
            op.state == DOWNLOADING
            and op.thread is not None
            and op.thread.is_alive()
        )
