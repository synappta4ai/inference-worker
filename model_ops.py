"""Gestión de pesos de modelos: descarga, borrado y estado.

El panel "Tu GPU" administra desde acá la caché de Hugging Face
(HF_HUB_CACHE), la misma que usan `snapshot_download` y `from_pretrained`:

  • start_download(key) → descarga en un hilo daemon, con progreso y
    cancelación: la barra de progreso de huggingface aborta los caminos HTTP
    y `cancel()` aborta además las transferencias Xet en vuelo (los bytes de
    esas los mueve Rust, así que la barra nunca ve la señal de parada).
  • cancel(key)         → pide abortar y espera hasta 2 s a que el hilo pare;
    descarta los bytes a medio bajar de ese intento.

El tamaño y el porcentaje que ve el panel salen de lo que el modelo tiene
COMPLETO en disco (pesos utilizables) más, sólo mientras hay una descarga
viva, los bytes a medio bajar de ESE intento: los restos de intentos que ya
no corren se excluyen y se borran (`purge_partial_downloads`): un modelo
cancelado no suma a su tamaño ni a su porcentaje los bytes que quedaron a
medio bajar.
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
# Corta a propósito: el back espera esta RPC con timeout de 6 s.
_CANCEL_GRACE_S = 2.0

# Hilos concurrentes de `snapshot_download` (1 hilo = 1 fichero a la vez).
_DOWNLOAD_WORKERS = max(1, int(os.environ.get("BM_DOWNLOAD_WORKERS", "4")))

# Modo de descarga de pesos (BM_PROVIDER_MODE):
#   - "" / "personal" (default): NADA se descarga solo. Los pesos bajan
#     únicamente cuando el admin pulsa Descargar en el panel /gpu (RPC
#     DownloadModel). Generar con un modelo sin pesos falla con un mensaje
#     claro en vez de saturar el PC con una descarga implícita de GB.
#   - "shared": el worker es un proveedor compartido en la nube: al
#     arrancar pre-descarga todo el catálogo y las generaciones pueden
#     descargar implícitamente lo que falte.
PROVIDER_MODE_PERSONAL = "personal"
PROVIDER_MODE_SHARED = "shared"


def provider_mode() -> str:
    """Modo de descarga actual (se lee en cada llamada: testeable)."""
    mode = os.environ.get("BM_PROVIDER_MODE", "").strip().lower()
    return mode if mode == PROVIDER_MODE_SHARED else PROVIDER_MODE_PERSONAL


def provider_shared() -> bool:
    """true si el worker corre como proveedor compartido en la nube."""
    return provider_mode() == PROVIDER_MODE_SHARED


def key_for_repo(repo: str | None) -> str | None:
    """Clave del catálogo para un repo (búsqueda inversa, o None)."""
    if not repo:
        return None
    for key, spec in MODEL_CATALOG.items():
        if spec.repo == repo:
            return key
    return None


def weights_complete(repo: str | None) -> bool:
    """true si los pesos del repo están completos en disco."""
    if not repo:
        return True  # mock / bridge: no hay pesos que esperar
    return inspect_repo(repo).complete


def require_weights(repo: str | None) -> None:
    """Guardián anti-descargas-implícitas para los motores.

    Llamar ANTES de `from_pretrained`: en modo personal, si los pesos no
    están completos, falla rápido con un mensaje accionable en vez de
    bajar GB en silencio durante una generación. En modo compartido no
    hace nada (ahí sí tiene sentido descargar al vuelo).
    """
    if provider_shared() or weights_complete(repo):
        return
    key = key_for_repo(repo)
    where = f"el modelo '{key}'" if key else f"los pesos de '{repo}'"
    raise RuntimeError(
        f"{where} no están descargados. Descargalos primero desde el "
        "panel Tu GPU (/gpu → Modelos del worker → Descargar) y reintentá "
        "la generación."
    )


def predownload_all(ops: "ModelOps") -> list[str]:
    """Lanza la descarga de todo el catálogo con repo (modo compartido).

    Devuelve las claves lanzadas; las que ya están en curso o completas se
    omiten en silencio. Lo llama serve() al arrancar con BM_PROVIDER_MODE.
    """
    launched: list[str] = []
    for key in ops.keys():
        try:
            spec = ops.spec(key)
        except KeyError:
            continue
        if not spec.repo or spec.engine == "v1_bridge":
            continue
        if inspect_repo(spec.repo).complete:
            continue
        try:
            ops.start_download(key)
        except ValueError:
            continue  # ya en curso: el hilo existente la termina
        launched.append(key)
    if launched:
        logger.info("[model_ops] pre-descarga (modo compartido): %s",
                    ", ".join(launched))
    return launched

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
    committed: int = 0             # pesos requeridos COMPLETOS en disco
    partial_bytes: int = 0         # de esos pesos, bytes a medio bajar
    partial_disk_bytes: int = 0    # TODOS los bytes a medio bajar del repo
    downloaded: int = 0            # committed + partial_bytes (intento vivo)
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
            stats.partial_disk_bytes += size
            stats.partial = True

    if required:
        stats.total = sum(int(m.get("size") or 0) for m in required.values())
        missing = False
        for name, meta in required.items():
            if name in present:
                stats.committed += present[name]
                continue
            # ¿Está bajándose todavía (blob a medio terminar)?
            blob = str(meta.get("blob_id") or "")
            if blob in partial:
                stats.partial_bytes += partial[blob]
            else:
                missing = True
        stats.downloaded = stats.committed + stats.partial_bytes
        stats.complete = stats.has_files and not missing and not stats.partial
    else:
        # Sin árbol (caché de versiones viejas): heurística — hay snapshot,
        # no hay .incomplete y hay algún peso. El total se desconoce. Los bytes
        # a medio bajar quedan fuera del "descargado": son restos.
        stats.committed = max(0, stats.on_disk - stats.partial_disk_bytes)
        stats.partial_bytes = stats.partial_disk_bytes
        stats.downloaded = stats.committed + stats.partial_bytes
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


# Contador de abortos de transferencias: `cancel()` corta las transferencias
# Xet de TODO el proceso (es el mismo mecanismo que usa Ctrl-C; hf_xet no
# ofrece aborto por descarga), así que un intento que muere justo después de
# un aborto ajeno sabe —comparando este contador— que no falló él.
_abort_lock = threading.Lock()
_abort_epoch = 0


def _abort_inflight_transfers() -> bool:
    """Corta las transferencias Xet en vuelo (lo que hace un Ctrl-C).

    La cancelación por la barra de tqdm sólo llega donde hay updates de bytes;
    en el camino Xet los mueve Rust y el hilo se queda bloqueado en
    `wait_to_finish`, así que sin esto la descarga seguía bajando después de
    apretar Cancelar. Devuelve true si pudo pedir el aborto.
    """
    global _abort_epoch
    try:
        # huggingface_hub lo expone para su propio `except KeyboardInterrupt`:
        # vive en un módulo privado porque es del motor Xet, no de la API.
        from huggingface_hub.utils._xet import abort_xet_session  # type: ignore
    except Exception:  # noqa: BLE001 — sin hf_xet (o módulo movido) no hay nada
        return False
    with _abort_lock:
        _abort_epoch += 1
    try:
        abort_xet_session()
    except Exception as exc:  # noqa: BLE001 — sin sesión viva no hay nada que abortar
        logger.debug("[model_ops] no pude abortar la sesión Xet: %s", exc)
        return False
    return True


def purge_partial_downloads(repo: str | None) -> int:
    """Borra los `.incomplete` que dejó un intento que ya no corre (bytes).

    huggingface_hub 1.33 baja cada archivo a un temporal único
    (`<etag>.<uuid>.incomplete`) que borra él mismo al fallar, así que lo que
    queda acá es de intentos que murieron sin poder limpiar (corte de luz, kill
    del proceso). Como el nombre es único por intento NO se reanudan: sólo
    ocupan disco y hacían que el panel contara tamaño y porcentaje de restos
    viejos. Los archivos completos (blobs + snapshots) no se tocan.
    """
    d = repo_cache_dir(repo)
    if d is None:
        return 0
    blobs = os.path.join(d, "blobs")
    if not os.path.isdir(blobs):
        return 0
    freed = 0
    for name in os.listdir(blobs):
        if not name.endswith(".incomplete"):
            continue
        path = os.path.join(blobs, name)
        try:
            freed += os.path.getsize(path)
            os.remove(path)
        except OSError:
            # En uso por otro proceso (Windows no deja borrar un archivo
            # abierto) o ya se fue: es best-effort, no frenamos la descarga.
            continue
    if freed:
        logger.info("[model_ops] %s: %d bytes de descargas sin terminar "
                    "descartados", repo, freed)
    return freed


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
        # Tamaño y porcentaje cuentan lo que este modelo tiene de verdad: los
        # pesos completos y, sólo mientras la descarga está viva, los bytes a
        # medio bajar de ESE intento. Los restos de intentos que ya no corren
        # (o que murieron) se excluyen y se borran al parar.
        live = stats.downloaded if running else stats.committed
        model.bytes_downloaded = live
        model.total_bytes = total or stats.total
        model.size_bytes = live

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
                99.9, 100.0 * model.bytes_downloaded / model.total_bytes
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
            # Arrancar limpio: los restos de un intento muerto no se reanudan y
            # sólo inflarían el tamaño/porcentaje que reporta el panel.
            purge_partial_downloads(spec.repo)
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

        # Doble vía: la barra de progreso aborta los caminos que pasan por ella
        # (HTTP) y el aborto de la sesión Xet corta lo que mueve Rust.
        aborted = _abort_inflight_transfers()
        if thread is not None:
            thread.join(_CANCEL_GRACE_S)
        with self._lock:
            still_running = self._running(op)
        if still_running:
            logger.warning("[model_ops] %s sigue transfiriendo "
                           "%s después de cancelar; el panel lo muestra "
                           "hasta que pare", key,
                           "tras el aborto de Xet" if aborted else "(sin Xet)")
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
            epoch = _abort_epoch

        try:
            snapshot_download(
                spec.repo,
                tqdm_class=progress_tqdm(op.cancel),
                max_workers=_DOWNLOAD_WORKERS,
            )
        except DownloadCancelled:
            self._stopped(key, spec, op, epoch, cancelled=True)
            return
        except BaseException as exc:  # noqa: BLE001 — ver _stopped()
            self._stopped(key, spec, op, epoch, exc=exc)
            return

        finish(IDLE)
        # Un intento muerto pudo dejar `.incomplete` atrás: si el modelo quedó
        # descargado, esos restos no tienen por qué ocupar disco ni contarse.
        purge_partial_downloads(spec.repo)
        stats = inspect_repo(spec.repo)
        logger.info("[model_ops] %s descargado (%d bytes)", key, stats.on_disk)

    def _stopped(self, key: str, spec: ModelSpec, op: _Op, epoch: int,
                 exc: BaseException | None = None,
                 cancelled: bool = False) -> None:
        """Cierra un intento que terminó antes de tiempo y limpia sus restos.

        Atrapa `BaseException` (y no `Exception`) porque el aborto de las
        transferencias Xet —lo que dispara `cancel()`— no viaja como error
        normal: puede llegar como `KeyboardInterrupt`, el mismo camino que usa
        Ctrl-C en huggingface_hub.
        """
        with self._lock:
            cancelled = cancelled or op.cancel.is_set()
            # `cancel()` aborta las transferencias Xet de TODO el proceso: un
            # intento ajeno que muere por ese aborto no falló, así que vuelve a
            # "sin descargar" (reintentable) en vez de quedar en error.
            interrupted = not cancelled and _abort_epoch != epoch
            op.state = IDLE if (cancelled or interrupted) else FAILED
            op.error = "" if (cancelled or interrupted) else (
                f"{type(exc).__name__}: {exc}")

        freed = purge_partial_downloads(spec.repo)
        if cancelled:
            logger.info("[model_ops] descarga cancelada: %s (%d bytes a medio "
                        "bajar descartados)", key, freed)
        elif interrupted:
            logger.warning("[model_ops] %s interrumpido por la cancelación de "
                           "otro modelo (%s); queda listo para reintentar",
                           key, exc)
        else:
            # Con traza: el estado sólo guarda type+message, y un mensaje
            # suelto ("[Errno 13] Permission denied") no dice si el fallo
            # fue al escribir la caché, al mover el archivo o al bajarlo.
            logger.warning("[model_ops] %s no se pudo descargar: %s", key, exc,
                           exc_info=True)

    # ── borrado ────────────────────────────────────────────────────────
    def delete_all(self) -> tuple[list[str], dict[str, str], int]:
        """Borra los pesos de TODO el catálogo (opción "desinstalar todo").

        Devuelve (borradas, fallidas, bytes_liberados). Las claves sin repo
        (mock) o con descarga en curso no se tocan: van a `fallidas` con el
        motivo, sin abortar el resto.
        """
        deleted: list[str] = []
        failed: dict[str, str] = {}
        freed = 0
        for key in self.keys():
            try:
                spec = self.spec(key)
            except KeyError:
                continue
            if not spec.repo:
                continue  # mock: no ocupa disco, nada que borrar
            on_disk = inspect_repo(spec.repo).on_disk
            try:
                self.delete(key)
            except ValueError as exc:
                failed[key] = str(exc)
                continue
            freed += on_disk
            deleted.append(key)
        if deleted:
            logger.info("[model_ops] desinstalado todo: %d modelos (%d bytes)",
                        len(deleted), freed)
        return deleted, failed, freed

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
