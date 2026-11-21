"""Configuración de pytest para el worker de inferencia.

Los tests importan `server.py` directamente (está en el directorio padre),
por lo que añadimos ese directorio al sys.path.
"""

import os
import sys

import pytest

# Los tests corren sin descargas: fuerza el motor mock antes de importar server.
os.environ.setdefault("BM_FORCE_MOCK", "1")

WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKER_DIR not in sys.path:
    sys.path.insert(0, WORKER_DIR)


@pytest.fixture(autouse=True)
def _no_remote_metadata(monkeypatch):
    """Ningún test consulta Hugging Face por el tamaño del repo.

    `_download` pide el total remoto para que el porcentaje sea exacto; en
    tests eso sería una llamada de red lenta y no determinista. Los tests que
    la necesitan la reemplazan por su propio valor.
    """
    import model_ops

    monkeypatch.setattr(model_ops, "remote_total_bytes", lambda repo: 0)


@pytest.fixture(autouse=True)
def _isolate_output_dir(tmp_path, monkeypatch):
    """Redirige los outputs del worker a un directorio temporal por test."""
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr("server.OUTPUT_DIR", str(out))
    yield


@pytest.fixture
def servicer():
    """Instancia fresca del servicer gRPC."""
    from server import InferenceServicer

    return InferenceServicer()
