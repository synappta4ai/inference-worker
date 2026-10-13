# -*- coding: utf-8 -*-
"""Launcher del worker local en modo mock (sin GPU, sin descargas).

Garantiza el entorno antes de importar server.py (registry.py lee
BM_FORCE_MOCK al importar). Uso:  py -3 run_mock.py
"""
import os

os.environ.setdefault("BM_FORCE_MOCK", "1")      # todos los modelos → mock_engine
os.environ.setdefault("GRPC_PORT", "50051")
os.environ.setdefault("BM_ARTIFACT_PORT", "0")   # artifact server off (monolito local)

import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py"), run_name="__main__")
