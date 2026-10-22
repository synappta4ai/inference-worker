# AGENTS.md — inference-worker

Worker de inferencia (gRPC + artifact server). Python (PyTorch/Diffusers) + gRPC. Repo propio (`main`).

## Reglas del repo

- Cada subcarpeta puede tener su propio `AGENTS.md`; leé el de la carpeta en la que vas a trabajar.
- Repos separados: `front-synapta`, `back-synapta` e `inference-worker` tienen su propio git; los commits se hacen por repo.
- Nunca commitear: `.env`, `.env.e2e`, logs (`*.log`), IPs públicas o secretos.

## Commits

- Convención `feat(scope): ...` en inglés, cuerpo con bullets, footer Codebuff. Un commit por feature.
- **Al commitear y hacer push, actualizá la versión antes de push** si el repo tiene versionamiento (`package.json`, `Cargo.toml`, `pyproject.toml`, `__version__`, etc.). Subí la versión en un commit `chore` dedicado, separado del commit de código.

## Verificación antes de cerrar

- `python3 -m py_compile` en los `.py` que tocaste.
- `go build ./...` + `go vet ./...` solo si tocás código en `back-synapta` (otro repo).

## Cómo se conecta

- `BM_WORKER_ADDR` (`host:port`, o `tls://host:port`) para que el back se conecte al gRPC `:50051`.
- `BM_WORKER_ARTIFACT_BASE` para re-hospedar artefactos del worker remoto (artifact server `:50052`).
- `BM_ARTIFACT_TOKEN` para proteger los endpoints HTTP del artifact server (modelos, artefactos, auto-update).
- Contrato gRPC: `proto/inference.proto` es la fuente; el back espeja el código generado en `back-synapta/internal/worker/proto/`. Si cambiás el contrato, regenerá ambos lados.

## Estado

- gRPC `:50051`, artifact server `:50052`.
- Versión reportada al back: `BRIDGE_VERSION` en `bridge/__init__.py`.
