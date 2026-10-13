# =============================================================================
# Worker de Inferencia Brain-Master — imagen GPU desplegable
#
# Build (en máquina con GPU o para registry):
#   docker build -t brainmaster-worker:latest .
#
# Run local (NVIDIA Container Toolkit requerido):
#   docker run -d --gpus all \
#     -e GRPC_PORT=50051 \
#     -e BM_OUTPUT_DIR=/app/outputs \
#     -v /ruta/cache/hf:/root/.cache/huggingface \
#     -v /ruta/app/models:/models:ro \        # opcional: familias bridge V1
#     -p 50051:50051 brainmaster-worker:latest
#
# Plataformas: Runpod · Vast.ai · Lambda · Fly.io (GPU) · EC2 g5/g6 · local
# =============================================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Cache de Hugging Face dentro del volumen montable
    HF_HOME=/models-cache/huggingface \
    TORCH_HOME=/models-cache/torch

WORKDIR /app

# Python 3.11 + ffmpeg (encoder de video del worker) + herramientas
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    ffmpeg git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# PyTorch con CUDA 12.4 + stack de IA completo
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r requirements.txt

# Código del worker (incluye engines/)
COPY . .

# Outputs + punto de montaje del cache de modelos + punto de montaje V1
RUN mkdir -p /app/outputs /models-cache /models
VOLUME ["/app/outputs", "/models-cache", "/models"]

EXPOSE 50051

# Healthcheck gRPC barato: intenta GetGpuTelemetry cada 60s
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import grpc,os; \
c=grpc.insecure_channel(f'127.0.0.1:{os.environ.get(\"GRPC_PORT\",\"50051\")}'); \
c.close()" || exit 1

CMD ["python", "server.py"]
