# syntax=docker/dockerfile:1
FROM python:3.11-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app


RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 torchvision==0.17.2


RUN python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"


RUN pip install --no-cache-dir \
    huggingface_hub==0.19.4 \
    transformers==4.36.2 \
    sentence-transformers==2.2.2


COPY requirements.txt ./
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt


COPY app ./app
COPY data ./data


RUN mkdir -p data/raw data/index debug_plots

# Env
ENV EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
