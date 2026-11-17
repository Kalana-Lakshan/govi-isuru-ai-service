# Govi Isuru AI service: slim CPU image (no torch, no tensorflow).
# Build from the repo root, with the serving bundle at ./bundle (model.int8.onnx + decision files):
#   docker build -t govi-ai-service .
#   docker run -p 8000:8000 -e GOVI_API_KEY=change-me -v govi-state:/app/state govi-ai-service

FROM python:3.12-slim AS build
WORKDIR /src
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --prefix=/install ".[service]"

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    GOVI_BUNDLE_DIR=/app/bundle GOVI_STATE_DIR=/app/state GOVI_THREADS=1 \
    OMP_NUM_THREADS=1
COPY --from=build /install /usr/local
WORKDIR /app
COPY service/app ./app
COPY bundle ./bundle
RUN useradd --create-home --uid 10001 govi && mkdir -p /app/state && chown govi /app/state
USER govi
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
