# service/

FastAPI + ONNX Runtime serving image. It depends only on `govi-ai[service]` (numpy, pillow, onnxruntime,
fastapi, uvicorn, pydantic): no torch, TensorFlow, pandas or scikit-learn.

## Build and run

```bash
# 1. produce a bundle (Colab, after training):  scripts/build_bundle.py --run <run> --out bundle
# 2. from the repo root, with ./bundle present:
docker build -f service/Dockerfile -t govi-ai-service .
docker run -p 8000:8000 -e GOVI_API_KEY=change-me -v govi-state:/app/state govi-ai-service
```

Only `model.int8.onnx` is shipped (the FP32 `model.onnx` is excluded by `.dockerignore`).
Officer-registered prototypes are written to `/app/state`; mount a volume there to keep them.

## API (all `/v1` routes need `X-API-Key` when `GOVI_API_KEY` is set)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/ready` | model loaded |
| GET | `/v1/metadata` | classes per crop, officer-added classes, statuses |
| POST | `/v1/disease/predict?crop=rice&heatmap=true` | multipart `file`; returns one of the statuses below |
| POST | `/v1/disease/prototypes` | officer-verified new disease: `{crop, label, embeddings_f16_base64[]}` |

Statuses: `KNOWN_DISEASE`, `UNKNOWN_DISEASE` (includes `embedding_f16_base64` for officer review),
`LOW_CONFIDENCE`, `WRONG_CROP` (with `detected_crop`), `UNSUPPORTED_PLANT`, `NOT_A_PLANT`.
The OpenAPI contract is served at `/openapi.json` and `/docs`.

## New-disease workflow (backend side)

1. `/v1/disease/predict` returns `UNKNOWN_DISEASE` plus an embedding. The backend stores it with the photo.
2. Officers review the stored cases, group them, and confirm a name (snake_case id).
3. The backend calls `/v1/disease/prototypes` with the confirmed cases' embeddings. The disease is recognised at once
   (`KNOWN_DISEASE`, `source: officer_prototype`).
4. Confirmed photos go into the next training round, where the disease becomes a trained class.
