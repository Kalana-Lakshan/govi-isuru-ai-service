# service/

FastAPI + ONNX Runtime serving image. It depends only on `govi-ai[service]` (numpy, pillow, onnxruntime,
fastapi, uvicorn, pydantic): no torch, TensorFlow, pandas or scikit-learn.

## Build and run

```bash
# The trained serving bundle is committed in ./bundle (rebuild: scripts/build_bundle.py, then
# scripts/compose_bundle.py). From the repo root:
docker build -f service/Dockerfile -t govi-ai-service .
docker run -p 8000:8000 -e GOVI_API_KEY=change-me -v govi-state:/app/state govi-ai-service
```

The service loads `model.onnx` (FP32, what the thresholds were calibrated on); set `GOVI_MODEL_FILE=model.int8.onnx` to use an INT8 model instead.
Officer-registered prototypes are written to `/app/state`; mount a volume there to keep them.

## API (all `/v1` routes need `X-API-Key` when `GOVI_API_KEY` is set)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/ready` | model loaded |
| GET | `/v1/metadata` | classes per crop, officer-added classes, statuses |
| POST | `/v1/disease/predict?crop=rice&heatmap=true` | multipart `file`; returns one of the statuses below |
| POST | `/v1/disease/prototypes` | officer-verified new disease: `{crop, label, embeddings_f16_base64[]}` |
| POST | `/v1/disease/unknowns/cluster` | group stored unknown cases into candidate new diseases for officers |
| GET | `/v1/classes?crop=` | class types, display names (en/si/ta) and agronomist advice |
| GET | `/metrics` | Prometheus counters per status, mean latency, uptime |
| GET | `/v1/yield/districts` | districts with forecasts |
| GET | `/v1/yield/forecast?district=&season=maha\|yala&area_ha=` | next-season paddy yield (p10/p50/p90 kg/ha, 80 % interval) and expected harvest |
| GET | `/v1/yield/history?district=&season=` | observed DCS yields per year |
| POST | `/v1/suitability` | rank rice / tea / chili (FAO EcoCrop model): `{district?, season, soil_ph, temperature_c?, season_rainfall_mm?, annual_rainfall_mm?, irrigated, drainage?, texture?, altitude_m?}` |

Statuses: `KNOWN_DISEASE`, `UNKNOWN_DISEASE` (includes `embedding_f16_base64` for officer review),
`LOW_CONFIDENCE`, `WRONG_CROP` (with `detected_crop`), `UNSUPPORTED_PLANT`, `NOT_A_PLANT`, `BAD_PHOTO`
(with `problems` and farmer-readable `advice`). Full integration guide: [../docs/INTEGRATION.md](../docs/INTEGRATION.md).
The OpenAPI contract is served at `/openapi.json` and `/docs`.

## New-disease workflow (backend side)

1. `/v1/disease/predict` returns `UNKNOWN_DISEASE` plus an embedding. The backend stores it with the photo.
2. Officers review the stored cases, group them, and confirm a name (snake_case id).
3. The backend calls `/v1/disease/prototypes` with the confirmed cases' embeddings. The disease is recognised at once
   (`KNOWN_DISEASE`, `source: officer_prototype`).
4. Confirmed photos go into the next training round, where the disease becomes a trained class.
