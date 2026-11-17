# Integrating the Govi Isuru AI service (for the backend)

The AI service is a small CPU container that **only the backend** calls. It doesn't talk to browsers or apps
directly. The full contract is in [`openapi.json`](openapi.json); interactive docs are at `/docs` when the service runs.

## 1. Run it

```bash
# the trained serving bundle is committed in ./bundle (version in bundle/decision.json)
GOVI_API_KEY=change-me docker compose up --build        # -> http://localhost:8000
```

When the trained bundle is ready, replace `./bundle` with it and rebuild. The API doesn't change.
Development-bundle responses contain `model.warning`; **never show their diagnoses to farmers**.

| Env var | Meaning | Default |
|---|---|---|
| `GOVI_API_KEY` | required header `X-API-Key` on every `/v1` route and `/metrics` | unset = open (dev only) |
| `GOVI_THREADS` | CPU threads for inference | 1 |
| `GOVI_MAX_UPLOAD_MB` | upload limit | 10 |
| `GOVI_STATE_DIR` | where officer-registered diseases are stored; **mount a volume** | `/app/state` |

Resources: ~300 MB RAM, 1 vCPU, ~20–60 ms per photo. Health: `GET /health` (liveness), `GET /ready` (model loaded).

## 2. Disease diagnosis: `POST /v1/disease/predict`

`multipart/form-data` with field `file` (JPEG/PNG). Query: `crop=rice|tea|chili` (what the farmer selected,
optional), `heatmap=true|false`.

Every response has `status`. Show the farmer something different for each:

| `status` | Meaning | Suggested app behaviour |
|---|---|---|
| `KNOWN_DISEASE` | `class`, `confidence`, `top3`, `heatmap_png_base64` | show diagnosis + heatmap + advice from `/v1/classes` |
| `LOW_CONFIDENCE` | `top3` but not sure (`reason`) | "Possibly X or Y. Please consult an officer"; offer booking |
| `UNKNOWN_DISEASE` | a leaf of the crop, but not a disease the model knows; has `embedding_f16_base64` | "Unrecognised problem, sent to an officer"; **store the case** (§4) |
| `WRONG_CROP` | `requested_crop` vs `detected_crop` | "This looks like a **chili** leaf, not paddy. Diagnose as chili?" and re-call with `crop=detected_crop` |
| `UNSUPPORTED_PLANT` | a plant, but not rice/tea/chili | "Only rice, tea and chili are supported" |
| `NOT_A_PLANT` | not a plant photo | "Please photograph a leaf" |
| `BAD_PHOTO` | `problems` + farmer-readable `advice` list | show the advice, ask for a retake |

All responses carry `photo_quality` measurements and `model` (`version`, `file`). Errors: `400` unreadable
image, `401` bad key, `413` too large, `422` invalid parameters, `503` model not loaded.

## 3. Class information: `GET /v1/classes?crop=rice`

Type (fungal/bacterial/viral/pest/abiotic/healthy), status (`known` / `unknown_eval`), display names
`name.en/si/ta` and `advice.en/si/ta`. Sinhala/Tamil names and advice are **empty until an agronomist fills
`configs/disease_info.yaml`**. Show the English class name with "advice coming soon" until then.

## 4. New-disease workflow (officers)

1. **Store** every `UNKNOWN_DISEASE` case: photo, district, date, crop and `embedding_f16_base64`.
2. **Group** cases for review: `POST /v1/disease/unknowns/cluster`
   `{"crop": "rice", "cases": [{"id": "<your case id>", "embedding_f16_base64": "..."}], "threshold": 0.8, "min_size": 3}`
   returns `groups` (member ids, `representative_id` = look at this photo first, `closest_known`) and
   `ungrouped_ids`. Many similar unknowns from one district may indicate an outbreak worth an alert.
3. An officer confirms a group as a disease and picks a snake_case id (e.g. `rice_false_smut`).
4. **Register** it: `POST /v1/disease/prototypes`
   `{"crop": "rice", "label": "rice_false_smut", "embeddings_f16_base64": ["...", "..."]}`.
   From then on matching photos return `KNOWN_DISEASE` with `source: "officer_prototype"`, with no retraining.
5. Keep the confirmed photos: they go into the next training round.

## 5. Paddy yield and crop suitability

* `GET /v1/yield/districts`, `GET /v1/yield/forecast?district=Polonnaruwa&season=maha&area_ha=2`
  returns p10/p50/p90 kg/ha (80 % range) and `expected_harvest_kg` for the area; source: official DCS statistics.
* `GET /v1/yield/history?district=...&season=...` returns observed yields per year (for charts).
* `POST /v1/suitability` with `{"district": "Kandy", "season": "maha", "soil_ph": 5.2, "irrigated": false,
  "drainage": "good"}` ranks rice/tea/chili with FAO classes S1/S2/S3/N and the `limiting_factor`.
  District climate is filled from ERA5 climatology; the user may override `temperature_c`, rainfall, `altitude_m`.

## 6. Operations

* `GET /metrics` (Prometheus text): predictions per status, error counts, mean latency, uptime.
* A sudden rise in `BAD_PHOTO`, `NOT_A_PLANT` or `UNKNOWN_DISEASE` usually means a UI or camera problem, or a
  real new disease. Both are worth a look.
* Back up the state volume (officer-registered diseases).
