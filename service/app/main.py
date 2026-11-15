"""Govi Isuru AI service (FastAPI + ONNX Runtime).

Environment
-----------
GOVI_BUNDLE_DIR   serving bundle (model + decision bundle)           default /app/bundle
GOVI_STATE_DIR    writable dir for officer-registered prototypes      default /app/state
GOVI_API_KEY      if set, every /v1 request needs header X-API-Key    default unset (dev only)
GOVI_THREADS      ONNX Runtime intra-op threads                       default 1
GOVI_MAX_UPLOAD_MB                                                    default 10

This service is meant to be called by the Govi Isuru backend only, never directly by browsers.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from govi_ai.inference.engine import Engine, InvalidImage

BUNDLE_DIR = Path(os.getenv("GOVI_BUNDLE_DIR", "/app/bundle"))
STATE_DIR = Path(os.getenv("GOVI_STATE_DIR", "/app/state"))
API_KEY = os.getenv("GOVI_API_KEY")
MAX_UPLOAD = int(float(os.getenv("GOVI_MAX_UPLOAD_MB", "10")) * 1024 * 1024)
THREADS = int(os.getenv("GOVI_THREADS", "1"))

_engine: Engine | None = None
_lock = threading.Lock()


def engine() -> Engine:
    if _engine is None:
        raise HTTPException(503, "model not loaded")
    return _engine


def _officer_file() -> Path:
    return STATE_DIR / "officer_prototypes.json"


def _load_officer_prototypes(eng: Engine) -> int:
    f = _officer_file()
    if not f.exists():
        return 0
    entries = json.loads(f.read_text(encoding="utf-8"))
    for e in entries:
        eng.decision.add_prototype(e["crop"], e["label"], np.asarray(e["mean"], dtype=np.float64))
    return len(entries)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = Engine(BUNDLE_DIR, threads=THREADS)
    app.state.officer_prototypes = _load_officer_prototypes(_engine)
    yield
    _engine = None


app = FastAPI(title="Govi Isuru AI service", version="1.0.0", lifespan=lifespan,
              description="Crop gate + open-set disease recognition for rice, tea and chili.")


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and not (x_api_key and secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(401, "missing or invalid X-API-Key")


# ------------------------------------------------------------------ schemas
Status = Literal["KNOWN_DISEASE", "UNKNOWN_DISEASE", "LOW_CONFIDENCE", "WRONG_CROP", "UNSUPPORTED_PLANT", "NOT_A_PLANT"]


class PredictResponse(BaseModel):
    status: Status
    crop: str | None = None
    requested_crop: str | None = None
    detected_crop: str | None = None
    class_: str | None = Field(default=None, alias="class")
    source: str | None = None
    confidence: float | None = None
    reason: str | None = None
    top3: list[dict] | None = None
    gate: dict | None = None
    novelty: dict | None = None
    heatmap_png_base64: str | None = None
    embedding_f16_base64: str | None = None
    model: dict
    latency_ms: float

    model_config = {"populate_by_name": True}


class PrototypeRequest(BaseModel):
    crop: str
    label: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$", description="snake_case id agreed by the officer")
    embeddings_f16_base64: list[str] = Field(min_length=1, max_length=200,
                                             description="embeddings returned by /v1/disease/predict for confirmed cases")


# ------------------------------------------------------------------ routes
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    e = engine()
    return {"status": "ready", "model": e.model_file, "version": e.decision.version}


@app.get("/v1/metadata", dependencies=[Depends(require_key)])
def metadata() -> dict:
    e = engine()
    return {
        "version": e.decision.version, "model": e.model_file, "gate_classes": e.decision.gate_classes,
        "crops": {c: {"classes": cal.classes, "officer_classes": [lab for lab, o in zip(cal.proto_labels, cal.proto_origin)
                                                                  if o == "officer"]}
                  for c, cal in e.decision.crops.items()},
        "statuses": list(Status.__args__),
    }


@app.post("/v1/disease/predict", response_model=PredictResponse, response_model_exclude_none=True,
          response_model_by_alias=True, dependencies=[Depends(require_key)])
def predict(file: UploadFile = File(...), crop: str | None = Query(default=None, description="crop the farmer selected"),
            heatmap: bool = True) -> dict:
    e = engine()
    if crop is not None and crop not in e.crops:
        raise HTTPException(422, f"crop must be one of {e.crops}")
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"image larger than {MAX_UPLOAD // (1024 * 1024)} MB")
    try:
        with _lock:  # decision state (officer prototypes) may change concurrently
            return e.predict(data, crop=crop, heatmap=heatmap)
    except InvalidImage as err:
        raise HTTPException(400, str(err)) from err


@app.post("/v1/disease/prototypes", dependencies=[Depends(require_key)])
def register_prototype(req: PrototypeRequest) -> dict:
    """Officer-verified new disease: adds a prototype so it is recognised immediately (no retraining)."""
    e = engine()
    if req.crop not in e.decision.crops:
        raise HTTPException(422, f"crop must be one of {list(e.decision.crops)}")
    try:
        embs = np.stack([np.frombuffer(base64.b64decode(s), dtype=np.float16) for s in req.embeddings_f16_base64])
    except ValueError as err:
        raise HTTPException(422, "embeddings must be base64 float16 vectors from /v1/disease/predict") from err
    dim = e.decision.crops[req.crop].means.shape[1]
    if embs.shape[1] != dim:
        raise HTTPException(422, f"embedding size {embs.shape[1]} != model's {dim}")
    mean = embs.astype(np.float64).mean(0)
    with _lock:
        e.decision.add_prototype(req.crop, req.label, mean)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        f = _officer_file()
        entries = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        entries.append({"crop": req.crop, "label": req.label, "n": len(embs), "mean": mean.round(6).tolist()})
        f.write_text(json.dumps(entries), encoding="utf-8")
    return {"status": "registered", "crop": req.crop, "label": req.label, "n_embeddings": len(embs)}
