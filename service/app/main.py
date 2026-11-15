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


_yield: dict = {}


def _load_yield() -> dict:
    """Optional yield files (bundle/yield/*.json from scripts/train_yield.py)."""
    d = BUNDLE_DIR / "yield"
    if not (d / "forecasts.json").exists():
        return {}
    f = json.loads((d / "forecasts.json").read_text(encoding="utf-8"))
    h = json.loads((d / "history.json").read_text(encoding="utf-8")) if (d / "history.json").exists() else {}
    index = {(r["district"], r["season"]): r for r in f["forecasts"]}
    return {"meta": {k: v for k, v in f.items() if k != "forecasts"}, "forecasts": index, "history": h.get("history", {})}


_suit: dict = {}


def _load_suitability() -> dict:
    """Optional suitability files (bundle/suitability/*.json from scripts/build_suitability.py)."""
    from govi_ai.tabular.suitability import CropRequirements

    d = BUNDLE_DIR / "suitability"
    if not (d / "crops.json").exists():
        return {}
    c = json.loads((d / "crops.json").read_text(encoding="utf-8"))
    dist = json.loads((d / "districts.json").read_text(encoding="utf-8")) if (d / "districts.json").exists() else {}
    reqs = {k: CropRequirements(**v) for k, v in c["crops"].items()}
    meta = {k: v for k, v in c.items() if k != "crops"} | {"climate_source": dist.get("climate_source")}
    return {"reqs": reqs, "districts": dist.get("districts", {}), "meta": meta}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _yield, _suit
    _engine = Engine(BUNDLE_DIR, threads=THREADS)
    app.state.officer_prototypes = _load_officer_prototypes(_engine)
    _yield = _load_yield()
    _suit = _load_suitability()
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


def _yield_data() -> dict:
    if not _yield:
        raise HTTPException(503, "yield model not available in this bundle")
    return _yield


@app.get("/v1/yield/districts", dependencies=[Depends(require_key)])
def yield_districts() -> dict:
    y = _yield_data()
    return {"districts": sorted({d for d, _ in y["forecasts"]}), "seasons": ["maha", "yala"], **y["meta"]}


@app.get("/v1/yield/forecast", dependencies=[Depends(require_key)])
def yield_forecast(district: str, season: Literal["maha", "yala"],
                   area_ha: float | None = Query(default=None, gt=0, le=100_000)) -> dict:
    """Next-season paddy yield for a district (official DCS data, 5-year climatology + calibrated 80 % interval).
    With ``area_ha`` the expected harvest in kg is returned as well."""
    y = _yield_data()
    r = y["forecasts"].get((district.strip().upper(), season))
    if r is None:
        raise HTTPException(404, f"no forecast for {district}/{season}")
    out = {**r, **y["meta"]}
    if area_ha is not None:
        out["expected_harvest_kg"] = {k: round(v * area_ha) for k, v in r["yield_kg_ha"].items()}
    return out


@app.get("/v1/yield/history", dependencies=[Depends(require_key)])
def yield_history(district: str, season: Literal["maha", "yala"] | None = None) -> dict:
    y = _yield_data()
    h = y["history"].get(district.strip().upper())
    if h is None:
        raise HTTPException(404, f"unknown district {district}")
    return {"district": district.strip().upper(), "history": {season: h.get(season, {})} if season else h,
            "unit": "kg per net hectare"}


class SuitabilityRequest(BaseModel):
    district: str | None = Field(default=None, description="fills climate defaults from NASA POWER climatology")
    season: Literal["maha", "yala"] = "maha"
    soil_ph: float = Field(ge=3.0, le=10.0)
    temperature_c: float | None = Field(default=None, ge=-5, le=45)
    season_rainfall_mm: float | None = Field(default=None, ge=0, le=6000)
    annual_rainfall_mm: float | None = Field(default=None, ge=0, le=8000)
    irrigated: bool = False
    drainage: Literal["poor", "moderate", "good", "excessive"] | None = None
    texture: Literal["clay", "clay_loam", "loam", "sandy_loam", "sandy", "organic"] | None = None
    altitude_m: float | None = Field(default=None, ge=0, le=3000)


@app.post("/v1/suitability", dependencies=[Depends(require_key)])
def suitability(req: SuitabilityRequest) -> dict:
    """Rank rice / tea / chili with the FAO EcoCrop limiting-factor model (real ECOCROP requirements)."""
    from govi_ai.tabular.suitability import recommend

    if not _suit:
        raise HTTPException(503, "suitability data not available in this bundle")
    clim = {}
    if req.district:
        clim = _suit["districts"].get(req.district.strip().upper())
        if clim is None:
            raise HTTPException(404, f"unknown district {req.district}")
    temp = req.temperature_c if req.temperature_c is not None else clim.get("temperature_c")
    season_rain = req.season_rainfall_mm if req.season_rainfall_mm is not None else clim.get(f"{req.season}_rainfall_mm")
    annual_rain = req.annual_rainfall_mm if req.annual_rainfall_mm is not None else clim.get("annual_rainfall_mm")
    if None in (temp, season_rain, annual_rain):
        raise HTTPException(422, "give a district, or temperature_c, season_rainfall_mm and annual_rainfall_mm")
    ranked = recommend(_suit["reqs"], season_rainfall_mm=season_rain, annual_rainfall_mm=annual_rain,
                       temperature_c=temp, soil_ph=req.soil_ph, irrigated=req.irrigated, drainage=req.drainage,
                       texture=req.texture, altitude_m=req.altitude_m)
    return {"ranking": ranked,
            "inputs_used": {"temperature_c": temp, "season_rainfall_mm": season_rain, "annual_rainfall_mm": annual_rain,
                            "season": req.season, "soil_ph": req.soil_ph, "irrigated": req.irrigated,
                            "drainage": req.drainage, "texture": req.texture, "altitude_m": req.altitude_m,
                            "climate_defaults_from": "NASA POWER" if clim else None},
            "method": "FAO EcoCrop limiting-factor model; classes S1>=0.8, S2>=0.6, S3>=0.4, else N",
            **_suit["meta"]}


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
