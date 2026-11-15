"""End-to-end: train (tiny) -> eval -> build_bundle (calibration, ONNX FP32+INT8, parity) -> Engine -> FastAPI.

Uses a random-weight backbone, 64 px images and a handful of steps, so it checks plumbing, not accuracy.
"""

import base64
import importlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")
pytest.importorskip("fastapi")

REPO = Path(__file__).resolve().parents[1]
CROPS = {"rice": ["healthy", "blast"], "tea": ["healthy", "blister_blight"]}


def _run(args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-3000:])
    return r.stdout


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    from PIL import Image

    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    rows = []
    for crop, labels in CROPS.items():
        for lab in labels:
            for split, n in (("train", 8), ("val", 4), ("test", 4)):
                rows += [dict(crop=crop, label=lab, role="known", source=f"{crop}_s", split=split)] * n
    for role in ("other_plant", "non_plant"):
        for split, n in (("train", 8), ("val", 4), ("test", 4)):
            rows += [dict(crop="", label=f"{role}:x", role=role, source=role, split=split)] * n
    rows += [dict(crop="rice", label="leaf_smut", role="unknown_eval", source="u", split="unknown_eval")] * 4
    df = pd.DataFrame(rows)
    df["sample_id"] = [f"s{i}" for i in range(len(df))]
    df["use_in_paper"] = True
    (root / "images").mkdir()
    df["path"] = [f"images/{i}.jpg" for i in range(len(df))]
    for p in df["path"]:
        Image.fromarray((rng.random((48, 48, 3)) * 255).astype(np.uint8)).save(root / p)
    df.to_csv(root / "manifest.csv", index=False)

    run, bundle = root / "run", root / "bundle"
    _run([REPO / "scripts/train_student.py", "--data-root", root, "--out", run, "--teacher", "none",
          "--student", "mnv4_s", "--no-pretrained", "--img-size", "64", "--epochs", "1", "--batch-size", "8",
          "--workers", "0", "--max-steps", "2"])
    _run([REPO / "scripts/eval_student.py", "--data-root", root, "--run", run, "--workers", "0"])
    _run([REPO / "scripts/build_bundle.py", "--data-root", root, "--run", run, "--out", bundle,
          "--calib-images", "8", "--parity-images", "8"])
    return bundle, root


def _jpeg(seed=1):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(seed).random((80, 60, 3)) * 255).astype(np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def test_bundle_contents_and_parity(bundle_dir):
    bundle, _ = bundle_dir
    for f in ("model.onnx", "model.int8.onnx", "decision.json", "preprocess.json", "prototypes_rice.npz",
              "bundle_report.json"):
        assert (bundle / f).exists(), f
    rep = json.loads((bundle / "bundle_report.json").read_text())
    assert rep["model.onnx"]["argmax_agreement"]["gate"] == 1.0  # FP32 ONNX == PyTorch
    assert rep["files"]["model.int8.onnx_mb"] < rep["files"]["model.onnx_mb"]


def test_engine_predict(bundle_dir):
    from govi_ai.inference.engine import Engine, InvalidImage

    eng = Engine(bundle_dir[0])
    r = eng.predict(_jpeg(), crop="rice", return_embedding=True)
    assert r["status"] in {"KNOWN_DISEASE", "UNKNOWN_DISEASE", "LOW_CONFIDENCE", "WRONG_CROP",
                           "UNSUPPORTED_PLANT", "NOT_A_PLANT"}
    assert r["model"]["file"] == "model.int8.onnx" and r["latency_ms"] > 0
    emb = np.frombuffer(base64.b64decode(r["embedding_f16_base64"]), dtype=np.float16)
    assert emb.shape == (eng.decision.crops["rice"].means.shape[1],)
    with pytest.raises(InvalidImage):
        eng.predict(b"not an image")


def test_fastapi_service(bundle_dir, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    bundle, _ = bundle_dir
    monkeypatch.setenv("GOVI_BUNDLE_DIR", str(bundle))
    monkeypatch.setenv("GOVI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("GOVI_API_KEY", "test-key")
    monkeypatch.syspath_prepend(str(REPO / "service"))
    main = importlib.reload(importlib.import_module("app.main"))

    with TestClient(main.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json()["status"] == "ready"
        assert client.get("/v1/metadata").status_code == 401
        h = {"X-API-Key": "test-key"}
        meta = client.get("/v1/metadata", headers=h).json()
        assert set(meta["crops"]) == {"rice", "tea", "chili"} or set(meta["crops"]) >= {"rice", "tea"}

        r = client.post("/v1/disease/predict?crop=rice", headers=h, files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 200, r.text
        assert "status" in r.json()
        assert client.post("/v1/disease/predict", headers=h,
                           files={"file": ("x.jpg", b"nope", "image/jpeg")}).status_code == 400
        assert client.post("/v1/disease/predict?crop=maize", headers=h,
                           files={"file": ("a.jpg", _jpeg(), "image/jpeg")}).status_code == 422

        eng = main.engine()
        emb = eng.predict(_jpeg(3), return_embedding=True)["embedding_f16_base64"]
        r = client.post("/v1/disease/prototypes", headers=h,
                        json={"crop": "rice", "label": "new_leaf_spot", "embeddings_f16_base64": [emb, emb]})
        assert r.status_code == 200, r.text
        assert "new_leaf_spot" in client.get("/v1/metadata", headers=h).json()["crops"]["rice"]["officer_classes"]
        assert (tmp_path / "state" / "officer_prototypes.json").exists()
        assert client.post("/v1/disease/prototypes", headers=h,
                           json={"crop": "rice", "label": "Bad Label", "embeddings_f16_base64": [emb]}).status_code == 422
