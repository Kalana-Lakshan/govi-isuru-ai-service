"""Phase 3: student model, losses, teacher targets and an end-to-end CPU smoke run."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from govi_ai.models.student import Student  # noqa: E402
from govi_ai.train.data import PAD, LabelSpace, balanced_weights  # noqa: E402
from govi_ai.train.losses import LossWeights, kd_kl, outlier_exposure, student_loss  # noqa: E402
from govi_ai.train.teacher import build_targets  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CROPS = {"rice": ["healthy", "blast", "brown_spot"], "tea": ["healthy", "blister_blight"]}
GATE = ["rice", "tea", "chili", "other_plant", "non_plant"]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Student("mnv4_s", CROPS, GATE, pretrained=False)


def test_forward_shapes_and_cam(model):
    model.eval()
    out = model(torch.randn(2, 3, 64, 64))
    assert out["gate"].shape == (2, 5)
    assert out["disease"]["rice"].shape == (2, 3) and out["disease"]["tea"].shape == (2, 2)
    assert out["emb"].shape == (2, model.embed_dim)
    cam = model.cam(out["fmap"], "rice", torch.tensor([1, 2]))
    assert cam.shape == out["fmap"].shape[:1] + out["fmap"].shape[2:]
    assert float(cam.min()) >= 0 and float(cam.max()) <= 1


def test_am_softmax_margin_lowers_true_class_logit(model):
    e = torch.randn(4, model.embed_dim)
    y = torch.tensor([0, 1, 2, 0])
    plain, margin = model.heads["rice"](e), model.heads["rice"](e, y)
    diff = (plain - margin)[torch.arange(4), y]
    torch.testing.assert_close(diff, torch.full((4,), model.heads["rice"].scale * 0.2))


def test_kd_and_oe_basics():
    z = torch.randn(8, 4)
    assert float(kd_kl(z, z, 2.0)) < 1e-5
    assert float(outlier_exposure(torch.zeros(3, 4))) == pytest.approx(np.log(4), rel=1e-5)


def test_student_loss_routes_rows_to_heads(model):
    model.train()
    b = {
        "image": torch.randn(6, 3, 64, 64),
        "gate": torch.tensor([0, 0, 1, 3, 4, 2]),
        "crop_idx": torch.tensor([0, 0, 1, -1, -1, 2]),
        "disease": torch.tensor([1, 2, 0, -1, -1, -1]),
        "t_gate": torch.randn(6, 5),
        "t_disease": torch.cat([torch.randn(6, 3), torch.full((6, 1), PAD)], 1),
    }
    out = model(b["image"])
    loss, logs = student_loss(model, out, b, LossWeights())
    loss.backward()
    assert {"gate", "kd_gate", "am_rice", "kd_rice", "oe_rice", "am_tea", "oe_tea"} <= set(logs)
    assert np.isfinite(logs["total"])


def _fake_df(n_per=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for crop, labels in CROPS.items():
        for lab in labels:
            for split, n in (("train", n_per), ("val", n_per // 2), ("test", n_per // 2)):
                rows += [dict(crop=crop, label=lab, role="known", source=f"{crop}_s", split=split)] * n
    for role in ("other_plant", "non_plant"):
        for split, n in (("train", n_per), ("val", n_per // 2), ("test", n_per // 2)):
            rows += [dict(crop="", label=f"{role}:x", role=role, source=role, split=split)] * n
    rows += [dict(crop="rice", label="leaf_smut", role="unknown_eval", source="u", split="unknown_eval")] * 5
    df = pd.DataFrame(rows)
    df["sample_id"] = [f"s{i}" for i in range(len(df))]
    space = LabelSpace(GATE, CROPS)
    df = space.encode(df)
    centers = rng.normal(0, 3, (20, 16))
    key = (df["gate_idx"] * 4 + df["disease_idx"].clip(lower=0)).to_numpy()
    X = centers[key] + rng.normal(0, 0.5, (len(df), 16))
    return df, X, space


def test_balanced_weights():
    df, _, _ = _fake_df()
    tr = df[df["split"] == "train"].reset_index(drop=True)
    w = balanced_weights(tr, negative_share=0.25)
    assert w.sum() == pytest.approx(1.0)
    assert w[tr["role"] != "known"].sum() == pytest.approx(0.25)


def test_teacher_targets_shapes_and_padding():
    df, X, space = _fake_df()
    t = build_targets(df, X, space, "split")
    assert t["gate"].shape == (len(df), 5)
    assert t["disease"].shape == (len(df), 3)
    tea_rows = (df["crop"] == "tea").to_numpy()
    assert (t["disease"][tea_rows, 2] == PAD).all()          # tea has 2 classes; slot 3 is padding
    rice_known = ((df["crop"] == "rice") & (df["role"] == "known") & (df["split"] == "test")).to_numpy()
    acc = (t["disease"][rice_known, :3].argmax(1) == df.loc[rice_known, "disease_idx"]).mean()
    assert acc > 0.9


def test_train_and_eval_scripts_end_to_end(tmp_path):
    """Tiny real run of train_student.py + eval_student.py (random backbone, 64 px, CPU)."""
    from PIL import Image

    df, X, _ = _fake_df(n_per=8)
    df = df.drop(columns=["gate_idx", "crop_idx", "disease_idx"])
    rng = np.random.default_rng(0)
    paths = []
    for i in range(len(df)):
        p = f"images/{i}.jpg"
        (tmp_path / "images").mkdir(exist_ok=True)
        Image.fromarray((rng.random((32, 32, 3)) * 255).astype(np.uint8)).save(tmp_path / p)
        paths.append(p)
    df["path"] = paths
    df["use_in_paper"] = True
    df.to_csv(tmp_path / "manifest.csv", index=False)
    np.savez(tmp_path / "feat.npz", feats=X.astype(np.float16), sample_ids=df["sample_id"].to_numpy(dtype=str))

    run = tmp_path / "run"
    cmd = [sys.executable, str(REPO / "scripts" / "train_student.py"), "--data-root", str(tmp_path),
           "--out", str(run), "--features", str(tmp_path), "--teacher", "feat", "--student", "mnv4_s",
           "--no-pretrained", "--img-size", "64", "--epochs", "2", "--batch-size", "8", "--workers", "0",
           "--max-steps", "3"]
    # The repo taxonomy has more classes than this fixture; that is fine (empty classes are skipped).
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (run / "best.pt").exists() and (run / "teacher_split.npz").exists()

    r = subprocess.run([sys.executable, str(REPO / "scripts" / "eval_student.py"), "--data-root", str(tmp_path),
                        "--run", str(run), "--workers", "0", "--batch-size", "16"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads((run / "report.json").read_text())
    assert "macro_f1" in rep["gate"] and "closed_set" in rep["crops"]["rice"]
