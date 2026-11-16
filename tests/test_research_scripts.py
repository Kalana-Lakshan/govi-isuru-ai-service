"""Research scripts end to end on a tiny dataset: near-duplicate audit -> clean report -> officer loop
-> XAI -> results tables. Checks plumbing, not accuracy (random-weight model, 48 px images)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("sklearn")

REPO = Path(__file__).resolve().parents[1]


def _run(args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-3000:])
    return r.stdout


@pytest.fixture(scope="module")
def govi(tmp_path_factory):
    from PIL import Image

    root = tmp_path_factory.mktemp("data")
    G = tmp_path_factory.mktemp("govi")
    rng = np.random.default_rng(0)
    rows = []
    for crop, labels in {"rice": ["healthy", "blast", "brown_spot"], "tea": ["healthy", "blister_blight"]}.items():
        for lab in labels:
            for split, n in (("train", 10), ("val", 4), ("test", 4)):
                rows += [dict(crop=crop, label=lab, role="known", source=f"{crop}_s", split=split)] * n
    for lab in ("white_spot", "anthracnose"):
        rows += [dict(crop="tea", label=lab, role="unknown_eval", source="u", split="unknown_eval")] * 6
    for role in ("other_plant", "non_plant"):
        for split, n in (("train", 6), ("val", 3), ("test", 3)):
            rows += [dict(crop="", label=f"{role}:x", role=role, source=role, split=split)] * n
    df = pd.DataFrame(rows)
    df["sample_id"] = [f"s{i:04d}" for i in range(len(df))]
    df["use_in_paper"] = True
    df["path"] = [f"images/{i}.jpg" for i in range(len(df))]
    (root / "images").mkdir()
    for p in df["path"]:
        small = (rng.random((6, 6, 3)) * 255).astype(np.uint8)
        Image.fromarray(small).resize((48, 48), Image.Resampling.BICUBIC).save(root / p)
    df.to_csv(root / "manifest.csv", index=False)

    # synthetic "DINOv2" features: the first test image is a near-copy of the first train image
    feats = rng.normal(0, 1, (len(df), 16)).astype(np.float32)
    first_test = df.index[(df["split"] == "test") & (df["crop"] == "rice")][0]
    first_train = df.index[(df["split"] == "train") & (df["crop"] == "rice")][0]
    feats[first_test] = feats[first_train] + 0.001
    (G / "features").mkdir()
    np.savez(G / "features" / "dinov2_b.npz", feats=feats.astype(np.float16), sample_ids=df["sample_id"].to_numpy(dtype=str))

    run = G / "runs" / "mnv4_s_split"
    _run([REPO / "scripts/train_student.py", "--data-root", root, "--out", run, "--teacher", "none", "--student", "mnv4_s",
          "--no-pretrained", "--img-size", "64", "--epochs", "1", "--batch-size", "8", "--workers", "0", "--max-steps", "2"])
    _run([REPO / "scripts/near_dup_audit.py", "--data-root", root, "--features", G / "features", "--out", G / "audit"])
    _run([REPO / "scripts/eval_student.py", "--data-root", root, "--run", run, "--workers", "0",
          "--exclude-file", G / "audit" / "near_dup_exclusions.csv"])
    return root, G, df.at[first_test, "sample_id"]


def test_near_dup_audit_flags_the_planted_copy(govi):
    _, G, planted = govi
    excl = pd.read_csv(G / "audit" / "near_dup_exclusions.csv")
    assert planted in set(excl.loc[excl["protocol"] == "split", "sample_id"])
    summ = json.loads((G / "audit" / "summary.json").read_text())
    assert summ["protocols"]["split"]["n_test"] > 0
    assert (G / "audit" / "pairs_top.png").exists()


def test_clean_report_excludes_flagged_rows(govi):
    _, G, _ = govi
    run = G / "runs" / "mnv4_s_split"
    clean = json.loads((run / "report_clean.json").read_text())
    full = json.loads((run / "report.json").read_text())
    assert clean["excluded_near_duplicates"] >= 1
    assert clean["crops"]["rice"]["n_test"] == full["crops"]["rice"]["n_test"] - 1


def test_officer_loop_experiment(govi):
    root, G, _ = govi
    _run([REPO / "scripts/officer_loop_experiment.py", "--data-root", root, "--run", G / "runs" / "mnv4_s_split",
          "--out", G / "officer", "--seeds", "2", "--min-train", "4", "--min-test", "2"])
    rep = json.loads((G / "officer" / "officer_loop.json").read_text())
    rice = rep["few_shot"]["rice"]
    assert set(rice) == {"healthy", "blast", "brown_spot"}
    one = next(iter(rice.values()))
    assert 0 <= one["flagged_unknown_before"] <= 1 and "5" in one["shots"]
    assert "recommended_threshold" in rep["clustering"]["tea"]


def test_xai_and_tables(govi):
    root, G, _ = govi
    _run([REPO / "scripts/xai_figures.py", "--data-root", root, "--run", G / "runs" / "mnv4_s_split", "--out", G / "xai",
          "--per-crop", "3", "--figure-rows", "2"])
    xai = json.loads((G / "xai" / "faithfulness.json").read_text())
    assert set(xai["crops"]["rice"]["deletion_auc_mean"]) == {"cam", "gradcam", "gradcam_pp", "random"}
    assert (G / "xai" / "xai_rice.png").exists()

    _run([REPO / "scripts/make_tables.py", "--govi", G, "--out", G / "results"])
    md = (G / "results" / "RESULTS.md").read_text(encoding="utf-8")
    for heading in ("Student (MobileNetV4-S", "near-duplicate-free", "Ablations", "Officer loop", "faithfulness",
                    "near-duplicate audit"):
        assert heading in md, heading
