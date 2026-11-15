"""Decision logic and calibration (numpy only)."""

import numpy as np
import pandas as pd

from govi_ai.calibrate import _low_conf_threshold, fit_bundle
from govi_ai.inference.decision import CropCalibration, DecisionBundle

GATE = ["rice", "tea", "chili", "other_plant", "non_plant"]


def _bundle():
    rice = CropCalibration(classes=["healthy", "blast"], temperature=1.0,
                           means=np.array([[0.0, 0.0], [10.0, 0.0]]), precision=np.eye(2),
                           unknown_threshold=-9.0, low_conf_threshold=0.7)
    return DecisionBundle(GATE, 1.0, 0.5, {"rice": rice})


def _gate(i, strength=10.0):
    z = np.zeros(5)
    z[i] = strength
    return z


def test_gate_outcomes():
    b = _bundle()
    d = {"rice": np.array([5.0, 0.0])}
    assert b.decide(_gate(4), d, np.zeros(2))["status"] == "NOT_A_PLANT"
    assert b.decide(_gate(3), d, np.zeros(2))["status"] == "UNSUPPORTED_PLANT"
    r = b.decide(_gate(1), d, np.zeros(2), requested_crop="rice")
    assert r["status"] == "WRONG_CROP" and r["detected_crop"] == "tea"
    assert b.decide(np.zeros(5), d, np.zeros(2))["status"] == "LOW_CONFIDENCE"  # uniform gate


def test_known_low_confidence_and_unknown():
    b = _bundle()
    r = b.decide(_gate(0), {"rice": np.array([5.0, 0.0])}, np.array([0.5, 0.0]), requested_crop="rice")
    assert r["status"] == "KNOWN_DISEASE" and r["class"] == "healthy" and r["source"] == "model"
    r = b.decide(_gate(0), {"rice": np.array([0.2, 0.0])}, np.array([0.5, 0.0]))
    assert r["status"] == "LOW_CONFIDENCE" and r["reason"] == "disease_uncertain"
    r = b.decide(_gate(0), {"rice": np.array([5.0, 0.0])}, np.array([5.0, 8.0]))  # far from both prototypes
    assert r["status"] == "UNKNOWN_DISEASE" and r["novelty"]["mahalanobis_score"] < -9


def test_officer_prototype_is_recognised_immediately_and_persists(tmp_path):
    b = _bundle()
    emb = np.array([5.0, 8.0])
    assert b.decide(_gate(0), {"rice": np.array([5.0, 0.0])}, emb)["status"] == "UNKNOWN_DISEASE"
    b.add_prototype("rice", "new_spot", emb[None] + np.random.default_rng(0).normal(0, 0.1, (5, 2)))
    r = b.decide(_gate(0), {"rice": np.array([5.0, 0.0])}, emb)
    assert r["status"] == "KNOWN_DISEASE" and r["class"] == "new_spot" and r["source"] == "officer_prototype"
    b.save(tmp_path)
    r2 = DecisionBundle.load(tmp_path).decide(_gate(0), {"rice": np.array([5.0, 0.0])}, emb)
    assert r2["class"] == "new_spot"


def test_low_conf_threshold_meets_target_risk():
    rng = np.random.default_rng(0)
    conf = rng.random(5000)
    correct = rng.random(5000) < conf  # higher confidence -> more often right
    t = _low_conf_threshold(conf, correct.astype(float), 0.05)
    acc = correct[conf >= t]
    assert 1 - acc.mean() <= 0.05 + 1e-9 and acc.size > 100


def test_fit_bundle_on_synthetic_outputs():
    rng = np.random.default_rng(1)
    rows, emb, gate, dis = [], [], [], []
    centers = {"healthy": np.array([0.0, 0, 0, 0]), "blast": np.array([6.0, 0, 0, 0]),
               "smut": np.array([0.0, 9, 0, 0])}
    for lab, role in (("healthy", "known"), ("blast", "known"), ("smut", "unknown_eval")):
        for split in (["train"] * 60 + ["val"] * 30 + ["test"] * 30) if role == "known" else ["unknown_eval"] * 30:
            rows.append({"crop": "rice", "label": lab, "role": role, "split": split,
                         "gate_idx": 0, "disease_idx": {"healthy": 0, "blast": 1}.get(lab, -1)})
            e = centers[lab] + rng.normal(0, 0.5, 4)
            emb.append(e)
            gate.append(_gate(0, 6.0) + rng.normal(0, 0.5, 5))
            dis.append(np.array([-np.linalg.norm(e - centers["healthy"]), -np.linalg.norm(e - centers["blast"])]) * 3)
    df = pd.DataFrame(rows)
    outs = {"emb": np.array(emb), "gate": np.array(gate), "disease_rice": np.array(dis)}
    bundle, rep = fit_bundle(df, outs, GATE, {"rice": ["healthy", "blast"]})
    assert rep["rice"]["unknown_disease_detected"] > 0.9
    assert 0.85 < 1 - rep["rice"]["test_known_flagged_unknown"] <= 1.0
    assert rep["rice"]["test_macro_f1"] > 0.95
