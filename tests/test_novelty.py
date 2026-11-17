"""Per-crop unknown-disease score: chosen by leave-one-class-out on validation, saved in the bundle, used at serving."""

import numpy as np

from govi_ai.calibrate import choose_novelty, coverage_table
from govi_ai.inference.decision import CropCalibration, DecisionBundle


def _crop(novelty=None):
    means = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    return CropCalibration(classes=["healthy", "blast", "spot"], temperature=1.0, means=means, precision=np.eye(2),
                           unknown_threshold=-9.0, low_conf_threshold=0.5,
                           novelty=novelty or {"score": "mahalanobis"})


def _bundle(crop):
    return DecisionBundle(["rice", "tea", "chili", "other_plant", "non_plant"], 1.0, 0.0, {"rice": crop})


GATE_RICE = np.array([10.0, 0, 0, 0, 0])


def test_msp_score_wins_when_distances_carry_no_signal():
    rng = np.random.default_rng(0)
    n, k = 300, 3
    y = rng.integers(0, k, n)
    dist = rng.random((n, k))                       # Mahalanobis distances: pure noise
    logits = rng.normal(0, 0.3, (n, k))
    logits[np.arange(n), y] += 4.0                   # the classifier is confident on known classes
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    cfg, loco = choose_novelty(dist, probs, y, known_tpr=0.95, maha_threshold=-1.0)
    assert cfg["score"] in ("msp", "fused") and loco[cfg["score"]] > loco["mahalanobis"]
    assert "threshold" in cfg


def test_msp_novelty_flags_an_uncertain_leaf_and_keeps_a_confident_one():
    crop = _crop({"score": "msp", "threshold": 0.8, "maha_mu": 0, "maha_sd": 1, "msp_mu": 0, "msp_sd": 1})
    b = _bundle(crop)
    sure = b.decide(GATE_RICE, {"rice": np.array([0.0, 8.0, 0.0])}, np.array([10.0, 0.5]), "rice")
    unsure = b.decide(GATE_RICE, {"rice": np.array([1.0, 1.1, 0.9])}, np.array([10.0, 0.5]), "rice")
    assert sure["status"] == "KNOWN_DISEASE" and sure["class"] == "blast"
    assert unsure["status"] == "UNKNOWN_DISEASE" and unsure["novelty"]["score"] == "msp"


def test_officer_prototype_still_recognised_with_msp_novelty():
    crop = _crop({"score": "msp", "threshold": 0.99, "maha_mu": 0, "maha_sd": 1, "msp_mu": 0, "msp_sd": 1})
    b = _bundle(crop)
    b.add_prototype("rice", "rice_false_smut", np.array([[20.0, 20.0]]))
    out = b.decide(GATE_RICE, {"rice": np.array([1.0, 1.0, 1.0])}, np.array([20.0, 20.5]), "rice")
    assert out["status"] == "KNOWN_DISEASE" and out["class"] == "rice_false_smut"


def test_novelty_survives_save_and_load(tmp_path):
    cfg = {"score": "fused", "threshold": -1.5, "maha_mu": -3.0, "maha_sd": 2.0, "msp_mu": 0.8, "msp_sd": 0.1}
    _bundle(_crop(cfg)).save(tmp_path)
    loaded = DecisionBundle.load(tmp_path)
    assert loaded.crops["rice"].novelty == cfg


def test_old_bundles_default_to_mahalanobis(tmp_path):
    _bundle(_crop()).save(tmp_path)
    assert DecisionBundle.load(tmp_path).crops["rice"].novelty["score"] == "mahalanobis"


def test_coverage_table_is_monotone():
    rng = np.random.default_rng(1)
    conf = rng.random(500)
    correct = (rng.random(500) < conf).astype(float)   # more confident -> more often right
    rows = coverage_table(conf, correct)
    assert [r["coverage"] for r in rows][-1] == 1.0
    assert rows[0]["risk"] <= rows[-1]["risk"]
    assert all(a["threshold"] >= b["threshold"] for a, b in zip(rows, rows[1:]))
