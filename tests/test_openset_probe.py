"""Open-set scores and the probe pipeline on well-separated Gaussian clusters (test fixtures)."""

import numpy as np
import pandas as pd
import pytest

from govi_ai.eval import metrics as M
from govi_ai.openset import Mahalanobis, energy, fit_temperature, knn_score, max_logit, msp, softmax
from govi_ai.probe import evaluate_crop, evaluate_gate


def test_energy_is_logsumexp_and_orders_like_max_logit_for_peaked_logits():
    z = np.array([[10.0, 0, 0], [2.0, 1.9, 1.8]])
    np.testing.assert_allclose(energy(z), np.log(np.exp(z).sum(1)))
    assert msp(z)[0] > msp(z)[1] and max_logit(z)[0] > max_logit(z)[1]


def test_mahalanobis_and_knn_separate_unknown_cluster():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 4, (3, 16))
    y = rng.integers(0, 3, 600)
    X = centers[y] + rng.normal(0, 1, (600, 16))
    unk = rng.normal(0, 4, (1, 16)) + 12 + rng.normal(0, 1, (200, 16))
    maha = Mahalanobis().fit(X, y)
    assert M.auroc(maha.score(X), maha.score(unk)) > 0.99
    assert M.auroc(knn_score(X, X[:300]), knn_score(X, unk)) > 0.95


def test_prototype_library_adds_class_without_refit():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (300, 8))
    y = np.zeros(300, int)
    maha = Mahalanobis().fit(X, y)
    new = rng.normal(8, 1, (50, 8))
    before = maha.score(new).mean()
    maha.add_class(1, new[:20])
    assert maha.score(new[20:]).mean() > before  # new disease now looks "known"


def test_fit_temperature_recovers_scale():
    rng = np.random.default_rng(2)
    true_logits = rng.normal(0, 2, (4000, 4))
    y = np.array([rng.choice(4, p=p) for p in softmax(true_logits)])
    t = fit_temperature(true_logits * 3.0, y)  # over-confident by 3x
    assert 2.0 < t < 4.5


def _fake_run(seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, (12, 32))
    rows, feats = [], []

    def add(n, center, **r):
        for _ in range(n):
            rows.append(r)
            feats.append(centers[center] + rng.normal(0, 1, 32))

    k = 0
    for crop, labels in {"rice": ["healthy", "blast", "brown_spot"], "tea": ["healthy", "blister_blight"]}.items():
        for lab in labels:
            for split, n in (("train", 60), ("val", 15), ("test", 15)):
                add(n, k, crop=crop, label=lab, role="known", source="s1", split=split)
            k += 1
        add(20, k, crop=crop, label="mystery", role="unknown_eval", source="s1", split="unknown_eval")
        k += 1
    for split, n in (("train", 60), ("val", 15), ("test", 15)):
        add(n, k, crop="", label="op:x", role="other_plant", source="op", split=split)
        add(n, k + 1, crop="", label="np:x", role="non_plant", source="np", split=split)
    df = pd.DataFrame(rows)
    df["sample_id"] = [f"s{i}" for i in range(len(df))]
    return df, np.asarray(feats)


def test_evaluate_crop_end_to_end():
    df, X = _fake_run()
    r = evaluate_crop(df, X, "rice", "split", ["healthy", "blast", "brown_spot", "tungro"])
    assert r["classes"] == ["healthy", "blast", "brown_spot"]  # tungro has no training rows
    assert r["closed_set"]["macro_f1"] > 0.95
    assert set(r["open_set"]) == {"unknown_disease", "other_crops", "other_plant", "non_plant"}
    assert r["open_set"]["unknown_disease"]["mahalanobis"]["auroc"] > 0.9


def test_evaluate_gate_end_to_end():
    df, X = _fake_run()
    g = evaluate_gate(df, X, ["rice", "tea", "chili", "other_plant", "non_plant"])
    assert g["accuracy"] > 0.9
    assert g["wrong_crop_rate"] < 0.1


def test_extract_features_with_tiny_random_backbone(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from PIL import Image

    from govi_ai.features import extract, load_features, save_features
    from govi_ai.models.backbones import create_backbone

    paths = []
    for i in range(5):
        p = tmp_path / f"{i}.jpg"
        Image.fromarray((np.random.default_rng(i).random((40, 30, 3)) * 255).astype(np.uint8)).save(p)
        paths.append(p)
    model, mean, std = create_backbone("resnet18", pretrained=False)
    f = extract(model, paths, mean, std, img_size=64, batch_size=2, num_workers=0, device="cpu")
    assert f.shape == (5, 512) and f.dtype == np.float16
    save_features(tmp_path / "r18.npz", f, [f"s{i}" for i in range(5)], {"backbone": "resnet18"})
    X = load_features(tmp_path, ["r18", "r18"], ["s4", "s0"])
    assert X.shape == (2, 1024)
    np.testing.assert_allclose(np.linalg.norm(X[:, :512], axis=1), 1, atol=1e-3)
