"""Photo-quality gate and unknown-case clustering (numpy only)."""

import numpy as np
from PIL import Image, ImageFilter

from govi_ai.inference.clustering import cluster_cases, describe
from govi_ai.inference.quality import QualityThresholds, assess, measure


def _leafy(seed=0, size=(400, 300)):
    rng = np.random.default_rng(seed)
    a = (rng.random((size[1] // 10, size[0] // 10, 3)) * 255).astype(np.uint8)
    return Image.fromarray(a).resize(size, Image.Resampling.BICUBIC)


def test_good_photo_passes():
    problems, m = assess(_leafy(), QualityThresholds())
    assert problems == [] and m["min_side_px"] == 300


def test_bad_photos_are_named():
    t = QualityThresholds()
    assert "too_dark" in assess(Image.new("RGB", (400, 300), (5, 5, 5)), t)[0]
    assert "overexposed" in assess(Image.new("RGB", (400, 300), (252, 252, 252)), t)[0]
    assert "too_small" in assess(_leafy(size=(80, 60)), t)[0]
    blurred = _leafy().filter(ImageFilter.GaussianBlur(25))
    assert "blurry" in assess(blurred, t)[0] or "no_detail" in assess(blurred, t)[0]


def test_measure_is_resolution_stable():
    a, b = measure(_leafy(size=(1200, 900))), measure(_leafy(size=(400, 300)))
    assert abs(a["brightness"] - b["brightness"]) < 5


def test_thresholds_roundtrip(tmp_path):
    t = QualityThresholds(min_sharpness=7.5)
    t.save(tmp_path / "q.json")
    assert QualityThresholds.load(tmp_path / "q.json").min_sharpness == 7.5
    assert QualityThresholds.load(tmp_path / "missing.json") == QualityThresholds()


def test_clustering_finds_groups_and_noise():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 1, (3, 32))
    emb = np.vstack([centers[k] + rng.normal(0, 0.05, (10, 32)) for k in range(3)]
                    + [rng.normal(0, 1, (4, 32))])  # 4 isolated cases
    clusters, noise = cluster_cases(emb, threshold=0.9, min_size=3)
    assert len(clusters) == 3
    assert sorted(len(c) for c in clusters) == [10, 10, 10]
    assert len(noise) == 4
    info = describe(emb, clusters)
    assert all(d["cohesion"] > 0.9 for d in info)


def test_clustering_empty_and_tiny():
    c, n = cluster_cases(np.zeros((0, 8)))
    assert c == [] and len(n) == 0
    c, n = cluster_cases(np.eye(2), threshold=0.9, min_size=3)
    assert c == [] and sorted(n.tolist()) == [0, 1]
