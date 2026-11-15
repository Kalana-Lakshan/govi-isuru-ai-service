import numpy as np
import pytest
from sklearn import metrics as skm

from govi_ai.eval import metrics as M


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 500)
    p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 5, 500))
    return y, p


def test_macro_f1_matches_sklearn(data):
    y, p = data
    assert M.macro_f1(y, p, 5) == pytest.approx(skm.f1_score(y, p, average="macro"))


def test_macro_f1_present_only_ignores_absent_classes():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 0, 1, 2])  # class 2 never true
    assert M.macro_f1(y, p, 3, present_only=True) == pytest.approx(
        skm.f1_score(y, p, labels=[0, 1], average="macro"))


def test_balanced_accuracy_matches_sklearn(data):
    y, p = data
    assert M.balanced_accuracy(y, p, 5) == pytest.approx(skm.balanced_accuracy_score(y, p))


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    k = np.round(rng.normal(1, 1, 300), 1)
    u = np.round(rng.normal(0, 1, 200), 1)
    y = np.r_[np.ones_like(k), np.zeros_like(u)]
    assert M.auroc(k, u) == pytest.approx(skm.roc_auc_score(y, np.r_[k, u]))


def test_fpr_at_tpr_perfect_and_random():
    assert M.fpr_at_tpr(np.ones(100), np.zeros(100)) == 0.0
    rng = np.random.default_rng(2)
    s = rng.random(20000)
    assert M.fpr_at_tpr(s[:10000], s[10000:]) == pytest.approx(0.95, abs=0.02)


def test_ece_perfectly_calibrated_is_small():
    rng = np.random.default_rng(3)
    conf = rng.uniform(0.5, 1, 20000)
    correct = rng.random(20000) < conf
    probs = np.stack([conf, 1 - conf], 1)
    y = np.where(correct, 0, 1)
    assert M.ece(probs, y) < 0.02


def test_oscr_bounds():
    k = np.array([0.9, 0.8, 0.7])
    assert M.oscr(k, [True, True, True], np.array([0.1, 0.2])) == pytest.approx(1.0)
    assert M.oscr(k, [False, False, False], np.array([0.1])) == pytest.approx(0.0)


def test_risk_coverage():
    conf = np.array([0.9, 0.8, 0.1])
    correct = np.array([1, 1, 0])
    assert M.risk_at_coverage(conf, correct, 2 / 3) == 0.0
    assert M.risk_at_coverage(conf, correct, 1.0) == pytest.approx(1 / 3)
