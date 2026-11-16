"""RQ3: how well does the human-in-the-loop new-disease workflow work, without retraining?

Uses a trained student's cached embeddings (outputs.npz from eval_student.py). Two experiments:

A. Few-shot prototype addition ("leave-one-class-out"). For every known class c with enough images:
   * fit class prototypes + shared covariance on the *other* classes' training embeddings and set the
     unknown threshold at 95 % known-class TPR on validation (exactly as calibrate.py does);
   * before: share of c's test images flagged UNKNOWN (= would a genuinely new disease be caught?);
   * after an officer registers c from k confirmed photos (k = 1, 3, 5, 10, 20; random, 5 seeds):
     recall of c (assigned to c), false acceptance (other classes assigned to c), accuracy on the old classes.

B. Clustering of real unseen disorders (the unknown_eval classes, e.g. tea anthracnose, chili BBWV2):
   purity, adjusted Rand index and share grouped, over a grid of cosine thresholds; this calibrates the
   default threshold of POST /v1/disease/unknowns/cluster.

    python scripts/officer_loop_experiment.py --data-root /content/data --run $GOVI/runs/mnv4_s_split --out $GOVI/officer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from govi_ai.inference.clustering import cluster_cases  # noqa: E402
from govi_ai.manifest import load_manifest  # noqa: E402
from govi_ai.openset import Mahalanobis  # noqa: E402

SHOTS = (1, 3, 5, 10, 20)
THRESHOLDS = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))


def few_shot(df, emb, crop: str, protocol: str, seeds: int, min_train: int, min_test: int) -> dict:
    known = ((df["crop"] == crop) & (df["role"] == "known")).to_numpy()
    lab = df["label"].to_numpy()
    tr = known & (df[protocol] == "train").to_numpy()
    va = known & (df[protocol] == "val").to_numpy()
    te = known & (df[protocol] == "test").to_numpy()
    classes = [c for c in sorted(set(lab[tr])) if (lab[tr] == c).sum() >= min_train and (lab[te] == c).sum() >= min_test]
    out = {}
    for c in classes:
        others = sorted(set(lab[tr]) - {c})
        idx = {k: i for i, k in enumerate(others)}
        m_tr, m_va = tr & (lab != c), va & (lab != c)
        maha = Mahalanobis().fit(emb[m_tr], np.array([idx[x] for x in lab[m_tr]]))
        thr = float(np.quantile(maha.score(emb[m_va]), 0.05))  # 95 % TPR on known classes
        te_c, te_o = te & (lab == c), te & (lab != c)
        before = float((maha.score(emb[te_c]) < thr).mean())
        res = {"n_test": int(te_c.sum()), "flagged_unknown_before": round(before, 4), "shots": {}}
        pool = np.flatnonzero(tr & (lab == c))
        for k in SHOTS:
            if k > len(pool):
                continue
            rec, fa, old = [], [], []
            for s in range(seeds):
                rng = np.random.default_rng(s)
                m = Mahalanobis().fit(emb[m_tr], np.array([idx[x] for x in lab[m_tr]]))
                m.add_class(len(others), emb[rng.choice(pool, k, replace=False)])
                names = np.array(others + [c])

                def assign(rows):
                    d = m.distances(emb[rows])
                    a = names[d.argmin(1)].astype(object)
                    a[-d.min(1) < thr] = "UNKNOWN"
                    return a

                a_c, a_o = assign(np.flatnonzero(te_c)), assign(np.flatnonzero(te_o))
                rec.append(np.mean(a_c == c))
                fa.append(np.mean(a_o == c))
                old.append(np.mean(a_o == lab[te_o]))
            res["shots"][str(k)] = {"recall_new": round(float(np.mean(rec)), 4),
                                    "false_accept_old_as_new": round(float(np.mean(fa)), 4),
                                    "old_class_accuracy": round(float(np.mean(old)), 4)}
        out[c] = res
    return out


def clustering_eval(df, emb, crop: str) -> dict:
    unk = ((df["crop"] == crop) & (df["role"] == "unknown_eval")).to_numpy()
    labels = df.loc[unk, "label"].to_numpy()
    if unk.sum() < 10 or len(set(labels)) < 2:
        return {"skipped": f"{int(unk.sum())} unknown images, {len(set(labels))} disorders"}
    e = emb[unk]
    res = {"disorders": {k: int((labels == k).sum()) for k in sorted(set(labels))}, "thresholds": {}}
    for t in THRESHOLDS:
        groups, noise = cluster_cases(e, threshold=t, min_size=3)
        pred = np.full(len(e), -1)
        for g, members in enumerate(groups):
            pred[members] = g
        grouped = pred >= 0
        purity = (sum(np.bincount(np.unique(labels, return_inverse=True)[1][m]).max() for m in groups) / grouped.sum()
                  if grouped.any() else 0.0)
        res["thresholds"][str(t)] = {"groups": len(groups), "share_grouped": round(float(grouped.mean()), 4),
                                     "purity": round(float(purity), 4),
                                     "ari": round(adjusted_rand(labels[grouped], pred[grouped]), 4) if grouped.sum() > 1 else 0.0}
    best = max(res["thresholds"].items(), key=lambda kv: kv[1]["ari"] * kv[1]["share_grouped"])
    res["recommended_threshold"] = float(best[0])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--min-train", type=int, default=30)
    ap.add_argument("--min-test", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    z = np.load(args.run / "outputs.npz")
    cfg = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    protocol = cfg["protocol"]
    df = load_manifest(args.data_root, paper_only=not cfg.get("all_sources", False))
    df = df.set_index("sample_id").loc[z["sample_ids"].tolist()].reset_index()
    emb = z["emb"].astype(np.float64)

    report = {"run": str(args.run), "protocol": protocol, "few_shot": {}, "clustering": {}}
    for crop in sorted(set(df["crop"]) - {""}):
        report["few_shot"][crop] = few_shot(df, emb, crop, protocol, args.seeds, args.min_train, args.min_test)
        report["clustering"][crop] = clustering_eval(df, emb, crop)
        fs = report["few_shot"][crop]
        if fs:
            k5 = [v["shots"].get("5", {}).get("recall_new") for v in fs.values() if "5" in v["shots"]]
            print(f"{crop}: {len(fs)} classes, mean new-class recall at 5 shots = {np.mean(k5):.3f}; "
                  f"flagged unknown before = {np.mean([v['flagged_unknown_before'] for v in fs.values()]):.3f}")
    (args.out / "officer_loop.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out / 'officer_loop.json'}")


if __name__ == "__main__":
    main()
