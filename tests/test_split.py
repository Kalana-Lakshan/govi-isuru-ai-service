from collections import Counter, defaultdict

from govi_ai.data.split import add_lodo_columns, in_domain_split, lodo_split


def _rows():
    rows = []
    gid = 0
    for src in ("s1", "s2", "s3"):
        for label in ("healthy", "blast", "brown_spot"):
            for k in range(60):
                # every 5th image has a duplicate partner in the same group
                gid += 0 if k % 5 == 1 else 1
                rows.append({"source": src, "role": "known", "crop": "rice", "label": label,
                             "group_id": f"g{gid}"})
    for k in range(40):
        rows.append({"source": "s1", "role": "unknown_eval", "crop": "rice", "label": "leaf_smut",
                     "group_id": f"u{k}"})
    for k in range(200):
        rows.append({"source": "neg", "role": "non_plant", "crop": "", "label": "neg:coco",
                     "group_id": f"n{k}"})
    return rows


def test_in_domain_split_is_group_safe_and_proportional():
    rows = _rows()
    in_domain_split(rows, seed=0)
    by_group = defaultdict(set)
    for r in rows:
        by_group[r["group_id"]].add(r["split"])
    assert all(len(v) == 1 for v in by_group.values())
    known = Counter(r["split"] for r in rows if r["role"] == "known")
    n = sum(known.values())
    assert 0.10 < known["test"] / n < 0.20
    assert 0.10 < known["val"] / n < 0.20
    assert all(r["split"] == "unknown_eval" for r in rows if r["role"] == "unknown_eval")


def test_train_only_rows_never_reach_val_or_test():
    rows = _rows()
    for r in rows:
        if r["source"] == "s3":
            r["train_only"] = "True"
    in_domain_split(rows, seed=0)
    assert {r["split"] for r in rows if r["source"] == "s3" and r["role"] == "known"} == {"train"}
    out = lodo_split(rows, "s1", seed=0)
    assert {v for r, v in zip(rows, out) if r["source"] == "s3" and r["role"] == "known"} == {"train"}
    # ...but a train_only source can still be the held-out LODO test set
    out = lodo_split(rows, "s3", seed=0)
    assert {v for r, v in zip(rows, out) if r["source"] == "s3" and r["role"] == "known"} == {"test"}


def test_lodo_holds_out_one_source():
    rows = _rows()
    in_domain_split(rows, seed=0)
    out = lodo_split(rows, "s2", seed=0)
    for r, v in zip(rows, out):
        if r["source"] == "s2" and r["role"] == "known":
            assert v == "test"
        elif r["source"] in ("s1", "s3") and r["role"] == "known":
            assert v in ("train", "val")
        elif r["role"] == "unknown_eval":
            assert v == "unknown_eval"
        elif r["role"] == "non_plant":
            assert v in ("train", "val")  # negatives never act as a test set here


def test_add_lodo_columns_skips_small_sources():
    rows = _rows()
    in_domain_split(rows, seed=0)
    cols = add_lodo_columns(rows, ["s1", "s2", "s3"], min_images=100)
    assert cols == ["lodo_s1", "lodo_s2", "lodo_s3"]
    assert add_lodo_columns(rows, ["s1"], min_images=10_000) == []
