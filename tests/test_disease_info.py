"""configs/disease_info.yaml: every entry is a real class, and carried-over advice reaches the catalogue."""

import yaml

from govi_ai.bundle import class_catalog
from govi_ai.taxonomy import CONFIG_DIR, load_taxonomy


def test_entries_match_taxonomy_and_advice_is_served():
    tax = load_taxonomy()
    info = yaml.safe_load((CONFIG_DIR / "disease_info.yaml").read_text(encoding="utf-8"))["crops"]
    for crop, classes in info.items():
        assert set(classes) <= set(tax.crops[crop]), (crop, set(classes) - set(tax.crops[crop]))
        for cid, row in classes.items():
            assert row.get("severity", "") in ("", "none", "low", "medium", "high"), (crop, cid)
            if row.get("advice_en"):
                assert row.get("source"), f"{crop}/{cid}: advice without a source"
    cat = class_catalog(tax)
    blast = cat["crops"]["rice"]["blast"]
    assert blast["name"]["si"] and "Tricyclazole" in blast["advice"]["en"]
    assert blast["severity"] == "high" and blast["description"]["en"]
