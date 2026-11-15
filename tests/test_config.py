import pytest

from govi_ai.data.registry import UnmappedLabelError, load_registry, norm_label
from govi_ai.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture(scope="module")
def reg(tax):
    return load_registry(tax)


def test_taxonomy_loads(tax):
    assert tax.gate_classes == ["rice", "tea", "chili", "other_plant", "non_plant"]
    for crop in ("rice", "tea", "chili"):
        assert "healthy" in tax.known_classes(crop)
        assert set(tax.known_classes(crop)).isdisjoint(tax.unknown_eval_classes(crop))


def test_every_crop_has_unknown_eval_classes(tax):
    # Needed for the unknown-disease evaluation protocol.
    for crop in tax.crop_names:
        assert tax.unknown_eval_classes(crop), crop


def test_registry_validates_against_taxonomy(reg):
    ids = [s.id for s in reg.sources]
    assert len(ids) == len(set(ids))
    assert "paddy_doctor" in ids and "csd_assam" in ids


def test_no_augmented_configs(reg):
    for s in reg.sources:
        assert str(s.fetch.get("config", "")).lower() != "augmented", s.id


def test_norm_label():
    assert norm_label("Leaf_Red Rust") == norm_label("leaf-red-rust") == "leafredrust"


def test_resolve_known_unknown_and_drop(reg, tax):
    paddy = reg.get("paddy_doctor")
    r = paddy.resolve("normal", tax)
    assert (r.role, r.crop, r.label) == ("known", "rice", "healthy")

    uci = reg.get("uci_rice")
    assert uci.resolve("leaf_smut", tax).role == "unknown_eval"

    bd = reg.get("rice_bangladesh")
    assert bd.resolve("Insect", tax) is None          # dropped on purpose
    assert bd.resolve("rice leaffolder", tax).label == "leaffolder"  # case/space-insensitive

    with pytest.raises(UnmappedLabelError):
        paddy.resolve("not_a_label", tax)


def test_resolve_negatives(reg, tax):
    pv = reg.get("op_plantvillage")
    assert pv.resolve("Pepper,_bell___healthy", tax) is None
    assert pv.resolve("Background_without_leaves", tax).role == "non_plant"
    r = pv.resolve("Tomato___healthy", tax)
    assert r.role == "other_plant" and r.crop is None and r.label.startswith("op_plantvillage:")
