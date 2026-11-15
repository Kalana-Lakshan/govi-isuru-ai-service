"""EcoCrop suitability model with the real FAO ECOCROP parameters for rice, tea and chili."""

import pytest

from govi_ai.tabular.suitability import CropRequirements, recommend, score_crop, trapezoid

# Values as published in FAO ECOCROP (EcoCrop_DB.csv) for the three crops.
ECOCROP = {
    "rice": {"TOPMN": 20, "TOPMX": 30, "TMIN": 10, "TMAX": 36, "ROPMN": 1500, "ROPMX": 2000, "RMIN": 1000,
             "RMAX": 4000, "PHOPMN": 5.5, "PHOPMX": 7.0, "PHMIN": 4.5, "PHMAX": 9.0, "ALTMX": 2500,
             "DRA": "poorly (saturated >50% of year)", "DRAR": "poorly (saturated >50% of year)",
             "TEXT": "wide", "TEXTR": "wide"},
    "tea": {"TOPMN": 20, "TOPMX": 30, "TMIN": 8, "TMAX": 35, "ROPMN": 1400, "ROPMX": 2000, "RMIN": 1000,
            "RMAX": 5000, "PHOPMN": 4.5, "PHOPMX": 5.5, "PHMIN": 4.0, "PHMAX": 6.0, "ALTMX": 2200,
            "DRA": "well (dry spells)", "DRAR": "well (dry spells), excessive (dry/moderately dry)",
            "TEXT": "heavy, medium, light", "TEXTR": "heavy, medium, light"},
    "chili": {"TOPMN": 17, "TOPMX": 30, "TMIN": 8, "TMAX": 35, "ROPMN": 600, "ROPMX": 1250, "RMIN": 500,
              "RMAX": 1700, "PHOPMN": 5.5, "PHOPMX": 6.8, "PHMIN": 4.5, "PHMAX": 7.0, "ALTMX": 3000,
              "DRA": "well (dry spells)", "DRAR": "well (dry spells)", "TEXT": "medium, organic",
              "TEXTR": "medium, light"},
}


@pytest.fixture(scope="module")
def reqs():
    return {k: CropRequirements.from_ecocrop_row(k, v) for k, v in ECOCROP.items()}


def test_trapezoid():
    assert trapezoid(25, 10, 20, 30, 36) == 1
    assert trapezoid(15, 10, 20, 30, 36) == pytest.approx(0.5)
    assert trapezoid(33, 10, 20, 30, 36) == pytest.approx(0.5)
    assert trapezoid(40, 10, 20, 30, 36) == 0


def test_tea_highland_acid_soil_beats_rice(reqs):
    # Hill-country conditions: cool, wet, acidic, well-drained.
    r = recommend(reqs, season_rainfall_mm=1100, annual_rainfall_mm=2200, temperature_c=19, soil_ph=5.0,
                  drainage="good")
    assert r[0]["crop"] == "tea"
    rice = next(x for x in r if x["crop"] == "rice")
    assert rice["limiting_factor"] in ("soil_ph", "drainage")


def test_dry_zone_irrigated_rice(reqs):
    # Dry-zone Maha with irrigation: rainfall and drainage no longer limit rice.
    r = score_crop(reqs["rice"], temperature_c=27.5, rainfall_mm=900, soil_ph=6.5, irrigated=True, drainage="good")
    assert r["score"] == 1 and r["class"] == "S1"
    r2 = score_crop(reqs["rice"], temperature_c=27.5, rainfall_mm=900, soil_ph=6.5, irrigated=False, drainage="good")
    assert r2["class"] == "N" and r2["limiting_factor"] in ("rainfall_mm", "drainage")


def test_chili_limited_by_too_much_rain(reqs):
    r = score_crop(reqs["chili"], temperature_c=27, rainfall_mm=1600, soil_ph=6.2, drainage="good")
    assert r["limiting_factor"] == "rainfall_mm" and 0 < r["score"] < 0.4


def test_altitude_limit(reqs):
    assert score_crop(reqs["rice"], temperature_c=22, rainfall_mm=1700, soil_ph=6, altitude_m=2600)["score"] == 0
