"""DCS sheet parsing on small frames that mirror the two real layouts."""

import numpy as np
import pandas as pd

from govi_ai.tabular.paddy_stats import parse_sheet, to_wide

nan = np.nan


def test_parse_sheet_with_water_columns_and_header_typo():
    raw = pd.DataFrame([
        [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
        [nan, "District", "1978/1979", nan, nan, nan, "2000/1980", nan, nan, nan],  # typo in 2nd label
        [nan, nan, "Major", "Minor", "Rainfed", "Total", "Major", "Minor", "Rainfed", "Total"],
        [nan, "COLOMBO", "-", "-", "-", "2038", "-", "-", "-", "2,713"],
        [nan, "POLONNARUWA", "5000", "4000", "-", "4800", "5100", "4100", "-", "4900"],
        [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    ])
    df = parse_sheet(raw, "avg_yield_kg_ha", "maha")
    assert set(df["season_label"]) == {"1978/1979", "1979/1980"}
    assert set(df["year"]) == {1979, 1980}
    col = df[(df.district == "COLOMBO") & (df.year == 1980)]
    assert col["water"].tolist() == ["total"] and col["value"].tolist() == [2713.0]
    assert len(df[df.district == "POLONNARUWA"]) == 6  # rainfed "-" skipped


def test_parse_totals_only_sheet_scales_production_and_fixes_year():
    raw = pd.DataFrame([
        [nan, nan, nan, nan],
        ["DISTRICT", 1979, 198, 1981],  # "198" is a typo for 1980 in the real Yala sheet
        ["COLOMBO", 6.1, 10.3, 12],
    ])
    df = parse_sheet(raw, "production_mt", "yala")
    assert df["year"].tolist() == [1979, 1980, 1981]
    assert df["season_label"].tolist() == ["1979", "1980", "1981"]
    assert df["value"].tolist() == [6100.0, 10300.0, 12000.0]  # thousand t -> t
    assert set(df["water"]) == {"total"}


def test_to_wide():
    long = pd.DataFrame([
        ("A", "yala", "2020", 2020, "total", "avg_yield_kg_ha", 4000.0),
        ("A", "yala", "2020", 2020, "total", "production_mt", 100.0),
        ("A", "yala", "2020", 2020, "major", "production_mt", 60.0),
    ], columns=["district", "season", "season_label", "year", "water", "measure", "value"])
    w = to_wide(long)
    assert len(w) == 1 and w.loc[0, "production_mt"] == 100.0 and w.loc[0, "avg_yield_kg_ha"] == 4000.0
