"""Yield model on a small long-format table shaped like the DCS data."""

import numpy as np
import pandas as pd

from govi_ai.tabular.yield_model import backtest, build_table, forecast_table, history_table, ratio_quantiles


def _long(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d, level in (("POLONNARUWA", 5000), ("AMPARA", 4500), ("KANDY", 3000)):
        for season in ("maha", "yala"):
            for year in range(1990, 2021):
                y = level + 10 * (year - 1990) + rng.normal(0, 300)
                sown = 50000 + rng.normal(0, 2000)
                for w, share in (("major", 0.5), ("minor", 0.3), ("rainfed", 0.2)):
                    rows.append((d, season, str(year), year, w, "sown_extent_ha", sown * share))
                rows += [(d, season, str(year), year, "total", "sown_extent_ha", sown),
                         (d, season, str(year), year, "total", "harvested_extent_ha", sown * 0.97),
                         (d, season, str(year), year, "total", "avg_yield_kg_ha", y)]
    rows.append(("SRI LANKA", "maha", "2000", 2000, "total", "avg_yield_kg_ha", 4000.0))
    return pd.DataFrame(rows, columns=["district", "season", "season_label", "year", "water", "measure", "value"])


def test_build_table_lags_and_shares():
    t = build_table(_long())
    assert "SRI LANKA" not in set(t["district"])
    r = t[(t.district == "KANDY") & (t.season == "yala") & (t.year == 2000)].iloc[0]
    prev = t[(t.district == "KANDY") & (t.season == "yala") & (t.year == 1999)].iloc[0]
    assert r["lag1"] == prev["y"]
    assert abs(r["share_major"] - 0.5) < 1e-9 and abs(r["fail_lag1"] - 0.03) < 1e-9
    # previous season of Yala Y is Maha Y
    maha = t[(t.district == "KANDY") & (t.season == "maha") & (t.year == 2000)].iloc[0]
    assert r["prev_season"] == maha["y"]


def test_backtest_and_forecast():
    t = build_table(_long())
    res, s = backtest(t, first_year=2015)
    assert s["n"] == len(res) == 3 * 2 * 6
    assert s["climatology_5yr_mean"]["mae_kg_ha"] < 600
    assert 0.5 < s["climatology_5yr_mean"]["interval_80"]["coverage"] <= 1.0
    f = forecast_table(t)
    assert len(f) == 6 and all(x["year"] == 2021 for x in f)
    assert all(x["yield_kg_ha"]["p10"] < x["yield_kg_ha"]["p50"] < x["yield_kg_ha"]["p90"] for x in f)
    assert history_table(t)["AMPARA"]["maha"][2020] > 0


def test_ratio_quantiles_order():
    t = build_table(_long())
    lo, hi = ratio_quantiles(t.dropna(subset=["y", "mean5"]))
    assert lo < 1 < hi
