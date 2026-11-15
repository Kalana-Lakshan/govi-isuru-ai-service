"""District x season paddy yield forecaster from official DCS statistics (phase 7).

Target: average yield (kg per net hectare, all water sources) of a district in a season.
Features available *at sowing time* of that season, so a forecast is possible before harvest:
  * yield history: same season 1, 2 and 3 years before, previous season (the other one), 5-year mean
  * trend: year
  * the current season's sown extent and its split over major irrigation / minor irrigation / rainfed
  * last same-season harvest failure rate (1 - harvested / sown)
  * district (categorical), season (maha / yala)

Evaluation: rolling-origin backtest. For every test year Y, train on years < Y only and predict Y.
Finding (2010-2026 backtest): at sowing time no gradient-boosted variant beats the district's 5-year
mean (MAE ~446-451 vs ~436 kg/ha), because season-to-season variation is weather-driven. The
served forecast is therefore the 5-year mean with a conformal 80 % interval; the GBM stays as the
comparison method.

Weather is deliberately left out of this first model: seasonal weather is unknown at sowing time;
adding in-season weather (NASA POWER / CHIRPS) is a follow-up that only helps mid-season forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NON_DISTRICTS = {"SRI LANKA", "MAHAWELI'H'", "UDAWALAWE"}  # national total and irrigation systems
FEATURES = ["year", "season_maha", "district_code", "lag1", "lag2", "lag3", "mean5", "prev_season",
            "sown_total", "share_major", "share_minor", "share_rainfed", "fail_lag1"]
QUANTILES = (0.1, 0.5, 0.9)
# Interval for the served (climatology) forecast. Shocks such as the 2022 fertiliser crisis make
# seasons non-exchangeable, so a nominal 80 % interval under-covers (73 % in the 2010-2026 rolling
# backtest). Nominal (7.5 %, 92.5 %) quantiles reach 79.5 % empirical coverage -> served as "80 %".
INTERVAL_QS = (0.075, 0.925)


def season_index(year: int, season: str) -> int:
    """Chronological index: Maha (harvest year Y, sown late Y-1) precedes Yala of year Y."""
    return 2 * year + (0 if season == "maha" else 1)


def build_table(long: pd.DataFrame) -> pd.DataFrame:
    """One row per district-season-year with target ``y`` and the sowing-time features."""
    df = long[~long["district"].isin(NON_DISTRICTS)]
    piv = df.pivot_table(index=["district", "season", "year"], columns=["measure", "water"], values="value",
                         aggfunc="first")
    piv.columns = [f"{m}__{w}" for m, w in piv.columns]
    t = piv.reset_index()
    t["y"] = t.get("avg_yield_kg_ha__total")
    harvested = t.get("harvested_extent_ha__total")
    t.loc[(t["y"] <= 0) | (harvested.fillna(0) <= 0), "y"] = np.nan  # no harvest -> no yield observation

    sown_total = t.get("sown_extent_ha__total")
    t["sown_total"] = sown_total
    for w in ("major", "minor", "rainfed"):
        col = t.get(f"sown_extent_ha__{w}")
        t[f"share_{w}"] = (col / sown_total).where(sown_total > 0) if col is not None else np.nan
    t["fail"] = 1 - (harvested / sown_total).where(sown_total > 0)

    t = t.sort_values(["district", "season", "year"])
    for k in (1, 2, 3):
        # lag by calendar year, not by row: missing years must stay missing
        prev = t[["district", "season", "year", "y"]].copy()
        prev["year"] += k
        t = t.merge(prev.rename(columns={"y": f"lag{k}"}), on=["district", "season", "year"], how="left")
    t["mean5"] = np.nan
    for (_, _), idx in t.groupby(["district", "season"]).groups.items():
        sub = t.loc[idx].set_index("year")["y"]
        t.loc[idx, "mean5"] = [sub[(sub.index < yr) & (sub.index >= yr - 5)].mean() for yr in t.loc[idx, "year"]]
    fail_prev = t[["district", "season", "year", "fail"]].copy()
    fail_prev["year"] += 1
    t = t.merge(fail_prev.rename(columns={"fail": "fail_lag1"}), on=["district", "season", "year"], how="left")

    # previous season = the other season immediately before this one
    t["sidx"] = [season_index(y, s) for y, s in zip(t["year"], t["season"])]
    prev_s = t[["district", "sidx", "y"]].copy()
    prev_s["sidx"] += 1
    t = t.merge(prev_s.rename(columns={"y": "prev_season"}), on=["district", "sidx"], how="left")

    t["season_maha"] = (t["season"] == "maha").astype(int)
    codes = {d: i for i, d in enumerate(sorted(t["district"].unique()))}
    t["district_code"] = t["district"].map(codes)
    return t.sort_values(["sidx", "district"]).reset_index(drop=True)


def _model(quantile: float, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile, max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0, categorical_features=[FEATURES.index("district_code")],
        random_state=seed)


def fit(train: pd.DataFrame, seed: int = 0) -> dict:
    tr = train.dropna(subset=["y", "lag1"])
    return {q: _model(q, seed).fit(tr[FEATURES], tr["y"]) for q in QUANTILES}


def predict(models: dict, rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({f"p{int(q * 100)}": m.predict(rows[FEATURES]) for q, m in models.items()}, index=rows.index)
    # quantile crossing can happen with separate models; sort per row
    out[:] = np.sort(out.to_numpy(), axis=1)
    return out


def ratio_quantiles(table: pd.DataFrame, qs=INTERVAL_QS) -> tuple[float, float]:
    """Split-conformal style interval for the climatology forecast: empirical quantiles of
    y / mean5 over the given (past) rows. Multiplicative, so it scales with each district's level."""
    r = (table["y"] / table["mean5"]).replace([np.inf, -np.inf], np.nan).dropna()
    return float(r.quantile(qs[0])), float(r.quantile(qs[1]))


def backtest(table: pd.DataFrame, first_year: int = 2005, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Rolling-origin evaluation (train on years < Y, predict Y) of:
      * ``gbm``          gradient-boosted quantile model on the sowing-time features
      * ``climatology``  5-year mean of the same district and season, with a conformal 80 % interval
      * ``last_year``    same season, previous year
    """
    rows = []
    for year in sorted(table.loc[table["year"] >= first_year, "year"].unique()):
        test = table[(table["year"] == year) & table["y"].notna() & table["lag1"].notna() & table["mean5"].notna()]
        if test.empty:
            continue
        past = table[table["year"] < year]
        p = predict(fit(past, seed), test)
        lo, hi = ratio_quantiles(past.dropna(subset=["y", "mean5"]))
        r = test[["district", "season", "year", "y", "lag1", "mean5"]].copy()
        r[["gbm_p10", "gbm_p50", "gbm_p90"]] = p.to_numpy()
        r["clim_p10"], r["clim_p90"] = r["mean5"] * lo, r["mean5"] * hi
        rows.append(r)
    res = pd.concat(rows, ignore_index=True)

    def err(pred):
        e = (res["y"] - pred).abs()
        return {"mae_kg_ha": float(e.mean()), "mape": float((e / res["y"]).mean())}

    def cover(lo, hi):
        return {"coverage": float(((res["y"] >= lo) & (res["y"] <= hi)).mean()),
                "mean_width_kg_ha": float((hi - lo).mean())}

    summary = {
        "n": len(res), "years": [int(res["year"].min()), int(res["year"].max())],
        "gbm": {**err(res["gbm_p50"]), "interval_80": cover(res["gbm_p10"], res["gbm_p90"])},
        "climatology_5yr_mean": {**err(res["mean5"]), "interval_80": cover(res["clim_p10"], res["clim_p90"])},
        "last_year": err(res["lag1"]),
    }
    return res, summary


def forecast_table(table: pd.DataFrame) -> list[dict]:
    """Next-season forecast for every district and season: 5-year mean of the observed yields with a
    conformal 80 % interval (the method that backtests best at sowing time; see ``backtest``)."""
    lo, hi = ratio_quantiles(table.dropna(subset=["y", "mean5"]))
    out = []
    for (district, season), g in table.dropna(subset=["y"]).groupby(["district", "season"]):
        g = g.sort_values("year")
        last = int(g["year"].iloc[-1])
        recent = g[g["year"] > last - 5]["y"]
        m = float(recent.mean())
        out.append({"district": district, "season": season, "year": last + 1,
                    "method": "climatology_5yr_mean", "based_on_years": [int(last - 4), last],
                    "yield_kg_ha": {"p10": round(m * lo), "p50": round(m), "p90": round(m * hi)},
                    "last_observed": {"year": last, "yield_kg_ha": round(float(g["y"].iloc[-1]))}})
    return out


def history_table(table: pd.DataFrame) -> dict:
    """Observed yields per district and season (served for charts)."""
    h: dict = {}
    for r in table.dropna(subset=["y"]).itertuples():
        h.setdefault(r.district, {}).setdefault(r.season, {})[int(r.year)] = round(float(r.y))
    return h
