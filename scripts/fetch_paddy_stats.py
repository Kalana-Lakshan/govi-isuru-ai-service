"""Download the official DCS paddy statistics (1978/79 – 2025) into tidy CSVs.

    python scripts/fetch_paddy_stats.py --out data/tabular

Writes paddy_dcs_long.csv (every measure x water source) and paddy_dcs_wide.csv (totals, one row
per district-season) plus a sanity report: yield x harvested extent should match production.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from govi_ai.tabular.paddy_stats import BASE, SERIES, fetch_all, to_wide  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/tabular"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    long = fetch_all()
    long.to_csv(args.out / "paddy_dcs_long.csv", index=False)
    wide = to_wide(long)
    wide.to_csv(args.out / "paddy_dcs_wide.csv", index=False)

    # Consistency: DCS computes production = NET harvested extent x yield per net hectare, while the
    # extent series are gross (incl. bunds). production / (yield x gross extent) should therefore be a
    # stable fraction below 1 (the net/gross ratio); a unit error would show up as ~0.001 or ~1000.
    w = wide.dropna(subset=["avg_yield_kg_ha", "harvested_extent_ha", "production_mt"])
    w = w[(w["production_mt"] > 0) & (w["district"] != "SRI LANKA")]
    ratio = w["production_mt"] / (w["avg_yield_kg_ha"] * w["harvested_extent_ha"] / 1000)
    report = {
        "source": BASE, "series": {f"{m}/{s}": slug for (m, s), slug in SERIES.items()},
        "fetched": date.today().isoformat(),
        "rows_long": len(long), "rows_wide": len(wide),
        "districts": sorted(wide["district"].unique().tolist()),
        "years": [int(wide["year"].min()), int(wide["year"].max())],
        "net_to_gross_check": {"n": len(w), "median_ratio": float(ratio.median()),
                               "p05": float(ratio.quantile(0.05)), "p95": float(ratio.quantile(0.95))},
    }
    (args.out / "paddy_dcs_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "series"}, indent=1))


if __name__ == "__main__":
    main()
