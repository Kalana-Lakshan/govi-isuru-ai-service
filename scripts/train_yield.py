"""Phase 7: backtest yield models on official DCS data and write the serving files.

    python scripts/fetch_paddy_stats.py --out data/tabular
    python scripts/train_yield.py --data data/tabular/paddy_dcs_long.csv --out bundle/yield

Writes:
  backtest.json   rolling-origin comparison (GBM vs 5-year climatology vs last year) for the paper
  forecasts.json  next-season forecast per district x season (served by /v1/yield/forecast)
  history.json    observed yields per district x season (served by /v1/yield/history)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from govi_ai.tabular.paddy_stats import BASE  # noqa: E402
from govi_ai.tabular.yield_model import backtest, build_table, forecast_table, history_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="paddy_dcs_long.csv from fetch_paddy_stats.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-test-year", type=int, default=2010)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table = build_table(pd.read_csv(args.data))
    res, summary = backtest(table, first_year=args.first_test_year)
    res.to_csv(args.out / "backtest_predictions.csv", index=False)
    meta = {"source": f"Department of Census and Statistics, Sri Lanka ({BASE})", "built": date.today().isoformat(),
            "unit": "kg per net hectare", "interval": "80 % (empirically calibrated in the rolling backtest)"}
    (args.out / "backtest.json").write_text(json.dumps({**meta, **summary}, indent=2), encoding="utf-8")
    (args.out / "forecasts.json").write_text(json.dumps({**meta, "forecasts": forecast_table(table)}, indent=1),
                                             encoding="utf-8")
    (args.out / "history.json").write_text(json.dumps({**meta, "history": history_table(table)}), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
