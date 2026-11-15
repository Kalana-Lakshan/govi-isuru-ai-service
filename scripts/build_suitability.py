"""Phase 7: build the crop-suitability files from real data (no training, no synthetic rows).

    python scripts/build_suitability.py --out bundle/suitability

Writes:
  crops.json      FAO ECOCROP requirements for rice, tea and chili
  districts.json  NASA POWER climatology per district reference point: mean temperature, annual,
                  Maha (Oct-Mar) and Yala (Apr-Sep) rainfall, grid-cell elevation
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from govi_ai import CONFIG_DIR  # noqa: E402
from govi_ai.tabular.suitability import CROPS, CropRequirements  # noqa: E402

ECOCROP_URL = "https://raw.githubusercontent.com/OpenCLIM/ecocrop/main/EcoCrop_DB.csv"
POWER_URL = ("https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=T2M,PRECTOTCORR"
             "&community=AG&longitude={lon}&latitude={lat}&format=JSON")
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
DAYS = dict(zip(MONTHS, [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))
MAHA = ["OCT", "NOV", "DEC", "JAN", "FEB", "MAR"]
YALA = ["APR", "MAY", "JUN", "JUL", "AUG", "SEP"]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "research-data-client"})
    with urllib.request.urlopen(req, timeout=90) as r:  # noqa: S310 - fixed public hosts
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ecocrop-csv", type=Path, help="local EcoCrop_DB.csv (default: download)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw = args.ecocrop_csv.read_bytes() if args.ecocrop_csv else _get(ECOCROP_URL)
    db = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    crops = {}
    for crop, (sci, _) in CROPS.items():
        rows = db[db["ScientificName"] == sci]
        if len(rows) != 1:
            raise SystemExit(f"expected exactly one ECOCROP row for {sci}, found {len(rows)}")
        crops[crop] = CropRequirements.from_ecocrop_row(crop, rows.iloc[0].to_dict()).to_dict()
    meta = {"built": date.today().isoformat(),
            "crop_source": "FAO ECOCROP database (via github.com/OpenCLIM/ecocrop EcoCrop_DB.csv)",
            "climate_source": "NASA POWER climatology API (power.larc.nasa.gov), community AG"}
    (args.out / "crops.json").write_text(json.dumps({**meta, "crops": crops}, indent=2), encoding="utf-8")

    points = yaml.safe_load((CONFIG_DIR / "districts.yaml").read_text(encoding="utf-8"))["districts"]
    districts = {}
    for name, p in points.items():
        d = json.loads(_get(POWER_URL.format(lat=p["lat"], lon=p["lon"])))
        t2m, pr = d["properties"]["parameter"]["T2M"], d["properties"]["parameter"]["PRECTOTCORR"]
        mm = {m: pr[m] * DAYS[m] for m in MONTHS}
        districts[name.upper()] = {
            "lat": p["lat"], "lon": p["lon"], "temperature_c": round(t2m["ANN"], 1),
            "annual_rainfall_mm": round(sum(mm.values())),
            "maha_rainfall_mm": round(sum(mm[m] for m in MAHA)), "yala_rainfall_mm": round(sum(mm[m] for m in YALA)),
            "elevation_m_grid": round(d["geometry"]["coordinates"][2]),
        }
        time.sleep(0.5)  # be gentle with the public API
    (args.out / "districts.json").write_text(json.dumps({**meta, "districts": districts}, indent=1), encoding="utf-8")
    print(json.dumps({k: crops[k]["params"] for k in crops}, indent=1))
    print(f"{len(districts)} districts -> {args.out}")


if __name__ == "__main__":
    main()
