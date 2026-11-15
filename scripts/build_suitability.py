"""Phase 7: build the crop-suitability files from real data (no training, no synthetic rows).

    python scripts/build_suitability.py --out bundle/suitability

Writes:
  crops.json      FAO ECOCROP requirements for rice, tea and chili
  districts.json  1995-2024 ERA5 climatology per district reference point (Open-Meteo archive API, which
                  downscales temperature to a 90 m DEM): mean temperature, annual / Maha (Oct-Mar) / Yala
                  (Apr-Sep) rainfall, dry months (< 60 mm), elevation.

Why ERA5 and not NASA POWER: POWER's ~0.5 deg grid flattened the hill country (Nuwara Eliya 22.7 C at 681 m
vs ~16 C at 1,868 m) and put 2,225 mm/yr on dry-zone Hambantota (~1,000 mm). ERA5 via Open-Meteo gave
Hambantota 1,034 mm, Colombo 2,614 mm, Nuwara Eliya 2,533 mm. Known weakness: it under-estimates some
dry-zone stations (Polonnaruwa 1,190 mm vs ~1,700 mm); CHIRPS (0.05 deg) is the planned upgrade.
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
ERA5_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            "&start_date={start}-01-01&end_date={end}-12-31&daily=precipitation_sum,temperature_2m_mean"
            "&models=era5&timezone=Asia%2FColombo")
MAHA = (10, 11, 12, 1, 2, 3)
YALA = (4, 5, 6, 7, 8, 9)


def _get(url: str, retries: int = 6) -> bytes:
    import urllib.error

    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": "research-data-client"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:  # noqa: S310 - fixed public hosts
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            time.sleep(60 * (attempt + 1))  # free API rate limit: back off
    raise RuntimeError("unreachable")


def era5_climatology(lat: float, lon: float, start: int, end: int) -> dict:
    d = json.loads(_get(ERA5_URL.format(lat=lat, lon=lon, start=start, end=end)))
    df = pd.DataFrame({"date": pd.to_datetime(d["daily"]["time"]), "p": d["daily"]["precipitation_sum"],
                       "t": d["daily"]["temperature_2m_mean"]}).dropna()
    years = end - start + 1
    monthly = df.groupby(df["date"].dt.month)["p"].sum() / years
    return {"temperature_c": round(float(df["t"].mean()), 1),
            "annual_rainfall_mm": round(float(monthly.sum())),
            "maha_rainfall_mm": round(float(monthly[list(MAHA)].sum())),
            "yala_rainfall_mm": round(float(monthly[list(YALA)].sum())),
            "dry_months_lt60mm": int((monthly < 60).sum()),
            "monthly_rainfall_mm": [round(float(monthly.get(m, 0))) for m in range(1, 13)],
            "elevation_m": d.get("elevation")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ecocrop-csv", type=Path, help="local EcoCrop_DB.csv (default: download)")
    ap.add_argument("--start", type=int, default=1995)
    ap.add_argument("--end", type=int, default=2024)
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
            "climate_source": f"ERA5 reanalysis {args.start}-{args.end} via Open-Meteo archive API (CC BY 4.0), "
                              "temperature downscaled to a 90 m DEM"}
    (args.out / "crops.json").write_text(json.dumps({**meta, "crops": crops}, indent=2), encoding="utf-8")

    points = yaml.safe_load((CONFIG_DIR / "districts.yaml").read_text(encoding="utf-8"))["districts"]
    districts = {}
    for name, p in points.items():
        districts[name.upper()] = {"lat": p["lat"], "lon": p["lon"], **era5_climatology(p["lat"], p["lon"], args.start, args.end)}
        print(name, districts[name.upper()]["temperature_c"], districts[name.upper()]["annual_rainfall_mm"], flush=True)
        time.sleep(15)  # stay well inside the free API's rate limit
    (args.out / "districts.json").write_text(json.dumps({**meta, "districts": districts}, indent=1), encoding="utf-8")
    print(json.dumps({k: crops[k]["params"] for k in crops}, indent=1))
    print(f"{len(districts)} districts -> {args.out}")


if __name__ == "__main__":
    main()
