"""Official paddy statistics from the Department of Census and Statistics (DCS), Sri Lanka.

Source: https://www.statistics.gov.lk/Agriculture/StaticalInformation/PaddyStatistics
Eight Excel series, district x season, 1978/79 Maha – 2025 (Agriculture and Environmental
Statistics Division, DCS). Each sheet has one row per district (plus the Udawalawe and
Mahaweli 'H' irrigation systems and a SRI LANKA total) and, per season, four columns:
Major (major irrigation schemes), Minor (minor irrigation), Rainfed, Total. "-" = not reported.

Output is a tidy long table:
    district, season (maha|yala), season_label (e.g. 2014/2015 or 2015), year, water (major|minor|rainfed|total),
    measure (sown_extent_ha|harvested_extent_ha|avg_yield_kg_ha|production_mt), value
For Maha seasons, ``year`` is the harvest year (the second year of "1978/1979").
"""

from __future__ import annotations

import io
import re
import urllib.request

import pandas as pd

BASE = "https://www.statistics.gov.lk/Agriculture/StaticalInformation/PaddyStatistics/"
# (measure, season) -> page slug. Slugs are copied verbatim from the DCS site (incl. its spelling).
# The production sheets report totals only, in thousand metric tons (scaled to t in SCALE);
# fetch_paddy_stats.py verifies this via production ≈ harvested extent x average yield.
SCALE = {"production_mt": 1000.0}
SERIES = {
    ("sown_extent_ha", "maha"): "Sown_Extent-Maha_Seasons-1979-2025",
    ("sown_extent_ha", "yala"): "Sown_Extent-Yala_Seasons-1979-2025",
    ("harvested_extent_ha", "maha"): "Harvested_Extent-Maha_Seasons-1979-2025",
    ("harvested_extent_ha", "yala"): "Harvested_Extent-Yala_Seasons-1979-2025",
    ("avg_yield_kg_ha", "maha"): "Average_Yeild-Maha_Season-1979-2025",
    ("avg_yield_kg_ha", "yala"): "Average_Yeild-Yala_Season-1979-2025",
    ("production_mt", "maha"): "Production-Maha_Season-1979-2025",
    ("production_mt", "yala"): "Production-Yala_Season-1979-2025",
}
WATER = {"major": "major", "minor": "minor", "rainfed": "rainfed", "total": "total"}
_SEASON_RE = re.compile(r"^\s*(\d{4})(?:\s*/\s*(\d{4}))?\s*$")


def download(slug: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(BASE + slug, headers={"User-Agent": "Mozilla/5.0 (research data fetch)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed official host
        data = r.read()
    if not data.startswith(b"PK"):
        raise ValueError(f"{slug}: expected an .xlsx file, got {data[:40]!r}")
    return data


def parse_sheet(raw: pd.DataFrame, measure: str, season: str) -> pd.DataFrame:
    """Parse one DCS sheet (read with ``header=None``) into the tidy long format."""
    # Locate the header rows: the row containing "District" holds season labels; the next one
    # holds Major / Minor / Rainfed / Total.
    hdr = next(i for i in range(len(raw)) if raw.iloc[i].astype(str).str.strip().str.lower().eq("district").any())
    dcol = int(raw.iloc[hdr].astype(str).str.strip().str.lower().eq("district").to_numpy().argmax())
    seasons_row, water_row = raw.iloc[hdr], raw.iloc[hdr + 1]
    # Some sheets (production) have no Major/Minor/Rainfed/Total row: every column is a total.
    has_water_row = water_row.astype(str).str.strip().str.lower().isin(WATER).any()

    columns: list[tuple[int, str, int, str]] = []  # (col index, season label, year, water)
    current = None
    for j in range(dcol + 1, raw.shape[1]):
        lab = seasons_row.iloc[j]
        if pd.notna(lab) and str(lab).strip():
            text = re.sub(r"\.0$", "", str(lab).strip())
            m = _SEASON_RE.match(text)
            year = int(m.group(2) or m.group(1)) if m else None
            if current and (year is None or year != current[1] + 1):
                # Typos in DCS headers (e.g. "198" for 1980): seasons are consecutive, so trust
                # the position when the label is malformed or breaks the sequence.
                if year is None and not re.fullmatch(r"\d{2,4}(/\d{2,4})?", text):
                    continue
                year = current[1] + 1
                text = f"{year - 1}/{year}" if season == "maha" else str(year)
            if year is not None:
                # Canonical label from the year (DCS headers contain typos such as "2000/2021").
                current = (f"{year - 1}/{year}" if season == "maha" else str(year), year)
        if not current:
            continue
        if has_water_row:
            w = str(water_row.iloc[j]).strip().lower()
            if w in WATER:
                columns.append((j, current[0], current[1], WATER[w]))
        elif not columns or columns[-1][0] != j:
            columns.append((j, current[0], current[1], "total"))
    if not columns:
        raise ValueError(f"{measure}/{season}: no season columns found")

    scale = SCALE.get(measure, 1.0)
    out = []
    for i in range(hdr + (2 if has_water_row else 1), len(raw)):
        name = raw.iloc[i, dcol]
        if not isinstance(name, str) or not name.strip():
            continue
        district = re.sub(r"\s+", " ", name.strip()).upper()
        for j, label, year, water in columns:
            v = pd.to_numeric(str(raw.iloc[i, j]).replace(",", "").strip(), errors="coerce")
            if pd.notna(v):
                out.append((district, season, label, year, water, measure, float(v) * scale))
    return pd.DataFrame(out, columns=["district", "season", "season_label", "year", "water", "measure", "value"])


def fetch_all(series: dict | None = None) -> pd.DataFrame:
    frames = []
    for (measure, season), slug in (series or SERIES).items():
        raw = pd.read_excel(io.BytesIO(download(slug)), header=None, sheet_name=0)
        frames.append(parse_sheet(raw, measure, season))
    return pd.concat(frames, ignore_index=True)


def to_wide(long: pd.DataFrame, water: str = "total") -> pd.DataFrame:
    """One row per (district, season, year) with the four measures as columns."""
    w = long[long["water"] == water].pivot_table(
        index=["district", "season", "season_label", "year"], columns="measure", values="value", aggfunc="first")
    return w.reset_index().rename_axis(columns=None)
