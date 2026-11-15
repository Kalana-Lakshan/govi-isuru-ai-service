"""Read the harmonised manifest and select rows for an evaluation protocol."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BOOL_COLS = ("use_in_paper", "train_only")


def load_manifest(data_root: str | Path, paper_only: bool = False) -> pd.DataFrame:
    """Load ``manifest.csv``. ``paper_only`` drops sources that may not appear in reported numbers
    (both for training and testing, so paper results never depend on uncitable data)."""
    df = pd.read_csv(Path(data_root) / "manifest.csv", dtype=str, keep_default_na=False)
    for c in BOOL_COLS:
        if c in df:
            df[c] = df[c].str.lower().isin(("true", "1"))
    if paper_only and "use_in_paper" in df:
        df = df[df["use_in_paper"]].reset_index(drop=True)
    return df


def protocols(df: pd.DataFrame) -> list[str]:
    """Available protocol columns: in-domain ``split`` first, then ``lodo_<source>``."""
    return ["split"] + sorted(c for c in df.columns if c.startswith("lodo_"))


def gate_label(row: pd.Series) -> str:
    """Target of the crop gate head: the crop for crop images, else the negative role."""
    return row["crop"] if row["role"] in ("known", "unknown_eval") else row["role"]
