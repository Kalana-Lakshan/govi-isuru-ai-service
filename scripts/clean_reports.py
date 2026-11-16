"""(Re)write report_clean.json for every finished run that lacks one (cheap: uses cached outputs).

    python scripts/clean_reports.py --data-root /content/data --govi $GOVI
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--govi", type=Path, required=True)
    args = ap.parse_args()
    excl = args.govi / "audit" / "near_dup_exclusions.csv"
    if not excl.exists():
        print("no near-duplicate audit yet; nothing to do")
        return
    for run in sorted((args.govi / "runs").glob("*")):
        if (run / "report.json").exists() and (run / "outputs.npz").exists() and not (run / "report_clean.json").exists():
            print(f"== {run.name}")
            subprocess.run([sys.executable, str(REPO / "scripts" / "eval_student.py"), "--data-root", str(args.data_root),
                            "--run", str(run), "--workers", "2", "--exclude-file", str(excl)], check=False)


if __name__ == "__main__":
    main()
