"""Run the next unfinished experiment steps (configs/pipeline.yaml) within a time budget.

Designed for Colab free tier: sessions disconnect and GPU time is limited, so every step writes its
results to Drive and is skipped once its `done` file exists; training resumes from last.pt.

    python scripts/pipeline.py --data-root /content/data --govi /content/drive/MyDrive/govi-ai --hours 3.5
    python scripts/pipeline.py --govi ... --status          # what is done / next

Team mode (several people, each on their own Colab account): shared inputs are read from --inputs (the
owner's govi-ai folder, shared read-only) and results go to --govi (the person's own team folder); --team
picks that person's steps from the `teams:` section of the plan.

A log per step goes to {GOVI}/logs/<step>.log and a summary to {GOVI}/logs/pipeline_status.json, which
Google Drive for desktop syncs back to the PC.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def load_plan(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_steps(path: Path, ctx: dict) -> list[dict]:
    cfg = load_plan(path)
    out = []
    for s in cfg["steps"]:
        s = dict(s)
        if "lodo" in s:
            t = cfg["lodo_template"]
            s.setdefault("cmd", t["cmd"].replace("{SRC}", s["lodo"]))
            s.setdefault("done", t["done"].replace("{SRC}", s["lodo"]))
        s["cmd"] = " ".join(s["cmd"].split()).format(**ctx)
        s["done"] = s["done"].format(**ctx)
        out.append(s)
    return out


def team_dirs(govi: Path) -> list[Path]:
    """The owner's sibling team folders (govi-ai-team1, ...), where team members' results land."""
    return sorted(govi.parent.glob(govi.name + "-team*"))


def is_done(step: dict, govi: Path, teams: list[Path]) -> bool:
    """Done here, or (for the owner) already done by a team member in their team folder."""
    done = Path(step["done"])
    if done.exists():
        return True
    try:
        rel = done.relative_to(govi)
    except ValueError:
        return False
    return any((t / rel).exists() for t in teams)


def has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def write_status(govi: Path, steps: list[dict], extra: dict) -> None:
    logs = govi / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status = {"updated": datetime.now().isoformat(timespec="seconds"), **extra,
              "steps": [{"id": s["id"], "done": Path(s["done"]).exists(), "gpu": s.get("gpu", False),
                         "hours": s.get("hours")} for s in steps]}
    (logs / "pipeline_status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")


def run_step(step: dict, govi: Path) -> int:
    log = govi / "logs" / f"{step['id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {step['id']} ===\n{step['cmd']}\n(log: {log})", flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n### {datetime.now().isoformat(timespec='seconds')} {step['cmd']}\n")
        f.flush()
        p = subprocess.Popen(step["cmd"], shell=True, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for line in p.stdout:  # stream to the notebook and to Drive
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
        return p.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/content/data"))
    ap.add_argument("--govi", type=Path, required=True, help="Drive project dir (…/MyDrive/govi-ai)")
    ap.add_argument("--plan", type=Path, default=REPO / "configs" / "pipeline.yaml")
    ap.add_argument("--hours", type=float, default=3.5, help="session budget; no step starts that cannot finish")
    ap.add_argument("--only", nargs="*", help="run just these step ids")
    ap.add_argument("--inputs", type=Path, help="folder with dataset/, features/, audit/ (default: --govi)")
    ap.add_argument("--team", help="run the steps assigned to this team member in the plan's `teams:` section")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    inputs = args.inputs or args.govi
    ctx = {"DATA": str(args.data_root), "GOVI": str(args.govi), "IN": str(inputs), "FEATS": str(inputs / "features"),
           "RUNS": str(args.govi / "runs")}
    steps = load_steps(args.plan, ctx)
    teams_done = [] if args.team else team_dirs(args.govi)
    done = lambda st: is_done(st, args.govi, teams_done)  # noqa: E731
    if args.team:
        teams = load_plan(args.plan).get("teams", {})
        if args.team not in teams:
            sys.exit(f"unknown team {args.team!r}; choose one of {sorted(teams)}")
        mine = set(teams[args.team])
        # a team member runs only their steps (plus the per-session summary steps)
        steps = [s for s in steps if s["id"] in mine or s.get("always")]
        print(f"team {args.team}: {sorted(mine)}")
    gpu = has_gpu()
    if args.status:
        for s in steps:
            mark = "*" if s.get("always") else ("x" if done(s) else " ")
            print(f"[{mark}] {s['id']:24s} gpu={s.get('gpu', False)}  ~{s.get('hours')} h")
        return

    by_id = {s["id"]: s for s in steps}
    t0 = time.time()
    ran, failed = [], []
    for s in steps:
        if args.only and s["id"] not in args.only:
            continue
        if s.get("always") or done(s):
            continue
        if s.get("gpu") and not gpu:
            print(f"[skip] {s['id']} needs a GPU runtime")
            continue
        waiting = [n for n in s.get("needs", []) if n in by_id and not done(by_id[n])]
        if waiting:
            print(f"[wait] {s['id']} needs {', '.join(waiting)} to finish first")
            continue
        left = args.hours - (time.time() - t0) / 3600
        # Training steps resume, so starting one late is fine as long as at least an epoch fits.
        if left < min(0.5, s.get("hours", 0)):
            print(f"[stop] {left:.1f} h left in this session's budget; next: {s['id']}")
            break
        write_status(args.govi, steps, {"running": s["id"], "gpu": gpu})
        rc = run_step(s, args.govi)
        (ran if rc == 0 else failed).append(s["id"])
        if rc != 0:
            print(f"[fail] {s['id']} exited with {rc}; see its log. Continuing with the next step.")
    # Summary steps (clean reports, tables) run at the end of every session, budget or not.
    for s in steps:
        if s.get("always") and (not args.only or s["id"] in args.only):
            rc = run_step(s, args.govi)
            (ran if rc == 0 else failed).append(s["id"])
    write_status(args.govi, steps, {"running": None, "gpu": gpu, "last_session": {"ran": ran, "failed": failed}})
    remaining = [s["id"] for s in steps if not done(s) and not s.get("always")]
    print(f"\nsession done. ran={ran} failed={failed}\nremaining={remaining}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
