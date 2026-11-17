"""The Colab pipeline runner: ordering, skipping finished steps, LODO expansion, failures."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable.replace("\\", "/")


def _plan(tmp: Path) -> Path:
    p = tmp / "plan.yaml"
    p.write_text(textwrap.dedent(f"""
        steps:
          - id: a
            gpu: false
            hours: 0.1
            cmd: {PY} -c "open(r'{{GOVI}}/a.done','w').write('1')"
            done: "{{GOVI}}/a.done"
          - id: broken
            gpu: false
            hours: 0.1
            cmd: {PY} -c "import sys; sys.exit(3)"
            done: "{{GOVI}}/broken.done"
          - id: needs_gpu
            gpu: true
            hours: 0.1
            cmd: {PY} -c "print('should not run on CPU')"
            done: "{{GOVI}}/gpu.done"
          - {{id: lodo_x, gpu: false, hours: 0.1, lodo: x}}
        lodo_template:
          cmd: {PY} -c "open(r'{{RUNS}}/lodo_{{SRC}}.done','w').write('1')"
          done: "{{RUNS}}/lodo_{{SRC}}.done"
    """), encoding="utf-8")
    return p


def _run(tmp, plan, *extra):
    return subprocess.run([sys.executable, str(REPO / "scripts" / "pipeline.py"), "--govi", str(tmp),
                           "--plan", str(plan), "--data-root", str(tmp / "data"), *extra],
                          capture_output=True, text=True, cwd=REPO)


def test_pipeline_runs_skips_and_reports(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    plan = _plan(tmp_path)
    r = _run(tmp_path, plan)
    assert r.returncode == 1, r.stdout + r.stderr          # 'broken' failed
    assert (tmp_path / "a.done").exists()
    assert (tmp_path / "runs" / "lodo_x.done").exists()    # later steps still ran after the failure
    status = json.loads((tmp_path / "logs" / "pipeline_status.json").read_text())
    assert status["last_session"]["failed"] == ["broken"]
    assert (tmp_path / "logs" / "a.log").exists()

    # Second session: finished steps are skipped, only the broken one is retried.
    r2 = _run(tmp_path, plan)
    assert "=== a ===" not in r2.stdout and "=== broken ===" in r2.stdout

    s = _run(tmp_path, plan, "--status").stdout
    assert "[x] a" in s and "[ ] broken" in s and "[x] lodo_x" in s


def test_budget_stops_before_starting(tmp_path):
    (tmp_path / "runs").mkdir()
    r = _run(tmp_path, _plan(tmp_path), "--hours", "0")
    assert "[stop]" in r.stdout and not (tmp_path / "a.done").exists()


def test_always_steps_run_even_when_budget_is_exhausted(tmp_path):
    (tmp_path / "runs").mkdir()
    plan = tmp_path / "plan2.yaml"
    plan.write_text(textwrap.dedent(f"""
        steps:
          - {{id: big, gpu: false, hours: 5, cmd: '{PY} -c "print(1)"', done: "{{GOVI}}/big.done"}}
          - id: summary
            gpu: false
            hours: 0.01
            always: true
            cmd: {PY} -c "open(r'{{GOVI}}/summary.txt','a').write('x')"
            done: "{{GOVI}}/never"
    """), encoding="utf-8")
    _run(tmp_path, plan, "--hours", "0")
    _run(tmp_path, plan, "--hours", "0")
    assert (tmp_path / "summary.txt").read_text() == "xx"   # ran in both sessions
    assert "[*] summary" in _run(tmp_path, plan, "--status").stdout


def test_needs_waits_for_prerequisite(tmp_path):
    (tmp_path / "runs").mkdir()
    plan = tmp_path / "plan3.yaml"
    plan.write_text(textwrap.dedent(f"""
        steps:
          - {{id: train, gpu: true, hours: 0.1, cmd: '{PY} -c "print(1)"', done: "{{GOVI}}/train.done"}}
          - id: bundle
            gpu: false
            hours: 0.1
            needs: [train]
            cmd: {PY} -c "import sys; sys.exit(5)"
            done: "{{GOVI}}/bundle.done"
    """), encoding="utf-8")
    r = _run(tmp_path, plan)
    assert "[wait] bundle needs train" in r.stdout
    assert r.returncode == 0, r.stdout + r.stderr


def _team_plan(tmp: Path) -> Path:
    p = tmp / "team_plan.yaml"
    p.write_text(textwrap.dedent(f"""
        steps:
          - {{id: shared, gpu: false, hours: 0.1, cmd: '{PY} -c "print(1)"', done: "{{IN}}/shared.done"}}
          - id: a
            gpu: false
            hours: 0.1
            cmd: {PY} -c "import os; os.makedirs(r'{{RUNS}}/a', exist_ok=True); open(r'{{RUNS}}/a/report.json','w').write('1')"
            done: "{{RUNS}}/a/report.json"
          - id: b
            gpu: false
            hours: 0.1
            cmd: {PY} -c "import os; os.makedirs(r'{{RUNS}}/b', exist_ok=True); open(r'{{RUNS}}/b/report.json','w').write('1')"
            done: "{{RUNS}}/b/report.json"
        teams:
          team1: [b]
    """), encoding="utf-8")
    return p


def test_team_member_runs_only_their_steps_into_their_folder(tmp_path):
    owner = tmp_path / "govi-ai"
    team = tmp_path / "govi-ai-team1"
    owner.mkdir(); team.mkdir()
    (owner / "shared.done").write_text("1")
    plan = _team_plan(tmp_path)
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "pipeline.py"), "--govi", str(team), "--inputs", str(owner),
                        "--team", "team1", "--plan", str(plan), "--data-root", str(tmp_path / "data")],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (team / "runs" / "b" / "report.json").exists()
    assert not (team / "runs" / "a").exists() and not (owner / "runs").exists()

    # The owner now sees b as done (it is in the team folder) and runs only a.
    r2 = _run(owner, plan)
    assert "=== a ===" in r2.stdout and "=== b ===" not in r2.stdout, r2.stdout
    assert "[x] b" in _run(owner, plan, "--status").stdout


def test_unknown_team_is_rejected(tmp_path):
    (tmp_path / "runs").mkdir()
    r = _run(tmp_path, _team_plan(tmp_path), "--team", "team9")
    assert r.returncode != 0 and "unknown team" in (r.stdout + r.stderr)
