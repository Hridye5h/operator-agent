#!/usr/bin/env python3
"""Record demo runs driven by the live Gemini planner, for the run viewer.

    python scripts/record_llm_demos.py                     # default set
    python scripts/record_llm_demos.py happy supplier      # chosen scenarios

Each scenario runs in its own process against a fresh environment, with the
browser recorded. The trace, screenshots and video are copied to
``docs/demo/runs/llm-<scenario>/``. Runs are recorded as they happened: a run
that fails or escalates is kept, not retried until it looks good.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_agent import load_env  # noqa: E402

RUN_TIMEOUT = 1800


def kill_tree(proc: subprocess.Popen) -> None:
    """A run starts a server and a browser; killing only the child leaves both."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True)
    else:
        os.killpg(proc.pid, signal.SIGKILL)


DEFAULT = ["happy", "ambiguous", "high_value", "supplier_fraud_check", "batch"]
DEST = ROOT / "docs" / "demo" / "runs"


def newest_run(before: set[pathlib.Path]) -> pathlib.Path | None:
    runs = [p for p in (ROOT / "runs").glob("*") if p.is_dir() and p not in before]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def main(argv: list[str]) -> int:
    load_env()
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("No GEMINI_API_KEY in the environment or .env — nothing to record.")
        return 1

    scenarios = argv or DEFAULT
    rows = []
    for scenario in scenarios:
        before = set((ROOT / "runs").glob("*")) if (ROOT / "runs").exists() else set()
        print(f"\n=== {scenario} (gemini) ===", flush=True)
        proc = subprocess.Popen(
            [sys.executable, "-I", str(ROOT / "run_agent.py"), "--scenario", scenario,
             "--planner", "gemini", "--max-steps", "30", "--record-video"],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", start_new_session=os.name != "nt",
        )
        try:
            out, err = proc.communicate(timeout=RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            out, err = proc.communicate()
            print(out[-2500:])
            print(f"timed out after {RUN_TIMEOUT}s; the run is not published")
            rows.append((scenario, "timed out", "", ""))
            continue
        print(out[-2500:])
        if proc.returncode not in (0, 1):
            print(err[-1500:])

        run = newest_run(before)
        if run is None or not (run / "trace.json").exists():
            rows.append((scenario, "no trace", "", ""))
            continue

        target = DEST / f"llm-{scenario}"
        if target.exists():
            shutil.rmtree(target)
        (target / "artifacts").mkdir(parents=True)
        shutil.copy2(run / "trace.json", target / "trace.json")
        for art in (run / "artifacts").glob("*"):
            if art.suffix in (".png", ".webm"):
                shutil.copy2(art, target / "artifacts" / art.name)

        trace = json.loads((run / "trace.json").read_text(encoding="utf-8"))
        planner = " -> ".join(s["summary"].split(",")[0] for s in trace["steps"]
                              if s["kind"] == "plan" and s["summary"].startswith("planner="))
        rows.append((scenario, trace.get("outcome") or "incomplete",
                     str(trace.get("step_count")), planner))

    print("\n" + "-" * 78)
    for scenario, outcome, steps, planner in rows:
        print(f"{scenario:22} {outcome:12} steps={steps:4} {planner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
