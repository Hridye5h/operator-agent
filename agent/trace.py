"""Run trace — the agent's audit log and the evidence it hands back.

Every step the agent takes is appended here as a structured record, so a run can
be replayed and audited after the fact rather than inferred from stdout. The
trace is also what the final report cites as evidence.
"""

from __future__ import annotations

import json
import pathlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

StepKind = Literal[
    "goal", "plan", "tool_call", "observation", "decision",
    "escalation", "verification", "complete", "error",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Step:
    index: int
    kind: StepKind
    summary: str
    at: str = field(default_factory=_now)
    detail: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int | None = None
    artifacts: list[str] = field(default_factory=list)


class Trace:
    """Append-only record of one agent run."""

    def __init__(self, goal: str, root: pathlib.Path | str = "runs") -> None:
        self.run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.dir = pathlib.Path(root) / self.run_id
        self.artifacts_dir = self.dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.goal = goal
        self.steps: list[Step] = []
        self.started = time.time()
        self.outcome: str | None = None
        self.add("goal", goal)

    # -- recording ---------------------------------------------------------
    def add(self, kind: StepKind, summary: str, **detail: Any) -> Step:
        artifacts = detail.pop("artifacts", []) or []
        elapsed = detail.pop("elapsed_ms", None)
        step = Step(
            index=len(self.steps),
            kind=kind,
            summary=summary,
            detail=detail,
            elapsed_ms=elapsed,
            artifacts=list(artifacts),
        )
        self.steps.append(step)
        self._echo(step)
        self._flush()
        return step

    def artifact_path(self, name: str) -> pathlib.Path:
        """Reserve a path inside this run's artifact directory."""
        return self.artifacts_dir / name

    def finish(self, outcome: str) -> None:
        self.outcome = outcome
        self._flush()

    # -- output ------------------------------------------------------------
    _ICON = {
        "goal": "GOAL", "plan": "PLAN", "tool_call": " -> ", "observation": " <- ",
        "decision": "THINK", "escalation": "ASK", "verification": "CHECK",
        "complete": "DONE", "error": "FAIL",
    }

    def _echo(self, step: Step) -> None:
        tag = self._ICON.get(step.kind, step.kind.upper())
        line = f"[{step.index:02d}] {tag:<5} {step.summary}"
        if step.elapsed_ms is not None:
            line += f"  ({step.elapsed_ms} ms)"
        print(line, flush=True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "started_at": self.steps[0].at if self.steps else _now(),
            "duration_s": round(time.time() - self.started, 2),
            "outcome": self.outcome,
            "step_count": len(self.steps),
            "steps": [asdict(s) for s in self.steps],
        }

    def _flush(self) -> None:
        (self.dir / "trace.json").write_text(
            json.dumps(self.as_dict(), indent=2, default=str), encoding="utf-8"
        )

    def write_markdown(self) -> pathlib.Path:
        """Human-readable run report, suitable for pasting into a PR or ticket."""
        d = self.as_dict()
        out = [
            f"# Run {d['run_id']}",
            "",
            f"**Goal** — {d['goal']}",
            "",
            f"**Outcome** — {d['outcome'] or 'incomplete'}  ",
            f"**Steps** — {d['step_count']}  ",
            f"**Duration** — {d['duration_s']}s",
            "",
            "| # | Step | What happened |",
            "|---|------|---------------|",
        ]
        for s in d["steps"]:
            summary = str(s["summary"]).replace("|", "\\|")
            out.append(f"| {s['index']} | `{s['kind']}` | {summary} |")

        arts = sorted(p.name for p in self.artifacts_dir.glob("*"))
        if arts:
            out += ["", "## Evidence", ""]
            out += [f"- `artifacts/{a}`" for a in arts]

        path = self.dir / "report.md"
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        return path
