"""Company procedures — how this organisation does a given kind of work.

A procedure is data, not code: which tool plays each role (discover, read,
write, verify) and the house rules for that kind of request. The agent selects
one from the goal and shows it to whichever planner is running. An LLM planner
reads it as guidance; the deterministic planner executes it.

This is the seam where company context enters. Teaching the operator a new
process means adding a procedure file and registering its tools — the loop and
the planners do not change.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

from agent.tools import ToolRegistry

ROLES = ("discover", "read", "write", "verify")


def load_procedures(directory: pathlib.Path | str) -> list[dict[str, Any]]:
    out = []
    for path in sorted(pathlib.Path(directory).glob("*.json")):
        proc = json.loads(path.read_text(encoding="utf-8"))
        proc.setdefault("name", path.stem)
        out.append(proc)
    return out


def select_procedure(goal: str, procedures: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the procedure whose triggers best match the goal.

    Returns None when nothing matches or two procedures match equally well, so
    an unfamiliar or ambiguous request is not forced into the wrong process.
    """
    lowered = goal.lower()
    scored = []
    for proc in procedures:
        score = sum(1 for t in proc.get("triggers", [])
                    if re.search(rf"\b{re.escape(t.lower())}\b", lowered))
        scored.append((score, proc))
    scored.sort(key=lambda s: s[0], reverse=True)
    if not scored or scored[0][0] == 0:
        return None
    if len(scored) > 1 and scored[1][0] == scored[0][0]:
        return None
    return scored[0][1]


def check_procedure(procedure: dict[str, Any], registry: ToolRegistry) -> list[str]:
    """Problems that would stop a procedure from running against this registry."""
    problems = []
    for role, tool_name in procedure.get("steps", {}).items():
        if role not in ROLES:
            problems.append(f"unknown role {role!r}")
            continue
        tool = registry.get(tool_name)
        if tool is None:
            problems.append(f"{role} step names {tool_name!r}, which is not registered")
        elif tool.kind != role:
            problems.append(f"{role} step names {tool_name!r}, whose kind is {tool.kind!r}")
    for role in ("read", "write", "verify"):
        if role not in procedure.get("steps", {}):
            problems.append(f"procedure has no {role} step")
    return problems
