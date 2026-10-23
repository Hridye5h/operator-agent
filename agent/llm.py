"""LLM access, behind a narrow interface.

Two providers:

* ``GeminiPlanner``   - real reasoning via google-genai.
* ``ScriptedPlanner`` - a deterministic rule-based planner used when no API key
  is configured, and in tests.

The scripted planner exists so the agent loop, the tools, the recovery paths and
the verification step can all be exercised and tested without a network call.
It is genuinely planning over the same observation buffer and the same tool
registry; it just reasons with rules instead of a model. Every step it takes is
still decided at runtime from what it has observed.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Decision:
    """One planner step: either call a tool, or stop."""
    action: str                      # tool name, or "finish"
    args: dict[str, Any]
    reasoning: str = ""
    confidence: float = 1.0

    @staticmethod
    def parse(raw: str) -> "Decision":
        blob = _extract_json(raw)
        if blob is None:
            raise ValueError(f"planner did not return JSON: {raw[:200]!r}")
        action = blob.get("action") or blob.get("tool") or ""
        if not action:
            raise ValueError(f"planner returned no action: {blob}")
        args = blob.get("args") or blob.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        return Decision(
            action=str(action),
            args=args,
            reasoning=str(blob.get("reasoning", "")),
            confidence=float(blob.get("confidence", 1.0)),
        )


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class Planner(Protocol):
    name: str

    def decide(self, system: str, context: str) -> Decision: ...


class GeminiPlanner:
    """Real LLM planner."""

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None) -> None:
        from google import genai  # imported lazily so the offline path needs no SDK

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self._client = genai.Client(api_key=key)
        self._model = model
        self.name = f"gemini:{model}"

    def decide(self, system: str, context: str) -> Decision:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=context,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        return Decision.parse(response.text or "")


class ScriptedPlanner:
    """Deterministic fallback planner.

    Reasons over the same observation buffer the LLM sees. It is not a recorded
    script: it inspects what has actually happened so far and picks the next
    action, including the recovery and escalation branches.
    """

    name = "scripted"

    # Pulled out so the agent can tell the operator why it is running offline.
    reason = "no GEMINI_API_KEY configured — using deterministic planner"

    def decide(self, system: str, context: str) -> Decision:
        # Only the HISTORY section counts as "what has happened". The tool
        # catalogue earlier in the prompt also contains every tool name, so
        # matching against the whole context would mark every step as done
        # before the agent had taken any.
        c = context.split("HISTORY:", 1)[1] if "HISTORY:" in context else ""
        seen = lambda s: s in c  # noqa: E731

        # 1. Locate candidate documents.
        if not seen("list_invoices("):
            return Decision("list_invoices", {}, "Need to see what documents exist.")

        # 2. Read the newest matching document.
        if not seen("read_invoice("):
            vendor = _vendor_from_goal(context)
            return Decision(
                "read_invoice",
                {"vendor": vendor, "latest": True},
                f"Read the most recent {vendor or 'matching'} invoice.",
            )

        # 3. If the document is internally inconsistent, ask rather than guess.
        if seen("AMBIGUITY") and not seen("ask_human("):
            return Decision(
                "ask_human",
                {"question": "This invoice shows two conflicting payable amounts. "
                             "Which should be recorded?",
                 "options_from_observation": True},
                "Two contradictory totals; guessing here would corrupt the ledger.",
                confidence=0.2,
            )

        # 4. Enter the record through the browser.
        if not seen("submit_payable("):
            return Decision("submit_payable", {"use_extracted": True},
                            "Enter the payable into the internal system.")

        # 5. Recover from a failed submission.
        if seen("ERROR (retriable)") and c.rfind("ERROR (retriable)") > c.rfind("submit_payable(") - 1:
            if c.count("submit_payable(") < 4:
                return Decision("submit_payable", {"use_extracted": True, "retry": True},
                                "Previous submission failed transiently; retrying.")

        # 6. Verify by reading back from the system of record.
        if not seen("verify_payable("):
            return Decision("verify_payable", {"use_extracted": True},
                            "Confirm the record actually landed and matches.")

        return Decision("finish", {}, "Outcome verified.")


def _vendor_from_goal(context: str) -> str:
    m = re.search(r"GOAL:\s*(.+)", context)
    goal = m.group(1) if m else context
    for name in ("Northwind", "Globex", "Initech"):
        if name.lower() in goal.lower():
            return name
    return ""


def build_planner(prefer: str = "auto") -> tuple[Planner, str | None]:
    """Return (planner, note). ``note`` explains any downgrade."""
    if prefer in ("auto", "gemini"):
        try:
            return GeminiPlanner(), None
        except Exception as exc:
            if prefer == "gemini":
                raise
            return ScriptedPlanner(), f"{ScriptedPlanner.reason} ({type(exc).__name__})"
    return ScriptedPlanner(), None
