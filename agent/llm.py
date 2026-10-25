"""LLM access, behind a narrow interface.

Two providers:

* ``GeminiPlanner``   - real reasoning via google-genai.
* ``ScriptedPlanner`` - a deterministic procedure executor used when no API key
  is configured, and in tests.

The scripted planner exists so the agent loop, the tools, the recovery paths and
the verification step can all be exercised and tested without a network call.
It executes the company procedure the agent selected, over the same observation
buffer and the same tool registry, and contains no task-specific names; every
step it takes is decided at runtime from what it has observed.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol


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


def _retry_after(exc: Exception) -> float | None:
    """The server's own retry delay (google.rpc.RetryInfo), if it sent one."""
    def find(node: Any) -> str | None:
        if isinstance(node, dict):
            if "retryDelay" in node:
                return str(node["retryDelay"])
            for value in node.values():
                hit = find(value)
                if hit:
                    return hit
        elif isinstance(node, list):
            for value in node:
                hit = find(value)
                if hit:
                    return hit
        return None

    raw = find(getattr(exc, "details", None))
    if not raw:
        return None
    m = re.match(r"^\s*([\d.]+)\s*s?\s*$", raw)
    return min(float(m.group(1)), 90.0) if m else None


def _is_daily_quota(exc: Exception) -> bool:
    text = f"{getattr(exc, 'message', '') or ''} {getattr(exc, 'details', '') or ''}"
    return bool(re.search(r"per ?day|PerDay|daily", text, re.I))


def _why_empty(response: Any) -> str:
    """Explain an empty model response using whatever the SDK exposed."""
    reason = getattr(response, "prompt_feedback", None)
    blocked = getattr(reason, "block_reason", None) if reason is not None else None
    if blocked:
        return f"prompt blocked: {blocked}"

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finish = getattr(candidates[0], "finish_reason", None)
        if finish:
            return f"finish_reason={finish}"
    return "no candidates returned"


class Planner(Protocol):
    name: str

    def decide(self, system: str, context: str) -> Decision: ...


#: Tried in order when no model is configured; the first one this key can use wins.
PREFERRED_MODELS = ("gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash")

#: HTTP codes worth waiting out: rate limits and transient server faults.
RETRYABLE = {429, 500, 502, 503, 504}


class GeminiPlanner:
    """Real LLM planner, hardened for a live API.

    * The model is configurable (``GEMINI_MODEL``). If unset, the planner asks
      the API which models this key can use and picks the first known-good one,
      so a retired model name does not break a run.
    * Rate limits and transient server errors are waited out, honouring the
      server's own retry delay when it gives one. A daily quota is not retried:
      waiting will not bring it back.
    * The output cap is generous. Newer models reason before answering and that
      reasoning counts against the cap; a tight cap returns an empty reply.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 client: Any = None, max_tokens: int = 8192, max_attempts: int = 5,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._max_tokens = max_tokens
        self._max_attempts = max_attempts
        self._sleep = sleep
        #: (http code, seconds) for every wait, so a run can report them.
        self.waits: list[tuple[int, float]] = []

        if client is None:
            from google import genai  # imported lazily so the offline path needs no SDK

            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            client = genai.Client(api_key=key)
        self._client = client

        self._model = model or os.environ.get("GEMINI_MODEL") or self._pick_model()
        self.name = f"gemini:{self._model}"

    def _pick_model(self) -> str:
        """The first preferred model this key can actually call."""
        try:
            available = {
                m.name.split("/")[-1] for m in self._client.models.list()
                if "generateContent" in (getattr(m, "supported_actions", None) or [])
            }
        except Exception:
            return PREFERRED_MODELS[0]
        for name in PREFERRED_MODELS:
            if name in available:
                return name
        flash = sorted(n for n in available if "flash" in n and "lite" not in n
                       and "image" not in n and "tts" not in n)
        return flash[-1] if flash else PREFERRED_MODELS[0]

    def decide(self, system: str, context: str) -> Decision:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            response_mime_type="application/json",
            max_output_tokens=self._max_tokens,
        )

        delay = 2.0
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=context, config=config)
                break
            except Exception as exc:
                code = getattr(exc, "code", None)
                if code not in RETRYABLE or attempt == self._max_attempts \
                        or _is_daily_quota(exc):
                    raise
                wait = _retry_after(exc) or delay
                self.waits.append((code, wait))
                print(f"       gemini {code} {getattr(exc, 'status', '') or ''}: "
                      f"waiting {wait:.0f}s (attempt {attempt}/{self._max_attempts})",
                      flush=True)
                self._sleep(wait)
                delay = min(delay * 2, 60.0)

        text = getattr(response, "text", None)
        if not text:
            # An empty body is not a parse problem, and reporting it as one
            # sends the agent into a pointless re-prompt loop. Say what the
            # model actually did: refused, ran out of tokens, or returned
            # nothing.
            raise RuntimeError(f"model returned no text ({_why_empty(response)})")
        return Decision.parse(text)


class ScriptedPlanner:
    """Deterministic planner: executes the company procedure in the context.

    It knows the *shape* of a procedure — discover, read, write, verify — and
    nothing about any particular task. Which tool fills each role comes from the
    procedure; which document to read comes from matching the goal against what
    discovery actually returned. It is not a recorded script: every branch is
    taken at runtime from what has been observed, including recovering from a
    transient failure, confirming instead of re-entering when the record already
    exists, and escalating when the source contradicts itself.

    It reasons with rules instead of a model, and it only covers documented
    procedures. A request with no matching procedure needs the LLM planner.
    """

    name = "scripted"

    # Pulled out so the agent can tell the operator why it is running offline.
    reason = "no GEMINI_API_KEY configured — using deterministic planner"

    human = "ask_human"

    def decide(self, system: str, context: str) -> Decision:
        roles = _procedure_roles(context)
        if not {"read", "write", "verify"} <= roles.keys():
            raise ValueError("no company procedure in context; the deterministic "
                             "planner only executes documented procedures")
        D, R, W, V = (roles.get("discover"), roles["read"], roles["write"],
                      roles["verify"])
        ask = self.human

        # Only the HISTORY section counts as "what has happened". The tool
        # catalogue and the procedure earlier in the prompt mention every tool
        # name, so matching the whole context would mark every step as done.
        history = context.split("HISTORY:", 1)[1] if "HISTORY:" in context else ""
        seen = lambda s: s in history  # noqa: E731

        # 1. Discover what exists.
        if D and not seen(f"{D}("):
            return Decision(D, {}, "See which documents exist.")

        # 2. Read the document the request refers to.
        if not seen(f"{R}("):
            chosen = _pick_document(_goal(context), history)
            if chosen is None:
                return Decision(ask, {"question": "I could not tell which document this "
                                                  "request refers to. Which one should I use?"},
                                "No document clearly matches the request.", confidence=0.2)
            return Decision(R, {"file": chosen}, f"The request points at {chosen}.")

        # 3. A contested value is never chosen here. Ask once.
        contested = _contested_field(history)
        if contested and not seen(f"{ask}("):
            return Decision(
                ask,
                {"question": f"The document gives conflicting values for '{contested}'. "
                             f"Which should be used?",
                 "options_from_observation": True},
                "The source contradicts itself; choosing would be a guess.",
                confidence=0.2,
            )

        # 4. Write.
        if not seen(f"{W}("):
            return Decision(W, {"use_extracted": True},
                            "Enter the record into the system of record.")

        # 5. React to how the last write went. A failure stays in the history for
        #    the rest of the run, so once verification has succeeded these
        #    branches must go quiet or the agent keeps reacting to a resolved error.
        last_write = history.rfind(f"{W}(")
        if last_write != -1 and "VERIFIED" not in history:
            outcome = history[last_write:]

            # Already exists: the work is done, so confirm rather than re-enter.
            if "ERROR (conflict)" in outcome:
                return Decision(V, {}, "Already on file; confirming the existing record "
                                       "instead of entering a duplicate.")

            # Temporary fault: the same call is worth repeating, within reason.
            if "ERROR (retriable)" in outcome and history.count(f"{W}(") < 4:
                return Decision(W, {"use_extracted": True, "retry": True},
                                "Previous attempt failed transiently; retrying.")

            # Input rejected: repeating it unchanged cannot help.
            if "ERROR (permanent)" in outcome and not seen(f"{ask}("):
                return Decision(
                    ask,
                    {"question": "The system rejected this submission and retrying "
                                 "unchanged will not help. How should I proceed?"},
                    "Permanent rejection needs a human decision.",
                    confidence=0.2,
                )

        # 6. Verify by reading back from the system of record.
        if not seen(f"{V}("):
            return Decision(V, {}, "Confirm the record landed and matches what was sent.")

        return Decision("finish", {}, "Outcome verified.")


def _goal(context: str) -> str:
    m = re.search(r"GOAL:\s*(.+)", context)
    return m.group(1) if m else ""


def _procedure_roles(context: str) -> dict[str, str]:
    m = re.search(r"Steps:\s*(.+)", context)
    if not m:
        return {}
    roles = {}
    for part in m.group(1).split("->"):
        if "=" in part:
            role, tool = part.split("=", 1)
            roles[role.strip()] = tool.strip()
    return roles


def _contested_field(history: str) -> str | None:
    m = re.search(r"CONTESTED \((\w+)\)", history)
    return m.group(1) if m else None


_RECENCY_LATEST = ("latest", "newest", "most recent")
_RECENCY_OLDEST = ("oldest", "earliest")


def _pick_document(goal: str, listing: str) -> str | None:
    """Choose the document a request refers to, from what discovery returned.

    Scores each listed document by the distinctive words of the goal it
    contains, with an exact reference (an invoice number or a GSTIN named in the
    goal) outranking everything. Ties are broken by recency only when the goal
    asks for one; otherwise a tie means the request is ambiguous and the answer
    is None, so the agent asks rather than picks.
    """
    rows = re.findall(r"^\s+(\S+\.pdf)\s*\|\s*(.+)$", listing, re.M)
    if not rows:
        return None

    lowered = goal.lower()
    words = set(re.findall(r"[a-z][a-z0-9]{3,}", lowered))
    references = set(re.findall(r"\b[A-Z]{2}-[A-Z0-9]{2,}(?:-[A-Z0-9]+)*\b"
                                r"|\b\d{2}[A-Z]{5}\d{4}[A-Z0-9]{3}\b", goal))

    scored = []
    for file, rest in rows:
        text = f"{file} {rest}"
        score = sum(1 for w in words if w in text.lower())
        if any(r in text for r in references):
            score += 100
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", rest)
        scored.append((score, dates[-1] if dates else "", file))

    best = max(s for s, _, _ in scored)
    if best == 0:
        return None
    top = [(d, f) for s, d, f in scored if s == best]
    if len(top) == 1:
        return top[0][1]
    if any(k in lowered for k in _RECENCY_LATEST):
        return max(top)[1]
    if any(k in lowered for k in _RECENCY_OLDEST):
        return min(top)[1]
    return None


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
