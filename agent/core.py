"""The agent loop.

    Goal -> Understand -> Plan -> Execute -> Observe -> Adapt -> Verify -> Complete

The loop itself knows nothing about invoices. It holds a goal, an observation
buffer and a tool registry, and repeats: ask the planner for the next action,
run it, record what came back, decide whether to continue. Task knowledge lives
in the tools and in the goal string, which is what lets the same loop drive a
different task without modification.

Three properties are deliberate:

* **Verification is mandatory.** A run cannot report success on the strength of
  a tool returning OK. It must pass an explicit verification step that reads the
  outcome back from the system of record.
* **Failure is an observation.** Tools never raise into the loop; they return a
  result the planner can reason about, including whether a retry is sensible.
* **Escalation beats guessing.** When the planner signals low confidence or a
  tool reports an ambiguity, the agent asks a human rather than picking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agent.llm import Decision, Planner
from agent.tools import ToolRegistry, ToolResult
from agent.trace import Trace

SYSTEM_PROMPT = """\
You are an autonomous operator completing a business task inside a company's systems.

You work in a loop. At each turn you receive the goal, the tools available, and \
everything observed so far. You reply with exactly one JSON object choosing the \
next single action:

  {"action": "<tool name or 'finish'>",
   "args": {...},
   "reasoning": "<one sentence>",
   "confidence": <0.0-1.0>}

Rules you must follow:
- Take ONE action per turn. Base it on what you have actually observed, not on \
assumption.
- Never invent a value you have not read. If a required value is missing or the \
source contradicts itself, call ask_human instead of choosing.
- If a tool fails and the error is marked retriable, retrying the same call is \
reasonable. If it is marked permanent, change the arguments or the approach — \
repeating it unchanged will fail again.
- Before finishing you MUST verify the outcome by reading it back from the \
system of record. Do not treat a success page as proof.
- Set confidence below 0.4 when you are unsure; low confidence will route the \
step to a human.
- Reply with the JSON object only.
"""


@dataclass
class AgentConfig:
    max_steps: int = 14
    confidence_floor: float = 0.4
    require_verification: bool = True
    verification_tool: str = "verify_payable"


@dataclass
class RunResult:
    status: str                       # completed | escalated | failed | exhausted
    summary: str
    verified: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)
    trace: Trace | None = None
    steps: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "completed" and self.verified


class Agent:
    def __init__(self, planner: Planner, registry: ToolRegistry,
                 config: AgentConfig | None = None) -> None:
        self.planner = planner
        self.tools = registry
        self.config = config or AgentConfig()

    # -- prompt assembly ---------------------------------------------------
    def _context(self, goal: str, observations: list[str], state: dict[str, Any]) -> str:
        parts = [f"GOAL: {goal}", "", "TOOLS:", self.tools.describe(), ""]
        if state:
            parts += ["WORKING STATE (values you have actually established):",
                      json.dumps(state, indent=2, default=str), ""]
        parts.append("HISTORY:")
        parts += observations if observations else ["  (nothing yet)"]
        parts += ["", "What is the next single action? Reply with JSON only."]
        return "\n".join(parts)

    # -- main loop ---------------------------------------------------------
    def run(self, goal: str, trace: Trace, state: dict[str, Any] | None = None) -> RunResult:
        observations: list[str] = []
        state = dict(state or {})
        verified = False
        evidence: dict[str, Any] = {}
        last_error: str | None = None

        trace.add("plan", f"planner={self.planner.name}, max_steps={self.config.max_steps}")

        for step in range(self.config.max_steps):
            context = self._context(goal, observations, state)

            try:
                decision = self.planner.decide(SYSTEM_PROMPT, context)
            except Exception as exc:
                trace.add("error", f"planner failed: {type(exc).__name__}: {exc}")
                return RunResult("failed", f"planner failed: {exc}", trace=trace, steps=step)

            # --- finish requested ---------------------------------------
            if decision.action == "finish":
                if self.config.require_verification and not verified:
                    nudge = ("Refused to finish: the outcome has not been verified. "
                             f"Call {self.config.verification_tool} first.")
                    trace.add("decision", nudge)
                    observations.append(f"SYSTEM: {nudge}")
                    continue
                trace.add("complete", decision.reasoning or "goal satisfied")
                summary = decision.reasoning or "Task completed and verified."
                trace.finish("completed")
                return RunResult("completed", summary, verified, evidence, trace, step)

            # --- low confidence routes to a human ------------------------
            if decision.confidence < self.config.confidence_floor \
                    and decision.action != "ask_human" and "ask_human" in self.tools:
                trace.add("decision",
                          f"confidence {decision.confidence:.2f} below floor; escalating instead",
                          intended=decision.action)
                decision = Decision(
                    "ask_human",
                    {"question": f"I was about to call {decision.action} but I am not "
                                 f"confident. {decision.reasoning} How should I proceed?"},
                    reasoning="low confidence",
                )

            trace.add("decision", decision.reasoning or decision.action,
                      action=decision.action, confidence=decision.confidence)

            # --- execute --------------------------------------------------
            args = self._bind(decision, state)
            trace.add("tool_call", f"{decision.action}({_fmt_args(args)})")
            result, elapsed = self.tools.invoke(decision.action, args)

            trace.add("observation", f"{decision.action}: {result.brief(240)}",
                      ok=result.ok, retriable=result.retriable,
                      elapsed_ms=elapsed, artifacts=result.artifacts or [])

            observations.append(f"{decision.action}({_fmt_args(args)}) -> {result.brief()}")

            # --- absorb the result into working state --------------------
            if result.ok:
                last_error = None
                self._absorb(decision.action, result, state)
                if decision.action == self.config.verification_tool:
                    verified = True
                    evidence = result.data if isinstance(result.data, dict) else {"result": result.data}
                    trace.add("verification", result.brief(240), **evidence)
            else:
                last_error = result.error
                if decision.action == self.config.verification_tool:
                    verified = False
                    trace.add("verification", f"FAILED: {result.error}")
                if decision.action == "ask_human" and not result.retriable:
                    trace.add("escalation", result.error or "unanswered escalation")
                    trace.finish("escalated")
                    return RunResult("escalated", result.error or "needs a human",
                                     False, {}, trace, step)

        trace.add("error", f"step budget exhausted after {self.config.max_steps} steps")
        trace.finish("exhausted")
        return RunResult("exhausted",
                         last_error or "ran out of steps before verifying the outcome",
                         verified, evidence, trace, self.config.max_steps)

    # -- helpers -----------------------------------------------------------
    def _bind(self, decision: Decision, state: dict[str, Any]) -> dict[str, Any]:
        """Resolve convenience arguments against established working state.

        The planner may say "use the values you extracted" rather than copying
        numbers into its reply. Binding here keeps extracted values authoritative
        and stops the planner from re-typing (and mistyping) a figure.
        """
        args = dict(decision.args)
        wants_state = args.pop("use_extracted", False)
        args.pop("retry", None)
        options_flag = args.pop("options_from_observation", False)

        if wants_state:
            for key in ("vendor", "invoice_number", "amount", "due_date"):
                if key in state and key not in args:
                    args[key] = state[key]
            if decision.action.startswith("verify"):
                if "amount" in args:
                    args["expect_amount"] = args.pop("amount")
                if "due_date" in args:
                    args["expect_due_date"] = args.pop("due_date")
                args.pop("vendor", None)

        if options_flag and state.get("amount_options"):
            args["options"] = state["amount_options"]

        return args

    def _absorb(self, action: str, result: ToolResult, state: dict[str, Any]) -> None:
        data = result.data
        if not isinstance(data, dict):
            return

        if action.startswith("read_"):
            for key in ("vendor", "invoice_number", "due_date"):
                if data.get(key):
                    state[key] = data[key]
            if data.get("amount") is not None:
                state["amount"] = f"{float(data['amount']):.2f}"
            if data.get("ambiguity"):
                state["unresolved"] = data["ambiguity"]
                state["amount_options"] = [
                    f"{c['label']}: {c['value']:.2f}" for c in data.get("amount_candidates", [])
                ]

        elif action == "ask_human":
            answer = str(data.get("answer", ""))
            number = _money_in(answer)
            if number is not None:
                state["amount"] = f"{number:.2f}"
                state.pop("unresolved", None)
                state.pop("amount_options", None)
            else:
                state["human_guidance"] = answer

        elif action.startswith("submit"):
            if data.get("record_id"):
                state["record_id"] = data["record_id"]


def _money_in(text: str) -> float | None:
    import re

    m = re.search(r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)", text.replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _fmt_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())
