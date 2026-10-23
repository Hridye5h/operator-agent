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
import re
from dataclasses import dataclass, field
from typing import Any, Callable

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


ApprovalPolicy = Callable[[str, dict[str, Any]], str | None]


@dataclass
class AgentConfig:
    max_steps: int = 14
    confidence_floor: float = 0.4
    require_verification: bool = True
    verification_tool: str = "verify_payable"
    #: How many times to re-prompt when a planner reply cannot be parsed.
    planner_retries: int = 2
    #: Called before every tool invocation. Return a reason string to require
    #: human approval for that specific call, or None to allow it. This is how
    #: value thresholds and other policy live outside the loop.
    approval_policy: ApprovalPolicy | None = None


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
        granted: set[str] = set()
        declined: dict[str, int] = {}
        stalls = 0

        trace.add("plan", f"planner={self.planner.name}, max_steps={self.config.max_steps}")

        for step in range(self.config.max_steps):
            context = self._context(goal, observations, state)

            # A model will occasionally reply with prose, a truncated object, or
            # nothing at all. That is a bad turn, not a dead run: show it the
            # parse error and let it correct itself before giving up.
            decision = None
            planner_error: str | None = None
            for attempt in range(self.config.planner_retries + 1):
                probe = context if attempt == 0 else (
                    f"{context}\n\nYOUR PREVIOUS REPLY COULD NOT BE USED: {planner_error}\n"
                    "Reply with a single JSON object and nothing else."
                )
                try:
                    decision = self.planner.decide(SYSTEM_PROMPT, probe)
                    break
                except Exception as exc:
                    planner_error = f"{type(exc).__name__}: {exc}"
                    trace.add("error",
                              f"planner reply unusable "
                              f"(attempt {attempt + 1}/{self.config.planner_retries + 1}): "
                              f"{planner_error}")

            if decision is None:
                trace.finish("failed")
                return RunResult("failed", f"planner failed: {planner_error}",
                                 trace=trace, steps=step)

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

            blocked = self._unresolved_block(decision.action, args, state)
            if blocked:
                stalls += 1
                if stalls > 2:
                    msg = ("a required value is still unresolved and the planner "
                           "kept trying to write anyway; stopping without writing.")
                    trace.add("escalation", msg)
                    trace.finish("escalated")
                    return RunResult("escalated", msg, False, {}, trace, step)
                trace.add("decision", f"blocked {decision.action}: {blocked}")
                observations.append(f"SYSTEM: {blocked}")
                continue

            gate = self._approval_gate(decision.action, args, granted, declined, trace)
            if gate is not None:
                if gate == "no_human":
                    trace.finish("escalated")
                    return RunResult("escalated",
                                     "action required approval but no one was available",
                                     False, {}, trace, step)
                if gate == "declined_loop":
                    msg = (f"{decision.action} was declined and the planner kept "
                           f"proposing it unchanged; stopping without writing.")
                    trace.add("escalation", msg)
                    trace.finish("escalated")
                    return RunResult("escalated", msg, False, {}, trace, step)
                observations.append(f"SYSTEM: {gate}")
                continue

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

    # -- guards ------------------------------------------------------------
    WRITE_PREFIXES = ("submit", "create", "update", "delete", "post", "send")

    def _unresolved_block(self, action: str, args: dict[str, Any],
                          state: dict[str, Any]) -> str | None:
        """Refuse a write while a value it depends on is still in question.

        Without this the agent submits a blank amount, the form rejects it, and
        the run recovers — but it has already touched the system of record with
        a request it knew was incomplete. Catching it here keeps a known-bad
        write from leaving the agent at all.
        """
        if not state.get("unresolved"):
            return None
        if not action.startswith(self.WRITE_PREFIXES):
            return None
        missing = [k for k in ("amount",) if not str(args.get(k, "")).strip()]
        if not missing:
            return None
        return (f"Refusing {action}: {', '.join(missing)} is still unresolved "
                f"({state['unresolved']}). Resolve it with ask_human before writing.")

    # -- approval ----------------------------------------------------------
    def _approval_gate(self, action: str, args: dict[str, Any],
                       granted: set[str], declined: dict[str, int],
                       trace: Trace) -> str | None:
        """Hold an action until a human approves it.

        Returns None to proceed, a message to feed back to the planner if the
        action was declined, or one of two sentinels: ``"no_human"`` when nobody
        was available to decide, and ``"declined_loop"`` when the planner keeps
        re-proposing something already refused.

        A decision is remembered for the run. Asking a person the same question
        six times because the planner will not take no for an answer is its own
        kind of failure, so a refused action is not re-escalated.

        Two things can require approval: a tool that declares
        ``requires_approval``, or the configured policy flagging this particular
        call — which is how value thresholds work without putting money logic in
        the loop.
        """
        tool = self.tools.get(action)
        if tool is None:
            return None

        reason: str | None = None
        if tool.requires_approval:
            reason = f"{action} is marked as requiring approval"
        elif self.config.approval_policy is not None:
            try:
                reason = self.config.approval_policy(action, args)
            except Exception as exc:                     # a broken policy must fail closed
                reason = f"approval policy raised {type(exc).__name__}: {exc}"

        if not reason:
            return None

        key = f"{action}:{json.dumps(args, sort_keys=True, default=str)}"
        if key in granted:
            return None

        # Already refused: answer from memory instead of asking again.
        if key in declined:
            declined[key] += 1
            if declined[key] > 2:
                return "declined_loop"
            return (f"A human already declined {action} with these exact arguments. "
                    f"Change the action or stop; it will not be approved by asking again.")

        if "ask_human" not in self.tools:
            trace.add("escalation", f"approval required ({reason}) but no human channel exists")
            return "no_human"

        trace.add("decision", f"approval required before {action}: {reason}")
        verdict, _ = self.tools.invoke("ask_human", {
            "question": (f"Approve this action?\n"
                         f"  action: {action}({_fmt_args(args)})\n"
                         f"  why approval is required: {reason}"),
            "options": ["approve", "decline"],
        })

        if not verdict.ok:
            trace.add("escalation", verdict.error or "approval unanswered")
            return "no_human"

        answer = str((verdict.data or {}).get("answer", "")).strip().lower()
        if answer.startswith(("approve", "yes", "y", "ok")):
            granted.add(key)
            trace.add("decision", f"approved by human: {action}")
            return None

        declined[key] = 1
        trace.add("decision", f"declined by human: {action} ({answer or 'no reason given'})")
        return (f"A human declined {action}. Do not retry it unchanged; "
                f"either adjust the action or stop.")

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
                candidates = data.get("amount_candidates", [])
                state["amount_candidates"] = candidates
                state["amount_options"] = [
                    f"{c['label']}: {c['value']:.2f}" for c in candidates
                ]

        elif action == "ask_human":
            answer = str(data.get("answer", ""))
            resolved = _resolve_amount(answer, state.get("amount_candidates") or [])
            if resolved is not None:
                state["amount"] = f"{resolved:.2f}"
                state.pop("unresolved", None)
                state.pop("amount_options", None)
                state.pop("amount_candidates", None)
            else:
                # The reply did not name one of the amounts in question, so it is
                # guidance, not a resolution. Inventing a figure from whatever
                # digits it happens to contain would corrupt the ledger silently
                # and still pass verification, since verification checks the
                # write against intent and intent would already be wrong.
                state["human_guidance"] = answer

        elif action.startswith("submit"):
            if data.get("record_id"):
                state["record_id"] = data["record_id"]


MONEY_TOKEN = re.compile(r"\d[\d,]*(?:\.\d{1,2})?")


def _numbers_in(text: str) -> list[float]:
    out = []
    for raw in MONEY_TOKEN.findall(text):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            pass
    return out


def _resolve_amount(answer: str, candidates: list[dict[str, Any]]) -> float | None:
    """Interpret a human reply as a choice between known amounts.

    A reply only resolves the ambiguity if it actually names one of the amounts
    in question — by value, or by the label they were offered under. Anything
    else is treated as guidance.

    This is deliberately strict. A loose reading turns "see the email from 15
    March" into a payable of 15.00, which would then pass verification, because
    verification compares the write against intent and the intent would already
    be wrong.
    """
    if not candidates:
        # Nothing to choose between: accept only a reply that is purely an
        # amount, so prose with a stray digit in it is not read as a figure.
        stripped = answer.strip().replace("INR", "").replace("Rs.", "").lstrip("₹$").strip()
        numbers = _numbers_in(stripped)
        if len(numbers) != 1:
            return None
        residue = MONEY_TOKEN.sub("", stripped).strip(" .:-")
        return numbers[0] if residue == "" else None

    values = [(c.get("label", ""), float(c["value"])) for c in candidates if "value" in c]

    # Named by value.
    for number in _numbers_in(answer):
        for _, value in values:
            if abs(number - value) < 0.005:
                return value

    # Named by label ("the amended one", "printed total").
    lowered = answer.lower()
    for label, value in values:
        words = [w for w in label.lower().replace("total", " ").split() if len(w) > 3]
        if words and any(w in lowered for w in words):
            return value

    return None


def _fmt_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())
