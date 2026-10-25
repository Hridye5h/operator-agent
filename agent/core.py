"""The agent loop.

    Goal -> Understand -> Plan -> Execute -> Observe -> Adapt -> Verify -> Complete

The loop holds a goal, an observation buffer, a working state and a tool
registry, and repeats: ask the planner for the next action, run it, record what
came back, decide whether to continue.

It reasons about tools by their declared **kind** (discover, read, write,
verify, ask) and their declared **parameter schema** — never by name, and never
by a fixed list of fields. A reader returns ``fields``; a writer declares the
fields it accepts; binding matches the two by name. So the loop carries no
knowledge of any particular task, and a new task is new tools plus a goal.

These properties are enforced here rather than requested of the planner, because
prompts are advice and a weaker or misbehaving planner must not be able to get
around them:

* **Verification is mandatory.** A run cannot report success until a verify
  tool has read the outcome back from the system of record.
* **Verification covers what was written.** Every successful write must be
  read back before the run can finish, and each verify call is checked against
  the arguments of the write it names, not against whatever the planner chose
  to pass, so a verify call cannot quietly check fewer fields or fewer records.
* **Contested values cannot be guessed.** A contest belongs to the record it
  was read from. Until a human answer names one of the candidates, no write to
  that record may set the field — blank or with a value of the planner's own —
  and reading other documents does not clear it. Once resolved, a write to that
  record must carry exactly the value the human chose.
* **What is written is what was read.** Every value a reader could have
  supplied must agree with one document read in the run. Verification proves
  the record matches the write; this proves the write matches a source, so a
  mis-copied figure is refused rather than written and then verified.
* **Failure is an observation.** Tools never raise into the loop; they return a
  result the planner can reason about, including whether a retry is sensible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable

from agent.llm import Decision, Planner, UnusableReply
from agent.tools import Tool, ToolRegistry, ToolResult
from agent.trace import Trace

SYSTEM_PROMPT = """\
You are an autonomous operator completing a business task inside a company's systems.

You work in a loop. At each turn you receive the goal, the company procedure for \
this kind of request, the tools available, and everything observed so far. You \
reply with exactly one JSON object choosing the next single action:

  {"action": "<tool name or 'finish'>",
   "args": {...},
   "reasoning": "<one sentence>",
   "confidence": <0.0-1.0>}

Rules you must follow:
- Take ONE action per turn. Base it on what you have actually observed, not on \
assumption.
- Follow the company procedure when one is given; it is how this organisation \
does this work.
- Never invent a value you have not read. If a required value is missing or the \
source contradicts itself, call ask_human instead of choosing. Writes that touch \
a contested value are refused until a human resolves it.
- To pass values you have already extracted, set "use_extracted": true in args \
rather than copying them.
- If a tool fails and the error is marked retriable, retrying the same call is \
reasonable. A conflict means the target already exists: confirm it rather than \
creating it again. If an error is permanent, change the arguments or the \
approach — repeating it unchanged will fail again.
- Before finishing you MUST verify every record you wrote by reading it back \
from the system of record. Do not treat a success page as proof.
- Set confidence below 0.4 when you are unsure; low confidence will route the \
step to a human.
- Reply with the JSON object only.
"""

ApprovalPolicy = Callable[[str, dict[str, Any]], str | None]

# Working-state keys the loop maintains for itself, as opposed to values the
# tools established. They are never bound into a tool's arguments.
_INTERNAL = "_"


@dataclass
class AgentConfig:
    max_steps: int = 14
    confidence_floor: float = 0.4
    require_verification: bool = True
    #: Restrict verification to one named tool. ``None`` accepts any tool whose
    #: declared kind is ``verify``.
    verification_tool: str | None = None
    #: Tool used to consult a person, for escalation and approval.
    human_tool: str = "ask_human"
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
    def _context(self, goal: str, observations: list[str], state: dict[str, Any],
                 procedure: dict[str, Any] | None,
                 pending: list[dict[str, Any]] | None = None) -> str:
        parts = [f"GOAL: {goal}", ""]
        if procedure:
            parts += [render_procedure(procedure), ""]
        parts += ["TOOLS:", self.tools.describe(), ""]
        if state:
            parts += ["WORKING STATE (values you have actually established):",
                      json.dumps(state, indent=2, default=str), ""]
        if pending:
            parts += ["WRITES NOT YET VERIFIED (read each back before finishing):"]
            parts += [f"  {w['tool']}({_fmt_args(w['args'])})" for w in pending]
            parts.append("")
        parts.append("HISTORY:")
        parts += observations if observations else ["  (nothing yet)"]
        parts += ["", "What is the next single action? Reply with JSON only."]
        return "\n".join(parts)

    def _is_verification(self, action: str) -> bool:
        if self.config.verification_tool:
            return action == self.config.verification_tool
        return self.tools.kind_of(action) == "verify"

    # -- main loop ---------------------------------------------------------
    def run(self, goal: str, trace: Trace, state: dict[str, Any] | None = None,
            procedure: dict[str, Any] | None = None) -> RunResult:
        observations: list[str] = []
        state = dict(state or {})
        verified = False
        evidence: dict[str, Any] = {}
        last_error: str | None = None
        writes: list[dict[str, Any]] = []       # every successful write, in order
        reads: list[dict[str, Any]] = []        # the fields of every document read
        granted: set[str] = set()
        declined: dict[str, int] = {}
        stalls = 0
        human = self.config.human_tool

        planner_name = self.planner.name
        trace.add("plan", f"planner={planner_name}, max_steps={self.config.max_steps}")
        if procedure:
            trace.add("plan", f"procedure={procedure.get('name')}: "
                              f"{procedure.get('purpose', '')}".strip())

        for step in range(self.config.max_steps):
            pending = [w for w in writes if not w["verified"]]
            context = self._context(goal, observations, state, procedure, pending)

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
                except UnusableReply as exc:
                    planner_error = f"{type(exc).__name__}: {exc}"
                    trace.add("error",
                              f"planner reply unusable "
                              f"(attempt {attempt + 1}/{self.config.planner_retries + 1}): "
                              f"{planner_error}")
                except Exception as exc:
                    # Not a bad answer but no answer: re-prompting cannot help.
                    planner_error = f"{type(exc).__name__}: {exc}"
                    trace.add("error", f"planner unavailable: {planner_error}")
                    break

            if self.planner.name != planner_name:
                trace.add("plan", f"planner={self.planner.name} "
                                  f"(switched from {planner_name})")
                planner_name = self.planner.name

            if decision is None:
                trace.finish("failed")
                return RunResult("failed", f"planner failed: {planner_error}",
                                 trace=trace, steps=step)

            # --- finish requested ---------------------------------------
            if decision.action == "finish":
                if self.config.require_verification and (not verified or pending):
                    which = self.config.verification_tool or "a verify tool"
                    if pending:
                        listed = "; ".join(f"{w['tool']}({_fmt_args(w['args'])})"
                                           for w in pending)
                        nudge = (f"Refused to finish: {len(pending)} write(s) not yet "
                                 f"read back from the system of record: {listed}. "
                                 f"Call {which} for each.")
                    else:
                        nudge = ("Refused to finish: the outcome has not been verified. "
                                 f"Call {which} first.")
                    trace.add("decision", nudge)
                    observations.append(f"SYSTEM: {nudge}")
                    continue
                trace.add("complete", decision.reasoning or "goal satisfied")
                summary = decision.reasoning or "Task completed and verified."
                trace.finish("completed")
                return RunResult("completed", summary, verified, evidence, trace, step)

            # --- low confidence routes to a human ------------------------
            if decision.confidence < self.config.confidence_floor \
                    and decision.action != human and human in self.tools:
                trace.add("decision",
                          f"confidence {decision.confidence:.2f} below floor; escalating instead",
                          intended=decision.action)
                decision = Decision(
                    human,
                    {"question": f"I was about to call {decision.action} but I am not "
                                 f"confident. {decision.reasoning} How should I proceed?"},
                    reasoning="low confidence",
                )

            trace.add("decision", decision.reasoning or decision.action,
                      action=decision.action, confidence=decision.confidence)

            # --- bind arguments -----------------------------------------
            tool = self.tools.get(decision.action)
            target = _target_write(tool, decision.args, writes)
            args = self._bind(decision, tool, state, target)

            # --- guards -------------------------------------------------
            blocked = self._contested_block(tool, args, state) \
                or self._provenance_block(tool, args, reads, state)
            if blocked:
                stalls += 1
                if stalls > 2:
                    msg = ("the planner kept proposing a write the loop refuses; "
                           f"stopping without writing. Last refusal: {blocked}")
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

            # --- execute ------------------------------------------------
            trace.add("tool_call", f"{decision.action}({_fmt_args(args)})")
            result, elapsed = self.tools.invoke(decision.action, args)

            trace.add("observation", f"{decision.action}: {result.brief(240)}",
                      ok=result.ok, retriable=result.retriable,
                      error_kind=result.error_kind,
                      elapsed_ms=elapsed, artifacts=result.artifacts or [])

            observations.append(f"{decision.action}({_fmt_args(args)}) -> {result.brief(1500)}")

            # --- absorb the result into working state --------------------
            kind = tool.kind if tool else None
            if result.ok:
                last_error = None
                self._absorb(kind, result, state)
                if kind == "read" and isinstance(result.data, dict) \
                        and isinstance(result.data.get("fields"), dict):
                    reads.append(dict(result.data["fields"]))

                if kind == "write":
                    writes.append({"tool": decision.action, "args": dict(args),
                                   "verified": False})

                if self._is_verification(decision.action):
                    verified = True
                    evidence = result.data if isinstance(result.data, dict) \
                        else {"result": result.data}
                    if target is not None:
                        _mark_verified(tool, target, writes)
                    coverage = _coverage(tool, args, target)
                    trace.add("verification", result.brief(240), **evidence, **coverage)
            else:
                last_error = result.error
                if self._is_verification(decision.action):
                    verified = False
                    trace.add("verification", f"FAILED: {result.error}")
                if decision.action == human and not result.retriable:
                    trace.add("escalation", result.error or "unanswered escalation")
                    trace.finish("escalated")
                    return RunResult("escalated", result.error or "needs a human",
                                     False, {}, trace, step)

        trace.add("error", f"step budget exhausted after {self.config.max_steps} steps")
        trace.finish("exhausted")
        fully = verified and all(w["verified"] for w in writes)
        return RunResult("exhausted",
                         last_error or "ran out of steps before verifying the outcome",
                         fully, evidence, trace, self.config.max_steps)

    # -- guards ------------------------------------------------------------
    def _contested_block(self, tool: Tool | None, args: dict[str, Any],
                         state: dict[str, Any]) -> str | None:
        """Refuse a write that would set a value a reader reported as contested.

        This holds whether the planner left the value blank or supplied one of
        its own. The first case would send a request known to be incomplete; the
        second is a guess, and a guess that would then verify cleanly, because
        verification checks the record against what was written. Only a human
        answer naming one of the candidates clears the contest.
        """
        if tool is None or tool.kind != "write":
            return None
        for contest in state.get("_contests", []):
            name = contest["field"]
            if name not in tool.params or not _same_record(contest["key"], args, tool):
                continue
            if contest["chosen"] is None:
                options = ", ".join(f"{c.get('label')}={c.get('value')}"
                                    for c in contest["candidates"])
                return (f"Refusing {tool.name}: '{name}' is contested ({options}) and "
                        f"has not been resolved by a human. Ask with ask_human before "
                        f"writing.")
            value = str(args.get(name, "") or "").strip()
            if value != str(contest["chosen"]).strip():
                return (f"Refusing {tool.name}: a human chose {contest['chosen']} for "
                        f"'{name}' on this record, and this call would write "
                        f"{value or 'nothing'}.")
        return None

    def _provenance_block(self, tool: Tool | None, args: dict[str, Any],
                          reads: list[dict[str, Any]],
                          state: dict[str, Any]) -> str | None:
        """Refuse a write whose values no single document read in this run gives.

        Only parameters some reader supplies are checked, and all of them must
        agree with the same document, so a mis-copied figure, a value the
        document does not give, and fields spliced from two documents are all
        refused. A human's choice on a contested field stands in for the
        document's value.
        """
        if tool is None or tool.kind != "write" or not reads:
            return None
        sourced = {k for r in reads for k in r}
        claimed = {p: args[p] for p in tool.params
                   if p in sourced and args.get(p) not in (None, "")}
        if not claimed:
            return None
        closest: list[str] | None = None
        for read in reads:
            doc = _resolved(read, state.get("_contests", []))
            misses = [f"{p}={v!r} (document: {doc.get(p)!r})"
                      for p, v in claimed.items() if not _same_value(v, doc.get(p))]
            if not misses:
                return None
            if closest is None or len(misses) < len(closest):
                closest = misses
        return (f"Refusing {tool.name}: no document read in this run gives "
                f"{'; '.join(closest or [])}. Write what the document says, or ask a "
                f"human.")

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

        human = self.config.human_tool
        if human not in self.tools:
            trace.add("escalation", f"approval required ({reason}) but no human channel exists")
            return "no_human"

        trace.add("decision", f"approval required before {action}: {reason}")
        verdict, _ = self.tools.invoke(human, {
            "question": (f"Approve this action?\n"
                         f"  action: {action}({_fmt_args(args)})\n"
                         f"  why approval is required: {reason}"),
            "options": ["approve", "decline"],
        })

        if not verdict.ok:
            trace.add("escalation", verdict.error or "approval unanswered")
            return "no_human"

        answer = str((verdict.data or {}).get("answer", "")).strip().lower()
        if _is_approval(answer):
            granted.add(key)
            trace.add("decision", f"approved by human: {action}")
            return None

        declined[key] = 1
        trace.add("decision", f"declined by human: {action} ({answer or 'no reason given'})")
        return (f"A human declined {action}. Do not retry it unchanged; "
                f"either adjust the action or stop.")

    # -- binding -----------------------------------------------------------
    def _bind(self, decision: Decision, tool: Tool | None, state: dict[str, Any],
              target: dict[str, Any] | None) -> dict[str, Any]:
        """Fill a tool's arguments from what the run has established.

        Driven entirely by the tool's declared parameters, so it works for any
        task: a parameter named ``x`` is filled from the established value ``x``,
        and a parameter named ``expect_x`` from the value of ``x`` that was
        actually written.

        For verify tools the expectations are always taken from the write being
        checked, even if the planner passed its own. The point of verification is to
        compare the record with what was sent, and letting the planner choose
        the expectations would let it check less than it wrote.
        """
        args = dict(decision.args)
        wants_state = args.pop("use_extracted", False)
        args.pop("retry", None)
        options_flag = args.pop("options_from_observation", False)

        if tool is None:
            return args

        written = (target or {}).get("args", {})

        if tool.kind == "verify":
            for p in tool.params:
                if p.startswith("expect_"):
                    source = p[len("expect_"):]
                    if source in written:
                        args[p] = written[source]
                    elif p not in args and _public(state, source):
                        args[p] = state[source]
                elif p not in args:
                    if p in written:
                        args[p] = written[p]
                    elif _public(state, p):
                        args[p] = state[p]

        elif wants_state:
            for p in tool.params:
                if p not in args and _public(state, p):
                    args[p] = state[p]

        if options_flag and "options" in tool.params and state.get("_contested"):
            args["options"] = [f"{c.get('label')}: {c.get('value')}"
                               for c in state["_contested"].get("candidates", [])]

        return args

    # -- absorbing results -------------------------------------------------
    def _record_keys(self) -> set[str]:
        """Parameters that name a record: what the verify tools look records up by."""
        return {p for t in self.tools.by_kind("verify") for p in _identity(t)}

    def _absorb(self, kind: str | None, result: ToolResult, state: dict[str, Any]) -> None:
        data = result.data
        if not isinstance(data, dict):
            return

        if kind == "read":
            fields = data.get("fields")
            fields = fields if isinstance(fields, dict) else {}
            for key, value in fields.items():
                if value not in (None, "") and not key.startswith(_INTERNAL):
                    state[key] = value

            # Contests are kept per record for the whole run. Reading another
            # document must not clear one, or a planner could read its way past
            # the guard; reading the same one again must not forget the answer.
            state.pop("_contested", None)
            contested = data.get("contested")
            if isinstance(contested, dict) and contested.get("field"):
                record = {k: str(fields[k]).strip() for k in self._record_keys()
                          if fields.get(k) not in (None, "")}
                candidates = contested.get("candidates", [])
                values = [str(c.get("value")) for c in candidates]
                contests = state.get("_contests", [])
                known = next((c for c in contests if c["field"] == contested["field"]
                              and c["key"] == record
                              and [str(x.get("value")) for x in c["candidates"]] == values),
                             None)
                if known is None:
                    known = {"field": contested["field"], "candidates": candidates,
                             "message": contested.get("message", ""), "key": record,
                             "chosen": None}
                    state["_contests"] = contests + [known]
                if known["chosen"] is None:
                    state["_contested"] = known
                    state.pop(known["field"], None)     # not established
                else:
                    state[known["field"]] = known["chosen"]

        elif kind == "ask":
            answer = str(data.get("answer", ""))
            current = state.get("_contested")
            open_ = [c for c in state.get("_contests", []) if c["chosen"] is None]
            for contest in sorted(open_, key=lambda c: c is not current):
                chosen = _resolve_choice(answer, contest["candidates"])
                if chosen is not None:
                    contest["chosen"] = chosen
                    if contest is current:
                        state[contest["field"]] = chosen
                        state.pop("_contested", None)
                    return
            # The reply did not name one of the values in question, so it is
            # guidance, not a resolution. Reading a figure out of whatever digits
            # it happens to contain would corrupt the record silently and still
            # pass verification.
            state["human_guidance"] = answer

        elif kind == "write":
            if data.get("record_id"):
                state["record_id"] = data["record_id"]


def render_procedure(procedure: dict[str, Any]) -> str:
    """Company procedure as the planner sees it."""
    steps = procedure.get("steps", {})
    order = [r for r in ("discover", "read", "write", "verify") if r in steps]
    lines = [f"COMPANY PROCEDURE: {procedure.get('name', '')}"]
    if procedure.get("purpose"):
        lines.append(f"  Purpose: {procedure['purpose']}")
    if order:
        lines.append("  Steps: " + " -> ".join(f"{r}={steps[r]}" for r in order))
    for rule in procedure.get("rules", []):
        lines.append(f"  Rule: {rule}")
    return "\n".join(lines)


def _public(state: dict[str, Any], key: str) -> bool:
    return not key.startswith(_INTERNAL) and state.get(key) not in (None, "")


def _identity(tool: Tool) -> list[str]:
    """A verify tool's non-expectation parameters: what names the record."""
    return [p for p in tool.params if not p.startswith("expect_")]


def _resolved(read: dict[str, Any], contests: list[dict[str, Any]]) -> dict[str, Any]:
    """A document's fields with any human resolution of its contests applied."""
    doc = dict(read)
    for c in contests:
        if c["chosen"] is not None and c["field"] in doc and all(
                str(doc.get(k) or "").strip() == v for k, v in c["key"].items()):
            doc[c["field"]] = c["chosen"]
    return doc


_PLAIN_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _same_value(written: Any, source: Any) -> bool:
    if source in (None, ""):
        return False
    a, b = str(written).strip(), str(source).strip()
    if a == b:
        return True
    # Money may drop trailing zeros ("73219" for "73219.00"); an identifier may
    # not lose leading ones, so only a source with a decimal point qualifies.
    if "." in b and _PLAIN_NUMBER.fullmatch(a) and _PLAIN_NUMBER.fullmatch(b):
        return Decimal(a) == Decimal(b)
    return False


def _same_record(key: dict[str, str], args: dict[str, Any], tool: Tool) -> bool:
    """Whether a write could be to the record a contest belongs to.

    Only a write that names a different record is exempt; one that leaves the
    identifying fields out counts as the same record.
    """
    for p, v in key.items():
        if p in tool.params and args.get(p) not in (None, "") \
                and str(args[p]).strip() != v:
            return False
    return True


_YES = {"approve", "approved", "yes", "y", "ok", "okay"}
_NO = re.compile(r"\b(no|not|don'?t|never|decline\w*|reject\w*|den(?:y|ied)|stop|hold|wait"
                 r"|cancel\w*|but|only|if|unless|until|after|first|later|check|ask)\b|n't\b")


def _is_approval(answer: str) -> bool:
    """Only a plain yes approves: a reply that opens with one and carries no
    negation or condition. Anything else is a refusal, because approval gates
    money and a conditional yes is not a yes."""
    text = answer.strip().lower()
    words = re.findall(r"[a-z]+", text)
    return bool(words) and words[0] in _YES and not _NO.search(text)


def _target_write(tool: Tool | None, planner_args: dict[str, Any],
                  writes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The write a verify call is checking.

    If the planner named the record, it is the latest write of that record, or
    nothing at all when no write matches. Otherwise it is the latest write not
    yet read back, so successive unkeyed verifies work through a batch.
    """
    if tool is None or tool.kind != "verify" or not writes:
        return None
    named = {p: str(planner_args[p]).strip() for p in _identity(tool)
             if planner_args.get(p) not in (None, "")}
    if named:
        for w in reversed(writes):
            if all(str(w["args"].get(p, "")).strip() == v for p, v in named.items()):
                return w
        return None
    pending = [w for w in writes if not w["verified"]]
    return (pending or writes)[-1]


def _mark_verified(tool: Tool, target: dict[str, Any],
                   writes: list[dict[str, Any]]) -> None:
    # Earlier writes of the same record were superseded by the one just checked.
    ident = {p: target["args"][p] for p in _identity(tool) if p in target["args"]}
    for w in writes:
        if w is target or (ident and all(w["args"].get(p) == v for p, v in ident.items())):
            w["verified"] = True


def _coverage(tool: Tool | None, args: dict[str, Any],
              target: dict[str, Any] | None) -> dict[str, Any]:
    """Which written fields the verification actually compared."""
    if tool is None or not target:
        return {}
    written = set(target.get("args", {}))
    compared = {p[len("expect_"):] for p in args if p.startswith("expect_")}
    keys = {p for p in args if not p.startswith("expect_")}
    checked = written & (compared | keys)
    return {
        "verified_fields": sorted(checked),
        "unverifiable_fields": sorted(written - checked),
    }


MONEY_TOKEN = re.compile(r"\d[\d,]*(?:\.\d{1,2})?")


def _numbers_in(text: str) -> list[float]:
    out = []
    for raw in MONEY_TOKEN.findall(text):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            pass
    return out


def _as_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


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

    # Named by value. A reply naming two of the candidates ("not 118,590 — use
    # 96,400") is not resolved here: picking the first one mentioned would follow
    # the word "not" straight into the wrong figure.
    named = {value for number in _numbers_in(answer)
             for _, value in values if abs(number - value) < 0.005}
    if len(named) == 1:
        return named.pop()
    if len(named) > 1:
        return None

    # Named by label ("the amended one", "printed total").
    index = _match_label(answer, [label for label, _ in values])
    return values[index][1] if index is not None else None


def _match_label(answer: str, labels: list[str]) -> int | None:
    """Index of the one label the reply refers to, or None.

    Only words that belong to a single label count. Labels such as "account on
    signed form" and "account from email update" share "account", and letting a
    shared word decide would resolve "use the account from the email" to the
    signed-form account — the opposite of what was said. A reply that points at
    more than one label, or at none, resolves nothing.
    """
    lowered = answer.lower()
    word_sets = [set(re.findall(r"[a-z]{4,}", label.lower())) for label in labels]
    hits = []
    for i, words in enumerate(word_sets):
        others = set().union(*(ws for j, ws in enumerate(word_sets) if j != i))
        distinctive = words - others - {"total", "value"}
        if any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in distinctive):
            hits.append(i)
    return hits[0] if len(hits) == 1 else None


def _resolve_choice(answer: str, candidates: list[dict[str, Any]]) -> str | None:
    """Pick one of the contested candidates from a human reply, or None.

    Numeric candidates are matched numerically (so "96400" picks "96400.00");
    anything else is matched by its exact value appearing in the reply, or by a
    distinctive word of its label. Returns the candidate's own value, so what is
    written is exactly what was offered.
    """
    if not candidates:
        return None

    if all(_as_float(c.get("value")) is not None for c in candidates):
        chosen = _resolve_amount(answer, [{"label": c.get("label", ""),
                                           "value": _as_float(c["value"])}
                                          for c in candidates])
        if chosen is None:
            return None
        for c in candidates:
            if abs(_as_float(c["value"]) - chosen) < 0.005:
                return str(c["value"])
        return None

    lowered = answer.lower()
    named = [str(c.get("value")) for c in candidates
             if str(c.get("value", "")) and str(c.get("value")).lower() in lowered]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        return None
    index = _match_label(answer, [str(c.get("label", "")) for c in candidates])
    return str(candidates[index].get("value")) if index is not None else None


def _fmt_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())
