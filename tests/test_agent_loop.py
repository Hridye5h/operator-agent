"""Agent-loop invariants.

These use stub tools rather than the browser, so they run in milliseconds and
assert the loop's guarantees directly: that it cannot claim success without
verification, that it distinguishes retriable from permanent failure, and that
it escalates instead of guessing.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent.core import Agent, AgentConfig, _resolve_amount  # noqa: E402
from agent.llm import Decision  # noqa: E402
from agent.tools import ToolRegistry, ToolResult  # noqa: E402
from agent.trace import Trace  # noqa: E402


class SequencePlanner:
    """Replays a fixed list of decisions; repeats the last one if it runs out."""

    name = "sequence"

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.seen_contexts = []
        self._i = 0

    def decide(self, system, context):
        self.seen_contexts.append(context)
        if self._i < len(self.decisions):
            d = self.decisions[self._i]
            self._i += 1
            return d
        return self.decisions[-1]


@pytest.fixture()
def trace(tmp_path):
    return Trace("test goal", root=tmp_path)


def make_registry(**tools):
    reg = ToolRegistry()
    for name, fn in tools.items():
        reg.tool(name, f"stub {name}", {"type": "object", "properties": {}})(fn)
    return reg


# -- verification is mandatory ---------------------------------------------

def test_cannot_finish_without_verifying(trace):
    reg = make_registry(verify_payable=lambda **k: ToolResult(True, data={"id": 1}))
    planner = SequencePlanner([Decision("finish", {})] * 3)
    agent = Agent(planner, reg, AgentConfig(max_steps=3))

    result = agent.run("do the thing", trace)

    assert result.status == "exhausted"
    assert not result.verified
    assert any("Refused to finish" in c for c in planner.seen_contexts[1:])


def test_finish_allowed_once_verified(trace):
    reg = make_registry(
        verify_payable=lambda **k: ToolResult(True, data={"id": 7}, display="VERIFIED 7")
    )
    planner = SequencePlanner([
        Decision("verify_payable", {}),
        Decision("finish", {}),
    ])
    agent = Agent(planner, reg, AgentConfig(max_steps=4))

    result = agent.run("do the thing", trace)

    assert result.status == "completed"
    assert result.verified
    assert result.evidence["id"] == 7


def test_failed_verification_does_not_count_as_success(trace):
    reg = make_registry(
        verify_payable=lambda **k: ToolResult(
            False, error="MISMATCH: amount stored 10.0 != expected 99.0")
    )
    planner = SequencePlanner([
        Decision("verify_payable", {}),
        Decision("finish", {}),
    ])
    agent = Agent(planner, reg, AgentConfig(max_steps=4))

    result = agent.run("do the thing", trace)

    assert not result.verified
    assert result.status == "exhausted"
    assert not result.ok


# -- failure handling -------------------------------------------------------

def test_retriable_failure_then_success(trace):
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return ToolResult(False, error="SYS-503 busy", retriable=True)
        return ToolResult(True, data={"record_id": "1"}, display="ok")

    reg = make_registry(
        submit_payable=flaky,
        verify_payable=lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"),
    )
    planner = SequencePlanner([
        Decision("submit_payable", {}),
        Decision("submit_payable", {}),
        Decision("verify_payable", {}),
        Decision("finish", {}),
    ])
    agent = Agent(planner, reg, AgentConfig(max_steps=6))

    result = agent.run("do the thing", trace)

    assert calls["n"] == 2
    assert result.status == "completed"
    assert result.verified


def test_tool_exception_is_an_observation_not_a_crash(trace):
    def explodes(**kwargs):
        raise RuntimeError("socket closed")

    reg = make_registry(
        submit_payable=explodes,
        verify_payable=lambda **k: ToolResult(True, data={"id": 2}, display="VERIFIED"),
    )
    planner = SequencePlanner([
        Decision("submit_payable", {}),
        Decision("verify_payable", {}),
        Decision("finish", {}),
    ])
    agent = Agent(planner, reg, AgentConfig(max_steps=6))

    result = agent.run("do the thing", trace)

    assert result.status == "completed"          # loop survived the crash
    assert any("socket closed" in c for c in planner.seen_contexts)


def test_unknown_tool_is_reported_not_fatal(trace):
    reg = make_registry(
        verify_payable=lambda **k: ToolResult(True, data={"id": 3}, display="VERIFIED"))
    planner = SequencePlanner([
        Decision("teleport", {}),
        Decision("verify_payable", {}),
        Decision("finish", {}),
    ])
    agent = Agent(planner, reg, AgentConfig(max_steps=6))

    result = agent.run("do the thing", trace)

    assert result.status == "completed"
    assert any("no such tool" in c for c in planner.seen_contexts)


# -- escalation -------------------------------------------------------------

def test_no_human_available_stops_safely(trace):
    reg = make_registry(
        ask_human=lambda **k: ToolResult(
            False, error="NO_HUMAN_AVAILABLE: stopping rather than guessing."),
        submit_payable=lambda **k: ToolResult(True, data={"record_id": "9"}),
    )
    planner = SequencePlanner([
        Decision("ask_human", {"question": "which amount?"}),
        Decision("submit_payable", {}),
    ])
    agent = Agent(planner, reg, AgentConfig(max_steps=5))

    result = agent.run("do the thing", trace)

    assert result.status == "escalated"
    assert not result.verified
    # It must NOT have gone on to write anything after the failed escalation.
    assert all(s.summary.count("submit_payable") == 0
               for s in trace.steps if s.kind == "tool_call")


def test_low_confidence_is_rerouted_to_a_human(trace):
    asked = {}

    def ask_human(question="", **k):
        asked["q"] = question
        return ToolResult(True, data={"answer": "proceed"}, display="human answered")

    reg = make_registry(
        ask_human=ask_human,
        submit_payable=lambda **k: ToolResult(True, data={"record_id": "4"}),
        verify_payable=lambda **k: ToolResult(True, data={"id": 4}, display="VERIFIED"),
    )
    planner = SequencePlanner([
        Decision("submit_payable", {}, reasoning="not sure about this", confidence=0.1),
        Decision("verify_payable", {}),
        Decision("finish", {}),
    ])
    agent = Agent(planner, reg, AgentConfig(max_steps=6, confidence_floor=0.4))

    agent.run("do the thing", trace)

    assert "submit_payable" in asked.get("q", "")
    assert any(s.kind == "decision" and "escalating" in s.summary for s in trace.steps)


# -- working state ----------------------------------------------------------

def test_extracted_values_bind_into_later_calls(trace):
    captured = {}

    def read_invoice(**kwargs):
        return ToolResult(True, data={
            "vendor": "Globex Industries",
            "invoice_number": "GX-1",
            "due_date": "2026-09-10",
            "amount": 96400.0,
        }, display="read ok")

    def submit_payable(**kwargs):
        captured.update(kwargs)
        return ToolResult(True, data={"record_id": "1"}, display="ok")

    reg = make_registry(
        read_invoice=read_invoice,
        submit_payable=submit_payable,
        verify_payable=lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"),
    )
    planner = SequencePlanner([
        Decision("read_invoice", {}),
        Decision("submit_payable", {"use_extracted": True}),
        Decision("verify_payable", {}),
        Decision("finish", {}),
    ])
    agent = Agent(planner, reg, AgentConfig(max_steps=6))

    agent.run("do the thing", trace)

    assert captured["invoice_number"] == "GX-1"
    assert captured["amount"] == "96400.00"      # normalised for the strict form
    assert captured["due_date"] == "2026-09-10"


def test_human_answer_overrides_unresolved_amount(trace):
    captured = {}

    reg = make_registry(
        read_invoice=lambda **k: ToolResult(True, data={
            "vendor": "Globex", "invoice_number": "GX-2", "due_date": "2026-09-10",
            "amount": None, "ambiguity": "AMBIGUITY: two totals",
            "amount_candidates": [{"label": "printed", "value": 118590.0},
                                  {"label": "amended", "value": 96400.0}],
        }, display="AMBIGUITY: two totals"),
        ask_human=lambda **k: ToolResult(True, data={"answer": "amended total: 96400.00"},
                                         display="human answered"),
        submit_payable=lambda **k: (captured.update(k), ToolResult(
            True, data={"record_id": "1"}, display="ok"))[1],
        verify_payable=lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"),
    )
    planner = SequencePlanner([
        Decision("read_invoice", {}),
        Decision("ask_human", {"question": "which amount?"}),
        Decision("submit_payable", {"use_extracted": True}),
        Decision("verify_payable", {}),
        Decision("finish", {}),
    ])
    agent = Agent(planner, reg, AgentConfig(max_steps=7))

    agent.run("do the thing", trace)

    assert captured["amount"] == "96400.00"


# -- approval gate ----------------------------------------------------------

def over_limit(limit):
    def policy(action, args):
        if action != "submit_payable":
            return None
        if float(str(args.get("amount", 0)).replace(",", "")) > limit:
            return f"amount exceeds the {limit:,.0f} limit"
        return None
    return policy


def test_policy_requires_approval_before_a_large_write(trace):
    written, asked = [], []

    reg = make_registry(
        ask_human=lambda **k: (asked.append(k.get("question", "")),
                               ToolResult(True, data={"answer": "approve"}))[1],
        submit_payable=lambda **k: (written.append(k),
                                    ToolResult(True, data={"record_id": "1"}))[1],
        verify_payable=lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"),
    )
    planner = SequencePlanner([
        Decision("submit_payable", {"amount": "126500.00"}),
        Decision("verify_payable", {}),
        Decision("finish", {}),
    ])
    agent = Agent(planner, reg,
                  AgentConfig(max_steps=6, approval_policy=over_limit(100000)))

    result = agent.run("record it", trace)

    assert len(asked) == 1 and "submit_payable" in asked[0]
    assert len(written) == 1                      # approved, so it went through
    assert result.status == "completed"


def test_small_write_is_not_gated(trace):
    asked = []
    reg = make_registry(
        ask_human=lambda **k: (asked.append(k), ToolResult(True, data={"answer": "approve"}))[1],
        submit_payable=lambda **k: ToolResult(True, data={"record_id": "1"}),
        verify_payable=lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"),
    )
    planner = SequencePlanner([
        Decision("submit_payable", {"amount": "500.00"}),
        Decision("verify_payable", {}),
        Decision("finish", {}),
    ])
    agent = Agent(planner, reg,
                  AgentConfig(max_steps=6, approval_policy=over_limit(100000)))

    agent.run("record it", trace)

    assert asked == []                            # under the limit, no interruption


def test_declined_action_is_never_executed(trace):
    written = []
    reg = make_registry(
        ask_human=lambda **k: ToolResult(True, data={"answer": "decline"}),
        submit_payable=lambda **k: (written.append(k), ToolResult(True, data={"record_id": "1"}))[1],
        verify_payable=lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"),
    )
    planner = SequencePlanner([Decision("submit_payable", {"amount": "126500.00"})] * 6)
    agent = Agent(planner, reg,
                  AgentConfig(max_steps=8, approval_policy=over_limit(100000)))

    result = agent.run("record it", trace)

    assert written == []                          # nothing was written
    assert result.status == "escalated"
    assert not result.verified


def test_a_refusal_is_not_re_escalated_to_the_human(trace):
    """Asking a person the same question six times is its own failure."""
    asked = []
    reg = make_registry(
        ask_human=lambda **k: (asked.append(k), ToolResult(True, data={"answer": "decline"}))[1],
        submit_payable=lambda **k: ToolResult(True, data={"record_id": "1"}),
        verify_payable=lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"),
    )
    planner = SequencePlanner([Decision("submit_payable", {"amount": "126500.00"})] * 8)
    agent = Agent(planner, reg,
                  AgentConfig(max_steps=10, approval_policy=over_limit(100000)))

    agent.run("record it", trace)

    assert len(asked) == 1                        # asked once, remembered after


def test_tool_declaring_requires_approval_is_gated(trace):
    written, asked = [], []
    reg = ToolRegistry()
    reg.tool("dangerous", "stub", {"type": "object", "properties": {}},
             requires_approval=True)(
        lambda **k: (written.append(k), ToolResult(True, data={"ok": 1}))[1])
    reg.tool("ask_human", "stub", {"type": "object", "properties": {}})(
        lambda **k: (asked.append(k), ToolResult(True, data={"answer": "approve"}))[1])
    reg.tool("verify_payable", "stub", {"type": "object", "properties": {}})(
        lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"))

    planner = SequencePlanner([
        Decision("dangerous", {}),
        Decision("verify_payable", {}),
        Decision("finish", {}),
    ])
    agent = Agent(planner, reg, AgentConfig(max_steps=6))

    agent.run("do it", trace)

    assert len(asked) == 1                        # the declaration alone gated it
    assert len(written) == 1


def test_approval_with_no_human_channel_stops_the_run(trace):
    written = []
    reg = make_registry(
        submit_payable=lambda **k: (written.append(k), ToolResult(True, data={"record_id": "1"}))[1],
        verify_payable=lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"),
    )
    planner = SequencePlanner([Decision("submit_payable", {"amount": "126500.00"})] * 3)
    agent = Agent(planner, reg,
                  AgentConfig(max_steps=5, approval_policy=over_limit(100000)))

    result = agent.run("record it", trace)

    assert written == []
    assert result.status == "escalated"


def test_a_broken_policy_fails_closed(trace):
    written = []

    def exploding_policy(action, args):
        raise ValueError("policy service down")

    reg = make_registry(
        ask_human=lambda **k: ToolResult(True, data={"answer": "decline"}),
        submit_payable=lambda **k: (written.append(k), ToolResult(True, data={"record_id": "1"}))[1],
        verify_payable=lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"),
    )
    planner = SequencePlanner([Decision("submit_payable", {"amount": "1.00"})] * 4)
    agent = Agent(planner, reg,
                  AgentConfig(max_steps=6, approval_policy=exploding_policy))

    agent.run("record it", trace)

    assert written == []        # a policy that errors must not wave the action through


# -- resolving an ambiguity safely -----------------------------------------

CANDIDATES = [{"label": "printed total", "value": 118590.0},
              {"label": "amended total", "value": 96400.0}]


@pytest.mark.parametrize("answer,expected", [
    ("amended total: 96400.00", 96400.0),
    ("96400", 96400.0),
    ("118,590.00", 118590.0),
    ("use the printed one", 118590.0),
    ("the amended figure", 96400.0),
])
def test_a_reply_naming_one_of_the_amounts_resolves_it(answer, expected):
    assert _resolve_amount(answer, CANDIDATES) == expected


@pytest.mark.parametrize("answer", [
    "see the email from 15 March",     # would once have meant 15.00
    "use invoice GX-90155 figure",     # would once have meant 90155.00
    "the one from Q3",                 # would once have meant 3.00
    "ask Priya in accounts",
    "99999.00",                        # a figure that is not on the invoice
])
def test_prose_is_never_mistaken_for_an_amount(answer):
    assert _resolve_amount(answer, CANDIDATES) is None


@pytest.mark.parametrize("answer,expected", [
    ("73219.00", 73219.0),
    ("INR 73,219.00", 73219.0),
    ("see the email from 15 March", None),
    ("roughly 400 or so", None),
])
def test_with_no_candidates_only_a_bare_amount_counts(answer, expected):
    assert _resolve_amount(answer, []) == expected


def test_unhelpful_answer_does_not_produce_a_write(trace):
    """The dangerous case: a reply with a stray digit becoming the payable.

    It would also have passed verification, since verification compares the
    write against intent and the intent would already be wrong.
    """
    written = []

    reg = make_registry(
        read_invoice=lambda **k: ToolResult(True, data={
            "vendor": "Globex", "invoice_number": "GX-2", "due_date": "2026-09-10",
            "amount": None, "ambiguity": "AMBIGUITY: two totals",
            "amount_candidates": CANDIDATES,
        }, display="AMBIGUITY: two totals"),
        ask_human=lambda **k: ToolResult(True, data={"answer": "see the email from 15 March"},
                                         display="human answered"),
        submit_payable=lambda **k: (written.append(k),
                                    ToolResult(True, data={"record_id": "1"}))[1],
        verify_payable=lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"),
    )
    planner = SequencePlanner([
        Decision("read_invoice", {}),
        Decision("ask_human", {"question": "which amount?"}),
        Decision("submit_payable", {"use_extracted": True}),
        Decision("submit_payable", {"use_extracted": True}),
        Decision("submit_payable", {"use_extracted": True}),
        Decision("submit_payable", {"use_extracted": True}),
    ])
    agent = Agent(planner, reg, AgentConfig(max_steps=8))

    result = agent.run("record it", trace)

    assert written == []                      # nothing reached the system
    assert result.status == "escalated"
    assert not result.verified


def test_a_resolving_answer_does_produce_the_right_write(trace):
    written = []

    reg = make_registry(
        read_invoice=lambda **k: ToolResult(True, data={
            "vendor": "Globex", "invoice_number": "GX-2", "due_date": "2026-09-10",
            "amount": None, "ambiguity": "AMBIGUITY: two totals",
            "amount_candidates": CANDIDATES,
        }, display="AMBIGUITY: two totals"),
        ask_human=lambda **k: ToolResult(True, data={"answer": "use the amended one"},
                                         display="human answered"),
        submit_payable=lambda **k: (written.append(k),
                                    ToolResult(True, data={"record_id": "1"}))[1],
        verify_payable=lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"),
    )
    planner = SequencePlanner([
        Decision("read_invoice", {}),
        Decision("ask_human", {"question": "which amount?"}),
        Decision("submit_payable", {"use_extracted": True}),
        Decision("verify_payable", {}),
        Decision("finish", {}),
    ])
    agent = Agent(planner, reg, AgentConfig(max_steps=8))

    result = agent.run("record it", trace)

    assert len(written) == 1
    assert written[0]["amount"] == "96400.00"
    assert result.status == "completed"
