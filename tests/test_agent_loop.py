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

from agent.core import Agent, AgentConfig  # noqa: E402
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
