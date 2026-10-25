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

from agent.core import Agent, AgentConfig, _is_approval, _resolve_amount  # noqa: E402
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


# Binding is driven by each tool's declared parameters, so stubs that take part
# in binding declare the same schema the real tools do.
def _props(*names):
    return {"type": "object", "properties": {n: {"type": "string"} for n in names}}


SCHEMAS = {
    "read_invoice": _props("file", "invoice_number", "vendor", "latest"),
    "submit_payable": _props("vendor", "invoice_number", "amount", "due_date"),
    "verify_payable": _props("invoice_number", "expect_vendor", "expect_amount",
                             "expect_due_date"),
    "ask_human": _props("question", "options"),
}


def make_registry(**tools):
    reg = ToolRegistry()
    for name, fn in tools.items():
        reg.tool(name, f"stub {name}",
                 SCHEMAS.get(name, {"type": "object", "properties": {}}))(fn)
    return reg


def contested_invoice():
    """A reader result whose amount the document itself contradicts."""
    return {
        "fields": {"vendor": "Globex", "invoice_number": "GX-2",
                   "due_date": "2026-09-10", "amount": None},
        "contested": {
            "field": "amount",
            "message": "AMBIGUITY: two totals",
            "candidates": [{"label": "printed total", "value": "118590.00"},
                           {"label": "amended total", "value": "96400.00"}],
        },
    }


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
        return ToolResult(True, data={"fields": {
            "vendor": "Globex Industries",
            "invoice_number": "GX-1",
            "due_date": "2026-09-10",
            "amount": "96400.00",
        }}, display="read ok")

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
    assert captured["amount"] == "96400.00"      # exactly as the reader established it
    assert captured["due_date"] == "2026-09-10"


def test_human_answer_overrides_unresolved_amount(trace):
    captured = {}

    reg = make_registry(
        read_invoice=lambda **k: ToolResult(True, data=contested_invoice(),
                                            display="CONTESTED (amount): two totals"),
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
        read_invoice=lambda **k: ToolResult(True, data=contested_invoice(),
                                            display="CONTESTED (amount): two totals"),
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
        read_invoice=lambda **k: ToolResult(True, data=contested_invoice(),
                                            display="CONTESTED (amount): two totals"),
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


# -- every write is read back -------------------------------------------------

def recording_payables():
    """Stub AP system that stores writes and verifies against what it stored."""
    stored, checks = {}, []

    def submit_payable(**k):
        stored[k["invoice_number"]] = dict(k)
        return ToolResult(True, data={"record_id": len(stored)})

    def verify_payable(invoice_number="", **expect):
        checks.append({"invoice_number": invoice_number, **expect})
        rec = stored.get(invoice_number)
        if rec is None:
            return ToolResult(False, error=f"no record {invoice_number}")
        for k, v in expect.items():
            if rec.get(k[len("expect_"):]) != v:
                return ToolResult(False, error=f"MISMATCH {k}")
        return ToolResult(True, data={"invoice_number": invoice_number}, display="VERIFIED")

    reg = make_registry(submit_payable=submit_payable, verify_payable=verify_payable)
    return reg, stored, checks


def write(invoice, amount):
    return Decision("submit_payable", {"vendor": "V", "invoice_number": invoice,
                                       "amount": amount, "due_date": "2026-01-01"})


def test_a_write_after_verification_must_be_verified_too(trace):
    reg, _, _ = recording_payables()
    planner = SequencePlanner([write("A-1", "10.00"), Decision("verify_payable", {}),
                               write("B-2", "20.00"), Decision("finish", {})])

    result = Agent(planner, reg, AgentConfig(max_steps=6)).run("record both", trace)

    assert result.status == "exhausted"
    assert not result.verified
    assert any("WRITES NOT YET VERIFIED" in c and "B-2" in c for c in planner.seen_contexts)
    assert any("not yet read back" in c for c in planner.seen_contexts)


def test_one_verify_does_not_cover_a_batch(trace):
    reg, _, _ = recording_payables()
    planner = SequencePlanner([write("A-1", "10.00"), write("B-2", "20.00"),
                               Decision("verify_payable", {}), Decision("finish", {})])

    result = Agent(planner, reg, AgentConfig(max_steps=6)).run("record both", trace)

    assert result.status == "exhausted"


def test_each_record_is_checked_against_its_own_write(trace):
    reg, _, checks = recording_payables()
    planner = SequencePlanner([write("A-1", "10.00"), write("B-2", "20.00"),
                               Decision("verify_payable", {"invoice_number": "A-1"}),
                               Decision("verify_payable", {"invoice_number": "B-2"}),
                               Decision("finish", {})])

    result = Agent(planner, reg, AgentConfig(max_steps=6)).run("record both", trace)

    assert result.status == "completed" and result.verified
    assert [c["expect_amount"] for c in checks] == ["10.00", "20.00"]


def test_unkeyed_verifies_work_through_the_batch(trace):
    reg, _, checks = recording_payables()
    planner = SequencePlanner([write("A-1", "10.00"), write("B-2", "20.00"),
                               Decision("verify_payable", {}), Decision("verify_payable", {}),
                               Decision("finish", {})])

    result = Agent(planner, reg, AgentConfig(max_steps=6)).run("record both", trace)

    assert result.status == "completed"
    assert [c["invoice_number"] for c in checks] == ["B-2", "A-1"]


def test_verifying_a_lookalike_record_does_not_count(trace):
    reg, stored, _ = recording_payables()
    stored["OLD-9"] = {"vendor": "V", "invoice_number": "OLD-9",
                       "amount": "10.00", "due_date": "2026-01-01"}
    planner = SequencePlanner([write("A-1", "10.00"),
                               Decision("verify_payable", {"invoice_number": "OLD-9"}),
                               Decision("finish", {})])

    result = Agent(planner, reg, AgentConfig(max_steps=5)).run("record A", trace)

    assert result.status == "exhausted"


def test_a_corrected_write_supersedes_the_earlier_one(trace):
    reg, _, checks = recording_payables()
    planner = SequencePlanner([write("A-1", "10.00"), write("A-1", "12.00"),
                               Decision("verify_payable", {}), Decision("finish", {})])

    result = Agent(planner, reg, AgentConfig(max_steps=6)).run("record A", trace)

    assert result.status == "completed"
    assert checks[0]["expect_amount"] == "12.00"


# -- only a clear yes approves ------------------------------------------------

@pytest.mark.parametrize("answer,approved", [
    ("approve", True), ("Approved.", True), ("yes", True), ("Y", True), ("ok", True),
    ("okay, go ahead", True),
    ("you decide", False), ("yesterday's figure was wrong, hold it", False),
    ("approve but not above 50,000", False), ("yes, but do not submit yet", False),
    ("no", False), ("decline", False), ("", False), ("okay no", False),
    ("yes please", True), ("confirm with finance first", False),
    ("ok, but check with finance", False), ("yes, only if it is under budget", False),
    ("approve after Friday", False), ("yes I don't mind", False),
])
def test_only_a_clear_yes_is_an_approval(answer, approved):
    assert _is_approval(answer) is approved


def test_an_unclear_reply_does_not_release_a_gated_write(trace):
    written = []
    reg = make_registry(
        ask_human=lambda **k: ToolResult(True, data={"answer": "you decide"}),
        submit_payable=lambda **k: (written.append(k),
                                    ToolResult(True, data={"record_id": "1"}))[1],
        verify_payable=lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"),
    )
    planner = SequencePlanner([Decision("submit_payable", {"amount": "126500.00"})])
    agent = Agent(planner, reg,
                  AgentConfig(max_steps=5, approval_policy=over_limit(100000)))

    result = agent.run("record it", trace)

    assert written == []
    assert result.status == "escalated"


# -- a crashed write is not retried blind --------------------------------------

def test_a_write_that_crashes_is_reported_as_outcome_unknown():
    reg = ToolRegistry()

    def submit_payable(**k):
        raise RuntimeError("screenshot failed after submit")

    reg.tool("submit_payable", "w", SCHEMAS["submit_payable"], kind="write")(submit_payable)
    result, _ = reg.invoke("submit_payable", {"amount": "10.00"})

    assert not result.ok and not result.retriable
    assert result.error_kind == "unknown"
    assert "read the record back" in result.error
    assert result.brief().startswith("ERROR (outcome unknown)")


def test_a_read_that_crashes_may_be_retried():
    reg = ToolRegistry()
    reg.tool("read_invoice", "r", SCHEMAS["read_invoice"], kind="read")(
        lambda **k: 1 / 0)
    result, _ = reg.invoke("read_invoice", {})
    assert result.retriable and result.error_kind == "transient"


def test_a_bug_inside_a_tool_is_not_reported_as_bad_arguments():
    reg = ToolRegistry()

    def read_invoice(file=""):
        return len(None)                      # TypeError from the tool's own code

    reg.tool("read_invoice", "r", SCHEMAS["read_invoice"], kind="read")(read_invoice)
    result, _ = reg.invoke("read_invoice", {"file": "x"})
    assert "bad arguments" not in result.error
    assert "TypeError" in result.error


def crashing_after_submit(landed):
    """An AP stub whose write raises after (or before) the record is stored."""
    reg, stored, checks = recording_payables()

    def submit_payable(**k):
        if landed:
            stored[k["invoice_number"]] = dict(k)
        raise RuntimeError("browser closed after submit")

    verify = reg.get("verify_payable").fn
    return make_registry(submit_payable=submit_payable, verify_payable=verify), checks


def test_a_crashed_write_must_be_read_back_before_finishing(trace):
    reg, checks = crashing_after_submit(landed=True)
    planner = SequencePlanner([write("A-1", "10.00"), Decision("finish", {}),
                               Decision("verify_payable", {}), Decision("finish", {})])

    result = Agent(planner, reg, AgentConfig(max_steps=6)).run("record A", trace)

    assert any("not yet read back" in c for c in planner.seen_contexts)
    assert result.status == "completed" and result.verified
    assert checks[0]["invoice_number"] == "A-1" and checks[0]["expect_amount"] == "10.00"


def test_a_crashed_write_that_did_not_land_cannot_complete(trace):
    reg, _ = crashing_after_submit(landed=False)
    planner = SequencePlanner([write("A-1", "10.00"), Decision("verify_payable", {}),
                               Decision("finish", {})])

    result = Agent(planner, reg, AgentConfig(max_steps=5)).run("record A", trace)

    assert result.status == "exhausted" and not result.verified


# -- a person is not interrogated -----------------------------------------------

def test_a_planner_that_stays_unsure_stops_asking(trace):
    asked = []
    reg = make_registry(
        ask_human=lambda **k: (asked.append(k), ToolResult(True, data={"answer": "go on"}))[1],
        verify_payable=lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"),
    )
    planner = SequencePlanner([Decision("verify_payable", {}, "not sure", 0.1)])

    result = Agent(planner, reg, AgentConfig(max_steps=20, max_questions=3)).run("x", trace)

    assert len(asked) == 3
    assert result.status == "escalated"


def test_a_planner_that_keeps_asking_is_stopped(trace):
    asked = []
    reg = make_registry(
        ask_human=lambda **k: (asked.append(k), ToolResult(True, data={"answer": "fine"}))[1])
    planner = SequencePlanner([Decision("ask_human", {"question": "and now?"})])

    result = Agent(planner, reg, AgentConfig(max_steps=20)).run("x", trace)

    assert len(asked) == AgentConfig().max_questions
    assert result.status == "escalated"
