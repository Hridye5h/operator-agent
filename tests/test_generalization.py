"""Generalization and the invariants that make it safe.

Three groups:

* The loop's guarantees hold against a planner that misbehaves — one that
  guesses a contested value, or verifies less than it wrote. These are the
  guarantees that used to depend on the deterministic planner behaving well.
* The same loop and the same deterministic planner run two different company
  procedures, with nothing task-specific in either.
* A static guard: the loop and the planner contain no task-specific names.
"""

import io
import pathlib
import re
import sys
from contextlib import redirect_stdout

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent.core import Agent, AgentConfig, _resolve_choice, render_procedure  # noqa: E402
from agent.llm import Decision, ScriptedPlanner, _pick_document  # noqa: E402
from agent.procedures import check_procedure, load_procedures, select_procedure  # noqa: E402
from agent.tools import ToolRegistry, ToolResult, infer_kind  # noqa: E402
from agent.trace import Trace  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


class SequencePlanner:
    name = "sequence"

    def __init__(self, decisions):
        self.decisions, self._i, self.contexts = list(decisions), 0, []

    def decide(self, system, context):
        self.contexts.append(context)
        d = self.decisions[min(self._i, len(self.decisions) - 1)]
        self._i += 1
        return d


def props(*names):
    return {"type": "object", "properties": {n: {"type": "string"} for n in names}}


@pytest.fixture()
def trace(tmp_path):
    return Trace("test goal", root=tmp_path)


def run(agent, goal, trace, **kw):
    with redirect_stdout(io.StringIO()):
        return agent.run(goal, trace, **kw)


CONTESTED_READ = {
    "fields": {"vendor": "Globex", "invoice_number": "GX-9", "due_date": "2026-09-10",
               "amount": None},
    "contested": {"field": "amount", "message": "AMBIGUITY: two totals",
                  "candidates": [{"label": "printed total", "value": "118590.00"},
                                 {"label": "amended total", "value": "96400.00"}]},
}


def payable_registry(written, verified_with, read=CONTESTED_READ, answer="x"):
    reg = ToolRegistry()
    reg.tool("read_invoice", "r", props("file"), kind="read")(
        lambda **k: ToolResult(True, data=read, display="CONTESTED (amount): two totals"))
    reg.tool("submit_payable", "w", props("vendor", "invoice_number", "amount", "due_date"),
             kind="write")(
        lambda **k: (written.append(k), ToolResult(True, data={"record_id": "1"}))[1])
    reg.tool("verify_payable", "v",
             props("invoice_number", "expect_vendor", "expect_amount", "expect_due_date"),
             kind="verify")(
        lambda **k: (verified_with.append(k), ToolResult(True, data={"id": 1},
                                                         display="VERIFIED"))[1])
    reg.tool("ask_human", "a", props("question", "options"), kind="ask")(
        lambda **k: ToolResult(True, data={"answer": answer}))
    return reg


# --------------------------------------------------------------------------
# Contested values cannot be guessed — enforced by the loop
# --------------------------------------------------------------------------

def test_a_planner_cannot_write_its_own_guess_for_a_contested_value(trace):
    """Regression: an LLM that ignored the ambiguity used to get its guess
    written, and the run then verified clean."""
    written, verified_with = [], []
    guessing = SequencePlanner([
        Decision("read_invoice", {}),
        Decision("submit_payable", {"vendor": "Globex", "invoice_number": "GX-9",
                                    "amount": "118590.00", "due_date": "2026-09-10"},
                 "the printed total looks official", 0.9),
    ] * 1 + [Decision("submit_payable", {"amount": "118590.00"}, "", 0.9)] * 4)

    result = run(Agent(guessing, payable_registry(written, verified_with),
                       AgentConfig(max_steps=8)), "record it", trace)

    assert written == []
    assert result.status == "escalated"
    assert not result.verified


def test_a_planner_cannot_write_a_contested_value_blank_either(trace):
    written, verified_with = [], []
    planner = SequencePlanner([Decision("read_invoice", {})]
                              + [Decision("submit_payable", {"use_extracted": True})] * 4)

    result = run(Agent(planner, payable_registry(written, verified_with),
                       AgentConfig(max_steps=8)), "record it", trace)

    assert written == []
    assert result.status == "escalated"


def test_only_a_human_naming_a_candidate_clears_the_contest(trace):
    written, verified_with = [], []
    planner = SequencePlanner([
        Decision("read_invoice", {}),
        Decision("ask_human", {"question": "which amount?"}),
        Decision("submit_payable", {"use_extracted": True}),
        Decision("verify_payable", {}),
        Decision("finish", {}),
    ])
    reg = payable_registry(written, verified_with, answer="use the amended total")

    result = run(Agent(planner, reg, AgentConfig(max_steps=8)), "record it", trace)

    assert [w["amount"] for w in written] == ["96400.00"]
    assert result.ok


# --------------------------------------------------------------------------
# Verification covers what was written — enforced by the loop
# --------------------------------------------------------------------------

CLEAN_READ = {"fields": {"vendor": "Northwind", "invoice_number": "NW-1",
                         "due_date": "2026-07-22", "amount": "73219.00"}}


def test_verification_compares_every_written_field_even_if_the_planner_passes_none(trace):
    written, verified_with = [], []
    planner = SequencePlanner([
        Decision("read_invoice", {}),
        Decision("submit_payable", {"use_extracted": True}),
        Decision("verify_payable", {}),                    # no expectations at all
        Decision("finish", {}),
    ])
    reg = payable_registry(written, verified_with, read=CLEAN_READ)

    run(Agent(planner, reg, AgentConfig(max_steps=6)), "record it", trace)

    assert verified_with[0] == {"invoice_number": "NW-1", "expect_vendor": "Northwind",
                                "expect_amount": "73219.00",
                                "expect_due_date": "2026-07-22"}


def test_verification_ignores_expectations_the_planner_invents(trace):
    """A planner cannot make verification check something other than the write."""
    written, verified_with = [], []
    planner = SequencePlanner([
        Decision("read_invoice", {}),
        Decision("submit_payable", {"use_extracted": True}),
        Decision("verify_payable", {"expect_amount": "1.00"}),
        Decision("finish", {}),
    ])
    reg = payable_registry(written, verified_with, read=CLEAN_READ)

    run(Agent(planner, reg, AgentConfig(max_steps=6)), "record it", trace)

    assert verified_with[0]["expect_amount"] == "73219.00"


def test_the_trace_records_which_written_fields_were_verified(trace):
    written, verified_with = [], []
    planner = SequencePlanner([
        Decision("read_invoice", {}),
        Decision("submit_payable", {"use_extracted": True}),
        Decision("verify_payable", {}),
        Decision("finish", {}),
    ])
    reg = payable_registry(written, verified_with, read=CLEAN_READ)

    run(Agent(planner, reg, AgentConfig(max_steps=6)), "record it", trace)

    check = next(s for s in trace.steps if s.kind == "verification")
    assert check.detail["verified_fields"] == ["amount", "due_date", "invoice_number",
                                               "vendor"]
    assert check.detail["unverifiable_fields"] == []


def test_verification_is_recognised_by_kind_not_name(trace):
    reg = ToolRegistry()
    reg.tool("audit_ledger", "v", props(), kind="verify")(
        lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"))
    planner = SequencePlanner([Decision("audit_ledger", {}), Decision("finish", {})])

    result = run(Agent(planner, reg, AgentConfig(max_steps=4)), "x", trace)

    assert result.ok


def test_a_tool_merely_named_verify_does_not_count(trace):
    reg = ToolRegistry()
    reg.tool("verify_something", "not really", props(), kind="other")(
        lambda **k: ToolResult(True, data={}, display="looks fine"))
    planner = SequencePlanner([Decision("verify_something", {}), Decision("finish", {})])

    result = run(Agent(planner, reg, AgentConfig(max_steps=3)), "x", trace)

    assert not result.verified


def test_kind_inference_is_only_a_default():
    assert infer_kind("verify_payable") == "verify"
    assert infer_kind("register_vendor") == "write"
    assert infer_kind("list_vendor_forms") == "discover"
    assert infer_kind("teleport") == "other"


# --------------------------------------------------------------------------
# Resolving a contested value from a human reply
# --------------------------------------------------------------------------

ACCOUNTS = [{"label": "account on signed form", "value": "918020004561237"},
            {"label": "account from email update", "value": "30481122334455"}]


@pytest.mark.parametrize("answer,expected", [
    ("use the account from the email update", "30481122334455"),
    ("go with the signed form", "918020004561237"),
    ("918020004561237", "918020004561237"),
    ("keep the account", None),                         # shared word: ambiguous
    ("not 30481122334455, use 918020004561237", None),  # names both: refuse
    ("the one from the form", None),                    # points at both labels
])
def test_overlapping_labels_and_negations_are_not_misread(answer, expected):
    assert _resolve_choice(answer, ACCOUNTS) == expected


def test_leading_zeros_survive_resolution():
    cands = [{"label": "account on signed form", "value": "001122334455"},
             {"label": "account from email update", "value": "009988776655"}]
    assert _resolve_choice("001122334455", cands) == "001122334455"


# --------------------------------------------------------------------------
# Company procedures
# --------------------------------------------------------------------------

PROCS = load_procedures(ROOT / "company" / "procedures")


def test_both_shipped_procedures_load():
    assert {p["name"] for p in PROCS} == {"record_vendor_invoice", "onboard_supplier"}


@pytest.mark.parametrize("goal,expected", [
    ("Find the latest invoice from Northwind Traders and record it as a payable.",
     "record_vendor_invoice"),
    ("Onboard Zenith Components as a new supplier in the Vendor Master.",
     "onboard_supplier"),
    ("Book a meeting room for Thursday.", None),
])
def test_the_goal_selects_the_procedure(goal, expected):
    chosen = select_procedure(goal, PROCS)
    assert (chosen or {}).get("name") == expected


def test_an_equally_matched_goal_selects_nothing():
    a = {"name": "a", "triggers": ["alpha"]}
    b = {"name": "b", "triggers": ["beta"]}
    assert select_procedure("alpha and beta", [a, b]) is None


def test_check_procedure_reports_missing_tools_and_wrong_kinds():
    reg = ToolRegistry()
    reg.tool("read_x", "r", props(), kind="read")(lambda **k: None)
    reg.tool("write_x", "w", props(), kind="other")(lambda **k: None)
    proc = {"steps": {"read": "read_x", "write": "write_x", "verify": "verify_x"}}

    problems = check_procedure(proc, reg)

    assert any("write_x" in p and "kind" in p for p in problems)
    assert any("verify_x" in p and "not registered" in p for p in problems)


# --------------------------------------------------------------------------
# One deterministic planner, two procedures
# --------------------------------------------------------------------------

def stub_task(discover, read, write, verify, listing, fields, written):
    reg = ToolRegistry()
    reg.tool(discover, "d", props(), kind="discover")(
        lambda **k: ToolResult(True, data=listing))
    reg.tool(read, "r", props("file"), kind="read")(
        lambda **k: ToolResult(True, data={"fields": fields}, display="read ok"))
    reg.tool(write, "w", props(*fields), kind="write")(
        lambda **k: (written.append(k), ToolResult(True, data={"record_id": "1"},
                                                   display="written"))[1])
    reg.tool(verify, "v", props(*[f"expect_{f}" for f in fields]), kind="verify")(
        lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"))
    return reg


@pytest.mark.parametrize("proc_name,goal,listing,fields", [
    ("record_vendor_invoice",
     "Find the latest invoice from Northwind Traders and record it as a payable.",
     "3 document(s):\n"
     "  northwind_NW-1.pdf | Northwind Traders | NW-1 | issued 2026-01-14\n"
     "  northwind_NW-2.pdf | Northwind Traders | NW-2 | issued 2026-06-22\n"
     "  globex_GX-1.pdf | Globex Industries | GX-1 | issued 2026-08-11",
     {"vendor": "Northwind Traders", "invoice_number": "NW-2", "amount": "73219.00",
      "due_date": "2026-07-22"}),
    ("onboard_supplier",
     "Onboard Zenith Components as a new supplier in the Vendor Master.",
     "2 document(s):\n"
     "  zenith_components_registration.pdf | Zenith Components Pvt Ltd | "
     "29AAACZ1234F1Z5 | submitted 2026-09-02\n"
     "  orbit_logistics_registration.pdf | Orbit Logistics LLP | 27AAFCO5678K1Z3 | "
     "submitted 2026-09-14",
     {"legal_name": "Zenith Components Pvt Ltd", "gstin": "29AAACZ1234F1Z5",
      "bank_account": "50200012345678"}),
])
def test_the_same_planner_executes_either_procedure(tmp_path, proc_name, goal,
                                                    listing, fields):
    proc = next(p for p in PROCS if p["name"] == proc_name)
    s = proc["steps"]
    written = []
    reg = stub_task(s["discover"], s["read"], s["write"], s["verify"],
                    listing, fields, written)

    result = run(Agent(ScriptedPlanner(), reg, AgentConfig(max_steps=10)), goal,
                 Trace(goal, root=tmp_path), procedure=proc)

    assert result.ok
    assert written == [fields]


def test_the_deterministic_planner_will_not_improvise_without_a_procedure():
    with pytest.raises(ValueError, match="procedure"):
        ScriptedPlanner().decide("sys", "GOAL: do something new\n\nHISTORY:\n")


def test_rendered_procedure_is_what_the_planner_reads():
    proc = next(p for p in PROCS if p["name"] == "onboard_supplier")
    text = render_procedure(proc)
    assert "Steps: discover=list_vendor_forms -> read=read_vendor_form" in text
    assert "fraud" in text


# --------------------------------------------------------------------------
# Choosing the document a request refers to
# --------------------------------------------------------------------------

LISTING = ("  northwind_NW-1.pdf | Northwind Traders | NW-1 | issued 2026-01-14\n"
           "  northwind_NW-2.pdf | Northwind Traders | NW-2 | issued 2026-06-22\n"
           "  globex_GX-88214.pdf | Globex Industries | GX-88214 | issued 2026-04-02\n")


@pytest.mark.parametrize("goal,expected", [
    ("the latest invoice from Northwind Traders", "northwind_NW-2.pdf"),
    ("the oldest invoice from Northwind Traders", "northwind_NW-1.pdf"),
    ("record invoice GX-88214 now", "globex_GX-88214.pdf"),
    ("an invoice from Northwind Traders", None),     # two match, no recency: ask
    ("an invoice from Umbrella Corp", None),         # nothing matches: ask
])
def test_document_choice(goal, expected):
    assert _pick_document(goal, LISTING) == expected


# --------------------------------------------------------------------------
# Static guard: nothing task-specific in the loop or the planner
# --------------------------------------------------------------------------

TASK_NAMES = re.compile(
    r"list_invoices|read_invoice|submit_payable|verify_payable|list_vendor_forms|"
    r"read_vendor_form|register_vendor|verify_vendor|Northwind|Globex|Initech|"
    r"Zenith|Orbit|Kestrel|invoice_number|due_date|gstin|bank_account")


@pytest.mark.parametrize("module", ["agent/core.py", "agent/llm.py", "agent/tools.py",
                                    "agent/procedures.py"])
def test_framework_code_names_no_task(module):
    source = (ROOT / module).read_text(encoding="utf-8")
    hits = sorted(set(TASK_NAMES.findall(source)))
    assert hits == [], f"{module} is coupled to a task: {hits}"


# --------------------------------------------------------------------------
# A contest belongs to its record, and the human's choice is binding
# --------------------------------------------------------------------------

def two_invoice_registry(written, answer="x"):
    """GX-9 states two totals; NW-1 is clean. Read by file."""
    docs = {"gx": CONTESTED_READ, "nw": CLEAN_READ}
    reg = ToolRegistry()
    reg.tool("read_invoice", "r", props("file"), kind="read")(
        lambda file="", **k: ToolResult(True, data=docs[file],
                                        display="CONTESTED (amount)" if file == "gx" else "ok"))
    reg.tool("submit_payable", "w", props("vendor", "invoice_number", "amount", "due_date"),
             kind="write")(
        lambda **k: (written.append(k), ToolResult(True, data={"record_id": "1"}))[1])
    reg.tool("verify_payable", "v",
             props("invoice_number", "expect_vendor", "expect_amount", "expect_due_date"),
             kind="verify")(lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"))
    reg.tool("ask_human", "a", props("question", "options"), kind="ask")(
        lambda **k: ToolResult(True, data={"answer": answer}))
    return reg


GUESS = {"vendor": "Globex", "invoice_number": "GX-9", "amount": "118590.00",
         "due_date": "2026-09-10"}


def test_reading_another_document_does_not_clear_a_contest(trace):
    written = []
    planner = SequencePlanner([Decision("read_invoice", {"file": "gx"}),
                               Decision("read_invoice", {"file": "nw"}),
                               Decision("submit_payable", GUESS, "", 0.9)])

    result = run(Agent(planner, two_invoice_registry(written), AgentConfig(max_steps=8)),
                 "record it", trace)

    assert written == []
    assert not result.ok


def test_a_clean_record_can_be_written_while_another_is_contested(trace):
    written = []
    planner = SequencePlanner([Decision("read_invoice", {"file": "gx"}),
                               Decision("read_invoice", {"file": "nw"}),
                               Decision("submit_payable", {"use_extracted": True}),
                               Decision("verify_payable", {}),
                               Decision("finish", {})])

    result = run(Agent(planner, two_invoice_registry(written), AgentConfig(max_steps=8)),
                 "record it", trace)

    assert [w["invoice_number"] for w in written] == ["NW-1"]
    assert result.ok


def test_the_write_must_carry_the_value_the_human_chose(trace):
    written = []
    planner = SequencePlanner([Decision("read_invoice", {"file": "gx"}),
                               Decision("ask_human", {"question": "which amount?"}),
                               Decision("submit_payable", GUESS, "", 0.9)])
    reg = two_invoice_registry(written, answer="use the amended total")

    result = run(Agent(planner, reg, AgentConfig(max_steps=8)), "record it", trace)

    assert written == []
    assert result.status == "escalated"


def test_the_human_choice_survives_reading_the_document_again(trace):
    written = []
    planner = SequencePlanner([Decision("read_invoice", {"file": "gx"}),
                               Decision("ask_human", {"question": "which amount?"}),
                               Decision("read_invoice", {"file": "gx"}),
                               Decision("submit_payable", {"use_extracted": True}),
                               Decision("verify_payable", {}),
                               Decision("finish", {})])
    reg = two_invoice_registry(written, answer="use the amended total")

    result = run(Agent(planner, reg, AgentConfig(max_steps=8)), "record it", trace)

    assert [w["amount"] for w in written] == ["96400.00"]
    assert result.ok


# --------------------------------------------------------------------------
# A verification must name a record and compare something
# --------------------------------------------------------------------------

def test_verifying_without_a_key_never_reaches_the_system(monkeypatch):
    from tools import browser

    monkeypatch.setattr(browser.httpx, "get", lambda *a, **k: pytest.fail("queried"))
    result = browser._verify("/api/payables", "invoice_number", "",
                             {"amount": "10.00"}, describe=str)
    assert not result.ok and not result.retriable
    assert "without the record's invoice_number" in result.error


def test_verifying_with_nothing_to_compare_is_not_a_verification(monkeypatch):
    from tools import browser

    monkeypatch.setattr(browser.httpx, "get", lambda *a, **k: pytest.fail("queried"))
    result = browser._verify("/api/payables", "invoice_number", "NW-1",
                             {"vendor": "", "amount": None}, describe=str)
    assert not result.ok
    assert "nothing to compare" in result.error
