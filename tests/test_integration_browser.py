"""Browser-level integration: the real app, a real Chromium, no stubs.

Covers the perception layer directly — filling forms by visible label and
reading results back — and the second task end to end. Marked ``integration``
and skipped by default because each test boots the app and a browser::

    pytest -m integration
"""

import io
import pathlib
import sqlite3
import sys
from contextlib import redirect_stdout

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.integration

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "mockenv" / "internal_app" / "ap.db"


@pytest.fixture()
def env(tmp_path):
    """Boot the app (with the transient failure disabled) and a browser."""
    import os

    import run_agent as ra
    from tools import browser as bt

    os.environ["FAIL_FIRST_SUBMIT"] = "0"
    proc = ra.start_environment()
    session = bt.BrowserSession(headless=True, artifacts=tmp_path)
    session.start()
    try:
        yield ra, bt, session, tmp_path
    finally:
        session.stop()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        os.environ.pop("FAIL_FIRST_SUBMIT", None)


def vendors():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute("SELECT * FROM vendors")]
    finally:
        con.close()


def test_result_rows_are_read_with_their_real_labels(env):
    """Regression: the result page's CSS uppercases table headers, and reading
    rendered text turned "Record ID" into "RECORD ID" so every lookup missed."""
    _, bt, session, _ = env
    tools = bt.make_tools(session)

    result = tools["submit_payable"](vendor="Northwind Traders",
                                     invoice_number="NW-2026-0671",
                                     amount="73219.00", due_date="2026-07-22")

    assert result.ok, result.error
    assert result.data == {"record_id": "1", "vendor": "Northwind Traders",
                           "invoice_number": "NW-2026-0671", "amount": "73219.00",
                           "due_date": "2026-07-22"}


def test_a_missing_label_is_reported_not_guessed(env):
    _, bt, session, _ = env
    result = session.submit_form("/new", {"Purchase order": "PO-1"},
                                 submit="Record payable", tag="x")
    assert not result.ok
    assert "labelled 'Purchase order'" in result.error


def test_validation_errors_are_classified_as_permanent(env):
    _, bt, session, _ = env
    tools = bt.make_tools(session)
    result = tools["register_vendor"](legal_name="Zenith Components Pvt Ltd",
                                      gstin="29AAACZ1234F1Z5", pan="ZZZZZ9999Z",
                                      bank_account="50200012345678", ifsc="HDFC0001234",
                                      contact_email="a@b.example")
    assert not result.ok
    assert result.error_kind == "invalid"
    assert "VND-004" in result.error          # PAN must match the GSTIN


def run_scenario(ra, bt, session, goal, answers):
    from agent.core import Agent, AgentConfig
    from agent.llm import ScriptedPlanner
    from agent.procedures import load_procedures, select_procedure
    from agent.trace import Trace
    from tools.human import HumanChannel

    registry = ra.build_registry(session, HumanChannel(interactive=False,
                                                       preset=answers or {}))
    procedure = select_procedure(goal, load_procedures(ra.PROCEDURES))
    trace = Trace(goal, root=session.artifacts)
    with redirect_stdout(io.StringIO()):
        return Agent(ScriptedPlanner(), registry, AgentConfig(max_steps=14)).run(
            goal, trace, procedure=procedure), trace


def test_leading_zeros_survive_the_whole_stack(env):
    """Extraction, form, database and verification all keep "001122334455"."""
    ra, bt, session, _ = env
    result, _ = run_scenario(
        ra, bt, session,
        "Onboard Kestrel Analytics as a new supplier in the Vendor Master.", {})

    assert result.ok
    assert result.evidence["bank_account"] == "001122334455"
    assert [v["bank_account"] for v in vendors()] == ["001122334455"]


def test_a_fraud_pattern_with_nobody_to_ask_creates_no_vendor(env):
    ra, bt, session, _ = env
    result, _ = run_scenario(
        ra, bt, session,
        "Onboard Orbit Logistics as a new supplier in the Vendor Master.", None)

    assert result.status == "escalated"
    assert vendors() == []


def test_the_person_chooses_the_account_and_only_that_one_is_written(env):
    ra, bt, session, _ = env
    result, trace = run_scenario(
        ra, bt, session,
        "Onboard Orbit Logistics as a new supplier in the Vendor Master.",
        {"bank_account": "use the account on the signed form"})

    assert result.ok
    assert [v["bank_account"] for v in vendors()] == ["918020004561237"]
    check = next(s for s in trace.steps if s.kind == "verification")
    assert "bank_account" in check.detail["verified_fields"]
