"""Covers the mock system's three deliberate failure modes.

These are not agent tests — they pin the environment's behaviour so the agent's
recovery paths are exercised against something that genuinely misbehaves.
"""

import sys
import pathlib

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mockenv.internal_app.app import app, init_db, _state  # noqa: E402


@pytest.fixture()
def client():
    init_db(reset=True)
    _state["submit_attempts"] = 0
    with TestClient(app) as c:
        yield c


GOOD = {
    "vendor": "Northwind Traders",
    "invoice_number": "NW-2026-0671",
    "amount": "73219.00",
    "due_date": "2026-07-22",
}


def test_first_submission_hits_transient_outage(client):
    r = client.post("/new", data=GOOD)
    assert r.status_code == 503
    assert "SYS-503" in r.text


def test_retry_after_transient_succeeds(client):
    client.post("/new", data=GOOD)              # absorbs the transient failure
    r = client.post("/new", data=GOOD)
    assert r.status_code == 200
    assert "Payable recorded" in r.text


def test_formatted_amount_is_rejected(client):
    client.post("/new", data=GOOD)              # absorb transient
    r = client.post("/new", data={**GOOD, "amount": "INR 73,219.00"})
    assert r.status_code == 400
    assert "VAL-003" in r.text


def test_bad_date_format_is_rejected(client):
    client.post("/new", data=GOOD)
    r = client.post("/new", data={**GOOD, "due_date": "22/07/2026"})
    assert r.status_code == 400
    assert "VAL-004" in r.text


def test_duplicate_invoice_is_rejected(client):
    client.post("/new", data=GOOD)              # absorb transient
    assert client.post("/new", data=GOOD).status_code == 200
    r = client.post("/new", data=GOOD)
    assert r.status_code == 400
    assert "VAL-006" in r.text


def test_verification_endpoint_reflects_what_was_stored(client):
    client.post("/new", data=GOOD)
    client.post("/new", data=GOOD)
    payload = client.get("/api/payables", params={"invoice_number": "NW-2026-0671"}).json()
    assert payload["count"] == 1
    rec = payload["records"][0]
    assert rec["vendor"] == "Northwind Traders"
    assert rec["amount"] == pytest.approx(73219.00)
    assert rec["due_date"] == "2026-07-22"
