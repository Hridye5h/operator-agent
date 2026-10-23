"""Meridian AP — the mock 'internal company system'.

Deliberately a real web app with a real HTML form rather than an API the agent
could shortcut to. The agent has to drive it through a browser.

Three failure modes are built in on purpose, so the agent's recovery behaviour
is exercised rather than asserted:

1. Strict amount format  - rejects "INR 73,219.00"; wants a bare decimal.
2. Duplicate guard       - rejects an invoice number already on file.
3. Transient outage      - the first submission of each process fails with a
                           503-style page when FAIL_FIRST_SUBMIT=1 (default on).
"""

from __future__ import annotations

import os
import pathlib
import re
import sqlite3
from contextlib import asynccontextmanager, closing
from datetime import datetime

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

BASE = pathlib.Path(__file__).parent
DB = BASE / "ap.db"
templates = Jinja2Templates(directory=str(BASE / "templates"))

AMOUNT_RE = re.compile(r"^\d+(\.\d{1,2})?$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_state = {"submit_attempts": 0}


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Meridian AP", lifespan=lifespan)


def fail_first_enabled() -> bool:
    return os.environ.get("FAIL_FIRST_SUBMIT", "1") == "1"


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def init_db(reset: bool = False) -> None:
    if reset and DB.exists():
        DB.unlink()
    with closing(connect()) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS payables (
                   id             INTEGER PRIMARY KEY AUTOINCREMENT,
                   vendor         TEXT NOT NULL,
                   invoice_number TEXT NOT NULL UNIQUE,
                   amount         REAL NOT NULL,
                   due_date       TEXT NOT NULL,
                   entered_at     TEXT NOT NULL
               )"""
        )
        con.commit()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    with closing(connect()) as con:
        rows = con.execute("SELECT * FROM payables ORDER BY id DESC").fetchall()
    return templates.TemplateResponse(
        request=request, name="index.html", context={"rows": rows}
    )


@app.get("/new", response_class=HTMLResponse)
def new_form(request: Request):
    return templates.TemplateResponse(
        request=request, name="new.html", context={"error": None, "values": {}}
    )


@app.post("/new", response_class=HTMLResponse)
def create(
    request: Request,
    vendor: str = Form(""),
    invoice_number: str = Form(""),
    amount: str = Form(""),
    due_date: str = Form(""),
):
    values = {
        "vendor": vendor,
        "invoice_number": invoice_number,
        "amount": amount,
        "due_date": due_date,
    }
    _state["submit_attempts"] += 1

    # Failure mode 3: transient outage on the very first submission.
    if fail_first_enabled() and _state["submit_attempts"] == 1:
        return templates.TemplateResponse(
            request=request,
            name="new.html",
            context={
                "error": "SYS-503: accounting service temporarily unavailable, "
                         "please retry in a moment.",
                "values": values,
                "transient": True,
            },
            status_code=503,
        )

    def reject(msg: str):
        return templates.TemplateResponse(
            request=request,
            name="new.html",
            context={"error": msg, "values": values},
            status_code=400,
        )

    if not vendor.strip():
        return reject("VAL-001: Vendor is required.")
    if not invoice_number.strip():
        return reject("VAL-002: Invoice number is required.")

    # Failure mode 1: strict amount format.
    if not AMOUNT_RE.match(amount.strip()):
        return reject(
            "VAL-003: Amount must be a plain decimal with no currency symbol, "
            "no commas and no spaces (example: 73219.00)."
        )
    if not DATE_RE.match(due_date.strip()):
        return reject("VAL-004: Due date must be in YYYY-MM-DD format.")

    value = float(amount)
    if value <= 0:
        return reject("VAL-005: Amount must be greater than zero.")

    # Failure mode 2: duplicate guard.
    with closing(connect()) as con:
        dup = con.execute(
            "SELECT id FROM payables WHERE invoice_number = ?", (invoice_number.strip(),)
        ).fetchone()
        if dup:
            return reject(
                f"VAL-006: Invoice {invoice_number.strip()} is already recorded "
                f"(payable #{dup['id']}). Duplicate entries are not permitted."
            )
        cur = con.execute(
            "INSERT INTO payables (vendor, invoice_number, amount, due_date, entered_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                vendor.strip(),
                invoice_number.strip(),
                value,
                due_date.strip(),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        con.commit()
        new_id = cur.lastrowid

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={"record_id": new_id, "values": values},
    )


@app.get("/api/payables")
def api_payables(invoice_number: str | None = None):
    """Read-only endpoint the agent uses to VERIFY what actually landed.

    Verification deliberately reads back from the system of record rather than
    trusting the success page it just saw.
    """
    with closing(connect()) as con:
        if invoice_number:
            rows = con.execute(
                "SELECT * FROM payables WHERE invoice_number = ?", (invoice_number,)
            ).fetchall()
        else:
            rows = con.execute("SELECT * FROM payables ORDER BY id DESC").fetchall()
    return {"count": len(rows), "records": [dict(r) for r in rows]}


@app.post("/api/_reset")
def api_reset():
    """Test helper: wipe the ledger and the transient-failure counter."""
    init_db(reset=True)
    _state["submit_attempts"] = 0
    return {"ok": True}
