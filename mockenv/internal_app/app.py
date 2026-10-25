"""Meridian Systems — the mock 'internal company systems'.

Two systems, both real web apps with real HTML forms rather than APIs the agent
could shortcut to. The agent has to drive them through a browser.

* **Accounts Payable** (``/``, ``/new``)          - record a vendor invoice
* **Vendor Master** (``/vendors``, ``/vendors/new``) - register a supplier

Failure modes are built in on purpose, so the agent's recovery behaviour is
exercised rather than asserted:

1. Strict input formats  - amounts must be bare decimals; GSTIN, PAN and IFSC
                           must follow their official formats, and the PAN must
                           match the one embedded in the GSTIN.
2. Duplicate guards      - an invoice number or GSTIN already on file is refused.
3. Transient outage      - the first submission of each process fails with a
                           503-style page when FAIL_FIRST_SUBMIT=1 (default on).

Errors are rendered in a ``role="alert"`` region and confirmations in a
``role="status"`` region, which is how the agent reads outcomes without relying
on element ids.
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
GSTIN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
PAN_RE = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
ACCOUNT_RE = re.compile(r"^\d{9,18}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

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
        con.execute(
            """CREATE TABLE IF NOT EXISTS vendors (
                   id             INTEGER PRIMARY KEY AUTOINCREMENT,
                   code           TEXT NOT NULL UNIQUE,
                   legal_name     TEXT NOT NULL,
                   gstin          TEXT NOT NULL UNIQUE,
                   pan            TEXT NOT NULL,
                   bank_account   TEXT NOT NULL,
                   ifsc           TEXT NOT NULL,
                   contact_email  TEXT NOT NULL,
                   registered_at  TEXT NOT NULL
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
        name="record.html",
        context={
            "title": "Payable recorded",
            "banner": f"Payable #{new_id} created for {vendor.strip()}, "
                      f"invoice {invoice_number.strip()}.",
            "rows": [("Record ID", new_id), ("Vendor", vendor.strip()),
                     ("Invoice", invoice_number.strip()), ("Amount", amount.strip()),
                     ("Due date", due_date.strip())],
            "back": "/",
        },
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


# ---------------------------------------------------------------------------
# Vendor Master
# ---------------------------------------------------------------------------

VENDOR_FIELDS = ("legal_name", "gstin", "pan", "bank_account", "ifsc", "contact_email")


@app.get("/vendors", response_class=HTMLResponse)
def vendors_index(request: Request):
    with closing(connect()) as con:
        rows = con.execute("SELECT * FROM vendors ORDER BY id DESC").fetchall()
    return templates.TemplateResponse(
        request=request, name="vendors.html", context={"rows": rows}
    )


@app.get("/vendors/new", response_class=HTMLResponse)
def vendor_form(request: Request):
    return templates.TemplateResponse(
        request=request, name="vendor_new.html", context={"error": None, "values": {}}
    )


@app.post("/vendors/new", response_class=HTMLResponse)
def vendor_create(
    request: Request,
    legal_name: str = Form(""),
    gstin: str = Form(""),
    pan: str = Form(""),
    bank_account: str = Form(""),
    ifsc: str = Form(""),
    contact_email: str = Form(""),
):
    values = {k: v.strip() for k, v in {
        "legal_name": legal_name, "gstin": gstin, "pan": pan,
        "bank_account": bank_account, "ifsc": ifsc, "contact_email": contact_email,
    }.items()}
    _state["submit_attempts"] += 1

    def page(msg: str, status: int, transient: bool = False):
        return templates.TemplateResponse(
            request=request, name="vendor_new.html",
            context={"error": msg, "values": values, "transient": transient},
            status_code=status,
        )

    if fail_first_enabled() and _state["submit_attempts"] == 1:
        return page("SYS-503: vendor service temporarily unavailable, "
                    "please retry in a moment.", 503, transient=True)

    if not values["legal_name"]:
        return page("VND-001: Legal name is required.", 400)
    if not GSTIN_RE.match(values["gstin"]):
        return page("VND-002: GSTIN must be 15 characters in the standard format "
                    "(e.g. 29AAACZ1234F1Z5).", 400)
    if not PAN_RE.match(values["pan"]):
        return page("VND-003: PAN must be 10 characters (e.g. AAACZ1234F).", 400)
    if values["gstin"][2:12] != values["pan"]:
        return page("VND-004: PAN does not match the PAN embedded in the GSTIN.", 400)
    if not ACCOUNT_RE.match(values["bank_account"]):
        return page("VND-005: Bank account number must be 9 to 18 digits.", 400)
    if not IFSC_RE.match(values["ifsc"]):
        return page("VND-006: IFSC must be 11 characters (e.g. HDFC0001234).", 400)
    if not EMAIL_RE.match(values["contact_email"]):
        return page("VND-007: Contact email is not a valid address.", 400)

    with closing(connect()) as con:
        dup = con.execute("SELECT code FROM vendors WHERE gstin = ?",
                          (values["gstin"],)).fetchone()
        if dup:
            return page(f"VND-409: A vendor with GSTIN {values['gstin']} is already "
                        f"registered (vendor {dup['code']}).", 409)
        count = con.execute("SELECT COUNT(*) AS n FROM vendors").fetchone()["n"]
        code = f"V-{count + 1:04d}"
        con.execute(
            "INSERT INTO vendors (code, legal_name, gstin, pan, bank_account, ifsc, "
            "contact_email, registered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (code, *(values[f] for f in VENDOR_FIELDS),
             datetime.now().isoformat(timespec="seconds")),
        )
        con.commit()

    return templates.TemplateResponse(
        request=request,
        name="record.html",
        context={
            "title": "Vendor registered",
            "banner": f"Vendor {code} registered: {values['legal_name']}.",
            "rows": [("Vendor code", code), ("Legal name", values["legal_name"]),
                     ("GSTIN", values["gstin"]), ("PAN", values["pan"]),
                     ("Bank account number", values["bank_account"]),
                     ("IFSC", values["ifsc"]), ("Contact email", values["contact_email"])],
            "back": "/vendors",
        },
    )


@app.get("/api/vendors")
def api_vendors(gstin: str | None = None):
    """Read-only endpoint the agent uses to VERIFY a registration."""
    with closing(connect()) as con:
        if gstin:
            rows = con.execute("SELECT * FROM vendors WHERE gstin = ?", (gstin,)).fetchall()
        else:
            rows = con.execute("SELECT * FROM vendors ORDER BY id DESC").fetchall()
    return {"count": len(rows), "records": [dict(r) for r in rows]}


@app.post("/api/_reset")
def api_reset():
    """Test helper: wipe the ledger and the transient-failure counter."""
    init_db(reset=True)
    _state["submit_attempts"] = 0
    return {"ok": True}
