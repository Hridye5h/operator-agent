"""Browser tools — operate the company's web systems the way a person would.

The agent finds form fields by their **visible label**, presses buttons by their
**visible name**, and reads outcomes from **ARIA roles** (``alert`` for errors,
``status`` for confirmations) and from the result page's own row labels. It does
not use element ids, so the same code operates any form a person could fill in,
and a new task needs a field-to-label map rather than new browser code.

Writes always go through the form. The systems also expose a JSON API, and it is
used only to *verify* afterwards — standing in for the independent read-back a
real deployment would do against a database or a reporting endpoint.

Every submission captures screenshots, and a session can also record video, so
the trace carries visual evidence of what the agent actually saw and did.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx
from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

from agent.tools import ToolResult

BASE_URL = "http://127.0.0.1:8099"


def _on_this_system(url: str) -> bool:
    try:
        here, there = urlsplit(BASE_URL), urlsplit(url)
        return (there.scheme, there.hostname, there.port) == \
            (here.scheme, here.hostname, here.port)
    except ValueError:                      # e.g. a port that is not a number
        return False

# Natural-language cues, not system-specific ids: the same classification works
# for every form, and for any system that words its errors like a human would.
TRANSIENT = re.compile(r"\bSYS-5\d\d\b|temporarily unavailable|try again later", re.I)
CONFLICT = re.compile(r"already (?:recorded|registered|exists)|duplicate", re.I)


@dataclass
class BrowserSession:
    """Owns the Playwright lifetime for one agent run."""

    headless: bool = True
    artifacts: pathlib.Path | None = None
    video_dir: pathlib.Path | None = None
    _pw: Any = None
    _browser: Any = None
    _context: Any = None
    _page: Any = None
    _shots: int = 0
    video_path: pathlib.Path | None = field(default=None)

    def start(self) -> None:
        if self._pw is not None:
            return
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        options: dict[str, Any] = {"viewport": {"width": 1280, "height": 900}}
        if self.video_dir is not None:
            self.video_dir.mkdir(parents=True, exist_ok=True)
            options["record_video_dir"] = str(self.video_dir)
            options["record_video_size"] = {"width": 1280, "height": 900}
        self._context = self._browser.new_context(**options)
        self._page = self._context.new_page()

    def stop(self) -> None:
        if self._page is not None and self.video_dir is not None:
            try:
                self.video_path = pathlib.Path(self._page.video.path())
            except Exception:
                self.video_path = None
        # Closing the context is what flushes a recorded video to disk.
        for closer in (getattr(self._context, "close", None),
                       getattr(self._browser, "close", None),
                       getattr(self._pw, "stop", None)):
            try:
                if closer:
                    closer()
            except Exception:
                pass
        self._pw = self._browser = self._context = self._page = None

    @property
    def page(self):
        if self._page is None:
            self.start()
        return self._page

    def screenshot(self, label: str) -> str | None:
        if self.artifacts is None:
            return None
        self._shots += 1
        safe = re.sub(r"[^a-z0-9_-]+", "-", label.lower()).strip("-")
        path = self.artifacts / f"{self._shots:02d}-{safe}.png"
        try:
            self.page.screenshot(path=str(path), full_page=True)
            return path.name
        except Exception:
            return None

    # -- generic perception ------------------------------------------------
    def submit_form(self, path: str, fields: dict[str, Any], submit: str,
                    tag: str) -> ToolResult:
        """Fill a form by its visible labels, submit it, and read the outcome.

        Returns the result page's rows as ``{row label: value}`` on success. On
        failure, classifies the error so the planner knows whether to retry,
        confirm an existing record, or change its input.
        """
        page = self.page
        shots: list[str] = []
        try:
            page.goto(f"{BASE_URL}{path}", wait_until="domcontentloaded", timeout=15000)
            for label, value in fields.items():
                box = page.get_by_label(label, exact=True)
                if box.count() != 1:
                    return ToolResult(
                        False, retriable=False, error_kind="invalid", artifacts=shots,
                        error=f"form at {path} has {box.count()} fields labelled "
                              f"{label!r}; expected exactly one")
                typed = "" if value is None else str(value)
                box.fill(typed)
                # A maxlength or a formatting script can change what was typed;
                # submitting anyway would store a value nobody intended.
                held = box.input_value()
                if held != typed:
                    return ToolResult(
                        False, retriable=False, error_kind="invalid", artifacts=shots,
                        error=f"the field labelled {label!r} holds {held!r} after "
                              f"typing {typed!r}; not submitted")
            shot = self.screenshot(f"form-filled-{tag}")
            if shot:
                shots.append(shot)

            button = page.get_by_role("button", name=submit, exact=True)
            if button.count() != 1:
                return ToolResult(False, retriable=False, error_kind="invalid",
                                  artifacts=shots,
                                  error=f"no single button named {submit!r} at {path}")
            button.click()
            page.wait_for_load_state("domcontentloaded", timeout=15000)
        except PWTimeout as exc:
            return ToolResult(False, error=f"browser timeout: {exc}", retriable=True,
                              error_kind="transient", artifacts=shots)

        shot = self.screenshot(f"after-submit-{tag}")
        if shot:
            shots.append(shot)

        alert = page.get_by_role("alert")
        if alert.count():
            message = alert.first.inner_text().strip()
            if TRANSIENT.search(message):
                return ToolResult(False, error=f"temporary: {message}", retriable=True,
                                  error_kind="transient", artifacts=shots)
            if CONFLICT.search(message):
                return ToolResult(False, error=f"already exists: {message}",
                                  retriable=False, error_kind="conflict", artifacts=shots)
            return ToolResult(False, error=f"rejected: {message}", retriable=False,
                              error_kind="invalid", artifacts=shots)

        if page.get_by_role("status").count():
            return ToolResult(True, data=read_rows(page), artifacts=shots)

        return ToolResult(False, error=f"unrecognised page at {page.url}",
                          retriable=False, error_kind="invalid", artifacts=shots)


def read_rows(page) -> dict[str, str]:
    """Read a result page's ``<th>label</th><td>value</td>`` rows into a dict.

    Uses the document text rather than the rendered text. Rendered text applies
    CSS, and a stylesheet that uppercases headers turns "Vendor code" into
    "VENDOR CODE" — every lookup misses, and only labels that happen to be
    uppercase already survive.
    """
    out: dict[str, str] = {}
    rows = page.locator("table tr")
    for i in range(rows.count()):
        row = rows.nth(i)
        if row.locator("th").count() and row.locator("td").count():
            label = " ".join((row.locator("th").first.text_content() or "").split())
            value = " ".join((row.locator("td").first.text_content() or "").split())
            out[label] = value
    return out


def wait_for_server(url: str = BASE_URL, timeout: float = 20.0) -> bool:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code < 500:
                return True
        except Exception:
            time.sleep(0.25)
    return False


def _compare(record: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Field-by-field comparison.

    A field is compared numerically only when the system of record stores it as
    a number. Deciding by whether both sides *parse* as numbers would call bank
    account "0012345678" equal to "12345678", which are different accounts.
    """
    problems = []
    for name, want in expected.items():
        if want in (None, ""):
            continue
        have = record.get(name)
        if isinstance(have, (int, float)) and not isinstance(have, bool):
            try:
                same = round(float(have), 2) == round(float(str(want).replace(",", "")), 2)
            except ValueError:
                same = False
        else:
            same = str(have).strip() == str(want).strip()
        if not same:
            problems.append(f"{name} stored {have!r} != expected {want!r}")
    return problems


def _verify(endpoint: str, key: str, value: str, expected: dict[str, Any],
            describe) -> ToolResult:
    """Read one record back by key and compare it with what was intended."""
    # With no key the endpoint lists every record, and with no expectations the
    # comparison is empty; either would "verify" whatever happens to be there.
    if not str(value or "").strip():
        return ToolResult(False, retriable=False, error_kind="invalid",
                          error=f"cannot verify without the record's {key}")
    if not any(v not in (None, "") for v in expected.values()):
        return ToolResult(False, retriable=False, error_kind="invalid",
                          error=f"nothing to compare for {key}={value}: no expected "
                                f"values. Read the source or write the record first.")
    try:
        resp = httpx.get(f"{BASE_URL}{endpoint}", params={key: value}, timeout=10.0)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return ToolResult(False, error=f"verification query failed: {exc}",
                          retriable=True, error_kind="transient")

    if payload.get("count", 0) == 0:
        return ToolResult(False, data=payload,
                          error=f"NOT FOUND: no record with {key}={value} — "
                                f"the submission did not persist")
    if payload["count"] > 1:
        return ToolResult(False, data=payload,
                          error=f"DUPLICATE: {payload['count']} records with {key}={value}")

    record = payload["records"][0]
    problems = _compare(record, expected)
    if problems:
        return ToolResult(False, data=record, error="MISMATCH: " + "; ".join(problems))

    checked = [k for k, v in expected.items() if v not in (None, "")]
    return ToolResult(True, data=record,
                      display=f"VERIFIED {describe(record)} "
                              f"(compared: {', '.join([key] + checked)})")


def make_tools(session: BrowserSession):
    """Build the browser-backed tool callables bound to one session."""

    # -- accounts payable ------------------------------------------------
    def submit_payable(vendor: str = "", invoice_number: str = "",
                       amount: str = "", due_date: str = "") -> ToolResult:
        result = session.submit_form(
            "/new",
            {"Vendor": vendor, "Invoice number": invoice_number,
             "Amount": amount, "Due date": due_date},
            submit="Record payable",
            tag=invoice_number or "payable",
        )
        if not result.ok:
            return result
        rows = result.data
        stored = {
            "record_id": rows.get("Record ID", ""),
            "vendor": rows.get("Vendor", ""),
            "invoice_number": rows.get("Invoice", ""),
            "amount": rows.get("Amount", ""),
            "due_date": rows.get("Due date", ""),
        }
        result.data = stored
        result.display = (f"submitted OK -> record #{stored['record_id']} "
                          f"({stored['invoice_number']}, {stored['amount']}, "
                          f"{stored['due_date']})")
        return result

    def verify_payable(invoice_number: str = "", expect_vendor: str = "",
                       expect_amount: str = "", expect_due_date: str = "") -> ToolResult:
        """Read back from the system of record and compare against intent.

        Verification is a *separate read*, not a reuse of the success page, so a
        submission that appeared to work but did not persist is still caught.
        """
        return _verify(
            "/api/payables", "invoice_number", invoice_number,
            {"vendor": expect_vendor, "amount": expect_amount,
             "due_date": expect_due_date},
            lambda r: (f"record #{r['id']}: {r['vendor']}, {r['invoice_number']}, "
                       f"{float(r['amount']):.2f}, due {r['due_date']}"),
        )

    # -- vendor master ---------------------------------------------------
    def register_vendor(legal_name: str = "", gstin: str = "", pan: str = "",
                        bank_account: str = "", ifsc: str = "",
                        contact_email: str = "") -> ToolResult:
        result = session.submit_form(
            "/vendors/new",
            {"Legal name": legal_name, "GSTIN": gstin, "PAN": pan,
             "Bank account number": bank_account, "IFSC": ifsc,
             "Contact email": contact_email},
            submit="Register vendor",
            tag=gstin or "vendor",
        )
        if not result.ok:
            return result
        rows = result.data
        stored = {
            "record_id": rows.get("Vendor code", ""),
            "legal_name": rows.get("Legal name", ""),
            "gstin": rows.get("GSTIN", ""),
            "bank_account": rows.get("Bank account number", ""),
        }
        result.data = stored
        result.display = (f"registered OK -> vendor {stored['record_id']} "
                          f"({stored['legal_name']}, {stored['gstin']})")
        return result

    def verify_vendor(gstin: str = "", expect_legal_name: str = "", expect_pan: str = "",
                      expect_bank_account: str = "", expect_ifsc: str = "",
                      expect_contact_email: str = "") -> ToolResult:
        return _verify(
            "/api/vendors", "gstin", gstin,
            {"legal_name": expect_legal_name, "pan": expect_pan,
             "bank_account": expect_bank_account, "ifsc": expect_ifsc,
             "contact_email": expect_contact_email},
            lambda r: f"vendor {r['code']}: {r['legal_name']}, {r['gstin']}",
        )

    # -- generic ---------------------------------------------------------
    def open_page(path: str = "/") -> ToolResult:
        # The path comes from the planner, and through it from whatever the
        # documents say. "@evil.example/x" appended to the base URL is a URL
        # whose host is evil.example, so only a path on this system is allowed.
        if not _on_this_system(f"{BASE_URL}{path}") or not path.startswith("/") \
                or path.startswith("//"):
            return ToolResult(False, retriable=False, error_kind="invalid",
                              error=f"open_page takes a path on the internal system, "
                                    f"starting with '/': {path!r}")
        page = session.page
        try:
            page.goto(f"{BASE_URL}{path}", wait_until="domcontentloaded", timeout=15000)
        except PWTimeout as exc:
            return ToolResult(False, error=f"navigation timeout: {exc}", retriable=True,
                              error_kind="transient")
        if not _on_this_system(page.url):
            return ToolResult(False, retriable=False, error_kind="invalid",
                              error=f"navigation left the internal system ({page.url})")
        shot = session.screenshot(f"page{path.replace('/', '-') or '-root'}")
        body = page.locator("body").inner_text()
        return ToolResult(True, data={"url": page.url, "text": body},
                          artifacts=[shot] if shot else [],
                          display=f"{page.url}\n{body[:500]}")

    return {
        "submit_payable": submit_payable,
        "verify_payable": verify_payable,
        "register_vendor": register_vendor,
        "verify_vendor": verify_vendor,
        "open_page": open_page,
    }
