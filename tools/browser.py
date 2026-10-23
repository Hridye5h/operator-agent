"""Browser tool — drives the internal system through a real browser.

The mock system exposes a JSON API, and the agent deliberately does *not* use it
to write. Entering a payable goes through Playwright filling the same HTML form
a person would, because the point of the exercise is operating software that was
never designed for an agent. The JSON API is used only to *verify* afterwards,
standing in for the independent read-back a real deployment would do.

Every submission captures a screenshot into the run's artifact directory, so the
trace carries visual evidence of what the agent actually saw.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import Any

import httpx
from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

from agent.tools import ToolResult

BASE_URL = "http://127.0.0.1:8099"

# Error codes the system reports as temporary — worth retrying unchanged.
TRANSIENT = re.compile(r"\bSYS-5\d\d\b", re.I)
# Codes that mean the input itself is wrong — retrying unchanged is pointless.
VALIDATION = re.compile(r"\bVAL-\d{3}\b", re.I)


@dataclass
class BrowserSession:
    """Owns the Playwright lifetime for one agent run."""

    headless: bool = True
    artifacts: pathlib.Path | None = None
    _pw: Any = None
    _browser: Any = None
    _page: Any = None
    _shots: int = 0

    def start(self) -> None:
        if self._pw is not None:
            return
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page(viewport={"width": 1280, "height": 900})

    def stop(self) -> None:
        for closer in (getattr(self._browser, "close", None), getattr(self._pw, "stop", None)):
            try:
                if closer:
                    closer()
            except Exception:
                pass
        self._pw = self._browser = self._page = None

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


def make_tools(session: BrowserSession):
    """Build the browser-backed tool callables bound to one session."""

    def submit_payable(vendor: str = "", invoice_number: str = "",
                       amount: str = "", due_date: str = "") -> ToolResult:
        page = session.page
        shots: list[str] = []
        try:
            page.goto(f"{BASE_URL}/new", wait_until="domcontentloaded", timeout=15000)
            page.fill("#vendor", str(vendor))
            page.fill("#invoice_number", str(invoice_number))
            page.fill("#amount", str(amount))
            page.fill("#due_date", str(due_date))
            shot = session.screenshot(f"form-filled-{invoice_number or 'payable'}")
            if shot:
                shots.append(shot)
            page.click("#submit-payable")
            page.wait_for_load_state("domcontentloaded", timeout=15000)
        except PWTimeout as exc:
            return ToolResult(False, error=f"browser timeout: {exc}", retriable=True,
                              artifacts=shots)

        shot = session.screenshot(f"after-submit-{invoice_number or 'payable'}")
        if shot:
            shots.append(shot)

        # Success page?
        if page.locator("#success-banner").count():
            record_id = (page.locator("#r-id").inner_text() or "").strip()
            stored = {
                "record_id": record_id,
                "vendor": page.locator("#r-vendor").inner_text().strip(),
                "invoice_number": page.locator("#r-invoice").inner_text().strip(),
                "amount": page.locator("#r-amount").inner_text().strip(),
                "due_date": page.locator("#r-due").inner_text().strip(),
            }
            return ToolResult(
                True, data=stored, artifacts=shots,
                display=f"submitted OK -> record #{record_id} "
                        f"({stored['invoice_number']}, {stored['amount']}, {stored['due_date']})",
            )

        # Error page: classify so the agent knows whether a plain retry helps.
        if page.locator("#form-error").count():
            message = page.locator("#form-error").inner_text().strip()
            transient = bool(TRANSIENT.search(message))
            return ToolResult(
                False,
                error=f"{'temporary' if transient else 'rejected'}: {message}",
                retriable=transient,
                artifacts=shots,
            )

        return ToolResult(False, error=f"unrecognised page at {page.url}",
                          retriable=False, artifacts=shots)

    def verify_payable(invoice_number: str = "", expect_amount: str = "",
                       expect_due_date: str = "") -> ToolResult:
        """Read back from the system of record and compare against intent.

        Verification is a *separate read*, not a reuse of the success page, so a
        submission that appeared to work but did not persist is still caught.
        """
        try:
            resp = httpx.get(f"{BASE_URL}/api/payables",
                             params={"invoice_number": invoice_number}, timeout=10.0)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            return ToolResult(False, error=f"verification query failed: {exc}", retriable=True)

        if payload.get("count", 0) == 0:
            return ToolResult(False, data=payload,
                              error=f"NOT FOUND: no record for {invoice_number} — "
                                    f"the submission did not persist")
        if payload["count"] > 1:
            return ToolResult(False, data=payload,
                              error=f"DUPLICATE: {payload['count']} records for {invoice_number}")

        record = payload["records"][0]
        problems = []
        if expect_amount:
            try:
                if round(float(record["amount"]), 2) != round(float(expect_amount), 2):
                    problems.append(f"amount stored {record['amount']} != expected {expect_amount}")
            except ValueError:
                problems.append(f"amount not comparable: {expect_amount!r}")
        if expect_due_date and record["due_date"] != expect_due_date:
            problems.append(f"due_date stored {record['due_date']} != expected {expect_due_date}")

        if problems:
            return ToolResult(False, data=record,
                              error="MISMATCH: " + "; ".join(problems))

        return ToolResult(
            True, data=record,
            display=f"VERIFIED record #{record['id']}: {record['vendor']}, "
                    f"{record['invoice_number']}, {record['amount']:.2f}, due {record['due_date']}",
        )

    def open_page(path: str = "/") -> ToolResult:
        page = session.page
        try:
            page.goto(f"{BASE_URL}{path}", wait_until="domcontentloaded", timeout=15000)
        except PWTimeout as exc:
            return ToolResult(False, error=f"navigation timeout: {exc}", retriable=True)
        shot = session.screenshot(f"page{path.replace('/', '-') or '-root'}")
        body = page.locator("body").inner_text()
        return ToolResult(True, data={"url": page.url, "text": body},
                          artifacts=[shot] if shot else [],
                          display=f"{page.url}\n{body[:500]}")

    return submit_payable, verify_payable, open_page
