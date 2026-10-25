"""Document tools: discover invoices and extract their fields.

Extraction is deterministic rather than model-driven, on purpose. The planner is
the part of the system that benefits from a language model; pulling a number off
a page does not, and a regex that fails loudly is easier to verify than a model
that quietly rounds a figure. When the deterministic pass cannot find a field,
that is reported as a gap instead of being papered over.

Crucially, when a document yields two *contradictory* payable amounts, the
extractor does not pick one. It reports an ambiguity and lets the agent escalate.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any

from pypdf import PdfReader

from agent.tools import ToolResult

INVOICE_DIR = pathlib.Path(__file__).resolve().parents[1] / "mockenv" / "invoices"

_MONEY = r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)"

NUMBER_LABELS = [r"Invoice Number", r"Reference", r"Document"]
ISSUED_LABELS = [r"Invoice Date", r"Raised on", r"Dated"]
DUE_LABELS = [r"Payment Due Date", r"Due Date", r"Settle by", r"Pay by"]
TOTAL_LABELS = [r"Total Amount Due", r"Balance payable", r"Net payable", r"Amount Due"]
AMENDED_LABELS = [r"Revised amount payable[^:]*", r"Revised total"]

VENDOR_HINTS = {
    "northwind": "Northwind Traders",
    "globex": "Globex Industries",
    "initech": "Initech Software Services",
}


@dataclass
class Extracted:
    source: str
    vendor: str | None = None
    invoice_number: str | None = None
    issued: str | None = None
    due_date: str | None = None
    amount: float | None = None
    candidates: list[tuple[str, float]] = field(default_factory=list)
    ambiguity: str | None = None
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "vendor": self.vendor,
            "invoice_number": self.invoice_number,
            "issued": self.issued,
            "due_date": self.due_date,
            "amount": self.amount,
            "amount_candidates": [{"label": l, "value": v} for l, v in self.candidates],
            "ambiguity": self.ambiguity,
            "missing_fields": self.missing,
        }


ALL_LABELS = NUMBER_LABELS + ISSUED_LABELS + DUE_LABELS + TOTAL_LABELS + AMENDED_LABELS


def _text(path: pathlib.Path) -> str:
    return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


def _is_label(line: str) -> bool:
    stripped = line.strip().rstrip(":").strip()
    return any(re.fullmatch(lbl, stripped, re.I) for lbl in ALL_LABELS)


def _columnar_map(text: str) -> dict[str, str]:
    """Pair labels with values in column-style layouts.

    A table drawn as two columns often extracts as every label followed by every
    value ("Reference / Raised on / Settle by / GX-88214 / 2026-04-02 / ...").
    Inline matching grabs the next *label* as the value, so for runs of stacked
    labels the values are paired positionally instead.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    found: dict[str, str] = {}
    i = 0
    while i < len(lines):
        run: list[str] = []
        j = i
        while j < len(lines) and _is_label(lines[j]):
            run.append(lines[j].rstrip(":").strip())
            j += 1
        if run:
            values = []
            k = j
            while k < len(lines) and len(values) < len(run) and not _is_label(lines[k]):
                values.append(lines[k])
                k += 1
            for label, value in zip(run, values):
                found.setdefault(label.lower(), value)
            i = max(k, i + 1)
        else:
            i += 1
    return found


def _first(text: str, labels: list[str], pattern: str,
           columnar: dict[str, str] | None = None) -> str | None:
    """Resolve one field, preferring positional pairing over inline matching.

    The columnar map is consulted first because it is only populated when labels
    genuinely stack without inline values — exactly the case where an inline
    regex would run past the label and capture the *next label* as the value.
    """
    if columnar:
        for label in labels:
            for key, value in columnar.items():
                if re.fullmatch(label, key, re.I):
                    m = re.search(pattern, value, re.I)
                    if m:
                        return m.group(1).strip()

    for label in labels:
        m = re.search(rf"{label}\s*[:\-]?\s*{pattern}", text, re.I)
        if m and not _is_label(m.group(1)):
            return m.group(1).strip()
    return None


def _money_to_float(raw: str) -> float:
    return float(raw.replace(",", "").strip())


def _vendor(text: str, path: pathlib.Path) -> str | None:
    head = text[:200].lower()
    for key, name in VENDOR_HINTS.items():
        if key in head or key in path.name.lower():
            return name
    first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    return first_line.title() if first_line else None


def parse_invoice(path: pathlib.Path) -> Extracted:
    text = _text(path)
    col = _columnar_map(text)
    out = Extracted(source=path.name)
    out.vendor = _vendor(text, path)
    out.invoice_number = _first(text, NUMBER_LABELS, r"([A-Z0-9][A-Z0-9\-/]+)", col)
    out.issued = _normalise_date(
        _first(text, ISSUED_LABELS, r"([0-9]{4}-[0-9]{2}-[0-9]{2})", col))
    out.due_date = _normalise_date(
        _first(text, DUE_LABELS, r"([0-9]{4}-[0-9]{2}-[0-9]{2})", col))

    money = rf"(?:INR|Rs\.?|₹)?\s*{_MONEY}"
    printed = _first(text, TOTAL_LABELS, money, col)
    if printed:
        out.candidates.append(("printed total", _money_to_float(printed)))

    amended = _first(text, AMENDED_LABELS, money, col)
    if amended:
        out.candidates.append(("amended total", _money_to_float(amended)))

    values = {round(v, 2) for _, v in out.candidates}
    if len(values) > 1:
        detail = ", ".join(f"{l} = {v:,.2f}" for l, v in out.candidates)
        out.ambiguity = (
            f"AMBIGUITY: the document states more than one payable amount ({detail}). "
            "The agent must not choose between them unaided."
        )
    elif out.candidates:
        out.amount = out.candidates[0][1]

    for name in ("invoice_number", "issued", "due_date"):
        if getattr(out, name) is None:
            out.missing.append(name)
    if not out.candidates:
        out.missing.append("amount")
    return out


def _normalise_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value).isoformat()
    except ValueError:
        return value


# --------------------------------------------------------------------------
# Tool entry points
# --------------------------------------------------------------------------

def list_invoices(vendor: str = "") -> ToolResult:
    """Catalogue available documents with just enough metadata to choose one."""
    if not INVOICE_DIR.exists():
        return ToolResult(False, error=f"invoice directory missing: {INVOICE_DIR}")

    rows = []
    for path in sorted(INVOICE_DIR.glob("*.pdf")):
        try:
            parsed = parse_invoice(path)
        except Exception as exc:
            rows.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if vendor and vendor.lower() not in (parsed.vendor or "").lower() \
                and vendor.lower() not in path.name.lower():
            continue
        rows.append({
            "file": path.name,
            "vendor": parsed.vendor,
            "invoice_number": parsed.invoice_number,
            "issued": parsed.issued,
            "due_date": parsed.due_date,
        })

    if not rows:
        return ToolResult(False, error=f"no invoices found for vendor {vendor!r}")

    listing = "\n".join(
        f"  {r['file']} | {r.get('vendor')} | {r.get('invoice_number')} | issued {r.get('issued')}"
        if "error" not in r else f"  {r['file']} | UNREADABLE: {r['error']}"
        for r in rows
    )
    return ToolResult(True, data=f"{len(rows)} document(s):\n{listing}")


def read_invoice(file: str = "", vendor: str = "", latest: bool = False,
                 invoice_number: str = "") -> ToolResult:
    """Extract fields from one invoice.

    Selected by filename, by invoice number, or by vendor plus ``latest``. If
    several invoices match and ``latest`` is not set, the call is refused with
    the candidates rather than answered with one of them.
    """
    path: pathlib.Path | None = None

    if file:
        candidate = INVOICE_DIR / pathlib.Path(file).name
        if not candidate.exists():
            return ToolResult(False, error=f"no such invoice file: {file}")
        path = candidate
    elif invoice_number:
        for p in sorted(INVOICE_DIR.glob("*.pdf")):
            if (parse_invoice(p).invoice_number or "").upper() == invoice_number.strip().upper():
                path = p
                break
        if path is None:
            return ToolResult(False, error=f"no invoice found with number {invoice_number!r}")
    else:
        pool = []
        for p in sorted(INVOICE_DIR.glob("*.pdf")):
            parsed = parse_invoice(p)
            if vendor and vendor.lower() not in (parsed.vendor or "").lower() \
                    and vendor.lower() not in p.name.lower():
                continue
            pool.append((parsed.issued or "", p))
        if not pool:
            return ToolResult(False, error=f"no invoices matched vendor {vendor!r}")
        pool.sort(key=lambda t: t[0])
        if latest or len(pool) == 1:
            path = pool[-1][1]
        else:
            names = ", ".join(p.name for _, p in pool)
            return ToolResult(False, retriable=False, error_kind="invalid",
                              error=f"{len(pool)} invoices match vendor {vendor!r} "
                                    f"({names}); name one by file or invoice_number, "
                                    f"or pass latest=true")

    parsed = parse_invoice(path)
    payload = parsed.to_dict()

    # The generic contract the agent loop consumes: canonical string values the
    # downstream form accepts as-is, and any contested field with its candidates.
    # Field names match the parameters of submit_payable, which is what lets the
    # loop bind them without knowing what an invoice is.
    payload["fields"] = {
        "vendor": parsed.vendor,
        "invoice_number": parsed.invoice_number,
        "due_date": parsed.due_date,
        "amount": None if parsed.amount is None else f"{parsed.amount:.2f}",
    }
    if parsed.ambiguity:
        payload["contested"] = {
            "field": "amount",
            "message": parsed.ambiguity,
            "candidates": [{"label": label, "value": f"{value:.2f}"}
                           for label, value in parsed.candidates],
        }

    lines = [
        f"source: {parsed.source}",
        f"vendor: {parsed.vendor}",
        f"invoice_number: {parsed.invoice_number}",
        f"issued: {parsed.issued}",
        f"due_date: {parsed.due_date}",
    ]
    if parsed.ambiguity:
        lines.append("amount: UNRESOLVED")
        lines.append(f"CONTESTED (amount): {parsed.ambiguity}")
    else:
        lines.append(f"amount: {parsed.amount:.2f}" if parsed.amount is not None
                     else "amount: (not found)")
    if parsed.missing:
        lines.append(f"missing_fields: {parsed.missing}")

    return ToolResult(True, data=payload, display="\n".join(lines))
