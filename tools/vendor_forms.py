"""Vendor registration documents: discover them and extract their fields.

Same contract as the invoice reader — ``fields`` named to match the write tool's
parameters, and a ``contested`` entry when the document contradicts itself — so
the agent loop binds and guards these values without knowing what a vendor is.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

from pypdf import PdfReader

from agent.tools import ToolResult

VENDOR_DIR = pathlib.Path(__file__).resolve().parents[1] / "mockenv" / "vendors"

PATTERNS = {
    "legal_name": r"Legal name:\s*(.+)",
    "gstin": r"GSTIN:\s*([0-9A-Z]{15})\b",
    "pan": r"PAN:\s*([A-Z0-9]{10})\b",
    "bank_account": r"Bank account number:\s*(\d{9,18})\b",
    "ifsc": r"IFSC:\s*([A-Z0-9]{11})\b",
    "contact_email": r"Contact email:\s*(\S+@\S+)",
}
SUBMITTED = r"Submitted on:\s*(\d{4}-\d{2}-\d{2})"
# An instruction to send money to an account, however it is phrased: "remit to
# account", "Remit to our new account no.", "transfer future payments to A/C".
# Missing a rephrased one would silently drop the change instead of flagging it.
EMAILED_ACCOUNT = re.compile(
    r"\b(?:remit|pay|transfer|send)\b[^.\n]{0,40}?\b(?:account|a/c)\b"
    r"(?:\s*(?:number|no\.?))?\s*[:\-]?\s*(\d{9,18})", re.I)


def _text(path: pathlib.Path) -> str:
    return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


def parse_vendor_form(path: pathlib.Path) -> dict[str, Any]:
    text = _text(path)
    fields: dict[str, str | None] = {}
    for name, pattern in PATTERNS.items():
        m = re.search(pattern, text)
        fields[name] = m.group(1).strip() if m else None

    submitted = re.search(SUBMITTED, text)
    out: dict[str, Any] = {
        "source": path.name,
        "submitted": submitted.group(1) if submitted else None,
        "fields": fields,
        "missing_fields": [k for k, v in fields.items() if not v],
    }

    # A bank account supplied outside the signed form, that disagrees with it,
    # is not a correction to apply. It is exactly what an impersonation attack
    # looks like, so it is reported for a human and never resolved here.
    emailed = EMAILED_ACCOUNT.search(text)
    if emailed and fields.get("bank_account") and emailed.group(1) != fields["bank_account"]:
        out["contested"] = {
            "field": "bank_account",
            "message": ("AMBIGUITY: the signed form gives bank account "
                        f"{fields['bank_account']}, but a change received by email asks "
                        f"for {emailed.group(1)}. Bank-detail changes by email are a "
                        "known fraud pattern; this must be confirmed by a person."),
            "candidates": [
                {"label": "account on signed form", "value": fields["bank_account"]},
                {"label": "account from email update", "value": emailed.group(1)},
            ],
        }
    return out


def list_vendor_forms(name: str = "") -> ToolResult:
    if not VENDOR_DIR.exists():
        return ToolResult(False, error=f"vendor directory missing: {VENDOR_DIR}")
    rows = []
    for path in sorted(VENDOR_DIR.glob("*.pdf")):
        parsed = parse_vendor_form(path)
        legal = parsed["fields"].get("legal_name") or ""
        if name and name.lower() not in legal.lower() and name.lower() not in path.name.lower():
            continue
        rows.append(f"  {path.name} | {legal} | {parsed['fields'].get('gstin')} | "
                    f"submitted {parsed['submitted']}")
    if not rows:
        return ToolResult(False, error=f"no vendor registration forms found for {name!r}")
    return ToolResult(True, data=f"{len(rows)} document(s):\n" + "\n".join(rows))


def read_vendor_form(file: str = "", legal_name: str = "") -> ToolResult:
    path: pathlib.Path | None = None
    if file:
        candidate = VENDOR_DIR / pathlib.Path(file).name
        if not candidate.exists():
            return ToolResult(False, error=f"no such vendor form: {file}")
        path = candidate
    else:
        matches = [p for p in sorted(VENDOR_DIR.glob("*.pdf"))
                   if legal_name and legal_name.lower()
                   in (parse_vendor_form(p)["fields"].get("legal_name") or "").lower()]
        if not matches:
            return ToolResult(False, error=f"no vendor form matched {legal_name!r}")
        if len(matches) > 1:
            return ToolResult(False, retriable=False, error_kind="invalid",
                              error=f"{len(matches)} vendor forms match {legal_name!r} "
                                    f"({', '.join(p.name for p in matches)}); "
                                    f"name one by file")
        path = matches[0]

    parsed = parse_vendor_form(path)
    f = parsed["fields"]
    contested = parsed.get("contested")
    lines = [f"source: {parsed['source']}"]
    for key in ("legal_name", "gstin", "pan", "bank_account", "ifsc", "contact_email"):
        if contested and contested["field"] == key:
            lines.append(f"{key}: UNRESOLVED")
        else:
            lines.append(f"{key}: {f.get(key)}")
    if contested:
        lines.append(f"CONTESTED ({contested['field']}): {contested['message']}")
    if parsed["missing_fields"]:
        lines.append(f"missing_fields: {parsed['missing_fields']}")
    return ToolResult(True, data=parsed, display="\n".join(lines))
