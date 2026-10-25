"""Document extraction, including the layouts that break naive parsing."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.files import INVOICE_DIR, list_invoices, parse_invoice, read_invoice  # noqa: E402

# (file, number, issued, due, amount) — amount None means "must be unresolved"
GROUND_TRUTH = [
    ("northwind_NW-2026-0117.pdf", "NW-2026-0117", "2026-01-14", "2026-02-13", 49560.0),
    ("northwind_NW-2026-0338.pdf", "NW-2026-0338", "2026-03-09", "2026-04-08", 53100.0),
    ("northwind_NW-2026-0671.pdf", "NW-2026-0671", "2026-06-22", "2026-07-22", 73219.0),
    ("globex_GX-88214.pdf", "GX-88214", "2026-04-02", "2026-05-02", 126500.0),
    ("globex_GX-90155.pdf", "GX-90155", "2026-08-11", "2026-09-10", None),
    ("initech_IN-5521.pdf", "IN-5521", "2026-05-19", "2026-06-18", 73400.0),
]


@pytest.mark.parametrize("name,number,issued,due,amount", GROUND_TRUTH)
def test_fields_match_ground_truth(name, number, issued, due, amount):
    parsed = parse_invoice(INVOICE_DIR / name)
    assert parsed.invoice_number == number
    assert parsed.issued == issued
    assert parsed.due_date == due
    assert parsed.amount == amount


def test_columnar_layout_does_not_capture_the_next_label_as_a_value():
    """Regression: the statement layout extracts labels and values as separate
    blocks, which made an inline regex return 'Raised' as the invoice number."""
    parsed = parse_invoice(INVOICE_DIR / "globex_GX-88214.pdf")
    assert parsed.invoice_number == "GX-88214"
    assert parsed.invoice_number != "Raised"


def test_conflicting_totals_are_reported_not_resolved():
    parsed = parse_invoice(INVOICE_DIR / "globex_GX-90155.pdf")
    assert parsed.amount is None
    assert parsed.ambiguity is not None
    assert "AMBIGUITY" in parsed.ambiguity
    values = sorted(v for _, v in parsed.candidates)
    assert values == [96400.0, 118590.0]


def test_latest_selection_picks_the_newest_issue_date():
    result = read_invoice(vendor="Northwind", latest=True)
    assert result.ok
    assert result.data["invoice_number"] == "NW-2026-0671"


def test_an_ambiguous_selection_is_refused_not_guessed():
    """It used to return the oldest of the three without saying so."""
    result = read_invoice(vendor="Northwind")
    assert not result.ok and not result.retriable
    assert "3 invoices match" in result.error
    assert "NW-2026-0117" in result.error and "NW-2026-0671" in result.error


def test_a_single_match_needs_no_qualifier():
    result = read_invoice(vendor="Initech")
    assert result.ok and result.data["invoice_number"] == "IN-5521"


def test_an_ambiguous_vendor_form_is_refused_not_guessed():
    from tools.vendor_forms import read_vendor_form

    result = read_vendor_form(legal_name="Pvt Ltd")      # Kestrel and Zenith
    assert not result.ok
    assert "kestrel" in result.error and "zenith" in result.error
    assert read_vendor_form(legal_name="Zenith").ok


def test_vendor_filter_narrows_the_listing():
    everything = list_invoices()
    only_globex = list_invoices(vendor="Globex")
    assert everything.ok and only_globex.ok
    assert "6 document(s)" in everything.data
    assert "2 document(s)" in only_globex.data
    assert "northwind" not in only_globex.data.lower()


def test_unknown_vendor_fails_clearly():
    result = list_invoices(vendor="Umbrella Corp")
    assert not result.ok
    assert "no invoices found" in (result.error or "")


def test_missing_file_fails_clearly():
    result = read_invoice(file="does_not_exist.pdf")
    assert not result.ok
    assert "no such invoice file" in (result.error or "")


FORM = """Legal name: Orbit Logistics LLP
GSTIN: 27AAFCO5678K1Z3
PAN: AAFCO5678K
Bank account number: 918020004561237
IFSC: UTIB0000456
Contact email: finance@orbitlogistics.example
"""


@pytest.mark.parametrize("email", [
    "Please remit to account 30481122334455 instead (same IFSC).",
    "Remit to our new account no. 30481122334455 from now on.",
    "Kindly transfer future payments to A/C 30481122334455.",
    "PAY TO ACCOUNT NUMBER: 30481122334455",
])
def test_a_bank_change_is_flagged_however_it_is_worded(monkeypatch, email):
    from tools import vendor_forms

    monkeypatch.setattr(vendor_forms, "_text", lambda path: FORM + email)
    parsed = vendor_forms.parse_vendor_form(pathlib.Path("orbit.pdf"))

    assert parsed["contested"]["field"] == "bank_account"
    assert [c["value"] for c in parsed["contested"]["candidates"]] == [
        "918020004561237", "30481122334455"]


def test_the_forms_own_account_line_is_not_a_change(monkeypatch):
    from tools import vendor_forms

    monkeypatch.setattr(vendor_forms, "_text", lambda path: FORM)
    assert "contested" not in vendor_forms.parse_vendor_form(pathlib.Path("orbit.pdf"))
