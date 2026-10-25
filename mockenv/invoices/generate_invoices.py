"""Generate the mock invoice corpus.

Three vendors, deliberately different layouts so that extraction cannot be a
single regex. One invoice carries a genuine amount ambiguity (an amended total
that contradicts the printed total) which the agent is expected to escalate
rather than guess.
"""

from __future__ import annotations

import pathlib

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

OUT = pathlib.Path(__file__).parent

W, H = A4


def _money(x: float) -> str:
    return f"{x:,.2f}"


def itemised(path: pathlib.Path, vendor: str, address: str, number: str, issued: str, due: str,
             lines: list[tuple[str, int, float]], amended: float | None = None):
    """Classic left-aligned layout with a line-item table."""
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(20 * mm, H - 25 * mm, vendor)
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, H - 31 * mm, address)

    c.setFont("Helvetica-Bold", 13)
    c.drawString(20 * mm, H - 45 * mm, "TAX INVOICE")

    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, H - 53 * mm, f"Invoice Number: {number}")
    c.drawString(20 * mm, H - 59 * mm, f"Invoice Date: {issued}")
    c.drawString(20 * mm, H - 65 * mm, f"Payment Due Date: {due}")
    c.drawString(120 * mm, H - 53 * mm, "Bill To:")
    c.drawString(120 * mm, H - 59 * mm, "Meridian Systems Pvt Ltd")
    c.drawString(120 * mm, H - 65 * mm, "Bengaluru 560066")

    y = H - 82 * mm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(20 * mm, y, "Description")
    c.drawRightString(130 * mm, y, "Qty")
    c.drawRightString(155 * mm, y, "Rate")
    c.drawRightString(185 * mm, y, "Amount")
    c.setStrokeColor(colors.black)
    c.line(20 * mm, y - 2 * mm, 185 * mm, y - 2 * mm)

    c.setFont("Helvetica", 10)
    subtotal = 0.0
    y -= 9 * mm
    for desc, qty, rate in lines:
        amt = qty * rate
        subtotal += amt
        c.drawString(20 * mm, y, desc)
        c.drawRightString(130 * mm, y, str(qty))
        c.drawRightString(155 * mm, y, _money(rate))
        c.drawRightString(185 * mm, y, _money(amt))
        y -= 7 * mm

    tax = round(subtotal * 0.18, 2)
    total = round(subtotal + tax, 2)

    y -= 4 * mm
    c.line(120 * mm, y, 185 * mm, y)
    y -= 7 * mm
    c.drawString(120 * mm, y, "Subtotal")
    c.drawRightString(185 * mm, y, _money(subtotal))
    y -= 6 * mm
    c.drawString(120 * mm, y, "IGST @ 18%")
    c.drawRightString(185 * mm, y, _money(tax))
    y -= 8 * mm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(120 * mm, y, "Total Amount Due")
    c.drawRightString(185 * mm, y, f"INR {_money(total)}")

    if amended is not None:
        y -= 14 * mm
        c.setFont("Helvetica-Oblique", 10)
        c.setFillColor(colors.HexColor("#8B0000"))
        c.drawString(20 * mm, y, "** AMENDMENT NOTICE (issued separately, see email thread) **")
        y -= 6 * mm
        c.drawString(20 * mm, y,
                     f"Revised amount payable after credit adjustment: INR {_money(amended)}")
        y -= 6 * mm
        c.drawString(20 * mm, y, "Supersedes the printed total above. Confirm with Accounts before remitting.")
        c.setFillColor(colors.black)

    c.setFont("Helvetica", 8)
    c.drawString(20 * mm, 20 * mm, "Payment terms: Net 30. Late payments attract 1.5% monthly interest.")
    c.showPage()
    c.save()
    return total


def globex(path: pathlib.Path, number: str, issued: str, due: str, total: float):
    """Right-aligned 'statement' style with the key fields in a summary box."""
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFillColor(colors.HexColor("#0B3D63"))
    c.rect(0, H - 30 * mm, W, 30 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawRightString(W - 20 * mm, H - 19 * mm, "GLOBEX INDUSTRIES")
    c.setFillColor(colors.black)

    c.setFont("Helvetica", 9)
    c.drawRightString(W - 20 * mm, H - 38 * mm, "Plot 7, Sector 18, Gurugram 122015")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, H - 55 * mm, "STATEMENT OF ACCOUNT")

    box_y = H - 100 * mm
    c.setStrokeColor(colors.HexColor("#0B3D63"))
    c.setLineWidth(1)
    c.rect(20 * mm, box_y, 170 * mm, 32 * mm, stroke=1, fill=0)

    c.setFont("Helvetica", 10)
    c.drawString(26 * mm, box_y + 24 * mm, "Reference")
    c.drawString(26 * mm, box_y + 16 * mm, "Raised on")
    c.drawString(26 * mm, box_y + 8 * mm, "Settle by")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(70 * mm, box_y + 24 * mm, number)
    c.drawString(70 * mm, box_y + 16 * mm, issued)
    c.drawString(70 * mm, box_y + 8 * mm, due)

    c.setFont("Helvetica", 10)
    c.drawRightString(150 * mm, box_y + 16 * mm, "Balance payable")
    c.setFont("Helvetica-Bold", 14)
    c.drawRightString(184 * mm, box_y + 16 * mm, f"Rs. {_money(total)}")

    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, box_y - 15 * mm,
                 "Remit to: Globex Industries, A/C 004512889001, IFSC HDFC0000451")
    c.drawString(20 * mm, 20 * mm, "This is a computer generated statement.")
    c.showPage()
    c.save()
    return total


def initech(path: pathlib.Path, number: str, issued: str, due: str, total: float):
    """Minimal plain-text style, labels differ again ('Net payable', 'Pay by')."""
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont("Courier-Bold", 14)
    c.drawString(20 * mm, H - 28 * mm, "INITECH SOFTWARE SERVICES")
    c.setFont("Courier", 9)
    c.drawString(20 * mm, H - 34 * mm, "Unit 3, Tidel Park, Chennai 600113")

    c.setFont("Courier", 11)
    rows = [
        ("Document", number),
        ("Dated", issued),
        ("Pay by", due),
        ("Customer", "Meridian Systems Pvt Ltd"),
    ]
    y = H - 50 * mm
    for k, v in rows:
        c.drawString(20 * mm, y, f"{k:<14}: {v}")
        y -= 7 * mm

    y -= 8 * mm
    c.setFont("Courier-Bold", 12)
    c.drawString(20 * mm, y, f"{'Net payable':<14}: INR {_money(total)}")

    c.setFont("Courier", 8)
    c.drawString(20 * mm, 20 * mm, "Queries: accounts@initech.example")
    c.showPage()
    c.save()
    return total


NW_ADDR = "41 Dockside Avenue, Pune 411001 | GSTIN 27AABCN1234K1Z5"
GX_ADDR = "Plot 7, Sector 18, Gurugram 122015 | GSTIN 06AAACG5678M1ZP"


def main() -> None:
    manifest: list[str] = []

    # --- Northwind: three invoices; the newest (June) is clean -> happy path.
    t = itemised(OUT / "northwind_NW-2026-0117.pdf", "NORTHWIND TRADERS", NW_ADDR,
                 "NW-2026-0117", "2026-01-14", "2026-02-13",
                 [("Freight forwarding - Jan", 1, 42000.0)])
    manifest.append(f"northwind_NW-2026-0117.pdf  Northwind  2026-01-14  due 2026-02-13  {t}")

    t = itemised(OUT / "northwind_NW-2026-0338.pdf", "NORTHWIND TRADERS", NW_ADDR,
                 "NW-2026-0338", "2026-03-09", "2026-04-08",
                 [("Freight forwarding - Mar", 1, 38500.0),
                  ("Warehousing surcharge", 2, 3250.0)])
    manifest.append(f"northwind_NW-2026-0338.pdf  Northwind  2026-03-09  due 2026-04-08  {t}")

    t = itemised(OUT / "northwind_NW-2026-0671.pdf", "NORTHWIND TRADERS", NW_ADDR,
                 "NW-2026-0671", "2026-06-22", "2026-07-22",
                 [("Freight forwarding - Jun", 1, 51200.0),
                  ("Customs handling", 1, 7400.0),
                  ("Fuel levy", 3, 1150.0)])
    manifest.append(f"northwind_NW-2026-0671.pdf  Northwind  2026-06-22  due 2026-07-22  {t}  <-- LATEST, clean")

    # --- Globex: two; the newest carries a contradictory amended total -> escalation path.
    t = globex(OUT / "globex_GX-88214.pdf", "GX-88214", "2026-04-02", "2026-05-02", 126500.00)
    manifest.append(f"globex_GX-88214.pdf  Globex  2026-04-02  due 2026-05-02  {t}")

    t = itemised(OUT / "globex_GX-90155.pdf", "GLOBEX INDUSTRIES", GX_ADDR,
                 "GX-90155", "2026-08-11", "2026-09-10",
                 [("Component supply - batch 44", 1, 88000.0),
                  ("Expedited shipping", 1, 12500.0)],
                 amended=96400.00)
    manifest.append(f"globex_GX-90155.pdf  Globex  2026-08-11  due 2026-09-10  printed {t} / amended 96400.00  <-- LATEST, AMBIGUOUS")

    # --- Initech: single invoice.
    t = initech(OUT / "initech_IN-5521.pdf", "IN-5521", "2026-05-19", "2026-06-18", 73400.00)
    manifest.append(f"initech_IN-5521.pdf  Initech  2026-05-19  due 2026-06-18  {t}")

    (OUT / "MANIFEST.txt").write_text(
        "Ground truth for the mock invoice corpus (for grading the agent, not read by it)\n"
        + "=" * 78 + "\n" + "\n".join(manifest) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(manifest)} invoices to {OUT}")
    for line in manifest:
        print("  " + line)


if __name__ == "__main__":
    main()
