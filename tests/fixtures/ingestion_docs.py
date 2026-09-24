"""
Programmatic document fixtures for the ingestion tests.

Every format the ingest route accepts is generated at test time with the same
libraries the processors read it with, so no binary fixtures live in the repo
and each test controls exactly what the document contains.
"""

from __future__ import annotations

import copy
from pathlib import Path

SAMPLE_TXT = """\
AURORA TECHNOLOGIES INC.
CONSOLIDATED FINANCIAL STATEMENTS
For the Fiscal Years Ended December 31, 2023 and December 31, 2022

================================================================================
CONSOLIDATED INCOME STATEMENT
(in millions of USD)
================================================================================

                                          FY2023          FY2022
                                          ------          ------
Revenue                                   $452.8          $387.1
Cost of Revenue                           $181.1          $158.3

Gross Profit                              $271.7          $228.8
  Gross Margin                             60.0%           59.1%

================================================================================
ARTICLE VIII — INDEMNIFICATION
================================================================================

Section 8.1 — Survival of Representations
The representations and warranties of the Company shall survive the Closing
for a period of eighteen (18) months. Claims must be asserted in writing
before the expiry of that period.

Section 8.2 — Indemnification Cap
The aggregate liability of the Company for breaches of representations and
warranties shall not exceed $174 million, being twenty-five percent of the
aggregate Merger Consideration.
"""


def write_txt(directory: Path, name: str = "aurora_sample.txt", text: str = SAMPLE_TXT) -> Path:
    """Writes a .txt file in the sample data-room style."""
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def write_pdf(directory: Path, name: str = "financial_report.pdf") -> Path:
    """
    Two-page PDF: page 1 has a large-font heading and prose; page 2 has more
    prose and a ruled 3x4 income-statement table pdfplumber can detect.
    """
    import fitz

    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 80), "Management Discussion", fontsize=20)
    body = (
        "Revenue grew seventeen percent year over year, driven by SaaS "
        "subscriptions. Operating margin expanded as restructuring completed."
    )
    y = 120
    for line in [body[i:i + 80] for i in range(0, len(body), 80)]:
        page1.insert_text((72, y), line, fontsize=11)
        y += 16

    page2 = doc.new_page()
    page2.insert_text((72, 80), "Customer concentration remained stable in the period.", fontsize=11)

    rows = [
        ["Line Item", "FY2022", "FY2023"],
        ["Revenue", "387.1", "452.8"],
        ["Gross Profit", "228.8", "271.7"],
        ["Net Income", "34.3", "46.4"],
    ]
    left, top, col_w, row_h = 72, 140, 150, 24
    n_rows, n_cols = len(rows), len(rows[0])
    for r in range(n_rows + 1):
        yy = top + r * row_h
        page2.draw_line((left, yy), (left + n_cols * col_w, yy))
    for c in range(n_cols + 1):
        xx = left + c * col_w
        page2.draw_line((xx, top), (xx, top + n_rows * row_h))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            page2.insert_text((left + c * col_w + 6, top + r * row_h + 16), cell, fontsize=11)

    path = directory / name
    doc.save(str(path))
    doc.close()
    return path


def write_docx(directory: Path, name: str = "merger_agreement_draft.docx") -> Path:
    """
    DOCX with a heading, prose, a tracked insertion (<w:ins>), a tracked
    deletion (<w:del>) and a table.
    """
    import docx
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    document = docx.Document()
    document.add_heading("Article VIII Indemnification", level=1)
    document.add_paragraph(
        "The representations and warranties shall survive the Closing for eighteen months."
    )

    para = document.add_paragraph("The indemnification cap shall be ")

    ins = OxmlElement("w:ins")
    ins.set(qn("w:id"), "1")
    ins.set(qn("w:author"), "Buyer Counsel")
    ins_run = OxmlElement("w:r")
    ins_text = OxmlElement("w:t")
    ins_text.text = "twenty-five percent of the purchase price"
    ins_run.append(ins_text)
    ins.append(ins_run)
    para._p.append(ins)

    deletion = OxmlElement("w:del")
    deletion.set(qn("w:id"), "2")
    deletion.set(qn("w:author"), "Buyer Counsel")
    del_run = OxmlElement("w:r")
    del_text = OxmlElement("w:delText")
    del_text.text = "ten percent of the purchase price"
    del_run.append(del_text)
    deletion.append(del_run)
    para._p.append(deletion)

    tail = copy.deepcopy(para.runs[0]._r)
    tail.find(qn("w:t")).text = "."
    para._p.append(tail)

    document.add_heading("Schedule of Escrow", level=2)
    table = document.add_table(rows=3, cols=3)
    for r, row in enumerate([
        ["Escrow Item", "FY2023", "FY2024"],
        ["General Escrow", "50.0", "25.0"],
        ["Special Escrow", "10.0", "5.0"],
    ]):
        for c, value in enumerate(row):
            table.cell(r, c).text = value

    path = directory / name
    document.save(str(path))
    return path


def write_xlsx(directory: Path, name: str = "income_statement_model.xlsx") -> Path:
    """Workbook with an income-statement sheet and a text-only notes sheet."""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Income Statement"
    ws.append(["USD in thousands"])
    ws.append(["Line Item", "FY2022", "FY2023"])
    ws.append(["Revenue", 387100, 452800])
    ws.append(["Gross Profit", 228800, 271700])
    ws.append(["Operating Income", 51400, 68000])
    ws.append(["Net Income", 34300, 46400])

    notes = wb.create_sheet("Notes")
    notes.append(["Note", "Detail"])
    notes.append(["Basis", "Figures are audited"])
    notes.append(["Auditor", "Deloitte"])

    path = directory / name
    wb.save(str(path))
    return path


def write_pptx(directory: Path, name: str = "board_deck.pptx") -> Path:
    """Deck with one slide: title, bullet text and a small numeric table."""
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # title only
    slide.shapes.title.text = "Preliminary Indications of Interest"

    box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(8), Inches(1))
    box.text_frame.text = "Three bidders submitted non-binding indications of interest."

    shape = slide.shapes.add_table(3, 3, Inches(0.5), Inches(3), Inches(8), Inches(1.5))
    for r, row in enumerate([
        ["Bidder", "Low", "High"],
        ["Vertex", "55.0", "58.0"],
        ["Meridian", "52.0", "54.0"],
    ]):
        for c, value in enumerate(row):
            shape.table.cell(r, c).text = value

    path = directory / name
    prs.save(str(path))
    return path
