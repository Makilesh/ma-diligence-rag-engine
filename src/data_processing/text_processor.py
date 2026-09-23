"""
Plain-text (.txt) processor — heading-aware sectioning with table detection.

Plain text has no pages, fonts or styles, so structure has to be recovered
from layout conventions. The sample data room (data/sample_deal/*.txt) uses:

- ALL CAPS headings ("SECTION 2: ADJUSTED EBITDA BRIDGE", "ARTICLE I — THE MERGER")
- numbered sub-sections ("Section 1.2 — Consideration")
- markdown headings ("# Overview") in hand-written notes
- "=====" rule lines around headings, "(in millions of USD)" qualifiers
- whitespace-aligned tables with "------" column rules

Each section carries the heading it actually sits under, so a citation names a
real section rather than whatever the first line of a paragraph happened to be.

Page numbers are deliberately NOT produced: a .txt file has no pages, and a
fabricated "p.N" in a citation is worse than none.
"""

import re

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# A rule line made only of one repeated separator character ("=====", "-----").
_RULE_LINE = re.compile(r"^\s*([=\-_*~#])\1{4,}\s*$")
# Column rules inside a table: two or more dash groups ("------   ------").
_COLUMN_RULE = re.compile(r"^\s*-{2,}(\s+-{2,})+\s*$")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+\S")
# "Section 1.2 — Consideration", "Article IV: Covenants", "Section 4.3. Taxes"
_NUMBERED_HEADING = re.compile(
    r"^(Section|SECTION|Article|ARTICLE)\s+[0-9IVXLC]+(\.\d+)*\s*[—–:.\-]\s+\S"
)
_PARENTHETICAL = re.compile(r"^\(.*\)$")
# A lone dash rule under a totals column ("   -------").
_SINGLE_DASH_RULE = re.compile(r"^\s*-{2,}\s*$")

MAX_HEADING_CHARS = 100


def _is_table_line(line: str) -> bool:
    """
    True when a line looks like a row of a whitespace-aligned table.

    A row has at least two cells separated by runs of 2+ spaces, and at least
    one cell after the label carries a digit (an amount, a year, a percentage).
    Column rule lines ("------   ------") also count.

    Args:
        line: Raw line (indentation preserved).

    Returns:
        True if the line belongs in a table block.
    """
    if _COLUMN_RULE.match(line):
        return True
    cells = re.split(r"\s{2,}", line.strip())
    if len(cells) < 2:
        return False
    return any(re.search(r"\d", c) for c in cells[1:])


def _is_heading_line(line: str, prev_line: str | None) -> bool:
    """
    True when a line is a section heading.

    A heading must start a block (first line, or preceded by a blank line, a
    rule line, or another heading) — this rejects wrapped body lines that
    happen to be upper case ("Q4 FY2023.").

    Args:
        line: Raw line.
        prev_line: The previous raw line, or None at the start of the file.

    Returns:
        True if the line should be treated as a heading.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return False
    if line[:1].isspace():  # indented lines are body or table content
        return False
    if _is_table_line(line):
        return False

    if _MARKDOWN_HEADING.match(stripped):
        return True
    if _NUMBERED_HEADING.match(stripped) and len(stripped) <= 80:
        return True

    letters = [ch for ch in stripped if ch.isalpha()]
    is_all_caps = len(letters) >= 4 and not any(ch.islower() for ch in letters)
    if not is_all_caps:
        return False

    starts_block = (
        prev_line is None
        or not prev_line.strip()
        or bool(_RULE_LINE.match(prev_line))
    )
    return starts_block


def _heading_level(stripped: str) -> int:
    """
    Level 2 for numbered sub-sections ("Section 1.2 — ..."), level 1 otherwise.

    Args:
        stripped: Heading text.

    Returns:
        1 or 2.
    """
    if _NUMBERED_HEADING.match(stripped) and not stripped.isupper():
        return 2
    return 1


def parse_text_sections(text: str) -> list[dict]:
    """
    Splits plain text into heading-scoped text and table sections.

    Args:
        text: Full document text.

    Returns:
        List of section dicts with text, section_heading, section_type
        ("text" | "table"), is_table and page_number=None.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    sections: list[dict] = []
    level1 = ""
    level2 = ""
    pending_heading_lines: list[str] = []  # heading lines not yet attached to a body
    body_lines: list[str] = []
    table_lines: list[str] = []
    prev_line: str | None = None
    prev_was_heading = False

    def current_heading() -> str:
        return level2 or level1

    def flush_body() -> None:
        nonlocal body_lines, pending_heading_lines
        body = "\n".join(body_lines).strip()
        if body:
            text_parts = pending_heading_lines + [body]
            sections.append({
                "text": "\n".join(text_parts).strip(),
                "section_heading": current_heading(),
                "section_type": "text",
                "is_table": False,
                "page_number": None,
            })
            pending_heading_lines = []
        body_lines = []

    def flush_table() -> None:
        nonlocal table_lines, pending_heading_lines
        while table_lines and not table_lines[-1].strip():
            table_lines.pop()
        # Strip only trailing whitespace — leading indentation is what keeps
        # the columns aligned.
        table_text = "\n".join(line.rstrip() for line in table_lines)
        if table_text.strip():
            heading = current_heading()
            # The heading (with its "(in millions of USD)" qualifier) is kept in
            # the table text: without it the numbers carry no unit or scale.
            prefix = f"{heading}\n" if heading else ""
            sections.append({
                "text": prefix + table_text,
                "section_heading": heading,
                "section_type": "table",
                "is_table": True,
                "page_number": None,
            })
            pending_heading_lines = []
        table_lines = []

    for line in lines:
        stripped = line.strip()

        # A dash rule under a totals column belongs to the table it sits in.
        if table_lines and _SINGLE_DASH_RULE.match(line):
            table_lines.append(line)
            prev_line = line
            continue

        # Separator rules end whatever block is open and carry no content.
        if _RULE_LINE.match(line) and not _COLUMN_RULE.match(line):
            if table_lines:
                flush_table()
            prev_line = line
            continue

        if table_lines:
            # Row-group labels such as "Operating Expenses:" sit between rows.
            is_label = (
                stripped.endswith(":")
                and len(stripped) <= 60
                and any(ch.islower() for ch in stripped)
            )
            # Key/value rows without digits ("Security:     First priority lien")
            # continue a table once one is open.
            is_multi_cell = len(re.split(r"\s{2,}", stripped)) >= 2
            # Deeply indented lines are a wrapped cell from the row above.
            is_wrapped_cell = len(line) - len(line.lstrip()) >= 8
            if (
                not stripped
                or _is_table_line(line)
                or is_label
                or is_multi_cell
                or is_wrapped_cell
            ):
                table_lines.append(line)
                prev_line = line
                continue
            flush_table()

        directly_under_heading = bool(pending_heading_lines) and not "".join(body_lines).strip()
        if directly_under_heading and _PARENTHETICAL.match(stripped):
            # "(in millions of USD)" qualifies the heading directly above it.
            if level2:
                level2 = f"{level2} {stripped}"
            else:
                level1 = f"{level1} {stripped}"
            pending_heading_lines.append(stripped)
            prev_line = line
            continue

        if _is_heading_line(line, prev_line if not prev_was_heading else ""):
            flush_body()
            heading_text = stripped.lstrip("#").strip()
            if _heading_level(heading_text) == 2:
                level2 = heading_text
            elif prev_was_heading and not level2:
                # Consecutive level-1 lines ("AURORA TECHNOLOGIES INC." /
                # "CONSOLIDATED FINANCIAL STATEMENTS") form one heading.
                level1 = f"{level1} — {heading_text}" if level1 else heading_text
            else:
                level1 = heading_text
                level2 = ""
            pending_heading_lines.append(stripped)
            prev_was_heading = True
            prev_line = line
            continue

        prev_was_heading = False

        if _is_table_line(line):
            flush_body()
            table_lines.append(line)
            prev_line = line
            continue

        body_lines.append(line)
        prev_line = line

    if table_lines:
        flush_table()
    flush_body()

    # A document that is nothing but headings still has content worth indexing.
    if not sections and pending_heading_lines:
        sections.append({
            "text": "\n".join(pending_heading_lines),
            "section_heading": current_heading(),
            "section_type": "text",
            "is_table": False,
            "page_number": None,
        })

    logger.info(
        "Text sectioning complete",
        extra={
            "sections": len(sections),
            "tables": sum(1 for s in sections if s["is_table"]),
        },
    )
    return sections
