"""
UI tests driven by Streamlit's own test harness.

`AppTest` runs the real app script and interacts with widgets through
Streamlit's session machinery, so these exercise the actual rerun semantics that
UI bugs in this project have hidden in — not a mock of them.

This layer exists because the two UI defects found so far were both invisible to
code review and to browser automation: the example-query buttons appeared to
fill the box while the widget still returned "", and the deal selector showed
"No deals found" against a fully populated index. Both are state bugs, which is
exactly what AppTest can see and a screenshot cannot.

Note on browser automation: synthetic CDP clicks do not reliably reach
Streamlit's React handlers, so a click that "does nothing" in an automated
browser is not evidence of a bug. AppTest is the reliable oracle here.
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

APP_PATH = str(_PROJECT_ROOT / "app" / "streamlit_app.py")

pytest.importorskip("streamlit.testing.v1")


def _fresh_app(deal_id: str = "aurora_vertex_2024"):
    """
    Runs the app with a deal already selected, without needing the API.

    The deal selector is populated from `GET /deals`, so tests that exercise
    anything past the sidebar used to require a running backend — they passed or
    failed depending on whether an unrelated server happened to be up, which is
    not a property a test should have. The `?deal_id=` deep link selects a deal
    without any network call, so these tests now exercise the UI and nothing
    else. Panels whose endpoints are unreachable degrade to empty, which is the
    behaviour the app is built for anyway.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    if deal_id:
        at.query_params["deal_id"] = deal_id
    at.run()
    return at


def test_app_renders_without_exception():
    """The script must run clean even with no deal selected and the API down."""
    at = _fresh_app(deal_id="")
    assert not at.exception, f"app raised on first render: {at.exception}"


def test_deep_link_selects_a_deal():
    """`?deal_id=` must scope the dashboard without any backend round-trip."""
    at = _fresh_app("aurora_vertex_2024")
    assert not at.exception
    assert any(
        b.key and b.key.startswith("example_") for b in at.button
    ), "query interface did not render for a deep-linked deal"


def test_example_query_fills_the_box_and_enables_run():
    """
    Clicking an example must populate the text area and enable the submit button.

    Guards the fix in Decision 23. The original bug wrote the example to a
    separate `query_text` key and passed it as `value=`; once a keyed widget
    exists Streamlit serves its stored state and ignores `value=`, so the box
    looked filled while the widget returned "" and `disabled=not query` kept
    Run Analysis greyed out. Asserting on the *widget's* value and the button's
    disabled flag is the whole point — asserting on what is displayed would
    have passed against the broken version.
    """
    at = _fresh_app()

    examples = [b for b in at.button if b.key and b.key.startswith("example_")]
    assert examples, "no example query buttons rendered"

    examples[0].click().run()
    assert not at.exception

    box = next(t for t in at.text_area if t.key == "query_input")
    assert box.value, "example click left the query box empty"

    run_buttons = [b for b in at.button if "Run Analysis" in b.label]
    assert run_buttons, "Run Analysis button missing"
    assert run_buttons[0].disabled is False, (
        "Run Analysis stayed disabled after an example was selected — the widget "
        "is not seeing the value that was written to its session_state key"
    )


def test_every_example_query_is_selectable():
    """One example per routed query type, and each must load independently."""
    from app.components.query_interface import EXAMPLE_QUERIES

    for category, expected in EXAMPLE_QUERIES.items():
        at = _fresh_app()
        button = next(b for b in at.button if b.key == f"example_{category}")
        button.click().run()

        box = next(t for t in at.text_area if t.key == "query_input")
        assert box.value == expected, f"{category} example did not load correctly"


class TestCitationHeadingCleanup:
    """
    Section headings come from the chunker and are sometimes not headings.

    A live answer rendered two citations as
    "============================================================" and
    "(a) by mutual written consent of Buyer and the Company;" — a divider rule
    and a mid-sentence fragment. Both are honest reflections of the chunk
    metadata and both make the citation list look broken precisely where the
    product is asking to be trusted.
    """

    def test_separator_rules_are_dropped(self):
        from app.components.citation_viewer import _clean_heading

        for junk in ["=" * 60, "-----", "___", "***", "  ==  ", "····"]:
            assert _clean_heading(junk) == "", f"{junk!r} should not be displayed"

    def test_real_headings_are_kept(self):
        from app.components.citation_viewer import _clean_heading

        for good in [
            "Section 8.2 — Indemnification Cap",
            "CONSOLIDATED INCOME STATEMENT",
            "Net Revenue Retention",
        ]:
            assert _clean_heading(good) == good

    def test_prose_captured_as_a_heading_is_dropped(self):
        from app.components.citation_viewer import _clean_heading

        prose = (
            "(a) by mutual written consent of Buyer and the Company, and subject "
            "always to the provisions of Section 9.3 hereof, provided that notice "
            "has been duly given."
        )
        assert _clean_heading(prose) == ""

    def test_trailing_decoration_is_trimmed(self):
        from app.components.citation_viewer import _clean_heading

        assert _clean_heading("ARTICLE VIII ====") == "ARTICLE VIII"

    def test_empty_and_none_are_safe(self):
        from app.components.citation_viewer import _clean_heading

        assert _clean_heading("") == ""
        assert _clean_heading(None) == ""


class TestTruncatedHeadingDetection:
    """
    Chunker output includes sentences cut mid-flow and labelled as headings.

    Measured across all 109 distinct section headings in the sample corpus: the
    rule drops 24 and keeps 85, with no legitimate heading caught. It is
    deliberately partial — a fragment ending on an ordinary noun cannot be told
    from a terse heading without parsing it — so this guards the precision, not
    the recall.
    """

    def test_sentences_cut_mid_flow_are_dropped(self):
        from app.components.citation_viewer import _clean_heading

        for prose in [
            'VERTEX CAPITAL PARTNERS LLC, a Delaware limited liability company ("Buyer"),',
            "The aggregate Merger Consideration is approximately $696 million, subject to",
            "Notwithstanding the foregoing, the Company may engage in discussions with a",
            "RESOLVED, that the Board of Directors hereby authorizes the Company to:",
            "Severance multiples and estimated cost, assuming all five executives are",
        ]:
            assert _clean_heading(prose) == "", f"should have been dropped: {prose[:50]}"

    def test_real_headings_with_commas_and_numbers_survive(self):
        """The rule must not punish legitimate headings that look busy."""
        from app.components.citation_viewer import _clean_heading

        for good in [
            "Note 1 - Restructuring Charges ($4.5M, FY2023)",
            "Note 5 - Capitalized Software Adjustment (($3.2M), FY2023)",
            "Agreement:          Amended and Restated Credit Agreement dated June 30, 2021",
            "Delta Ridge Energy               $12.2M    Expires Dec 31, 2024",
            "Section 9.2 — Effect on Material Contracts",
        ]:
            assert _clean_heading(good) == good.strip(), f"wrongly dropped: {good[:50]}"

    def test_short_labels_are_never_treated_as_prose(self):
        """Below the length floor, no truncation check applies at all."""
        from app.components.citation_viewer import _clean_heading

        for short in ["Net Revenue Retention", "Section 3.5", "EBITDA and Adjusted EBITDA"]:
            assert _clean_heading(short) == short
