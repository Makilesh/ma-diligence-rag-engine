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


def _fresh_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def test_app_renders_without_exception():
    """The script must run clean even with no deal selected and the API down."""
    at = _fresh_app()
    assert not at.exception, f"app raised on first render: {at.exception}"


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
