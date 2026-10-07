"""Headless smoke test of the Streamlit dashboard (no browser needed)."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _no_exceptions(at):
    assert not at.exception, [e.value for e in at.exception]


def test_app_renders_and_main_flows_work():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    _no_exceptions(at)

    at.button(key="quick_massage").click().run()
    _no_exceptions(at)
    assert at.text_input(key="keyword").value == "massage"

    at.button(key="btn_lda").click().run()
    _no_exceptions(at)

    at.radio(key="tl_order").set_value("Corpus order").run()
    _no_exceptions(at)

    at.text_input(key="sub_search").input("training").run()
    _no_exceptions(at)

    at.text_input(key="keyword").input("zzzqqq").run()
    _no_exceptions(at)
    assert any("No emails contain" in i.value for i in at.info)


def test_untrusted_email_text_is_escaped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["keyword"] = "massage"
    at.run()
    _no_exceptions(at)
    # Corpus bodies contain raw fragments such as "Thai Massage Sacred Bodywork <
    # wrote:"; they must reach the page escaped, never as live markup.
    rendered = "\n".join(m.value for m in at.markdown)
    assert "Bodywork &lt;" in rendered
    assert "Bodywork <" not in rendered
