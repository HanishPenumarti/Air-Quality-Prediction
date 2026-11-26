"""UI smoke test: renders every page of the Streamlit app headlessly (streamlit.testing.AppTest).

Needs pipeline outputs (models + processed data), from a full run or from the quick demo
(README: AQ_OUT / AQ_STATIONS). Skipped automatically when they are missing.
    python -m pytest tests/test_app.py -q          (set AQ_OUT first to test a demo bundle)
"""
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import MODELS, PROC  # noqa: E402

PAGES = ["page_forecast", "page_map", "page_whatif", "page_benchmark", "page_analyses", "page_data"]
HAVE = (PROC / "features_full.parquet").exists() and (MODELS / "production.json").exists()


@pytest.mark.skipif(not HAVE, reason="pipeline outputs not found (run the pipeline or the quick demo first)")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, tmp_path):
    from streamlit.testing.v1 import AppTest
    wrapper = tmp_path / "wrap.py"
    wrapper.write_text(textwrap.dedent(f"""
        import streamlit as st
        class _P:
            def __init__(self, fn, **k): self.fn = fn
        class _N:
            def __init__(self, pages): self.pages = pages
            def run(self): [p for p in self.pages if p.fn.__name__ == {page!r}][0].fn()
        st.Page = _P
        st.navigation = lambda pages, **k: _N(pages)
        src = open(r"{ROOT / 'app' / 'app.py'}", encoding="utf-8").read()
        exec(compile(src, "app.py", "exec"), {{"__name__": "__main__", "__file__": r"{ROOT / 'app' / 'app.py'}"}})
    """))
    at = AppTest.from_file(str(wrapper), default_timeout=900).run()
    assert not at.exception, [e.value for e in at.exception]
