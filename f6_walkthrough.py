from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

import notebook_render

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
_BUNDLED = DASHBOARD_DIR / "forecast_data" / "R7_forecaster_walkthrough.ipynb"
_UPSTREAM = PROJECT_ROOT / "outputs" / "r7" / "R7_forecaster_walkthrough.ipynb"
DEFAULT_NOTEBOOK = _BUNDLED if _BUNDLED.exists() else _UPSTREAM
NOTEBOOK_PATH = Path(os.getenv("R7_NOTEBOOK", DEFAULT_NOTEBOOK))


def render():
    st.markdown(
        '<div class="x-h1">How the forecaster was built</div>', unsafe_allow_html=True
    )

    if not NOTEBOOK_PATH.exists():
        st.error(f"Walkthrough notebook not found at `{NOTEBOOK_PATH}`.")
        st.info("Set the R7_NOTEBOOK environment variable to point at the .ipynb.")
        return

    cells = notebook_render.load_cells(str(NOTEBOOK_PATH), NOTEBOOK_PATH.stat().st_mtime)
    if not any(c["outputs"] for c in cells):
        st.info(
            "The notebook is saved with its outputs cleared, so the figures it "
            "describes are not in the file and are not reproduced here. Re-run and "
            "save it with outputs, and they will appear in place automatically."
        )

    show_code = st.toggle("Show the code behind each step", value=False)
    notebook_render.render(NOTEBOOK_PATH, show_code=show_code)
    st.caption(f"Rendered from {NOTEBOOK_PATH.name}")
