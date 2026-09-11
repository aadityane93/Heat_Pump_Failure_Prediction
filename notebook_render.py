from __future__ import annotations

import base64
import json
from pathlib import Path

import streamlit as st

import theme

# Streamlit renders every element in its own container, so a wrapper <div>
# spanning several st.* calls never encloses them. These selectors target the
# containers themselves, which is the only scoping that actually applies.
_MD = '[data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"]'
_MAIN = '[data-testid="stMainBlockContainer"]'


NB_CSS = f"""
<style>
{_MD} h1 {{ font-size:23px; font-weight:600; color:inherit; margin:24px 0 8px; }}
{_MD} h2 {{ font-size:19px; font-weight:600; color:inherit; margin:26px 0 6px; }}
{_MD} h3 {{ font-size:15px; font-weight:600; color:inherit; margin:18px 0 4px; }}
{_MD} p, {_MD} li {{ color:{theme.MUTED}; font-size:14px; line-height:1.62; }}
{_MD} strong {{ color:inherit; }}
{_MD} p code, {_MD} li code {{ font-family:"IBM Plex Mono",monospace; font-size:12.5px;
  background:{theme.SUNK}; border:1px solid {theme.LINE}; border-radius:3px;
  padding:1px 5px; color:inherit; }}
{_MD} hr {{ border:0; border-top:1px solid {theme.LINE}; margin:26px 0 0; }}
{_MAIN} table {{ border-collapse:collapse; margin:10px 0; font-size:12.5px;
  display:block; overflow-x:auto; max-width:100%; }}
{_MAIN} th, {_MAIN} td {{ border:1px solid {theme.LINE}; padding:5px 9px;
  color:{theme.MUTED}; text-align:right; white-space:nowrap; }}
{_MAIN} th {{ background:{theme.SUNK}; color:inherit; font-weight:600; }}
{_MAIN} [data-testid="stImage"] img {{ background:#ffffff;
  border:1px solid {theme.LINE}; border-radius:7px; padding:6px; }}
</style>
"""


def _outputs(cell: dict) -> list[tuple[str, object]]:
    """One renderable item per output, richest representation first.

    Jupyter emits several mime types for the same output; picking the richest is
    what the notebook viewer does too. Nothing is rewritten, only selected.
    """
    items: list[tuple[str, object]] = []
    for output in cell.get("outputs", []):
        kind = output.get("output_type")

        if kind == "stream":
            text = "".join(output.get("text", []))
            if text.strip():
                items.append(("text", text))
            continue

        if kind == "error":
            trace = "\n".join(output.get("traceback", []))
            if trace.strip():
                items.append(("text", trace))
            continue

        data = output.get("data", {})
        if "image/png" in data:
            payload = data["image/png"]
            if isinstance(payload, list):
                payload = "".join(payload)
            items.append(("image", base64.b64decode(payload)))
        elif "image/svg+xml" in data:
            payload = data["image/svg+xml"]
            items.append(("svg", "".join(payload) if isinstance(payload, list) else payload))
        elif "text/html" in data:
            payload = data["text/html"]
            items.append(("html", "".join(payload) if isinstance(payload, list) else payload))
        elif "text/plain" in data:
            payload = data["text/plain"]
            text = "".join(payload) if isinstance(payload, list) else payload
            if text.strip():
                items.append(("text", text))
    return items


@st.cache_data(show_spinner="Reading the notebook...")
def load_cells(path: str, mtime: float):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        {
            "type": cell.get("cell_type"),
            "source": "".join(cell.get("source", [])),
            "outputs": _outputs(cell),
        }
        for cell in notebook.get("cells", [])
    ]


def render(path: Path, show_code: bool = False, drop_leading_title: bool = True):
    """Render a notebook's markdown, figures and outputs. Content is unchanged."""
    st.markdown(NB_CSS, unsafe_allow_html=True)
    cells = load_cells(str(path), path.stat().st_mtime)

    dropped = not drop_leading_title
    for cell in cells:
        if cell["type"] == "markdown":
            body = cell["source"]
            if not dropped and body.lstrip().startswith("# "):
                body = body.lstrip().split("\n", 1)[1] if "\n" in body else ""
                dropped = True
            if body.strip():
                st.markdown(body)
            continue

        if show_code and cell["source"].strip():
            st.code(cell["source"], language="python")

        for kind, payload in cell["outputs"]:
            if kind == "image":
                st.image(payload)
            elif kind == "svg":
                st.markdown(payload, unsafe_allow_html=True)
            elif kind == "html":
                st.markdown(payload, unsafe_allow_html=True)
            else:
                st.code(payload)
