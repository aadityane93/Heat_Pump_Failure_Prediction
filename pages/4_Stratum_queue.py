from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DASHBOARD_DIR))

import notebook_render  # noqa: E402
import theme  # noqa: E402

# The live model tree wins when it is present, so a re-run reaches the page; the
# bundled snapshot keeps a fresh clone runnable.
_UPSTREAM = DASHBOARD_DIR.parent / "STRATUM-LEVEL RESIDUAL MODEL"
_BUNDLED = DASHBOARD_DIR / "stratum_model"
_DEFAULT = _UPSTREAM if (_UPSTREAM / "outputs" / "s12").exists() else _BUNDLED
SM_DIR = Path(os.getenv("STRATUM_DIR", _DEFAULT))
QUEUE = SM_DIR / "outputs" / "s12" / "data" / "s12_dashboard_queue.parquet"
NOTEBOOK = SM_DIR / "outputs" / "s12" / "S12_fleet_split.ipynb"

st.set_page_config(page_title="Stratum visit queue", layout="wide")

# This page's own letters. B names no fault, and C is not "fine": neutral, never green.
CATEGORIES = {
    "A_configuration_fault_candidate": (
        "A — check a setting", theme.SERIES2,
        "The setting to check names which one.",
    ),
    "B_high_consumption_no_fault_identified": (
        "B — high, cause unknown", theme.SERIES1,
        "Call or check remotely first — do not dispatch blind.",
    ),
    "C_nothing_material": ("C — nothing material", theme.AXIS_INK, ""),
}
DASH = '<span class="faint">—</span>'

SQ_CSS = f"""
<style>
.pb-frame {{ background:{theme.PANEL}; border:1px solid {theme.LINE}; border-radius:9px;
  padding:4px 8px; overflow-x:auto; margin-bottom:6px; }}
.pb-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.pb-table th {{ text-align:left; font-weight:600; font-size:11.5px; color:{theme.MUTED};
  padding:9px 10px; border-bottom:1px solid {theme.LINE}; vertical-align:bottom;
  max-width:150px; }}
.pb-table th.num {{ white-space:normal; }}
.pb-table td {{ padding:7px 10px; border-bottom:1px solid {theme.LINE};
  vertical-align:baseline; }}
.pb-table tr:last-child td {{ border-bottom:0; }}
.pb-table .num {{ text-align:right; font-family:"IBM Plex Mono",monospace;
  font-variant-numeric:tabular-nums; white-space:nowrap; }}
.pb-table .id {{ font-family:"IBM Plex Mono",monospace; }}
.pb-table .faint {{ color:{theme.FAINT}; }}
.sq-cat {{ display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin:18px 0 6px; }}
.sq-cat b {{ font-size:16px; font-weight:600; color:inherit; }}
.sq-cat .sw {{ align-self:center; }}
.sq-cat span.n, .sq-cat span.note {{ color:{theme.FAINT}; font-size:13px; }}
</style>
"""


@st.cache_data(show_spinner="Reading the stratum queue...")
def load_queue(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def _kwh(value) -> str:
    if value is None or pd.isna(value):
        return DASH
    return f"{value:,.0f}"


def _table(headers: list[tuple[str, str]], rows: list[list[str]]) -> str:
    head = "".join(f'<th class="{cls}">{html.escape(text)}</th>' for text, cls in headers)
    body = "".join(
        "<tr>"
        + "".join(f'<td class="{headers[i][1]}">{cell}</td>' for i, cell in enumerate(row))
        + "</tr>"
        for row in rows
    )
    return (
        f'<div class="pb-frame"><table class="pb-table"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


QUEUE_HEADERS = [
    ("Rank", "num"), ("Household", "id"), ("Peer group", ""),
    ("Excess energy (kWh/yr)", "num"), ("Where the excess is", ""),
    ("Setting to check (A only)", ""),
]


def _queue_rows(group: pd.DataFrame) -> list[list[str]]:
    rows = []
    for _, r in group.sort_values("rank_in_category").iterrows():
        setting = r["candidate_setting"]
        rows.append([
            f"{int(r['rank_in_category'])}",
            f'<span class="id">{int(r["Household_ID"])}</span>',
            html.escape(str(r["cell"])),
            _kwh(r["recoverable_kwh_yr"]),
            html.escape(str(r["driver"])),
            html.escape(str(setting)) if pd.notna(setting) and str(setting) else DASH,
        ])
    return rows


def queue(frame: pd.DataFrame):
    st.markdown('<div class="x-h1">Stratum visit queue</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="x-sub">Houses consuming more than comparable houses, ranked by excess '
        "energy. One baseline at a time.</p>",
        unsafe_allow_html=True,
    )

    sizes = frame["baseline"].value_counts()
    baseline = st.selectbox("Baseline", list(sizes.index))
    group = frame[frame["baseline"] == baseline]

    for code, (label, colour, note) in CATEGORIES.items():
        rows = group[group["category"] == code]
        if code == "C_nothing_material":
            with st.expander(f"{label}  ·  {len(rows)} households", expanded=False):
                st.markdown(_table(QUEUE_HEADERS, _queue_rows(rows)), unsafe_allow_html=True)
            continue
        st.markdown(
            f'<div class="sq-cat"><span class="sw" style="background:{colour}"></span>'
            f"<b>{html.escape(label)}</b><span class=\"n\">{len(rows)} households</span>"
            f'<span class="note">{html.escape(note)}</span></div>',
            unsafe_allow_html=True,
        )
        if rows.empty:
            st.markdown(f'<p class="x-sub">None in this baseline.</p>', unsafe_allow_html=True)
        else:
            st.markdown(_table(QUEUE_HEADERS, _queue_rows(rows)), unsafe_allow_html=True)


def summary(frame: pd.DataFrame):
    st.markdown('<div class="x-h1">Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="x-sub">Households and excess energy per category, for each '
        "baseline separately.</p>",
        unsafe_allow_html=True,
    )
    baselines = list(frame["baseline"].value_counts().index)
    for col, baseline in zip(st.columns(len(baselines)), baselines):
        group = frame[frame["baseline"] == baseline]
        rows = []
        for code, (label, colour, _) in CATEGORIES.items():
            cat = group[group["category"] == code]
            rows.append([
                f'<span class="sw" style="background:{colour};margin-right:7px"></span>'
                f"{html.escape(label)}",
                f"{len(cat):,}",
                _kwh(cat["recoverable_kwh_yr"].sum(min_count=1)),
            ])
        with col:
            st.markdown(
                f'<div class="sq-cat"><b>{html.escape(baseline)}</b></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                _table([("Category", ""), ("Households", "num"), ("Excess energy (kWh/yr)", "num")], rows),
                unsafe_allow_html=True,
            )


def notebook_view():
    if not NOTEBOOK.exists():
        st.error(f"Notebook not found at `{NOTEBOOK}`.")
        st.info("Set the STRATUM_DIR environment variable to the model directory.")
        return
    cells = notebook_render.load_cells(str(NOTEBOOK), NOTEBOOK.stat().st_mtime)
    # render() drops the notebook's own title, so it is shown here as the heading.
    title = next(
        (c["source"].lstrip()[2:].splitlines()[0] for c in cells
         if c["type"] == "markdown" and c["source"].lstrip().startswith("# ")),
        "Data workflow and results",
    )
    st.markdown(f'<div class="x-h1">{html.escape(title)}</div>', unsafe_allow_html=True)
    show_code = st.toggle("Show code", value=False)
    notebook_render.render(NOTEBOOK, show_code=show_code)


def main():
    st.markdown(theme.BASE_CSS, unsafe_allow_html=True)
    st.markdown(SQ_CSS, unsafe_allow_html=True)

    view = st.sidebar.radio("View", ["Queue", "Summary", "Data workflow and results"])
    st.sidebar.divider()

    if view == "Data workflow and results":
        notebook_view()
        return

    if not QUEUE.exists():
        st.error(f"Queue not found at `{QUEUE}`.")
        st.info("Set the STRATUM_DIR environment variable to the model directory.")
        return

    frame = load_queue(str(QUEUE), QUEUE.stat().st_mtime)
    if view == "Queue":
        queue(frame)
    else:
        summary(frame)


main()
