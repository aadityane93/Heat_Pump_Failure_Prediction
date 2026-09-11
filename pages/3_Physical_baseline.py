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

# The live model tree wins when it is present: a re-run of the model moves its
# values, and the page should follow. The bundled snapshot keeps a fresh clone
# runnable.
_UPSTREAM = DASHBOARD_DIR.parent / "PHYSICAL-BASELINE MODEL"
_BUNDLED = DASHBOARD_DIR / "physical_baseline"
_DEFAULT = _UPSTREAM if (_UPSTREAM / "outputs" / "export").exists() else _BUNDLED
PB_DIR = Path(os.getenv("PHYS_BASELINE_DIR", _DEFAULT))
EXPORT = PB_DIR / "outputs" / "export" / "physical_baseline_v1.parquet"
NOTEBOOK = PB_DIR / "outputs" / "deliverable" / "DELIVERABLE_physical_baseline.ipynb"

st.set_page_config(page_title="Physical baseline", layout="wide")

# No green anywhere: the model can confirm an excess but never its absence, so
# no state may read as "fine". Both no-result states share the neutral grey.
NEUTRAL = theme.AXIS_INK
PHYS_STATES = {
    "HIGH_EXCESS": ("Act — worst 25% of its era", theme.POS),
    "EXCESS_CONFIRMED": ("Confirmed, lower priority", theme.SERIES2),
    "INCONCLUSIVE": ("No finding", NEUTRAL),
    "INCOMPLETE_AUDIT": ("Data missing — can be collected", NEUTRAL),
}
VISIT_STATES = {
    "SLOPE_EXCESS": ("Ranked", theme.SERIES2),
    "NO_SLOPE_EXCESS": ("Nothing on this axis", NEUTRAL),
    "NOT_ASSESSABLE": ("Not computed", NEUTRAL),
}

ERAS = {
    "pre1975": "Before 1975",
    "1976_1990": "1976–1990",
    "1991_2000": "1991–2000",
    "2001_2010": "2001–2010",
    "post2010": "After 2010",
}
NO_ERA = "No era recorded"
ERA_ORDER = [*ERAS.values(), NO_ERA]
DASH = '<span class="faint">—</span>'

PB_CSS = f"""
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
.pb-st {{ display:inline-flex; align-items:center; gap:7px; white-space:nowrap; }}
.pb-era {{ display:flex; align-items:baseline; gap:10px; margin:20px 0 6px; }}
.pb-era b {{ font-size:16px; font-weight:600; color:inherit; }}
.pb-era span {{ color:{theme.FAINT}; font-size:13px; }}
</style>
"""


@st.cache_data(show_spinner="Reading the physical baseline export...")
def load_export(path: str, mtime: float) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["era_label"] = frame["phys_era"].map(ERAS).fillna(NO_ERA)
    return frame


def _num(value, signed: bool = False, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return DASH
    text = f"{value:+,.0f}" if signed else f"{value:,.0f}"
    return text.replace("-", "−") + suffix


def _status(code: str, states: dict) -> str:
    label, colour = states.get(code, (code, NEUTRAL))
    return (
        f'<span class="pb-st"><span class="sw" style="background:{colour}"></span>'
        f"{html.escape(label)}</span>"
    )


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


def _filters(frame: pd.DataFrame, status_col: str, states: dict, key: str):
    """Era and status buttons, all on by default."""
    era_sizes = frame["era_label"].value_counts()
    eras = [era for era in ERA_ORDER if era_sizes.get(era, 0)]
    picked_eras = st.pills(
        "Construction era", eras, selection_mode="multi", default=eras, key=f"{key}_era",
    )
    picked_status = st.pills(
        "Status", list(states), selection_mode="multi", default=list(states),
        key=f"{key}_status", format_func=lambda code: states[code][0],
    )
    return picked_eras or [], picked_status or []


def _ordered(group: pd.DataFrame, rank_col: str, status_col: str, states: dict,
             tiebreak: str) -> pd.DataFrame:
    """Ranked rows in the export's own order, then the unranked states in turn."""
    ranked = group[group[rank_col].notna()].sort_values(
        [rank_col, tiebreak, "household_id"], ascending=[True, False, True]
    )
    order = {code: i for i, code in enumerate(states)}
    rest = group[group[rank_col].isna()].assign(_o=lambda f: f[status_col].map(order))
    rest = rest.sort_values(["_o", "household_id"]).drop(columns="_o")
    return pd.concat([ranked, rest])


def _stacked(frame: pd.DataFrame, eras: list[str], statuses: list[str], rank_col: str,
             status_col: str, states: dict, tiebreak: str,
             headers: list[tuple[str, str]], row) -> None:
    shown = frame[frame[status_col].isin(statuses)]
    drawn = False
    # One table per era: the model never ranks one era against another.
    for era in (era for era in ERA_ORDER if era in eras):
        group = shown[shown["era_label"] == era]
        if group.empty:
            continue
        drawn = True
        ordered = _ordered(group, rank_col, status_col, states, tiebreak)
        st.markdown(
            f'<div class="pb-era"><b>{html.escape(era)}</b>'
            f"<span>{len(group)} households</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(_table(headers, [row(r) for _, r in ordered.iterrows()]),
                    unsafe_allow_html=True)
    if not drawn:
        st.info("No households match the selected filters.")


BUILDING_HEADERS = [
    ("Rank", "num"), ("Household", "id"), ("Status", ""),
    ("Excess over code, at least (kWh/yr)", "num"),
    ("Excess over code, at least (%)", "num"),
]


def _building_row(r: pd.Series) -> list[str]:
    ranked = pd.notna(r["phys_rank_in_era"])
    return [
        f"{int(r['phys_rank_in_era'])}" if ranked else DASH,
        f'<span class="id">{int(r["household_id"])}</span>',
        _status(r["phys_status"], PHYS_STATES),
        _num(r["phys_deficiency_lo_kwh_yr"]) if ranked else DASH,
        _num(r["phys_deficiency_pct_lo"], signed=True, suffix="%") if ranked else DASH,
    ]


# Two kWh columns, never one: the first is a setting a technician can change,
# the second needs a builder.
VISIT_HEADERS = [
    ("Rank", "num"), ("Household", "id"), ("Status", ""),
    ("Technician side (kWh/yr)", "num"), ("Fault probability", "num"),
    ("Named fault", ""), ("Building score, at least (kWh/yr)", "num"),
]


def _visit_row(r: pd.Series) -> list[str]:
    ranked = pd.notna(r["visit_rank_in_era"])
    prob = r["visit_fault_probability"]
    fault = r["visit_named_fault"]
    return [
        f"{int(r['visit_rank_in_era'])}" if ranked else DASH,
        f'<span class="id">{int(r["household_id"])}</span>',
        _status(r["visit_status"], VISIT_STATES),
        _num(r["visit_technician_recoverable_kwh"]) if ranked else DASH,
        # Absence is not a negative: an unscored house says so rather than 0%.
        f"{prob * 100:.0f}%" if pd.notna(prob) else '<span class="faint">not scored</span>',
        html.escape(str(fault)) if pd.notna(fault) and str(fault) else DASH,
        _num(r["phys_deficiency_lo_kwh_yr"]) if pd.notna(r["phys_rank_in_era"]) else DASH,
    ]


def building_score(frame: pd.DataFrame):
    st.markdown('<div class="x-h1">Building score</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="x-sub">How far each house sits above a code-compliant version of '
        "itself, ranked within each construction era.</p>",
        unsafe_allow_html=True,
    )
    eras, statuses = _filters(frame, "phys_status", PHYS_STATES, "pb_building")
    _stacked(frame, eras, statuses, "phys_rank_in_era", "phys_status", PHYS_STATES,
             "phys_deficiency_lo_kwh_yr", BUILDING_HEADERS, _building_row)


def visit_queue(frame: pd.DataFrame):
    st.markdown('<div class="x-h1">Visit queue</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="x-sub">Heating the building does not justify, ranked within each '
        "construction era. The fault probability is shown for the technician and does "
        "not set the order.</p>",
        unsafe_allow_html=True,
    )
    eras, statuses = _filters(frame, "visit_status", VISIT_STATES, "pb_visit")
    _stacked(frame, eras, statuses, "visit_rank_in_era", "visit_status", VISIT_STATES,
             "visit_technician_recoverable_kwh", VISIT_HEADERS, _visit_row)


def notebook_view():
    st.markdown(
        '<div class="x-h1">Physical baseline model — data workflow and results</div>',
        unsafe_allow_html=True,
    )
    if not NOTEBOOK.exists():
        st.error(f"Notebook not found at `{NOTEBOOK}`.")
        st.info("Set the PHYS_BASELINE_DIR environment variable to the model directory.")
        return
    show_code = st.toggle("Show code", value=False)
    notebook_render.render(NOTEBOOK, show_code=show_code)


def main():
    st.markdown(theme.BASE_CSS, unsafe_allow_html=True)
    st.markdown(PB_CSS, unsafe_allow_html=True)

    view = st.sidebar.radio("View", ["Building score", "Visit queue", "Data workflow and results"])
    st.sidebar.divider()

    if view == "Data workflow and results":
        notebook_view()
        return

    if not EXPORT.exists():
        st.error(f"Export not found at `{EXPORT}`.")
        st.info("Set the PHYS_BASELINE_DIR environment variable to the model directory.")
        return

    frame = load_export(str(EXPORT), EXPORT.stat().st_mtime)
    if view == "Building score":
        building_score(frame)
    else:
        visit_queue(frame)


main()
