from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DASHBOARD_DIR))

import notebook_render  # noqa: E402
import theme  # noqa: E402
from constants import FEATURE_DESCRIPTIONS  # noqa: E402
from peer_relative_features import (  # noqa: E402
    COL_DHW_BY_HP,
    COL_HAS_EV,
    COL_HAS_FAULT,
    DEFAULT_FEATURE_COLS,
    add_peer_relative_features,
    assign_stratum,
)

ARTIFACTS = DASHBOARD_DIR / "artifacts"
ANALYSIS_NOTEBOOK = (
    DASHBOARD_DIR / "notebooks_and_codes" / "Abdul_Fault_Detection_Analysis_RUN.ipynb"
)

st.set_page_config(page_title="Peer comparison", layout="wide")


# The peer readout is a four-column row; BASE_CSS only defines the two-column one.
PEER_CSS = f"""
<style>
.readout .row {{ grid-template-columns:1fr auto auto auto; gap:22px; }}
.num {{ font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
  font-size:16px; text-align:right; white-space:nowrap; color:inherit;
  min-width:88px; }}
.num u {{ text-decoration:none; display:block; font-size:10px; color:{theme.FAINT};
  letter-spacing:.06em; text-transform:uppercase; }}
.num.none {{ color:{theme.FAINT}; font-size:13px; }}
.ratio {{ font-weight:600; font-size:19px; }}
.readout .row-k .sw {{ margin-right:7px; vertical-align:baseline; }}
.readout .row-k small {{ margin-left:16px; }}
</style>
"""




@st.cache_data(show_spinner="Building peer-relative features...")
def load_peer_table(min_peers: int):
    dataset = pd.read_parquet(ARTIFACTS / "dataset.parquet")
    dataset["Household_ID"] = pd.to_numeric(
        dataset["Household_ID"], errors="coerce"
    ).astype("Int64")
    enriched = add_peer_relative_features(dataset, min_peers=min_peers)
    enriched["stratum"] = assign_stratum(enriched)
    return enriched


def pool_size(frame: pd.DataFrame, row: pd.Series) -> int:
    """Households that qualify as peers for this row, by the module's own rules."""
    same = frame[frame["stratum"] == row["stratum"]]
    if COL_DHW_BY_HP in frame.columns:
        same = same[same[COL_DHW_BY_HP] == row[COL_DHW_BY_HP]]
    if COL_HAS_EV in frame.columns:
        same = same[same[COL_HAS_EV] != True]  # noqa: E712
    if COL_HAS_FAULT in frame.columns:
        same = same[same[COL_HAS_FAULT] != 1]
    return max(len(same) - 1, 0)


def readout(row: pd.Series, features: list[str]) -> str:
    colors = theme.feature_colors(len(features))
    rows = []
    for col, hue in zip(features, colors):
        ratio = row.get(f"{col}_vs_peer")
        own = row.get(col)
        label = FEATURE_DESCRIPTIONS.get(col, col.replace("_", " "))

        if ratio is None or pd.isna(ratio) or own is None or pd.isna(own):
            body = (
                f'<div class="num none">—</div><div class="num none">—</div>'
                f'<div class="num none">no peer match</div>'
            )
        else:
            peer_mean = own / ratio if ratio else np.nan
            colour = theme.POS if ratio > 1 else theme.NEG
            body = (
                f'<div class="num">{own:,.2f}<u>own</u></div>'
                f'<div class="num">{peer_mean:,.2f}<u>peers</u></div>'
                f'<div class="num ratio" style="color:{colour}">{ratio:,.2f}<u>ratio</u></div>'
            )
        rows.append(
            f'<div class="row"><div class="row-k">'
            f'<span class="sw" style="background:{hue}"></span>{col}'
            f"<small>{label}</small></div>{body}</div>"
        )
    return f'<div class="readout">{"".join(rows)}</div>'


def ratio_chart(row: pd.Series, features: list[str]):
    frame = pd.DataFrame(
        {
            "feature": features,
            "ratio": [row.get(f"{c}_vs_peer") for c in features],
        }
    ).dropna(subset=["ratio"])

    if frame.empty:
        return None

    frame["direction"] = np.where(frame["ratio"] > 1, "above peers", "below peers")

    bars = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("ratio:Q", title="household ÷ peer mean",
                    scale=alt.Scale(zero=True),
                    axis=alt.Axis(labelFontSize=11, titleFontSize=11)),
            y=alt.Y("feature:N", title=None, sort=features,
                    axis=alt.Axis(labelFontSize=12)),
            # One hue per feature, in fixed slot order. The y-axis names every
            # bar, so identity is carried by text and the legend would only
            # repeat the axis.
            color=alt.Color(
                "feature:N",
                scale=alt.Scale(domain=features, range=theme.feature_colors(len(features))),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("ratio:Q", title="Ratio", format=".3f"),
                alt.Tooltip("direction:N", title="Versus peers"),
            ],
        )
    )
    parity = (
        alt.Chart(pd.DataFrame({"x": [1.0]}))
        .mark_rule(color=theme.AXIS_INK, strokeWidth=1.5, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    layered = alt.layer(bars, parity).properties(height=alt.Step(46))
    return theme.chart_config(layered)


def data_overview():
    st.markdown(
        '<div class="x-h1">HEAPO fault detection — validation analysis</div>',
        unsafe_allow_html=True,
    )
    if not ANALYSIS_NOTEBOOK.exists():
        st.error(f"Notebook not found at `{ANALYSIS_NOTEBOOK}`.")
        return
    show_code = st.toggle("Show the code behind each task", value=False)
    notebook_render.render(ANALYSIS_NOTEBOOK, show_code=show_code)
    st.caption(f"Rendered from {ANALYSIS_NOTEBOOK.name} — outputs as saved in the notebook.")


def main():
    st.markdown(theme.BASE_CSS, unsafe_allow_html=True)
    st.markdown(PEER_CSS, unsafe_allow_html=True)

    view = st.sidebar.radio("View", ["Peer comparison", "Data overview"])
    st.sidebar.divider()
    if view == "Data overview":
        data_overview()
        return

    st.markdown('<div class="x-h1">Peer-relative comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="x-sub">Each feature divided by the mean of the household\'s matched '
        "peers — same heat-pump type and distribution system, same hot-water method, "
        "no EV, no confirmed fault. A ratio of 1.00 means the household sits exactly on "
        "its peer group.</p>",
        unsafe_allow_html=True,
    )

    if not (ARTIFACTS / "dataset.parquet").exists():
        st.error("`artifacts/dataset.parquet` is missing. Run prepare_data.py first.")
        return

    min_peers = st.sidebar.slider("Minimum peers", 1, 6, 3)
    st.sidebar.caption(
        "Below this many matched peers the ratio is left empty rather than computed "
        "on a handful of noisy peers."
    )

    table = load_peer_table(min_peers)
    features = [c for c in DEFAULT_FEATURE_COLS if f"{c}_vs_peer" in table.columns]
    matched = table[table[f"{features[0]}_vs_peer"].notna()]

    only_matched = st.sidebar.checkbox("Only households with a peer match", value=True)
    shown = matched if only_matched else table
    if shown.empty:
        st.warning("No household has enough matched peers at this threshold.")
        return

    c1, c2, c3 = st.columns([2, 3, 5], vertical_alignment="bottom")
    with c1:
        house = st.selectbox(
            "Household", shown["Household_ID"].tolist(), format_func=lambda h: str(int(h))
        )
    row = table.set_index("Household_ID").loc[house]
    row["Household_ID"] = house

    with c2:
        st.markdown(
            f'<div class="chips"><span class="chip">{pool_size(table, row)} matched peers</span>'
            f'<span class="chip">{"fault found" if row.get(COL_HAS_FAULT) == 1 else "no fault"}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="chips" style="justify-content:flex-end">'
            f'<span class="chip">{row["stratum"]}</span>'
            f'<span class="chip">DHW by HP {row.get(COL_DHW_BY_HP)}</span>'
            f'<span class="chip">EV {row.get(COL_HAS_EV)}</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(readout(row, features), unsafe_allow_html=True)
    st.write("")

    chart = ratio_chart(row, features)
    if chart is None:
        st.info(
            f"Household {int(house)} has fewer than {min_peers} matched peers, so no "
            "ratio is defined for it. That is the module's stance, not a failure: an "
            "unstable ratio on one or two peers is worse than none."
        )
    else:
        st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.subheader("Coverage")

    pool = table
    if COL_HAS_FAULT in pool.columns:
        pool = pool[pool[COL_HAS_FAULT] != 1]
    if COL_HAS_EV in pool.columns:
        pool = pool[pool[COL_HAS_EV] != True]  # noqa: E712

    m1, m2, m3 = st.columns(3)
    m1.metric("Households", f"{len(table)}")
    m2.metric("Eligible as peers", f"{len(pool)}")
    m3.metric("With a comparison", f"{len(matched)}")

    st.caption(
        f"Only {len(pool)} of {len(table)} households can serve as peers at all: a peer "
        f"must be fault-free and EV-free, and {int((table[COL_HAS_FAULT] == 1).sum())} of "
        f"these households have a confirmed fault. That is why {len(matched)} get a ratio. "
        "The scarcity is a property of this labelled sample — households were visited "
        "because something was suspected — not of the peer method, which was written for "
        "the full fleet where most households are unvisited."
    )

    breakdown = (
        pool.groupby(["stratum", COL_DHW_BY_HP], dropna=False)
        .size()
        .reset_index(name="eligible peers")
        .sort_values("eligible peers", ascending=False)
    )
    with st.expander("Eligible peers by stratum and hot-water method"):
        st.dataframe(breakdown, hide_index=True, use_container_width=True)


main()
