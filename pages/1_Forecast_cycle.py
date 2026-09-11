from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import f6_walkthrough
import theme

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DASHBOARD_DIR.parent
BUNDLED = DASHBOARD_DIR / "forecast_data"
# The bundled copy makes a fresh clone runnable; the project-root layout is
# where these files live in the F-series working tree.
UPSTREAM = PROJECT_ROOT / "Forecast_consumption_cycle" / "outputs" / "f6" / "results"
F6_RESULTS = BUNDLED if (BUNDLED / "f6_household_features.parquet").exists() else UPSTREAM
EXPORT_PATH = Path(os.getenv("F6_EXPORT", F6_RESULTS / "f6_dashboard_export_DRAFT.parquet"))

LABEL_TEXT = {
    "valid": "valid",
    "pv_contaminated": "PV contaminated",
    "low_confidence_fit": "low confidence fit",
    "outside_fitted_window": "outside fitted window",
    "insufficient_training_data": "insufficient training data",
    "no_usable_measurement": "no usable measurement",
}

st.set_page_config(page_title="Heat Pump Day Readout", layout="wide")




def as_id(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


@st.cache_data(show_spinner="Reading the F6 export...")
def load_export(path: str):
    frame = pd.read_parquet(path)
    frame["Household_ID"] = as_id(frame["Household_ID"])
    frame["local_date"] = pd.to_datetime(frame["local_date"])
    meta = pq.ParquetFile(path).schema_arrow.metadata or {}
    notes = {k.decode(): v.decode() for k, v in meta.items() if k.decode().lower() != "pandas"}
    return frame, notes


@st.cache_resource(show_spinner="Resolving validity labels...")
def load_resolver():
    # f6_validity.py is the single source of the precedence order; it is
    # imported rather than reimplemented so the labels cannot drift.
    for candidate in (BUNDLED, PROJECT_ROOT):
        if (candidate / "f6_validity.py").exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break
    try:
        from f6_validity import ValidityResolver

        return ValidityResolver(results_dir=F6_RESULTS), None
    except Exception as exc:
        return None, f"{exc.__class__.__name__}: {exc}"


def fmt(value, digits=1):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return f"{value:,.{digits}f}"


def readout_card(day: pd.Series, label: str) -> str:
    color = theme.LABELS.get(label, theme.AXIS_INK)
    text = LABEL_TEXT.get(label, label)
    date_text = day["local_date"].strftime("%Y-%m-%d")

    actual = fmt(day.get("actual_kwh"))
    temp = fmt(day.get("temp_mean"))
    pred = fmt(day.get("predicted_kwh"))
    dev = day.get("deviation_kwh")
    dev_pct = day.get("deviation_pct")

    def row(key, sub, value, unit, extra="", cls=""):
        if value is None:
            body = '<div class="row-v none">not available</div>'
        else:
            body = (
                f'<div class="row-v">{value}<u>{unit}</u>'
                + (f"<i>{extra}</i>" if extra else "")
                + "</div>"
            )
        return (
            f'<div class="row {cls}"><div class="row-k">{key}<small>{sub}</small></div>'
            f"{body}</div>"
        )

    if dev is None or (isinstance(dev, float) and np.isnan(dev)):
        margin = row("Error margin", "predicted &minus; consumption &middot; positive = over-prediction",
                     None, "", "", "margin")
    else:
        direction = "over-prediction" if dev >= 0 else "under-prediction"
        pct_text = ""
        if dev_pct is not None and not (isinstance(dev_pct, float) and np.isnan(dev_pct)):
            pct_text = f"{dev_pct:+.1f}% &middot; {direction}"
        else:
            pct_text = direction
        margin = row("Error margin", "predicted &minus; consumption &middot; positive = over-prediction",
                     f"{dev:+,.1f}", "kWh", pct_text, "margin")

    return f"""
<div class="readout">
  <div class="readout-head">
    <div class="rd-date">{date_text}</div>
    <div class="lab"><span class="sw" style="background:{color}"></span>{html.escape(text)}</div>
  </div>
  <div class="rows">
    {row("Consumption", "what the meter recorded", actual, "kWh")}
    {row("Mean temperature", "Europe/Zurich day, the model's regressor", temp, "&deg;C")}
    {row("Predicted", "stored W2_hl6 coefficients on that temperature", pred, "kWh")}
    {margin}
  </div>
</div>
"""


def record_chart(one: pd.DataFrame, selected: pd.Timestamp):
    wide = one[["local_date", "actual_kwh", "predicted_kwh", "deviation_kwh", "temp_mean"]]

    long = (
        wide.melt(
            "local_date", value_vars=["actual_kwh", "predicted_kwh"],
            var_name="series", value_name="kwh",
        )
        .dropna(subset=["kwh"])
        .replace({"series": {"actual_kwh": "measured", "predicted_kwh": "predicted"}})
    )

    scale = alt.Scale(domain=["measured", "predicted"], range=[theme.SERIES1, theme.SERIES2])
    dashes = alt.Scale(domain=["measured", "predicted"], range=[[1, 0], [4, 3]])

    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=1.1)
        .encode(
            x=alt.X("local_date:T", title=None, axis=alt.Axis(labelFontSize=11)),
            y=alt.Y("kwh:Q", title="kWh/day", axis=alt.Axis(labelFontSize=11, titleFontSize=11)),
            color=alt.Color("series:N", scale=scale, title=None,
                            legend=alt.Legend(orient="top-right", labelFontSize=11)),
            strokeDash=alt.StrokeDash("series:N", scale=dashes, legend=None),
        )
    )

    marker = (
        alt.Chart(pd.DataFrame({"local_date": [selected]}))
        .mark_rule(color=theme.SERIES2, strokeWidth=1.5, opacity=0.9)
        .encode(x="local_date:T")
    )

    hover = alt.selection_point(
        fields=["local_date"], nearest=True, on="mouseover", empty=False
    )
    probe = (
        alt.Chart(wide)
        .mark_rule(color=theme.AXIS_INK, strokeWidth=1)
        .encode(
            x="local_date:T",
            opacity=alt.condition(hover, alt.value(0.5), alt.value(0)),
            tooltip=[
                alt.Tooltip("local_date:T", title="Date"),
                alt.Tooltip("actual_kwh:Q", title="Measured kWh", format=".1f"),
                alt.Tooltip("predicted_kwh:Q", title="Predicted kWh", format=".1f"),
                alt.Tooltip("deviation_kwh:Q", title="Error kWh", format="+.1f"),
                alt.Tooltip("temp_mean:Q", title="Mean temp °C", format=".1f"),
            ],
        )
        .add_params(hover)
    )

    layered = alt.layer(lines, marker, probe).properties(height=460)
    return theme.chart_config(layered).interactive()


def main():
    st.markdown(theme.BASE_CSS, unsafe_allow_html=True)

    view = st.sidebar.radio("View", ["Day readout", "Walkthrough"])
    st.sidebar.divider()
    if view == "Walkthrough":
        f6_walkthrough.render()
        return

    st.markdown('<div class="x-h1">Heat Pump Day Readout</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="x-sub">Pick a household, pick a date. The page returns what the meter '
        "recorded, the weather that day, what the model predicted, and the gap between "
        "them.</p>",
        unsafe_allow_html=True,
    )

    if not EXPORT_PATH.exists():
        st.error(f"F6 export not found at `{EXPORT_PATH}`.")
        return

    data, notes = load_export(str(EXPORT_PATH))
    resolver, resolver_error = load_resolver()

    meta = (
        data.groupby("Household_ID")
        .agg(
            days=("local_date", "count"),
            band=("fitquality_band", "first"),
            pv=("pv_flag", "first"),
            group=("group", "first"),
        )
        .reset_index()
    )

    bar = st.columns([2, 0.5, 0.5, 1.6, 3.4], vertical_alignment="bottom")

    with bar[0]:
        house = st.selectbox(
            "Household", meta["Household_ID"].tolist(),
            format_func=lambda hid: str(int(hid)),
        )

    one = data[data["Household_ID"] == house].sort_values("local_date").reset_index(drop=True)
    dates = one["local_date"]
    first, last = dates.iloc[0], dates.iloc[-1]

    key = f"f6_date_{int(house)}"
    if key not in st.session_state:
        predicted_days = one.loc[one["predicted_kwh"].notna(), "local_date"]
        st.session_state[key] = (
            predicted_days.iloc[len(predicted_days) // 2] if len(predicted_days) else last
        ).date()

    with bar[1]:
        if st.button("‹", use_container_width=True, help="Previous day"):
            step = pd.Timestamp(st.session_state[key]) - pd.Timedelta(days=1)
            if step >= first:
                st.session_state[key] = step.date()
    with bar[2]:
        if st.button("›", use_container_width=True, help="Next day"):
            step = pd.Timestamp(st.session_state[key]) + pd.Timedelta(days=1)
            if step <= last:
                st.session_state[key] = step.date()
    with bar[3]:
        st.date_input(
            "Date", key=key, min_value=first.date(), max_value=last.date(),
        )
    with bar[4]:
        info = meta.set_index("Household_ID").loc[house]
        st.markdown(
            '<div class="chips" style="justify-content:flex-end">'
            f'<span class="chip">{info["band"]}</span>'
            f'<span class="chip">{info["group"]}</span>'
            f'<span class="chip">PV {info["pv"]}</span>'
            f'<span class="chip">{int(info["days"]):,} days</span>'
            f'<span class="chip">{first:%Y-%m-%d} &rarr; {last:%Y-%m-%d}</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    chosen = pd.Timestamp(st.session_state[key])
    match = one[one["local_date"] == chosen]

    if match.empty:
        # The export carries only days the spine holds; a gap is itself the
        # answer, not a missing row to paper over.
        st.markdown(
            readout_card(
                pd.Series({"local_date": chosen, "actual_kwh": np.nan,
                           "temp_mean": np.nan, "predicted_kwh": np.nan,
                           "deviation_kwh": np.nan, "deviation_pct": np.nan}),
                "no_usable_measurement",
            ),
            unsafe_allow_html=True,
        )
    else:
        day = match.iloc[0]
        if resolver is not None:
            label = resolver.label(int(house), chosen)
        else:
            label = "valid" if bool(day["has_prediction"]) else "outside_fitted_window"
        st.markdown(readout_card(day, label), unsafe_allow_html=True)

    st.write("")
    st.altair_chart(record_chart(one, chosen), use_container_width=True)

    if resolver_error:
        st.caption(f"Validity labels fell back to the export's own flags — {resolver_error}")


main()
