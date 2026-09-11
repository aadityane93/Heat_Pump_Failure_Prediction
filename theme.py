from __future__ import annotations

# Streamlit switches theme in the browser without re-running the Python script,
# so anything whose colour is decided in Python goes stale until the next rerun.
# Aditya's page never has that problem because it injects no CSS at all and lets
# Streamlit own every colour. This module does the same thing two ways:
#
#   1. CSS inherits Streamlit's text colour and builds surfaces from translucent
#      grey, so it repaints instantly on a theme switch.
#   2. Charts use ONE palette validated against BOTH surfaces, on a transparent
#      background, so no series colour has to be chosen per theme.
#
# The palette is the reference dark steps: all checks pass on the dark surface,
# and on the light surface every check passes too (worst contrast 2.99 on the
# yellow, relieved by the axis labelling every mark).
CAT = ["#3987e5", "#d95926", "#199e70", "#c98500"]

SERIES1 = CAT[0]
SERIES2 = CAT[1]
POS = "#e66767"
NEG = "#3987e5"

# Neutral chrome that reads on either surface.
AXIS_INK = "#8a8a85"
GRID = "rgba(128,128,128,0.28)"

LABELS = {
    "valid": "#0ca30c",
    "pv_contaminated": "#e66767",
    "low_confidence_fit": "#c98500",
    "outside_fitted_window": "#3987e5",
    "insufficient_training_data": "#7fafd8",
    "no_usable_measurement": "#8a8a85",
}

# Surfaces as translucent grey: identical rule on light and dark, and the
# browser resolves them against whatever Streamlit has painted.
PANEL = "rgba(128,128,128,0.07)"
SUNK = "rgba(128,128,128,0.14)"
LINE = "rgba(128,128,128,0.28)"
MUTED = "color-mix(in srgb, currentColor 72%, transparent)"
FAINT = "color-mix(in srgb, currentColor 55%, transparent)"

BASE_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
.x-h1 {{ font-size:25px; font-weight:600; letter-spacing:-.016em; color:inherit;
  margin:2px 0 4px; font-family:"IBM Plex Sans",system-ui,sans-serif; }}
.x-sub {{ color:{MUTED}; max-width:70ch; font-size:14px; margin:0 0 6px; }}
.chips {{ display:flex; gap:6px; flex-wrap:wrap; align-items:center; margin:0 0 10px; }}
.chip {{ font-family:"IBM Plex Mono",monospace; font-size:11px; padding:3px 8px;
  border-radius:4px; background:{SUNK}; color:{MUTED};
  border:1px solid {LINE}; white-space:nowrap; }}
.chip b {{ color:inherit; font-weight:600; font-size:13px; }}
.readout {{ background:{PANEL}; border:1px solid {LINE}; border-radius:9px;
  overflow:hidden; }}
.readout-head {{ display:flex; align-items:center; justify-content:space-between;
  gap:14px; flex-wrap:wrap; padding:13px 16px; border-bottom:1px solid {LINE}; }}
.rd-date {{ font-family:"IBM Plex Mono",monospace; font-size:19px; font-weight:600;
  color:inherit; }}
.lab {{ display:inline-flex; align-items:center; gap:6px;
  font-family:"IBM Plex Mono",monospace; font-size:11px; padding:3px 8px;
  border-radius:4px; background:{SUNK}; border:1px solid {LINE}; color:{MUTED}; }}
.sw {{ width:9px; height:9px; border-radius:2px; flex:none; display:inline-block; }}
.row {{ display:grid; grid-template-columns:1fr auto; align-items:baseline; gap:16px;
  padding:11px 16px; border-bottom:1px solid {LINE}; }}
.row:last-child {{ border-bottom:0; }}
.row.margin {{ background:{SUNK}; }}
.row-k {{ color:{MUTED}; font-size:13px; }}
.row-k small {{ display:block; color:{FAINT}; font-size:11px; line-height:1.4;
  margin-top:1px; }}
.row-v {{ font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
  font-size:19px; font-weight:500; text-align:right; white-space:nowrap;
  color:inherit; }}
.row-v u {{ text-decoration:none; font-size:12px; color:{FAINT}; font-weight:400;
  margin-left:3px; }}
.row-v i {{ display:block; font-style:normal; font-size:11.5px; color:{FAINT};
  font-weight:400; margin-top:1px; }}
.row-v.none {{ color:{FAINT}; font-weight:400; font-size:15px; }}
</style>
"""


def chart_config(chart):
    """Transparent surface plus neutral chrome, so a chart needs no theme rerun."""
    return (
        chart.configure(
            background="transparent",
            padding={"left": 14, "right": 22, "top": 16, "bottom": 14},
        )
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=AXIS_INK, titleColor=AXIS_INK, gridColor=GRID,
            domainColor=GRID, tickColor=GRID,
        )
        .configure_legend(labelColor=AXIS_INK, titleColor=AXIS_INK)
    )


def feature_colors(n: int) -> list[str]:
    """Fixed slot order, never cycled."""
    return [CAT[i % len(CAT)] for i in range(n)]
