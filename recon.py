"""HEAPO reconnaissance: data availability from metadata/overview files only.

Never touches the per-household time series under smart_meter_data/{15min,daily,monthly}/.
Run:  python recon.py
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration -- the only place a path is hardcoded.
# ---------------------------------------------------------------------------
DATA_ROOT = Path(r"C:\Users\migue\OneDrive\Escritorio\SF AI forecast fauls\heapo_data")

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

FILES = {
    "households": DATA_ROOT / "meta_data" / "households.csv",
    "daily": DATA_ROOT / "smart_meter_data" / "overview" / "smart_meter_data_daily_overview.csv",
    "15min": DATA_ROOT / "smart_meter_data" / "overview" / "smart_meter_data_15min_overview.csv",
    "protocols": DATA_ROOT / "reports" / "protocols.csv",
}

HISTORY_THRESHOLDS = [365, 730]
BALANCE_THRESHOLDS = [90, 180, 365]

# Buffer for the plain-text report; every reported line goes through say().
_report: list[str] = []


def say(line: str = "") -> None:
    print(line)
    _report.append(line)


def load(key: str) -> pd.DataFrame:
    """Read one semicolon-separated HEAPO CSV, failing loudly with the full path."""
    path = FILES[key]
    if not path.is_file():
        raise FileNotFoundError(
            f"Required HEAPO file not found: {path}\n"
            f"Check DATA_ROOT at the top of {Path(__file__).name} (currently: {DATA_ROOT})"
        )
    # European CSV: ';' separator. Decimals are already '.', so no `decimal=` needed.
    return pd.read_csv(path, sep=";")


def to_date(series: pd.Series) -> pd.Series:
    """ISO timestamps with a +00:00 offset -> tz-naive calendar dates (UTC day).

    Daily rows are stamped 23:59:59 of the day they describe and 15min rows
    00:00:00 / 23:45:00, so taking the date component is correct for both.
    """
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()


# ---------------------------------------------------------------------------
# STEP 1 -- inspection
# ---------------------------------------------------------------------------
def inspect(frames: dict[str, pd.DataFrame]) -> None:
    say("=" * 78)
    say("STEP 1 -- STRUCTURE INSPECTION")
    say("=" * 78)

    for key, df in frames.items():
        say("")
        say(f"--- {key}: {FILES[key].name}  shape={df.shape} ---")

        if key == "protocols":
            # 106 columns; only dump the handful this script actually reasons about.
            cols = ["Report_ID", "Household_ID", "Visit_Year", "Visit_Date"]
            say("dtypes (selected):")
            say(df[cols].dtypes.to_string())
            say(f"unique Household_ID: {df['Household_ID'].nunique()}   "
                f"rows with missing Household_ID: {int(df['Household_ID'].isna().sum())}")
            say(f"Visit_Year range: {df['Visit_Year'].min()}-{df['Visit_Year'].max()}")
            continue

        say("dtypes:")
        say(df.dtypes.to_string())
        say("head(3):")
        say(df.head(3).to_string())

        say("low-cardinality columns (<= 10 distinct non-null values):")
        for col in df.columns:
            uniques = df[col].dropna().unique()
            if len(uniques) <= 10:
                vals = sorted(str(v) for v in uniques)
                say(f"  {col}: {vals}  (nulls={int(df[col].isna().sum())})")

    # --- targeted verifications the four questions depend on ---
    say("")
    say("--- targeted checks ---")

    hh = frames["households"]
    say(f"Group encoding: {dict(hh['Group'].value_counts())}"
        f"  -> exact strings 'treatment' / 'control', no nulls")

    for key in ("daily", "15min"):
        df = frames[key]
        for col in [c for c in df.columns if "Timestamp" in c]:
            parsed = pd.to_datetime(df[col], utc=True, errors="coerce")
            say(f"{key}: {col} parses {parsed.notna().sum()}/{len(df)} "
                f"[{parsed.min()} .. {parsed.max()}]")
        meas = [c for c in df.columns if "MeasurementsAvailable" in c]
        say(f"{key}: MeasurementsAvailable_* dtypes -> "
            f"{sorted({str(df[c].dtype) for c in meas})} "
            f"=> BOOLEAN FLAGS (channel present yes/no), NOT counts")
        say(f"{key}: duplicate Household_ID rows = {int(df['Household_ID'].duplicated().sum())}")

    # The daily file has a naming inconsistency worth knowing before writing any
    # column lookup by hand.
    odd = [c for c in frames["daily"].columns if c.startswith("SMDdaily_")]
    if odd:
        say(f"NOTE: daily overview naming inconsistency -- {len(odd)} columns are prefixed "
            f"'SMDdaily_' (missing underscore) instead of 'SMD_daily_': {odd}")

    say("")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def complete_winters(first, last) -> float:
    """Heating seasons (1 Oct - 31 Mar) fully inside [first, last].

    UPPER BOUND ON COVERAGE: derived from the date range endpoints only. A season
    is counted when it falls entirely inside the household's span -- this does NOT
    verify that the days in between actually carry measurements. Gaps, dropouts
    and missing days inside the range are invisible here and would only show up
    by reading the time series themselves.
    """
    if pd.isna(first) or pd.isna(last):
        return float("nan")
    n = 0
    # A season is labelled by the year it starts in; test every candidate year.
    for year in range(first.year - 1, last.year + 1):
        start = pd.Timestamp(year=year, month=10, day=1)
        end = pd.Timestamp(year=year + 1, month=3, day=31)
        if first <= start and last >= end:
            n += 1
    return n


def build_inventory(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    hh, daily, q15 = frames["households"], frames["daily"], frames["15min"]

    # Left joins keep all 1408 households; a household absent from an overview file
    # gets NaN there, which is itself the answer to "is this data available?".
    inv = hh.merge(daily, on="Household_ID", how="left").merge(q15, on="Household_ID", how="left")

    for prefix in ("SMD_daily", "SMD_15min"):
        first = to_date(inv[f"{prefix}_TimeAvailable_EarliestTimestamp"])
        last = to_date(inv[f"{prefix}_TimeAvailable_LatestTimestamp"])
        inv[f"{prefix}_FirstDate"] = first
        inv[f"{prefix}_LastDate"] = last
        inv[f"{prefix}_CompleteWinters"] = [complete_winters(a, b) for a, b in zip(first, last)]

    return inv


# ---------------------------------------------------------------------------
# STEP 2 -- the four questions
# ---------------------------------------------------------------------------
def q1_submetering(inv: pd.DataFrame) -> None:
    say("=" * 78)
    say("Q1 -- SUB-METERING (separate heat pump meter)")
    say("=" * 78)
    say("MeasurementsAvailable_HeatPump is a boolean flag: True = a dedicated heat")
    say("pump channel exists for that household. NaN = household absent from the file.")
    say("")

    for label, flag_col, present_col in (
        ("daily", "SMD_daily_MeasurementsAvailable_HeatPump", "SMD_daily_TimeAvailable_NumberDays"),
        ("15min", "SMD_15min_MeasurementsAvailable_HeatPump", "SMD_15min_TimeAvailable_NumberDays"),
    ):
        covered = inv[present_col].notna()
        # fillna(False): a household missing from the file has no sub-meter there.
        flag = inv[flag_col].fillna(False).astype(bool)
        say(f"[{label} overview]  households present in file: {int(covered.sum())} / {len(inv)}")
        say(f"  with heat pump sub-meter: {int(flag.sum())}")
        say(pd.crosstab(inv["Group"], flag,
                        rownames=["Group"], colnames=["HasHeatPumpMeter"]).to_string())
        say("")

    # Agreement is only meaningful where both files describe the same household.
    both = inv[inv["SMD_daily_TimeAvailable_NumberDays"].notna()
               & inv["SMD_15min_TimeAvailable_NumberDays"].notna()]
    d = both["SMD_daily_MeasurementsAvailable_HeatPump"].astype(bool)
    q = both["SMD_15min_MeasurementsAvailable_HeatPump"].astype(bool)
    say(f"Agreement on the {len(both)} households present in BOTH files:")
    say(pd.crosstab(d, q, rownames=["daily"], colnames=["15min"]).to_string())
    say(f"  disagreements: {int((d != q).sum())}")

    only15 = inv[inv["SMD_daily_TimeAvailable_NumberDays"].isna()
                 & inv["SMD_15min_MeasurementsAvailable_HeatPump"].fillna(False).astype(bool)]
    say(f"  sub-metered in 15min but ABSENT from the daily file: {len(only15)}")
    say("")


def q2_history(inv: pd.DataFrame) -> None:
    say("=" * 78)
    say("Q2 -- HISTORY LENGTH (SMD_daily_TimeAvailable_NumberDays)")
    say("=" * 78)
    days = inv["SMD_daily_TimeAvailable_NumberDays"].dropna()
    say(days.describe().to_string())
    say("")
    for t in HISTORY_THRESHOLDS:
        n = int((days >= t).sum())
        say(f"  >= {t:>4} days: {n:>5} households "
            f"({n / len(days):.1%} of the {len(days)} with daily data)")
    say("")
    say("15min overview, for comparison:")
    d15 = inv["SMD_15min_TimeAvailable_NumberDays"].dropna()
    say(f"  n={len(d15)}  mean={d15.mean():.1f}  median={d15.median():.0f}  "
        f"min={int(d15.min())}  max={int(d15.max())}")
    say("  " + "   ".join(f">={t}d: {int((d15 >= t).sum())}" for t in HISTORY_THRESHOLDS))
    say("")


def q3_winters(inv: pd.DataFrame) -> None:
    say("=" * 78)
    say("Q3 -- WINTER COVERAGE (complete heating seasons, 1 Oct - 31 Mar)")
    say("=" * 78)
    say("*** UPPER BOUND: counted from the earliest/latest timestamps only. A season")
    say("*** counts when it lies entirely inside the household's date range. This does")
    say("*** NOT account for gaps or missing days within that range -- the true number")
    say("*** of usable winters can only be lower, never higher.")
    say("")
    for label, col in (("daily", "SMD_daily_CompleteWinters"),
                       ("15min", "SMD_15min_CompleteWinters")):
        w = inv[col].dropna()
        say(f"[{label} overview]  n={len(w)}")
        for k, v in w.value_counts().sort_index().items():
            say(f"  {int(k)} complete winter(s): {v:>5} households")
        say(f"  mean={w.mean():.2f}  median={w.median():.0f}  max={int(w.max())}")
        say(f"  >= 1 complete winter : {int((w >= 1).sum())}")
        say(f"  >= 2 complete winters: {int((w >= 2).sum())}")
        say("")


def q4_balance(inv: pd.DataFrame, protocols: pd.DataFrame) -> None:
    say("=" * 78)
    say("Q4 -- BEFORE/AFTER BALANCE (households with protocol data)")
    say("=" * 78)
    # Protocols_Available is True for exactly the 'treatment' households. Control
    # households were never visited and carry DaysBefore/AfterVisit = 0 by
    # construction, so including them would silently deflate every count below.
    say(f"protocols.csv: {len(protocols)} visit rows, "
        f"{protocols['Household_ID'].nunique()} distinct households "
        f"({int(protocols['Household_ID'].isna().sum())} rows carry no Household_ID)")
    say(f"households with Protocols_Available=True: {int(inv['Protocols_Available'].sum())} "
        f"(exactly the Group=='treatment' set)")

    for label, prefix in (("daily", "SMD_daily"), ("15min", "SMD_15min")):
        sub = inv[inv["Protocols_Available"] & inv[f"{prefix}_TimeAvailable_NumberDays"].notna()]
        before = sub[f"{prefix}_TimeAvailable_DaysBeforeVisit"]
        after = sub[f"{prefix}_TimeAvailable_DaysAfterVisit"]
        say("")
        say(f"[{label} overview]  base: {len(sub)} protocol households with {label} data")
        rows = [
            {
                "threshold_days": t,
                "before>=t": int((before >= t).sum()),
                "after>=t": int((after >= t).sum()),
                "BOTH>=t": int(((before >= t) & (after >= t)).sum()),
                "pct_both": f"{((before >= t) & (after >= t)).mean():.1%}",
            }
            for t in BALANCE_THRESHOLDS
        ]
        say(pd.DataFrame(rows).to_string(index=False))
    say("")


def main() -> None:
    frames = {key: load(key) for key in FILES}
    inspect(frames)

    inv = build_inventory(frames)

    say("=" * 78)
    say("STEP 2 -- ANSWERS")
    say("=" * 78)
    say("")
    q1_submetering(inv)
    q2_history(inv)
    q3_winters(inv)
    q4_balance(inv, frames["protocols"])

    OUTPUT_DIR.mkdir(exist_ok=True)
    inv_path = OUTPUT_DIR / "household_inventory.csv"
    inv.to_csv(inv_path, index=False, sep=";")

    summary_path = OUTPUT_DIR / "recon_summary.txt"
    summary_path.write_text("\n".join(_report), encoding="utf-8")

    print("=" * 78)
    print(f"merged inventory -> {inv_path}  ({inv.shape[0]} rows x {inv.shape[1]} cols)")
    print(f"summary          -> {summary_path}")


if __name__ == "__main__":
    main()
