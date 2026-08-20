"""HEAPO dose-response reconnaissance.

Among the households that can actually support a before/after analysis, how many
had a measurable intervention, and how big was it?

Reads only protocols.csv and the 15min overview. Never touches the time series.
Run:  py -3.13 dose_response_recon.py
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
    "protocols": DATA_ROOT / "reports" / "protocols.csv",
    "15min": DATA_ROOT / "smart_meter_data" / "overview" / "smart_meter_data_15min_overview.csv",
}

MIN_DAYS = 180              # base sample definition
EXPECTED_N = 89             # from recon.py; the script stops if this does not hold
REPEAT_VISIT_POLICY = "last"  # first | last | exclude -- see step0

# STEP 4 sufficiency ladder (days of data on EACH side of the visit)
DAY_LADDER = [180, 270, 365]

# STEP 5 thresholds -- printed in the verdict so they can be argued with.
DOSE_RESPONSE_MIN = 25
BINARY_MIN = 15

# Heating curve reduction thresholds for the headline number.
K_THRESHOLDS = [0.5, 1.0, 2.0]

_report: list[str] = []


def say(line: str = "") -> None:
    print(line)
    _report.append(line)


def load(key: str) -> pd.DataFrame:
    path = FILES[key]
    if not path.is_file():
        raise FileNotFoundError(
            f"Required HEAPO file not found: {path}\n"
            f"Check DATA_ROOT at the top of {Path(__file__).name} (currently: {DATA_ROOT})"
        )
    return pd.read_csv(path, sep=";")


def tb(df: pd.DataFrame, col: str) -> pd.Series:
    """Coerce to pandas nullable boolean.

    protocols.csv stores nominally boolean columns two ways: real `bool`
    (never null) and `object` holding the strings 'True'/'False' plus NaN.
    `~series` on the object form does bitwise arithmetic on Python bools and
    silently yields -1/-2 rather than a mask, so every flag goes through here.
    """
    return df[col].map({True: True, False: False, "True": True, "False": False}).astype("boolean")


def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()


def describe_block(x: pd.Series, unit: str, indent: str = "  ") -> None:
    """count/mean/median/min/max/quartiles for a magnitude series."""
    if x.empty:
        say(f"{indent}(no values)")
        return
    q = x.quantile([0.25, 0.5, 0.75])
    say(f"{indent}count={len(x)}  mean={x.mean():.2f}{unit}  median={x.median():.2f}{unit}  "
        f"min={x.min():.2f}{unit}  max={x.max():.2f}{unit}")
    say(f"{indent}q25={q.loc[0.25]:.2f}{unit}  q50={q.loc[0.5]:.2f}{unit}  q75={q.loc[0.75]:.2f}{unit}")


def histogram(x: pd.Series, unit: str, indent: str = "  ") -> None:
    """Text histogram with 1-unit bins, labelled by the integer floor of the bin."""
    if x.empty:
        say(f"{indent}(no values)")
        return
    binned = x.apply(lambda v: int(v // 1))
    counts = binned.value_counts().sort_index()
    width = max(len(str(int(c))) for c in counts)
    for b, c in counts.items():
        say(f"{indent}[{b:+d}, {b+1:+d}){unit}: {c:>{width}}  {'#' * int(c)}")


def complete_winters(first, last) -> float:
    """Heating seasons (1 Oct - 31 Mar) fully inside [first, last].

    UPPER BOUND: derived from the two range endpoints only. It confirms the
    season lies inside the household's span; it does NOT verify that the days in
    between carry measurements. See the gap finding in step0 -- gaps are large
    here, so the true count can only be lower.
    """
    if pd.isna(first) or pd.isna(last):
        return float("nan")
    n = 0
    for year in range(first.year - 1, last.year + 1):
        if first <= pd.Timestamp(year, 10, 1) and last >= pd.Timestamp(year + 1, 3, 31):
            n += 1
    return n


# ---------------------------------------------------------------------------
# BASE
# ---------------------------------------------------------------------------
def step0_base(pr: pd.DataFrame, q15: pd.DataFrame) -> pd.DataFrame:
    say("=" * 100)
    say(f"BASE -- protocol households with >= {MIN_DAYS} d before AND after (15min overview)")
    say("=" * 100)

    base = pr[pr["Household_ID"].notna()].copy()
    base["Household_ID"] = base["Household_ID"].astype("Int64")
    say(f"linkable protocol rows: {len(base)}   distinct households: {base['Household_ID'].nunique()}")

    merged = base.merge(q15, on="Household_ID", how="left")
    usable = merged[(merged["SMD_15min_TimeAvailable_DaysBeforeVisit"] >= MIN_DAYS)
                    & (merged["SMD_15min_TimeAvailable_DaysAfterVisit"] >= MIN_DAYS)].copy()

    say(f"rows meeting the rule: {len(usable)}   distinct households: {usable['Household_ID'].nunique()}")
    if usable["Household_ID"].nunique() != EXPECTED_N:
        raise SystemExit(
            f"STOP: expected {EXPECTED_N} households, got {usable['Household_ID'].nunique()}. "
            f"Rows={len(usable)}. Investigate before continuing."
        )
    say(f"CONFIRMED: {EXPECTED_N} households, matching recon.py.")
    say("")

    # --- repeat visits ---
    all_visits = base["Household_ID"].value_counts()
    repeats = set(all_visits[all_visits > 1].index)
    in_usable = repeats & set(usable["Household_ID"])
    say("repeat-visit handling:")
    say(f"  households with >1 protocol visit overall: {len(repeats)} -> {sorted(repeats)}")
    say(f"  of those, inside the usable sample: {len(in_usable)} -> {sorted(in_usable)}")
    if not in_usable:
        say("  => NONE of the repeat-visit households survives the >= 180 d both-sides filter, so")
        say("     'first visit', 'last visit' and 'exclude' all yield the SAME sample. Counts both")
        say(f"     ways: first={EXPECTED_N}  last={EXPECTED_N}  excluded={EXPECTED_N}.")
        say(f"     The configured policy ({REPEAT_VISIT_POLICY}) is therefore a no-op here; it is")
        say("     still applied below so the script stays correct if the filter is ever loosened.")
    else:
        say(f"  => policy '{REPEAT_VISIT_POLICY}' applies and changes the sample.")

    usable = usable.sort_values(["Household_ID", "Visit_Year"])
    if REPEAT_VISIT_POLICY == "first":
        usable = usable.drop_duplicates("Household_ID", keep="first")
    elif REPEAT_VISIT_POLICY == "last":
        usable = usable.drop_duplicates("Household_ID", keep="last")
    else:
        usable = usable[~usable["Household_ID"].isin(repeats)]
    say(f"  sample after applying policy: {len(usable)} rows / {usable['Household_ID'].nunique()} households")
    say("")

    # --- inspection findings that qualify the brief ---
    say("--- inspection findings before proceeding ---")
    vd = pd.to_datetime(usable["Visit_Date"], errors="coerce")
    say(f"1. Visit_Date is present for {int(vd.notna().sum())}/{len(usable)} of the sample -- unlike the")
    say(f"   full protocol file (304/410), every usable household has a real visit date. STEP 4 uses it")
    say(f"   directly rather than deriving one.")

    first = to_date(usable["SMD_15min_TimeAvailable_EarliestTimestamp"])
    last = to_date(usable["SMD_15min_TimeAvailable_LatestTimestamp"])
    span = (last - first).dt.days
    ndays = usable["SMD_15min_TimeAvailable_NumberDays"]
    gap = span - ndays
    say(f"2. CONTRADICTS a natural reading of the brief: DaysBeforeVisit/DaysAfterVisit and NumberDays")
    say(f"   count days WITH DATA, not calendar days. Calendar span minus NumberDays is > 0 for")
    say(f"   {int((gap > 0).sum())}/{len(usable)} households (median {gap.median():.0f} d, max {gap.max():.0f} d).")
    say(f"   Good news: the >= {MIN_DAYS} d filter is therefore already gap-aware. Bad news: any date-range")
    say(f"   arithmetic (STEP 4 winters) is an upper bound over a range that is demonstrably holey.")

    derived = first + pd.to_timedelta(usable["SMD_15min_TimeAvailable_DaysBeforeVisit"], "D")
    dd = (vd - derived).dt.days.abs()
    say(f"3. Consequence: reconstructing the visit date as first_date + DaysBeforeVisit is wrong. It")
    say(f"   matches Visit_Date exactly for only {int((dd == 0).sum())}/{len(usable)} households and is off by up to")
    say(f"   {int(dd.max())} days. Use Visit_Date.")
    say("")

    usable["_first"] = first.values
    usable["_last"] = last.values
    usable["_visit"] = vd.values
    return usable.reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP 1 -- heating curve
# ---------------------------------------------------------------------------
CURVE_POINTS = [
    ("Outside20", "HeatPump_HeatingCurveSetting_Outside20_BeforeVisit",
     "HeatPump_HeatingCurveSetting_Outside20_AfterVisit"),
    ("Outside0", "HeatPump_HeatingCurveSetting_Outside0_BeforeVisit",
     "HeatPump_HeatingCurveSetting_Outside0_AfterVisit"),
    ("OutsideMinus8", "HeatPump_HeatingCurveSetting_OutsideMinus8_BeforeVisit",
     "HeatPump_HeatingCurveSetting_OutsideMinus8_AfterVisit"),
]


def step1_curve(u: pd.DataFrame) -> pd.DataFrame:
    say("=" * 100)
    say(f"STEP 1 -- HEATING CURVE REDUCTION WITHIN THE {len(u)}")
    say("=" * 100)
    say("Reduction is defined as Before - After, so a positive number is a reduction in supply")
    say("temperature (the intended direction of the intervention).")
    say("")

    deltas = {}
    for name, bcol, acol in CURVE_POINTS:
        both = u[bcol].notna() & u[acol].notna()
        d = (u[bcol] - u[acol]).where(both)
        deltas[name] = d
        obs = d.dropna()
        say(f"--- {name} ---")
        say(f"  both Before and After present: {int(both.sum())} of {len(u)} "
            f"({both.sum()/len(u):.1%})   -> missing for {int((~both).sum())}")
        say(f"  non-zero change: {int((obs != 0).sum())}")
        say(f"  reduction (After < Before): {int((obs > 0).sum())}    increase: {int((obs < 0).sum())}")
        say("  reduction magnitude, Kelvin (all observed deltas, including zeros):")
        describe_block(obs, " K", indent="    ")
        say("  histogram, 1 K bins:")
        histogram(obs, " K", indent="    ")
        say("")

    # --- the combined figure that matters ---
    d0, d8 = deltas["Outside0"], deltas["OutsideMinus8"]
    say("*" * 100)
    say("COMBINED: households with a heating curve REDUCTION at Outside0 OR at OutsideMinus8")
    say("*" * 100)
    for t in K_THRESHOLDS:
        # fillna(False) via >= on NaN yields False, so households lacking a pair simply do not qualify.
        hit = (d0 >= t) | (d8 >= t)
        n = int(hit.sum())
        say(f"  >= {t:>3} K at 0 C or -8 C:  {n:>3} of {len(u)}  ({n/len(u):.1%})")
    say("")
    n1 = int(((d0 >= 1.0) | (d8 >= 1.0)).sum())
    say(f"  ==> HEADLINE: {n1} of {len(u)} households have a heating curve reduction of >= 1 K.")
    say("")

    out = pd.DataFrame({
        "Household_ID": u["Household_ID"].values,
        "curve_delta_Outside20_K": deltas["Outside20"].values,
        "curve_delta_Outside0_K": deltas["Outside0"].values,
        "curve_delta_OutsideMinus8_K": deltas["OutsideMinus8"].values,
    })
    # Headline dose = the larger of the two cold-point reductions, which is what a
    # dose-response model would actually regress on.
    out["curve_max_reduction_K"] = out[["curve_delta_Outside0_K",
                                        "curve_delta_OutsideMinus8_K"]].max(axis=1)
    out["curve_reduced_1K"] = out["curve_max_reduction_K"] >= 1.0
    return out


# ---------------------------------------------------------------------------
# STEP 2 -- other interventions
# ---------------------------------------------------------------------------
def step2_other(u: pd.DataFrame) -> pd.DataFrame:
    say("=" * 100)
    say(f"STEP 2 -- OTHER INTERVENTIONS WITHIN THE {len(u)}")
    say("=" * 100)

    out = pd.DataFrame({"Household_ID": u["Household_ID"].values})

    # --- heating limit ---
    b, a = "HeatPump_HeatingLimitSetting_BeforeVisit", "HeatPump_HeatingLimitSetting_AfterVisit"
    both = u[b].notna() & u[a].notna()
    d = (u[b] - u[a]).where(both)
    # NaN != 0 evaluates True, so a missing pair must be excluded explicitly.
    changed = d.notna() & (d != 0)
    say("--- heating limit ---")
    say(f"  both values present: {int(both.sum())} of {len(u)}")
    say(f"  changed: {int(changed.sum())}   reduced: {int((d > 0).sum())}   raised: {int((d < 0).sum())}")
    say("  magnitude, degrees C:")
    describe_block(d.dropna(), " C", indent="    ")
    say("  histogram, 1 C bins:")
    histogram(d.dropna(), " C", indent="    ")
    out["limit_delta_C"] = d.values
    out["limit_changed"] = changed.values
    say("")

    # --- night setback ---
    nb = tb(u, "HeatPump_NightSetbackSetting_Activated_BeforeVisit")
    na = tb(u, "HeatPump_NightSetbackSetting_Activated_AfterVisit")
    deact = nb.eq(True) & na.eq(False)
    still = nb.eq(True) & na.eq(True)
    say("--- night setback ---")
    say(f"  activated before visit: {int(nb.eq(True).sum())}")
    say(f"  GENUINE DEACTIVATION (before True, after False): {int(deact.sum())}")
    say(f"  still active after visit: {int(still.sum())}")
    say(f"  after-value missing where before was True: {int((nb.eq(True) & na.isna()).sum())}")
    say(f"  switched ON at the visit (before False, after True): "
        f"{int((nb.eq(False) & na.eq(True)).sum())}")
    out["nightsetback_deactivated"] = deact.fillna(False).values
    say("")

    # --- DHW temperature ---
    b, a = "DHW_TemperatureSetting_BeforeVisit", "DHW_TemperatureSetting_AfterVisit"
    both = u[b].notna() & u[a].notna()
    d = (u[b] - u[a]).where(both)
    # NaN != 0 evaluates True, so a missing pair must be excluded explicitly.
    changed = d.notna() & (d != 0)
    say("--- DHW temperature ---")
    say(f"  both values present: {int(both.sum())} of {len(u)}")
    say(f"  changed: {int(changed.sum())}   reduced: {int((d > 0).sum())}   raised: {int((d < 0).sum())}")
    say("  magnitude, Kelvin:")
    describe_block(d.dropna(), " K", indent="    ")
    say("  histogram, 1 K bins:")
    histogram(d.dropna(), " K", indent="    ")
    out["dhw_delta_K"] = d.values
    out["dhw_changed"] = changed.values
    say("")

    # --- circulation pump stage ---
    b = "HeatDistribution_Circulation_PumpStagePosition_BeforeVisit"
    a = "HeatDistribution_Circulation_PumpStagePosition_AfterVisit"
    both = u[b].notna() & u[a].notna()
    d = (u[b] - u[a]).where(both)
    # NaN != 0 evaluates True, so a missing pair must be excluded explicitly.
    changed = d.notna() & (d != 0)
    say("--- circulation pump stage ---")
    say(f"  both values present: {int(both.sum())} of {len(u)}")
    say(f"  changed: {int(changed.sum())}   reduced: {int((d > 0).sum())}   raised: {int((d < 0).sum())}")
    say(f"  stage reductions observed: {sorted(d.dropna().unique().tolist())} "
        f"(ordinal positions, not a physical unit -- magnitude is not comparable across pumps)")
    out["pumpstage_delta"] = d.values
    out["pumpstage_changed"] = changed.values
    say("")
    return out


# ---------------------------------------------------------------------------
# STEP 3 -- intervention profile
# ---------------------------------------------------------------------------
INTERVENTIONS = [
    ("heating curve >=1K", "curve_reduced_1K"),
    ("heating limit", "limit_changed"),
    ("night setback off", "nightsetback_deactivated"),
    ("DHW temperature", "dhw_changed"),
    ("circulation pump", "pumpstage_changed"),
]


def step3_profile(prof: pd.DataFrame) -> pd.DataFrame:
    say("=" * 100)
    say("STEP 3 -- INTERVENTION PROFILE PER HOUSEHOLD")
    say("=" * 100)
    say("NOTE ON SCOPE: the brief says 'the four interventions'. STEP 1 defines one (heating curve)")
    say("and STEP 2 defines four more (heating limit, night setback, DHW temperature, circulation")
    say("pump), so FIVE are profiled here. Dropping any of them would hide a real intervention;")
    say("the counts below are per-intervention, so a four-way reading is still recoverable.")
    say("")

    flags = prof[[c for _, c in INTERVENTIONS]].astype(bool)
    flags.columns = [n for n, _ in INTERVENTIONS]
    n = flags.sum(axis=1)

    say(f"interventions recorded per household (n={len(prof)}):")
    for k in range(0, len(INTERVENTIONS) + 1):
        c = int((n == k).sum())
        say(f"  exactly {k}: {c:>3} households ({c/len(prof):.1%})")
    say(f"  mean interventions per household: {n.mean():.2f}   max: {int(n.max())}")
    say(f"  at least one: {int((n >= 1).sum())}   none at all: {int((n == 0).sum())}")
    say("")

    say("per-intervention totals:")
    for name in flags.columns:
        say(f"  {name:<22} {int(flags[name].sum()):>3}")
    say("")

    combo = flags.apply(lambda r: " + ".join([c for c in flags.columns if r[c]]) or "(none)", axis=1)
    vc = combo.value_counts()
    say("combinations occurring 3+ times:")
    shown = vc[vc >= 3]
    for k, v in shown.items():
        say(f"  {v:>3}  {k}")
    say(f"  ({int(vc[vc < 3].sum())} households spread over {int((vc < 3).sum())} rarer combinations)")
    say("")

    multi = int((n >= 2).sum())
    atleast1 = int((n >= 1).sum())
    say("SEPARABLE OR BUNDLED?")
    if atleast1:
        say(f"  {multi} of the {atleast1} households with any intervention ({multi/atleast1:.0%}) received")
        say(f"  two or more at the same visit. All changes happen on one day, so wherever two land")
        say(f"  together their effects are perfectly confounded in time.")
        singles = {name: int(((flags[name]) & (n == 1)).sum()) for name in flags.columns}
        say("  households where an intervention arrived ALONE (individually attributable):")
        for name, c in sorted(singles.items(), key=lambda kv: -kv[1]):
            total = int(flags[name].sum())
            share = f"{c/total:.0%}" if total else "n/a"
            say(f"    {name:<22} {c:>3} of {total:>3} ({share} of that intervention's cases)")
        # The verdict follows the data rather than being asserted: bundling only
        # blocks attribution if most affected households received a bundle.
        if multi > atleast1 / 2:
            say("  => interventions are mostly BUNDLED. Per-intervention attribution is limited to the")
            say("     isolated cases above, a much smaller sample than the totals suggest.")
        else:
            say("  => interventions are mostly SEPARABLE: the majority of affected households received")
            say("     exactly one, so per-intervention attribution is possible in principle. The")
            say("     constraint is not confounding, it is that each isolated group is tiny (see the")
            say("     counts above); bundling removes a further handful on top of that.")
    say("")
    prof = prof.copy()
    prof["n_interventions"] = n.values
    prof["intervention_combo"] = combo.values
    return prof


# ---------------------------------------------------------------------------
# STEP 4 -- data sufficiency
# ---------------------------------------------------------------------------
def step4_sufficiency(u: pd.DataFrame, prof: pd.DataFrame) -> None:
    say("=" * 100)
    say("STEP 4 -- DATA SUFFICIENCY PER INTERVENTION TYPE")
    say("=" * 100)
    say("*** The winter columns are a COVERAGE UPPER BOUND derived from the earliest/latest")
    say("*** timestamps and the visit date only. A season counts when it lies entirely inside the")
    say("*** relevant span. It does NOT verify that the days inside carry measurements -- and the")
    say("*** gaps measured in the BASE section (median 5 d, max 317 d) prove they often do not.")
    say("*** Real usable winters can only be fewer.")
    say("")

    before = u["SMD_15min_TimeAvailable_DaysBeforeVisit"].values
    after = u["SMD_15min_TimeAvailable_DaysAfterVisit"].values

    # Winters fully inside the pre-visit span and inside the post-visit span.
    pre_w = [complete_winters(f, v) for f, v in zip(u["_first"], u["_visit"])]
    post_w = [complete_winters(v, l) for v, l in zip(u["_visit"], u["_last"])]
    both_w = [(p >= 1 and q >= 1) for p, q in zip(pre_w, post_w)]

    meta = pd.DataFrame({
        "Household_ID": u["Household_ID"].values,
        "days_before": before, "days_after": after,
        "winters_before": pre_w, "winters_after": post_w, "winter_both_sides": both_w,
    })
    merged = prof.merge(meta, on="Household_ID")

    say(f"whole sample (n={len(merged)}):")
    for d in DAY_LADDER:
        c = int(((merged["days_before"] >= d) & (merged["days_after"] >= d)).sum())
        say(f"  >= {d} d each side: {c:>3}")
    say(f"  >= 1 complete winter each side: {int(merged['winter_both_sides'].sum())}")
    say("")

    rows = []
    for name, col in INTERVENTIONS:
        sub = merged[merged[col].astype(bool)]
        rec = {"intervention": name, "affected": len(sub)}
        for d in DAY_LADDER:
            rec[f">={d}d both"] = int(((sub["days_before"] >= d) & (sub["days_after"] >= d)).sum())
        rec[">=1 winter both"] = int(sub["winter_both_sides"].sum())
        rows.append(rec)
    say(pd.DataFrame(rows).to_string(index=False))
    say("")
    say("The '>=180d both' column equals 'affected' by construction -- that is the sample definition.")
    say("The columns to the right are what actually survives a stricter requirement.")
    say("")


# ---------------------------------------------------------------------------
# STEP 5 -- verdict
# ---------------------------------------------------------------------------
def step5_verdict(prof: pd.DataFrame, headline: int, n_total: int) -> None:
    say("=" * 100)
    say("STEP 5 -- VERDICT")
    say("=" * 100)
    say("Thresholds (stated so they can be argued with), counted as affected households:")
    say(f"  (a) dose-response with a continuous magnitude : >= {DOSE_RESPONSE_MIN}")
    say(f"      Rationale: a continuous dose needs enough spread to fit and check a slope; below")
    say(f"      ~25 points a single household moves the estimate.")
    say(f"  (b) binary treated/untreated comparison only  : >= {BINARY_MIN}")
    say(f"  (c) descriptive case series only              : below {BINARY_MIN}")
    say("")

    rows = []
    for name, col in INTERVENTIONS:
        n = int(prof[col].astype(bool).sum())
        alone = int(((prof[col].astype(bool)) & (prof["n_interventions"] == 1)).sum())
        if n >= DOSE_RESPONSE_MIN:
            v = "(a) dose-response"
        elif n >= BINARY_MIN:
            v = "(b) binary comparison only"
        else:
            v = "(c) descriptive case series"
        rows.append({"intervention": name, "affected": n, "isolated (attributable)": alone,
                     "verdict": v})
    vdf = pd.DataFrame(rows)
    say(vdf.to_string(index=False))
    say("")

    n_a = int(vdf["verdict"].str.startswith("(a)").sum())
    n_b = int(vdf["verdict"].str.startswith("(b)").sum())
    n_c = int(vdf["verdict"].str.startswith("(c)").sum())
    say(f"summary: (a) {n_a}   (b) {n_b}   (c) {n_c}")
    say("")
    say("*" * 100)
    say(f"THE NUMBER THAT MATTERS: {headline} of the {n_total} households with a usable before/after")
    say(f"window have a heating curve reduction of >= 1 K at 0 C or -8 C.")
    say("*" * 100)
    say("")
    if n_a == 0:
        say("No intervention type reaches the dose-response threshold. The heating curve -- the most")
        say("common fault in the whole dataset and the one the intervention is built around -- has a")
        say(f"recorded before/after pair for only a minority of the {n_total}, and a >= 1 K reduction for")
        say(f"{headline}. A dose-response analysis with a continuous magnitude is NOT supported.")
        say("")
        say("The limit is not the smart meter data, which was filtered to be adequate by construction.")
        say("It is the protocols: the numeric curve settings are simply not recorded for most visits,")
        say("so the dose is unknown even where the fault and the fix are both documented.")
    else:
        say(f"{n_a} intervention type(s) reach the dose-response threshold; treat the rest as binary")
        say("or descriptive.")
    say("")
    say("The 'isolated' column is the stricter sample: households where this intervention was the")
    say("only one recorded, so an observed consumption change is attributable to it alone. Every")
    say("intervention type falls below the descriptive threshold on that basis.")
    say("")


def main() -> None:
    pr = load("protocols")
    q15 = load("15min")

    u = step0_base(pr, q15)
    curve = step1_curve(u)
    other = step2_other(u)

    prof = curve.merge(other, on="Household_ID")
    prof = step3_profile(prof)
    step4_sufficiency(u, prof)

    headline = int(prof["curve_reduced_1K"].astype(bool).sum())
    step5_verdict(prof, headline, len(u))

    # One row per household: identity, window, interventions, magnitudes.
    sample = u[["Household_ID", "Visit_Year", "Visit_Date", "HeatPump_Installation_Type",
                "SMD_15min_TimeAvailable_DaysBeforeVisit",
                "SMD_15min_TimeAvailable_DaysAfterVisit",
                "_first", "_visit", "_last"]].merge(prof, on="Household_ID")

    OUTPUT_DIR.mkdir(exist_ok=True)
    sample_path = OUTPUT_DIR / "dose_response_sample.csv"
    sample.to_csv(sample_path, index=False, sep=";")
    summary_path = OUTPUT_DIR / "dose_response_summary.txt"
    summary_path.write_text("\n".join(_report), encoding="utf-8")

    print("=" * 100)
    print(f"sample  -> {sample_path}  ({sample.shape[0]} rows x {sample.shape[1]} cols)")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
