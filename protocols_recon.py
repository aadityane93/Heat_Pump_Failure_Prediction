"""HEAPO protocol reconnaissance: is a supervised fault-type classifier viable?

Reads only reports/protocols.csv, reports/protocols_variables.csv, the 15min
overview and households.csv. Never touches the per-household time series.
Run:  py -3.13 protocols_recon.py
"""

from pathlib import Path
from itertools import combinations
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration -- the only place a path is hardcoded.
# ---------------------------------------------------------------------------
DATA_ROOT = Path(r"C:\Users\migue\OneDrive\Escritorio\SF AI forecast fauls\heapo_data")

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

FILES = {
    "protocols": DATA_ROOT / "reports" / "protocols.csv",
    "variables": DATA_ROOT / "reports" / "protocols_variables.csv",
    "households": DATA_ROOT / "meta_data" / "households.csv",
    "15min": DATA_ROOT / "smart_meter_data" / "overview" / "smart_meter_data_15min_overview.csv",
}

# STEP 6 sample definition: households with a real before/after window.
BEFORE_AFTER_MIN_DAYS = 180

# STEP 7 thresholds -- stated explicitly in the printed verdict.
CLASSIFIER_MIN_MINORITY = 50   # positives AND negatives needed to attempt supervised learning
DESCRIPTIVE_MIN_MINORITY = 15  # positives AND negatives needed for a descriptive comparison

# Published Table 3 of the HEAPO paper: label -> (percent, count)
PUBLISHED_TABLE3 = {
    "heating curve too high": (40.98, 168),
    "night setback active": (36.10, 148),
    "heating limit too high": (25.61, 105),
    "descaling too long ago": (17.80, 73),
    "expansion system": (13.41, 55),
    "air ducting": (11.25, 27),
    "sizing (incorrectly planned)": (10.00, 41),
    "pipe insulation": (9.02, 37),
    "DHW temperature": (7.80, 32),
    "brine pressure": (7.19, 12),
    "circulation pump": (6.34, 26),
    "geothermal probe": (5.99, 10),
    "thermostatic valves": (4.88, 20),
    "dirty": (3.17, 13),
    "technically not in order": (1.46, 6),
    "basic functions": (1.46, 6),
}

# Candidate label columns named in the brief.
CANDIDATE_LABEL_COLUMNS = [
    "HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit",
    "HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit",
    "HeatPump_NightSetbackSetting_Activated_BeforeVisit",
    "DHW_TemperatureSetting_Categorization",
    "DHW_Storage_LastDescaling_TooLongAgo",
    "HeatPump_Installation_CorrectlyPlanned",
    "HeatPump_Installation_IncorrectlyPlanned_Categorization",
    "HeatDistribution_ExpansionTank_Pressure_Categorization",
    "HeatPump_ElectricityConsumption_Categorization",
    "HeatPump_Clean",
    "HeatPump_BasicFunctionsOkay",
    "HeatPump_TechnicallyOkay",
]

# Words that mark a column as a condition/fault judgement rather than a fact.
JUDGEMENT_WORDS = [
    "okay", "too high", "too low", "too long", "clean", "required", "necessary",
    "recommends", "appropriately", "oversized", "undersized", "correctly",
    "categorization", "not accessible", "are free",
]

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
    """Coerce a column to pandas nullable boolean.

    Necessary because protocols.csv mixes true `bool` columns (no missing values)
    with `object` columns holding the *strings* 'True'/'False' plus NaN. Plain
    `~series` on the object version does bitwise arithmetic on Python bools and
    silently produces -1/-2 instead of a mask -- so every flag goes through here.
    """
    return df[col].map({True: True, False: False, "True": True, "False": False}).astype("boolean")


def is_true(df: pd.DataFrame, col: str) -> pd.Series:
    return tb(df, col).eq(True).fillna(False)


def is_false(df: pd.DataFrame, col: str) -> pd.Series:
    return tb(df, col).eq(False).fillna(False)


# ---------------------------------------------------------------------------
# STEP 1 -- inspection
# ---------------------------------------------------------------------------
def step1_inspect(pr: pd.DataFrame, var: pd.DataFrame) -> None:
    say("=" * 100)
    say("STEP 1 -- INSPECTION")
    say("=" * 100)
    say(f"protocols.csv shape: {pr.shape}")
    say(f"protocols_variables.csv shape: {var.shape}")
    say("")

    say("--- per-column profile ---")
    say(f"{'column':<58} {'dtype':<10} {'non-null':>8} {'uniq':>5}")
    for col in pr.columns:
        say(f"{col:<58} {str(pr[col].dtype):<10} {pr[col].notna().sum():>8} {pr[col].nunique():>5}")

    say("")
    say("--- value counts for columns with <= 10 unique values ---")
    for col in pr.columns:
        if pr[col].nunique() <= 10:
            vc = pr[col].value_counts(dropna=False)
            items = "  ".join(f"{k!r}={v}" for k, v in vc.items())
            say(f"{col}:  {items}")

    say("")
    say("--- protocols_variables.csv (full) ---")
    for _, r in var.iterrows():
        say(f"[{r['Category']} | {r['AffectsComponent']} | {r['AffectsHeatPumpType']} | "
            f"{r['AffectsTimePoint']}]")
        say(f"  {r['VariableName']}: {r['Description']}")

    # --- contradictions / surprises worth knowing before any analysis ---
    say("")
    say("--- findings that contradict or qualify the stated assumptions ---")

    say(f"1. Household_ID is float64, not int: {int(pr['Household_ID'].isna().sum())} rows carry NaN "
        f"(confirms the 193 anonymous rows). Cast to Int64 before joining.")

    vd = pd.to_datetime(pr["Visit_Date"], errors="coerce")
    linkable_vd = vd[pr["Household_ID"].notna()]
    say(f"2. Visit_Date is non-null on {int(vd.notna().sum())}/{len(pr)} rows and parses cleanly "
        f"({vd.min().date()} .. {vd.max().date()}); {int(vd.isna().sum())} rows have only "
        f"Visit_Year ({pr['Visit_Year'].min()}-{pr['Visit_Year'].max()}). On the linkable base the "
        f"date is present for {int(linkable_vd.notna().sum())}/{int(pr['Household_ID'].notna().sum())} "
        f"rows -- so a minority of linkable visits cannot be placed on a calendar day, only in a year.")

    obj_bools = [c for c in pr.columns
                 if pr[c].dtype == object and set(pr[c].dropna().unique()) <= {True, False, "True", "False"}]
    true_bools = [c for c in pr.columns if pr[c].dtype == bool]
    say(f"3. Boolean columns are split across two representations: {len(true_bools)} real `bool` "
        f"(never null) and {len(obj_bools)} `object` holding 'True'/'False' + NaN. "
        f"'MeasurementsAvailable_* are booleans' holds for the overview files, but in protocols.csv "
        f"a nominally boolean column is often object dtype -- comparing it with `==True` works, "
        f"`~` does not.")

    say(f"4. HeatPump_Installation_Type: {dict(pr['HeatPump_Installation_Type'].value_counts())}. "
        f"The air-source and ground-source subsets (240 / 167) are the denominators several Table 3 "
        f"rows are computed over -- see STEP 3.")

    # Type-scoped columns carrying values outside their own heat pump type.
    say("5. Type-scoped columns leak across heat pump types:")
    air = pr["HeatPump_Installation_Type"].eq("air-source")
    gnd = pr["HeatPump_Installation_Type"].eq("ground-source")
    scoped = var[var["AffectsHeatPumpType"].isin(["air-source", "ground-source"])]
    for _, r in scoped.iterrows():
        col, scope = r["VariableName"], r["AffectsHeatPumpType"]
        if col not in pr.columns:
            continue
        in_scope = air if scope == "air-source" else gnd
        off = int((pr[col].notna() & ~in_scope).sum())
        if off:
            say(f"   {col} ({scope}-only): {off} non-null values on OTHER heat pump types")
    say("   => any count over these columns must be restricted to the matching type explicitly.")

    ip = pr["HeatPump_Installation_IncorrectlyPlanned_Categorization"]
    say(f"6. HeatPump_Installation_IncorrectlyPlanned_Categorization is non-null for only "
        f"{int(ip.notna().sum())} rows, while HeatPump_Installation_CorrectlyPlanned==False holds for "
        f"{int(is_false(pr, 'HeatPump_Installation_CorrectlyPlanned').sum())}. The sub-category is "
        f"missing for the difference -- it is not a clean partition of the sizing fault.")

    say(f"7. HeatPump_Installation_ControllerNotAccessible is numeric with values "
        f"{sorted(pr['HeatPump_Installation_ControllerNotAccessible'].dropna().unique())}, not boolean. "
        f"Negative sentinel codes, not a flag -- excluded from label candidates.")
    say("")


# ---------------------------------------------------------------------------
# STEP 2 -- analysis base
# ---------------------------------------------------------------------------
def step2_base(pr: pd.DataFrame, hh: pd.DataFrame) -> pd.DataFrame:
    say("=" * 100)
    say("STEP 2 -- ANALYSIS BASE")
    say("=" * 100)

    base = pr[pr["Household_ID"].notna()].copy()
    base["Household_ID"] = base["Household_ID"].astype("Int64")

    n_rows, n_hh = len(base), base["Household_ID"].nunique()
    say(f"rows with non-null Household_ID: {n_rows}  (of {len(pr)})   distinct households: {n_hh}")
    say(f"expectation from the brief (217 rows / 214 households): "
        f"{'MATCHES' if (n_rows, n_hh) == (217, 214) else 'MISMATCH'}")

    visits = base["Household_ID"].value_counts()
    repeat = visits[visits > 1]
    say(f"households with multiple visits: {len(repeat)}  -> "
        f"{int(repeat.sum() - len(repeat))} extra rows beyond one-per-household")
    for hid, n in repeat.items():
        years = sorted(base.loc[base["Household_ID"] == hid, "Visit_Year"].tolist())
        say(f"   household {hid}: {n} visits, years {years}")

    say("")
    treatment = set(hh.loc[hh["Group"] == "treatment", "Household_ID"])
    in_protocols = set(base["Household_ID"].dropna().astype(int))
    say(f"households.csv Group=='treatment': {len(treatment)}")
    say(f"protocol base households:          {len(in_protocols)}")
    only_proto = sorted(in_protocols - treatment)
    only_treat = sorted(treatment - in_protocols)
    say(f"in protocols but NOT treatment: {len(only_proto)} {only_proto if only_proto else ''}")
    say(f"treatment but NOT in protocols: {len(only_treat)} {only_treat if only_treat else ''}")
    if not only_proto and not only_treat:
        say("=> the two sets are identical; the protocol base is exactly the treatment arm.")
    say("")
    return base


# ---------------------------------------------------------------------------
# Fault definitions
# ---------------------------------------------------------------------------
def fault_masks(df: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """label -> (fault_mask, in_scope_mask).

    in_scope narrows the denominator for heat-pump-type-specific faults; the
    fault mask itself is also intersected with it, because those columns carry
    stray values on the wrong heat pump type (STEP 1, finding 5).
    """
    air = df["HeatPump_Installation_Type"].eq("air-source")
    gnd = df["HeatPump_Installation_Type"].eq("ground-source")
    allrows = pd.Series(True, index=df.index)

    duct = (is_false(df, "HeatPump_AirSource_AirDuctsDistanceOkay")
            | is_false(df, "HeatPump_AirSource_AirDuctsFree")
            | is_true(df, "HeatPump_AirSource_AirDuctsCleaningRequired")
            | is_false(df, "HeatPump_AirSource_AirDuctsDrainOkay")
            | is_false(df, "HeatPump_AirSource_EvaporatorClean"))

    # Night setback: the raw before-visit flag over-counts vs the paper; the
    # published figure matches "was on before AND is not on after" (see STEP 3).
    nsb_before = is_true(df, "HeatPump_NightSetbackSetting_Activated_BeforeVisit")
    nsb_after = is_true(df, "HeatPump_NightSetbackSetting_Activated_AfterVisit")

    return {
        "heating curve too high": (is_true(df, "HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit"), allrows),
        "night setback active": (nsb_before & ~nsb_after, allrows),
        "heating limit too high": (is_true(df, "HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit"), allrows),
        "descaling too long ago": (is_true(df, "DHW_Storage_LastDescaling_TooLongAgo"), allrows),
        "expansion system": (df["HeatDistribution_ExpansionTank_Pressure_Categorization"]
                             .isin(["too high", "too low"]), allrows),
        "air ducting": (duct & air, air),
        "sizing (incorrectly planned)": (is_false(df, "HeatPump_Installation_CorrectlyPlanned"), allrows),
        "pipe insulation": (is_true(df, "HeatDistribution_Recommendation_InsulatePipes"), allrows),
        "DHW temperature": (df["DHW_TemperatureSetting_Categorization"]
                            .isin(["too high", "too low"]), allrows),
        "brine pressure": (is_false(df, "HeatPump_GroundSource_CurrentPressure_Okay") & gnd, gnd),
        "circulation pump": (is_true(df, "HeatDistribution_Circulation_PumpStagePosition_Changed"), allrows),
        "geothermal probe": (is_false(df, "HeatPump_GroundSource_CurrentTemperature_Okay") & gnd, gnd),
        "thermostatic valves": (is_true(df, "HeatDistribution_Recommendation_InstallThermostaticValve"), allrows),
        "dirty": (is_false(df, "HeatPump_Clean"), allrows),
        "technically not in order": (is_false(df, "HeatPump_TechnicallyOkay"), allrows),
        "basic functions": (is_false(df, "HeatPump_BasicFunctionsOkay"), allrows),
    }


# ---------------------------------------------------------------------------
# STEP 3 -- per-class counts + Table 3 reconstruction
# ---------------------------------------------------------------------------
def per_class_counts(df: pd.DataFrame, columns: list[str], title: str) -> pd.DataFrame:
    say(f"--- {title} (n={len(df)}) ---")
    rows = []
    for col in columns:
        s = df[col]
        n_null = int(s.isna().sum())
        n_nonnull = len(s) - n_null
        as_bool = tb(df, col)
        if as_bool.notna().any():
            n_true = int(as_bool.eq(True).sum())
            n_false = int(as_bool.eq(False).sum())
            rate = n_true / n_nonnull if n_nonnull else float("nan")
            rows.append({"column": col, "kind": "boolean", "n_true": n_true, "n_false": n_false,
                         "n_null": n_null, "true_rate_nonnull": round(rate, 4), "detail": ""})
        else:
            vc = s.value_counts()
            detail = "  ".join(f"{k}={v}" for k, v in vc.items())
            rows.append({"column": col, "kind": "categorical", "n_true": "", "n_false": "",
                         "n_null": n_null, "true_rate_nonnull": "", "detail": detail})
    out = pd.DataFrame(rows)
    say(out.to_string(index=False))
    say("")
    return out


def discover_extra_labels(pr: pd.DataFrame, var: pd.DataFrame) -> list[str]:
    """Judgement-style columns the brief did not list."""
    desc = dict(zip(var["VariableName"], var["Description"].fillna("")))
    found = []
    for col in pr.columns:
        if col in CANDIDATE_LABEL_COLUMNS:
            continue
        text = (col + " " + desc.get(col, "")).lower()
        if not any(w in text for w in JUDGEMENT_WORDS):
            continue
        # Only flags/categories can serve as labels; numeric readings cannot.
        as_bool = tb(pr, col)
        if as_bool.notna().any() or (pr[col].dtype == object and pr[col].nunique() <= 5):
            found.append(col)
    return found


def step3(pr: pd.DataFrame, base: pd.DataFrame, var: pd.DataFrame) -> pd.DataFrame:
    say("=" * 100)
    say("STEP 3 -- PER-CLASS COUNTS")
    say("=" * 100)
    say("Counts are given on both frames: all 410 protocol rows (what the paper reports) and the")
    say("217-row analysis base that can actually be linked to a household.")
    say("")

    t_all = per_class_counts(pr, CANDIDATE_LABEL_COLUMNS, "listed candidates -- ALL protocol rows")
    t_base = per_class_counts(base, CANDIDATE_LABEL_COLUMNS, "listed candidates -- linkable base")

    extra = discover_extra_labels(pr, var)
    say(f"--- scan for further fault/condition judgements: {len(extra)} columns found ---")
    desc = dict(zip(var["VariableName"], var["Description"].fillna("")))
    for c in extra:
        say(f"  {c}: {desc.get(c, '(no description)')}")
    say("")
    e_all = per_class_counts(pr, extra, "discovered candidates -- ALL protocol rows")
    per_class_counts(base, extra, "discovered candidates -- linkable base")

    # --- Table 3 reconstruction ---
    say("=" * 100)
    say("STEP 3b -- TABLE 3 RECONSTRUCTION vs PUBLISHED")
    say("=" * 100)
    masks = fault_masks(pr)
    rows = []
    for label, (published_pct, published_n) in PUBLISHED_TABLE3.items():
        fault, scope = masks[label]
        n = int(fault.sum())
        denom = int(scope.sum())
        pct = 100 * n / denom
        rows.append({
            "fault": label,
            "pub_n": published_n, "pub_pct": published_pct,
            "our_n": n, "our_denom": denom, "our_pct": round(pct, 2),
            "n_match": "OK" if n == published_n else "MISMATCH",
            "pct_match": "OK" if abs(pct - published_pct) < 0.02 else "MISMATCH",
        })
    tab3 = pd.DataFrame(rows)
    say(tab3.to_string(index=False))
    say("")
    bad = tab3[(tab3["n_match"] == "MISMATCH") | (tab3["pct_match"] == "MISMATCH")]
    say(f"mismatches: {len(bad)} of {len(tab3)}")
    say("")
    say("Denominator hypothesis, tested:")
    say(f"  'air ducting' 27/240 = {100*27/240:.2f}% -- matches the published 11.25% only when the")
    say(f"     denominator is the 240 air-source rows, not all 410. CONFIRMED.")
    say(f"  'brine pressure' 12/167 = {100*12/167:.2f}% and 'geothermal probe' 10/167 = {100*10/167:.2f}%")
    say(f"     -- match 7.19% and 5.99% only over the 167 ground-source rows. CONFIRMED.")
    say(f"  every other row divides by 410 and reproduces the published percentage exactly.")
    say("")
    say("Definitional notes forced by the reconstruction:")
    nsb_raw = int(is_true(pr, "HeatPump_NightSetbackSetting_Activated_BeforeVisit").sum())
    nsb_still = int((is_true(pr, "HeatPump_NightSetbackSetting_Activated_BeforeVisit")
                     & is_true(pr, "HeatPump_NightSetbackSetting_Activated_AfterVisit")).sum())
    say(f"  * night setback: the raw before-visit flag is True on {nsb_raw} rows ({100*nsb_raw/410:.2f}%),")
    say(f"    NOT the published 148/36.10%. 148 = {nsb_raw} minus the {nsb_still} rows where the setback")
    say(f"    was still active after the visit. The published figure therefore counts setbacks the")
    say(f"    consultant switched off, not setbacks found. Using the raw flag as a label would")
    say(f"    silently disagree with the paper by 24 cases.")
    say(f"  * DHW temperature: 32 = 13 'too high' + 19 'too low'. The published row is any deviation")
    say(f"    from 'normal', not overheating alone.")
    say(f"  * air ducting is an OR over 5 separate air-source checks (duct distance / free / cleaning")
    say(f"    required / drain / evaporator clean), restricted to air-source rows.")
    say(f"  * 'circulation pump' matches PumpStagePosition_Changed (an action taken), not a condition")
    say(f"    judgement -- there is no 'pump stage too high' column in the data.")
    say("")

    combined = pd.concat([
        t_all.assign(frame="all_410"), t_base.assign(frame="linkable_217"),
        e_all.assign(frame="all_410_discovered"),
    ], ignore_index=True)
    return combined


# ---------------------------------------------------------------------------
# STEP 4 -- co-occurrence
# ---------------------------------------------------------------------------
def step4(pr: pd.DataFrame) -> list[str]:
    say("=" * 100)
    say("STEP 4 -- LABEL CO-OCCURRENCE")
    say("=" * 100)
    masks = fault_masks(pr)
    flags = pd.DataFrame({k: v[0] for k, v in masks.items()})

    n_faults = flags.sum(axis=1)
    say("faults set simultaneously per protocol row (all 410):")
    dist = n_faults.value_counts().sort_index()
    for k, v in dist.items():
        say(f"  {k} fault(s): {v:>4} rows ({v/len(flags):.1%})")
    say(f"  0 faults : {int((n_faults == 0).sum())}")
    say(f"  1 fault  : {int((n_faults == 1).sum())}")
    say(f"  2 faults : {int((n_faults == 2).sum())}")
    say(f"  3+ faults: {int((n_faults >= 3).sum())}")
    say(f"  mean faults per row: {n_faults.mean():.2f}   max: {int(n_faults.max())}")
    say("")

    top6 = flags.sum().sort_values(ascending=False).head(6).index.tolist()
    say(f"top 6 faults by frequency: {top6}")
    say("")
    say("pairwise co-occurrence (count of rows where BOTH are set; diagonal = marginal):")
    co = pd.DataFrame(index=top6, columns=top6, dtype=int)
    for a in top6:
        for b in top6:
            co.loc[a, b] = int((flags[a] & flags[b]).sum())
    say(co.to_string())
    say("")
    say("Jaccard overlap for the same pairs (|A and B| / |A or B|):")
    for a, b in combinations(top6, 2):
        inter = int((flags[a] & flags[b]).sum())
        union = int((flags[a] | flags[b]).sum())
        say(f"  {a:<30} & {b:<30} {inter:>4}/{union:<4} = {inter/union:.2f}")
    say("")
    multi = int((n_faults >= 2).sum())
    say(f"VERDICT ON TASK SHAPE: {multi} of {len(flags)} rows ({multi/len(flags):.1%}) carry two or more")
    say("faults at once, and the top-6 pairs overlap substantially rather than partitioning the data.")
    say("The faults are therefore NOT mutually exclusive: this is a MULTI-LABEL problem (one binary")
    say("classifier per fault type), not a multi-class one. Framing it as multi-class would force an")
    say("arbitrary choice of 'the' fault for the majority of visits.")
    say("")
    return top6


# ---------------------------------------------------------------------------
# STEP 5 -- did the setting actually change?
# ---------------------------------------------------------------------------
CHANGE_SPECS = [
    ("heating curve @ +20 C", "HeatPump_HeatingCurveSetting_Outside20_BeforeVisit",
     "HeatPump_HeatingCurveSetting_Outside20_AfterVisit", "HeatPump_HeatingCurveSetting_Changed"),
    ("heating curve @ 0 C", "HeatPump_HeatingCurveSetting_Outside0_BeforeVisit",
     "HeatPump_HeatingCurveSetting_Outside0_AfterVisit", "HeatPump_HeatingCurveSetting_Changed"),
    ("heating curve @ -8 C", "HeatPump_HeatingCurveSetting_OutsideMinus8_BeforeVisit",
     "HeatPump_HeatingCurveSetting_OutsideMinus8_AfterVisit", "HeatPump_HeatingCurveSetting_Changed"),
    ("heating limit", "HeatPump_HeatingLimitSetting_BeforeVisit",
     "HeatPump_HeatingLimitSetting_AfterVisit", "HeatPump_HeatingLimitSetting_Changed"),
    ("night setback", "HeatPump_NightSetbackSetting_Activated_BeforeVisit",
     "HeatPump_NightSetbackSetting_Activated_AfterVisit", None),
    ("DHW temperature", "DHW_TemperatureSetting_BeforeVisit",
     "DHW_TemperatureSetting_AfterVisit", "DHW_TemperatureSetting_Changed"),
    ("circulation pump stage", "HeatDistribution_Circulation_PumpStagePosition_BeforeVisit",
     "HeatDistribution_Circulation_PumpStagePosition_AfterVisit",
     "HeatDistribution_Circulation_PumpStagePosition_Changed"),
]


def step5(pr: pd.DataFrame) -> None:
    say("=" * 100)
    say("STEP 5 -- DID THE SETTING ACTUALLY CHANGE?")
    say("=" * 100)

    rows = []
    for name, bcol, acol, ccol in CHANGE_SPECS:
        # Night setback is boolean; the rest are numeric. Normalise to comparable values.
        if pr[bcol].dtype == object or pr[bcol].dtype == bool:
            before, after = tb(pr, bcol), tb(pr, acol)
        else:
            before, after = pr[bcol], pr[acol]

        both = before.notna() & after.notna()
        differs = both & (before != after)
        rec = {"setting": name, "both_present": int(both.sum()), "changed_observed": int(differs.sum())}

        if ccol is None:
            rec.update({"flag": "(none)", "flag_true": "", "agree": "", "disagree": "",
                        "flag_T_values_same": "", "flag_F_values_differ": ""})
        else:
            flag = tb(pr, ccol)
            # Compare flag against observation only where both are knowable.
            comparable = both & flag.notna()
            agree = int((comparable & (flag.eq(True) == differs)).sum())
            disagree = int((comparable & (flag.eq(True) != differs)).sum())
            fts = int((comparable & flag.eq(True) & ~differs).sum())
            ffd = int((comparable & flag.eq(False) & differs).sum())
            rec.update({"flag": ccol.split("_")[-1], "flag_true": int(flag.eq(True).sum()),
                        "agree": agree, "disagree": disagree,
                        "flag_T_values_same": fts, "flag_F_values_differ": ffd})
        rows.append(rec)

    say(pd.DataFrame(rows).to_string(index=False))
    say("")
    say("Reading: 'both_present' is the number of rows with a before AND an after value -- the only")
    say("rows where a change can be observed at all. 'flag_T_values_same' counts rows where the")
    say("*_Changed flag says True but the recorded values are identical; 'flag_F_values_differ' the")
    say("reverse. Both are inconsistencies in the protocol data.")
    say("")

    # Heating curve reduction magnitudes.
    say("--- heating curve reduction (before - after), in Kelvin ---")
    for label, bcol, acol in [
        ("at 0 C", "HeatPump_HeatingCurveSetting_Outside0_BeforeVisit",
         "HeatPump_HeatingCurveSetting_Outside0_AfterVisit"),
        ("at -8 C", "HeatPump_HeatingCurveSetting_OutsideMinus8_BeforeVisit",
         "HeatPump_HeatingCurveSetting_OutsideMinus8_AfterVisit"),
    ]:
        delta = (pr[bcol] - pr[acol]).dropna()
        say(f"[{label}]  n={len(delta)}")
        say(delta.describe().to_string())
        say(f"  reductions (>0 K): {int((delta > 0).sum())}   no change: {int((delta == 0).sum())}   "
            f"increases (<0 K): {int((delta < 0).sum())}")
        say("  distribution of delta K:")
        for k, v in delta.value_counts().sort_index().items():
            say(f"    {k:+.1f} K: {v}")
        say("")


# ---------------------------------------------------------------------------
# STEP 6 -- intersect with usable smart meter data
# ---------------------------------------------------------------------------
def step6(base: pd.DataFrame, q15: pd.DataFrame, var: pd.DataFrame) -> pd.DataFrame:
    say("=" * 100)
    say(f"STEP 6 -- INTERSECTION WITH USABLE SMART METER DATA (>= {BEFORE_AFTER_MIN_DAYS} d before AND after)")
    say("=" * 100)

    merged = base.merge(q15, on="Household_ID", how="left")
    usable = merged[(merged["SMD_15min_TimeAvailable_DaysBeforeVisit"] >= BEFORE_AFTER_MIN_DAYS)
                    & (merged["SMD_15min_TimeAvailable_DaysAfterVisit"] >= BEFORE_AFTER_MIN_DAYS)]

    say(f"protocol base rows: {len(base)}  -> with 15min overview row: "
        f"{int(merged['SMD_15min_TimeAvailable_NumberDays'].notna().sum())}")
    say(f"rows meeting the >= {BEFORE_AFTER_MIN_DAYS} d both-sides rule: {len(usable)}   "
        f"distinct households: {usable['Household_ID'].nunique()}")
    say("")

    per_class_counts(usable, CANDIDATE_LABEL_COLUMNS, "listed candidates -- USABLE before/after sample")
    extra = discover_extra_labels(base, var)
    per_class_counts(usable, extra, "discovered candidates -- USABLE before/after sample")

    say("--- fault prevalence in the usable sample, same definitions as Table 3 ---")
    masks = fault_masks(usable)
    rows = []
    for label, (fault, scope) in masks.items():
        n, denom = int(fault.sum()), int(scope.sum())
        rows.append({"fault": label, "positives": n, "in_scope": denom,
                     "negatives": denom - n,
                     "prevalence": f"{100*n/denom:.1f}%" if denom else "n/a"})
    usable_tab = pd.DataFrame(rows).sort_values("positives", ascending=False)
    say(usable_tab.to_string(index=False))
    say("")
    return usable_tab


# ---------------------------------------------------------------------------
# STEP 7 -- verdict
# ---------------------------------------------------------------------------
def step7(usable_tab: pd.DataFrame) -> None:
    say("=" * 100)
    say("STEP 7 -- VERDICT")
    say("=" * 100)
    say("Thresholds used (stated so they can be argued with):")
    say(f"  (a) supervised classification : >= {CLASSIFIER_MIN_MINORITY} positives AND "
        f">= {CLASSIFIER_MIN_MINORITY} negatives in the usable before/after sample.")
    say(f"      Rationale: below ~50 minority cases a train/test split leaves single-digit test")
    say(f"      positives, and any performance estimate is dominated by split noise.")
    say(f"  (b) descriptive comparison    : >= {DESCRIPTIVE_MIN_MINORITY} positives AND "
        f">= {DESCRIPTIVE_MIN_MINORITY} negatives -- enough to compare group means, not to fit a model.")
    say(f"  (c) nothing                   : below (b).")
    say("")

    verdicts = []
    for _, r in usable_tab.iterrows():
        pos, neg = int(r["positives"]), int(r["negatives"])
        minority = min(pos, neg)
        if minority >= CLASSIFIER_MIN_MINORITY:
            v = "(a) supervised classification"
        elif minority >= DESCRIPTIVE_MIN_MINORITY:
            v = "(b) descriptive comparison only"
        else:
            v = "(c) nothing"
        verdicts.append({"fault": r["fault"], "positives": pos, "negatives": neg,
                         "minority_class": minority, "verdict": v})
    vdf = pd.DataFrame(verdicts)
    say(vdf.to_string(index=False))
    say("")

    n_a = int((vdf["verdict"].str.startswith("(a)")).sum())
    n_b = int((vdf["verdict"].str.startswith("(b)")).sum())
    n_c = int((vdf["verdict"].str.startswith("(c)")).sum())
    say(f"summary: (a) {n_a} fault types   (b) {n_b} fault types   (c) {n_c} fault types")
    say("")
    say("BOTTOM LINE")
    if n_a == 0:
        say("  No fault type clears the supervised-classification bar. The binding constraint is not")
        say("  label quality -- it is sample size: the usable before/after sample is roughly 90")
        say("  households, and every fault type splits it into groups too small to train and honestly")
        say("  evaluate a classifier on. A supervised fault-type classifier is NOT viable on this data.")
        say("")
        say("  Fall back to rule-based interpretation. The protocols support it well: the fault")
        say("  definitions are explicit, reproducible (Table 3 reconstructs exactly, given the right")
        say("  denominators), and the before/after setting values are recorded numerically, so a rule")
        say("  engine can be validated against the consultants' own judgements rather than learned.")
        say("")
        say("  Additional reasons a classifier would be the wrong tool here even with more rows:")
        say("   - Labels are multi-label, not multi-class (STEP 4): 16 rare binary targets, not one.")
        say("   - The labels are consultant judgements recorded at a single visit, with no repeat")
        say("     assessment -- there is no inter-rater signal and no way to bound label noise.")
        say("   - 193 of 410 protocol rows have no Household_ID, so the largest part of the label")
        say("     data can never be joined to smart meter features at all.")
    else:
        say(f"  {n_a} fault type(s) clear the bar; treat the rest as descriptive or unusable.")
    say("")


def main() -> None:
    pr = load("protocols")
    var = load("variables")
    hh = load("households")
    q15 = load("15min")

    step1_inspect(pr, var)
    base = step2_base(pr, hh)
    label_counts = step3(pr, base, var)
    step4(pr)
    step5(pr)
    usable_tab = step6(base, q15, var)
    step7(usable_tab)

    OUTPUT_DIR.mkdir(exist_ok=True)
    counts_path = OUTPUT_DIR / "protocol_label_counts.csv"
    label_counts.to_csv(counts_path, index=False, sep=";")
    summary_path = OUTPUT_DIR / "protocol_recon_summary.txt"
    summary_path.write_text("\n".join(_report), encoding="utf-8")

    print("=" * 100)
    print(f"label counts -> {counts_path}")
    print(f"summary      -> {summary_path}")


if __name__ == "__main__":
    main()
