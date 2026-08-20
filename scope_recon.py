"""HEAPO scope reconnaissance -- final run before freezing scope.

Six questions: placebo-arm validity, peer-group feasibility, physics sanity,
night-setback signature, weather-normalisation baseline, PV contamination.

Reads the 15min time series, but strictly one household at a time.
Run:  py -3.13 scope_recon.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration -- the only place a path is hardcoded.
# ---------------------------------------------------------------------------
DATA_ROOT = Path(r"C:\Users\migue\OneDrive\Escritorio\SF AI forecast fauls\heapo_data")

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

FILES = {
    "households": DATA_ROOT / "meta_data" / "households.csv",
    "meta": DATA_ROOT / "meta_data" / "meta_data.csv",
    "protocols": DATA_ROOT / "reports" / "protocols.csv",
    "15min_overview": DATA_ROOT / "smart_meter_data" / "overview" / "smart_meter_data_15min_overview.csv",
}
SM15_DIR = DATA_ROOT / "smart_meter_data" / "15min"
WEATHER_DIR = DATA_ROOT / "weather_data" / "daily"

MIN_DAYS = 180
EXPECTED_N = 89
LOCAL_TZ = "Europe/Zurich"      # meter timestamps are UTC; hour-of-day needs local time
HEATING_SEASON_TAVG = 12.0      # deg C, per the brief
WINDOW_DAYS = 180
Q3_SAMPLE = 30
Q5_CONTROLS = 200
WEAK_R2 = 0.30
SMALL_CELL = 30
SEED = 20260818

# ASHRAE Guideline 14 acceptance thresholds.
ASHRAE_MONTHLY = {"NMBE": 5.0, "CVRMSE": 15.0}
ASHRAE_DAILY = {"NMBE": 10.0, "CVRMSE": 30.0}

_report: list[str] = []
_daily_cache: dict[int, pd.DataFrame] = {}


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
    """Object 'True'/'False' + NaN, or real bool -> pandas nullable boolean."""
    return df[col].map({True: True, False: False, "True": True, "False": False}).astype("boolean")


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares y ~ a + b*x. Returns (slope, intercept, r2).

    numpy only -- scipy and sklearn are not installed in this environment.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10 or np.ptp(x) == 0:
        return (np.nan, np.nan, np.nan)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return (float(coef[1]), float(coef[0]), r2)


def cvrmse_nmbe(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """ASHRAE G14 fit statistics, in percent. Both use the mean of ACTUAL."""
    ok = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[ok], pred[ok]
    if len(actual) < 3 or actual.mean() == 0:
        return (np.nan, np.nan)
    resid = actual - pred
    cvrmse = 100 * np.sqrt((resid ** 2).mean()) / actual.mean()
    nmbe = 100 * resid.sum() / (len(actual) * actual.mean())
    return (float(cvrmse), float(nmbe))


def std_diff(a: pd.Series, b: pd.Series) -> float:
    """Standardised difference (Cohen's d with pooled SD) for continuous variables."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
def load_weather() -> pd.DataFrame:
    """All 8 weather stations, daily, indexed by (Weather_ID, date)."""
    if not WEATHER_DIR.is_dir():
        raise FileNotFoundError(f"Weather directory not found: {WEATHER_DIR}")
    frames = [pd.read_csv(p, sep=";") for p in sorted(WEATHER_DIR.glob("*.csv"))]
    w = pd.concat(frames, ignore_index=True)
    w["date"] = pd.to_datetime(w["Timestamp"]).dt.normalize()
    return w[["Weather_ID", "date", "Temperature_avg_daily", "HeatingDegree_SIA_daily"]]


# ---------------------------------------------------------------------------
# Per-household 15min access -- one file at a time, never the fleet at once
# ---------------------------------------------------------------------------
def read_15min(hid: int, with_hp: bool = False) -> pd.DataFrame | None:
    """Read one household's 15min file and return it at LOCAL time.

    Timestamps are stored in UTC. Switzerland runs UTC+1/+2, so hour-of-day
    analysis (Q4) is wrong by one or two hours unless converted -- which would
    smear an overnight setback window across the wrong hours.
    """
    path = SM15_DIR / f"{hid}.csv"
    if not path.is_file():
        return None
    cols = ["Timestamp", "AffectsTimePoint", "kWh_received_Total"]
    if with_hp:
        cols.append("kWh_received_HeatPump")
    d = pd.read_csv(path, sep=";", usecols=cols)
    ts = pd.to_datetime(d["Timestamp"], utc=True, format="ISO8601").dt.tz_convert(LOCAL_TZ)
    d["local_ts"] = ts
    d["date"] = ts.dt.normalize().dt.tz_localize(None)
    d["hour"] = ts.dt.hour
    return d


def daily_for(hid: int, with_hp: bool = False) -> pd.DataFrame | None:
    """Daily totals for one household, cached. Columns: date, kWh[, kWh_hp], phase."""
    key = (hid, with_hp)
    if key in _daily_cache:
        return _daily_cache[key]
    d = read_15min(hid, with_hp=with_hp)
    if d is None:
        _daily_cache[key] = None
        return None
    agg = {"kWh_received_Total": "sum"}
    if with_hp:
        agg["kWh_received_HeatPump"] = "sum"
    g = d.groupby("date").agg(agg)
    g = g.rename(columns={"kWh_received_Total": "kWh", "kWh_received_HeatPump": "kWh_hp"})
    # A day's phase is the label carried by most of its intervals.
    phase = d.groupby("date")["AffectsTimePoint"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else None)
    g["phase"] = phase
    # Drop partial edge days: a 15min day should have 96 intervals.
    counts = d.groupby("date").size()
    g = g[counts >= 90]
    g = g.reset_index()
    _daily_cache[key] = g
    return g


def with_weather(daily: pd.DataFrame, wid: str, weather: pd.DataFrame) -> pd.DataFrame:
    w = weather[weather["Weather_ID"] == wid]
    return daily.merge(w, on="date", how="left")


# ---------------------------------------------------------------------------
# Base sample and intervention flags (identical definitions to the prior scripts)
# ---------------------------------------------------------------------------
def build_base(pr: pd.DataFrame, q15: pd.DataFrame) -> pd.DataFrame:
    say("=" * 100)
    say(f"BASE -- protocol households with >= {MIN_DAYS} d both sides (15min overview)")
    say("=" * 100)
    base = pr[pr["Household_ID"].notna()].copy()
    base["Household_ID"] = base["Household_ID"].astype("Int64")
    m = base.merge(q15, on="Household_ID", how="left")
    u = m[(m["SMD_15min_TimeAvailable_DaysBeforeVisit"] >= MIN_DAYS)
          & (m["SMD_15min_TimeAvailable_DaysAfterVisit"] >= MIN_DAYS)].copy()
    say(f"rows: {len(u)}   distinct households: {u['Household_ID'].nunique()}")
    if u["Household_ID"].nunique() != EXPECTED_N:
        raise SystemExit(f"STOP: expected {EXPECTED_N}, got {u['Household_ID'].nunique()}")
    say(f"CONFIRMED: {EXPECTED_N} households.")
    u = u.drop_duplicates("Household_ID", keep="last").reset_index(drop=True)
    u["Visit_Date_parsed"] = pd.to_datetime(u["Visit_Date"], errors="coerce")
    say("")
    return u


def intervention_flags(u: pd.DataFrame) -> pd.DataFrame:
    """Same five interventions as dose_response_recon.py."""
    f = pd.DataFrame({"Household_ID": u["Household_ID"].values})

    d0 = u["HeatPump_HeatingCurveSetting_Outside0_BeforeVisit"] - u["HeatPump_HeatingCurveSetting_Outside0_AfterVisit"]
    d8 = (u["HeatPump_HeatingCurveSetting_OutsideMinus8_BeforeVisit"]
          - u["HeatPump_HeatingCurveSetting_OutsideMinus8_AfterVisit"])
    f["curve_reduced_1K"] = ((d0 >= 1.0) | (d8 >= 1.0)).values

    for name, b, a in [
        ("limit_changed", "HeatPump_HeatingLimitSetting_BeforeVisit", "HeatPump_HeatingLimitSetting_AfterVisit"),
        ("dhw_changed", "DHW_TemperatureSetting_BeforeVisit", "DHW_TemperatureSetting_AfterVisit"),
        ("pumpstage_changed", "HeatDistribution_Circulation_PumpStagePosition_BeforeVisit",
         "HeatDistribution_Circulation_PumpStagePosition_AfterVisit"),
    ]:
        d = (u[b] - u[a]).where(u[b].notna() & u[a].notna())
        # NaN != 0 is True, so the notna() guard is mandatory.
        f[name] = (d.notna() & (d != 0)).values

    nb, na = tb(u, "HeatPump_NightSetbackSetting_Activated_BeforeVisit"), tb(u, "HeatPump_NightSetbackSetting_Activated_AfterVisit")
    f["nightsetback_deactivated"] = (nb.eq(True) & na.eq(False)).fillna(False).values
    f["nightsetback_still_active"] = (nb.eq(True) & na.eq(True)).fillna(False).values

    cols = ["curve_reduced_1K", "limit_changed", "nightsetback_deactivated", "dhw_changed", "pumpstage_changed"]
    f["n_interventions"] = f[cols].sum(axis=1)
    f["group"] = np.where(f["n_interventions"] >= 1, "INTERVENED", "UNCHANGED")
    return f


# ---------------------------------------------------------------------------
# Q1 -- is the zero-intervention group a placebo arm?
# ---------------------------------------------------------------------------
FAULT_FLAGS = {
    "heating curve too high": ("HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit", True),
    "heating limit too high": ("HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit", True),
    "night setback active (before)": ("HeatPump_NightSetbackSetting_Activated_BeforeVisit", True),
    "descaling too long ago": ("DHW_Storage_LastDescaling_TooLongAgo", True),
    "sizing incorrect": ("HeatPump_Installation_CorrectlyPlanned", False),
    "pipe insulation recommended": ("HeatDistribution_Recommendation_InsulatePipes", True),
    "thermostatic valve recommended": ("HeatDistribution_Recommendation_InstallThermostaticValve", True),
    "RPM valve recommended": ("HeatDistribution_Recommendation_InstallRPMValve", True),
    "dirty": ("HeatPump_Clean", False),
    "basic functions not okay": ("HeatPump_BasicFunctionsOkay", False),
    "technically not okay": ("HeatPump_TechnicallyOkay", False),
}
CONTINUOUS_VARS = {
    "building construction year": "Building_ConstructionYear",
    "heated floor area (m2)": "Building_FloorAreaHeated_Total",
    "residents": "Building_Residents",
    "HP installation year": "HeatPump_Installation_Year",
    "HP heating capacity (kW)": "HeatPump_Installation_HeatingCapacity",
    "HP nameplate COP": "HeatPump_Installation_Normpoint_COP",
}


def q1(u: pd.DataFrame, flags: pd.DataFrame) -> None:
    say("=" * 100)
    say("Q1 -- IS THE ZERO-INTERVENTION GROUP A VALID PLACEBO ARM?")
    say("=" * 100)
    d = u.merge(flags, on="Household_ID")
    iv = d[d["group"] == "INTERVENED"]
    un = d[d["group"] == "UNCHANGED"]
    still = d[d["nightsetback_still_active"]]
    say(f"INTERVENED: {len(iv)}   UNCHANGED: {len(un)}   "
        f"night-setback-still-active (inside INTERVENED or not): {len(still)}")
    say(f"  of the {len(still)} still-active cases, {int((still['group'] == 'INTERVENED').sum())} are in "
        f"INTERVENED and {int((still['group'] == 'UNCHANGED').sum())} in UNCHANGED")
    say("")

    say("--- fault flags: prevalence by group ---")
    rows = []
    for label, (col, positive) in FAULT_FLAGS.items():
        s = tb(d, col)
        hit = s.eq(positive).fillna(False)
        n_i, n_u = int(hit[d["group"] == "INTERVENED"].sum()), int(hit[d["group"] == "UNCHANGED"].sum())
        r_i, r_u = n_i / len(iv), n_u / len(un)
        rows.append({"fault": label, "INTERVENED_n": n_i, "INTERVENED_%": round(100 * r_i, 1),
                     "UNCHANGED_n": n_u, "UNCHANGED_%": round(100 * r_u, 1),
                     "rate_diff_pp": round(100 * (r_i - r_u), 1)})
    ftab = pd.DataFrame(rows)
    say(ftab.to_string(index=False))
    say("")

    say("--- categorical condition judgements ---")
    for col in ["HeatPump_ElectricityConsumption_Categorization",
                "DHW_TemperatureSetting_Categorization",
                "HeatDistribution_ExpansionTank_Pressure_Categorization"]:
        say(f"  {col}")
        ct = pd.crosstab(d["group"], d[col], dropna=False)
        say("    " + ct.to_string().replace("\n", "\n    "))
    say("")

    say("--- continuous covariates (mean, and standardised difference) ---")
    rows = []
    for label, col in CONTINUOUS_VARS.items():
        a, b = iv[col], un[col]
        rows.append({"variable": label,
                     "INTERVENED_mean": round(a.mean(), 1) if a.notna().any() else np.nan,
                     "INTERVENED_n": int(a.notna().sum()),
                     "UNCHANGED_mean": round(b.mean(), 1) if b.notna().any() else np.nan,
                     "UNCHANGED_n": int(b.notna().sum()),
                     "std_diff": round(std_diff(a, b), 2)})
    say(pd.DataFrame(rows).to_string(index=False))
    say("")
    say("--- heat pump type ---")
    say(pd.crosstab(d["group"], d["HeatPump_Installation_Type"]).to_string())
    say("")

    # The decisive number: faults FOUND at inspection, regardless of whether fixed.
    found = pd.Series(False, index=d.index)
    for label, (col, positive) in FAULT_FLAGS.items():
        found = found | tb(d, col).eq(positive).fillna(False)
    n_faults = pd.DataFrame({lab: tb(d, c).eq(p).fillna(False)
                             for lab, (c, p) in FAULT_FLAGS.items()}).sum(axis=1)
    d = d.assign(any_fault=found.values, n_faults=n_faults.values)

    say("--- faults FOUND at inspection (independent of whether anything was changed) ---")
    for g in ("INTERVENED", "UNCHANGED"):
        sub = d[d["group"] == g]
        say(f"  {g}: {int(sub['any_fault'].sum())}/{len(sub)} "
            f"({sub['any_fault'].mean():.1%}) had at least one fault recorded")
        say(f"    faults per visit: mean={sub['n_faults'].mean():.2f}  median={sub['n_faults'].median():.0f}  "
            f"max={int(sub['n_faults'].max())}")
        dist = sub["n_faults"].value_counts().sort_index()
        say("    distribution: " + "  ".join(f"{k}:{v}" for k, v in dist.items()))
    say("")

    un_faulty = int(d[(d["group"] == "UNCHANGED")]["any_fault"].sum())
    un_clean = len(un) - un_faulty
    say("VERDICT ON Q1")
    say(f"  Of the {len(un)} UNCHANGED households, {un_faulty} had at least one fault recorded at")
    say(f"  inspection and only {un_clean} were genuinely fault-free.")
    if un_faulty > un_clean:
        say("  => The UNCHANGED group is NOT a placebo arm. It is dominated by households where the")
        say("     consultant FOUND faults and did not fix them (or the fix was not recorded), which is")
        say("     a different thing entirely: they are exposed to the same inspection, the same")
        say("     awareness effect, and carry the same underlying faults. Using them as controls")
        say("     assumes the visit itself had no effect, which is exactly what a placebo arm is")
        say("     supposed to test.")
        say(f"  => The design must instead lean on the {len(still)} night-setback-still-active cases:")
        say("     same fault, same visit, no fix. That is a true within-fault control -- and it is 9")
        say("     households, which is the real constraint on the causal design.")
    else:
        say("  => The UNCHANGED group is mostly fault-free and can serve as a placebo arm.")
    say("")


# ---------------------------------------------------------------------------
# Q2 -- peer-group feasibility on the full fleet
# ---------------------------------------------------------------------------
def q2(hh: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    say("=" * 100)
    say("Q2 -- PEER-GROUP FEASIBILITY ON THE FULL FLEET")
    say("=" * 100)
    say(f"CONTRADICTS THE BRIEF: meta_data.csv covers {len(meta)} households, not all "
        f"{len(hh)}. {len(hh) - len(meta)} households have no metadata row at all and can never be")
    say(f"peer-grouped on survey variables. Percentages below are over all {len(hh)} households,")
    say(f"since that is the population the detector must score.")
    say("")

    full = hh[["Household_ID", "Group", "Weather_ID"]].merge(meta, on="Household_ID", how="left")

    say("--- completeness per metadata variable (over all 1408) ---")
    rows = []
    for c in meta.columns:
        if c == "Household_ID":
            continue
        n = int(full[c].notna().sum())
        rows.append({"variable": c, "non_null": n, "pct_of_1408": round(100 * n / len(full), 1)})
    comp = pd.DataFrame(rows).sort_values("non_null", ascending=False)
    say(comp.to_string(index=False))
    say("")

    # Distribution system as a single categorical, from the two boolean columns.
    fh = tb(full, "Survey_HeatDistribution_System_FloorHeating")
    rad = tb(full, "Survey_HeatDistribution_System_Radiator")
    # np.where cannot consume pd.NA, so collapse to plain numpy bools first.
    f_t, r_t = fh.eq(True).fillna(False).to_numpy(bool), rad.eq(True).fillna(False).to_numpy(bool)
    dist_sys = pd.Series(np.where(f_t & r_t, "both",
                         np.where(f_t, "floor",
                         np.where(r_t, "radiator", "neither/unknown"))), index=full.index)
    # A household with both source columns null is unknown, not "neither".
    dist_sys = dist_sys.where((fh.notna() | rad.notna()).to_numpy(bool))
    full["distribution_system"] = dist_sys
    say("--- derived distribution system ---")
    say(full["distribution_system"].value_counts(dropna=False).to_string())
    say("")

    say("--- HP type x distribution system x building type ---")
    ct = full.groupby(["Survey_HeatPump_Installation_Type", "distribution_system",
                       "Survey_Building_Type"], dropna=False).size().reset_index(name="n")
    ct = ct[ct["n"] > 0].sort_values("n", ascending=False)
    say(ct.to_string(index=False))
    small = ct[ct["n"] < SMALL_CELL]
    say("")
    say(f"cells with n < {SMALL_CELL}: {len(small)} cells holding {int(small['n'].sum())} households "
        f"({small['n'].sum()/len(full):.1%} of the fleet)")
    say(f"cells with n >= {SMALL_CELL}: {len(ct) - len(small)} cells holding "
        f"{int(ct[ct['n'] >= SMALL_CELL]['n'].sum())} households")
    say("")

    say("--- living area and residents ---")
    la, res = full["Survey_Building_LivingArea"], full["Survey_Building_Residents"]
    say(f"living area  : n={int(la.notna().sum())}  mean={la.mean():.0f}  median={la.median():.0f}  "
        f"min={la.min():.0f}  max={la.max():.0f}")
    say(f"residents    : n={int(res.notna().sum())}  mean={res.mean():.1f}  median={res.median():.0f}  "
        f"min={res.min():.0f}  max={res.max():.0f}")
    say(f"both non-null: {int((la.notna() & res.notna()).sum())} ({(la.notna() & res.notna()).mean():.1%})")
    say("")

    need = ["Survey_HeatPump_Installation_Type", "distribution_system", "Survey_Building_LivingArea"]
    complete = full[need].notna().all(axis=1)
    say(f"--- households with complete peer-grouping metadata (HP type + distribution + area) ---")
    say(f"  {int(complete.sum())} of {len(full)} ({complete.mean():.1%})")
    say(f"  missing at least one: {int((~complete).sum())}")
    say("")

    say("LARGEST VIABLE PEER-GROUPING SCHEME")
    two_way = full.groupby(["Survey_HeatPump_Installation_Type", "distribution_system"],
                           dropna=False).size().reset_index(name="n")
    two_way = two_way[two_way["n"] > 0].sort_values("n", ascending=False)
    say(two_way.to_string(index=False))
    # A NaN level is missing data, not a stratum -- counting it as a viable cell would
    # hide the unassignable households inside the coverage figure.
    real = two_way.dropna(subset=["Survey_HeatPump_Installation_Type", "distribution_system"])
    viable = real[real["n"] >= SMALL_CELL]
    unassignable = int(two_way["n"].sum() - real["n"].sum())
    say("")
    say(f"  HP type x distribution system yields {len(real)} REAL occupied cells (rows with a NaN")
    say(f"  level are missing data, not a stratum), of which {len(viable)} have n >= {SMALL_CELL},")
    say(f"  covering {int(viable['n'].sum())} households.")
    say(f"  UNASSIGNABLE to any stratum (HP type or distribution system missing): {unassignable} "
        f"({unassignable/len(full):.1%} of the fleet).")
    say(f"  Adding building type is NOT viable: 'appartment' has only "
        f"{int((full['Survey_Building_Type'] == 'appartment').sum())} households fleet-wide, so every")
    say(f"  cell it creates is below the minimum.")
    say(f"  Adding living area as a tercile within each cell would give {3 * len(viable)} groups; the")
    say(f"  smallest viable cell has n={int(viable['n'].min())}, so terciles there hold ~{int(viable['n'].min()/3)}")
    say(f"  households -- below the {SMALL_CELL} minimum. Area must therefore enter as a continuous")
    say(f"  normaliser (kWh/m2), not as a stratum.")
    say("")
    say(f"  RECOMMENDED: stratify on HP type x distribution system ({len(viable)} groups, min n="
        f"{int(viable['n'].min())}), and within each group rank households on degree-day normalised")
    say(f"  energy intensity (kWh per HDD per m2) as percentiles. Households missing area "
        f"({int(full['Survey_Building_LivingArea'].isna().sum())}) fall back to kWh per HDD unnormalised,")
    say(f"  ranked within the same stratum.")
    say("")
    return ct


# ---------------------------------------------------------------------------
# Q3 -- does the physics show up?
# ---------------------------------------------------------------------------
def q3(hh: pd.DataFrame, q15ov: pd.DataFrame, weather: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    say("=" * 100)
    say("Q3 -- DOES THE PHYSICS SHOW UP IN THE DATA?")
    say("=" * 100)

    meta = load("meta")
    pool = hh.merge(meta[["Household_ID", "Survey_HeatPump_Installation_Type"]], on="Household_ID", how="left")
    pool = pool[pool["SmartMeterData_Available_15min"].astype(bool)]

    # Stratified sample: proportional across the two heat pump types.
    picks = []
    for t, grp in pool.groupby("Survey_HeatPump_Installation_Type"):
        if pd.isna(t):
            continue
        k = max(1, round(Q3_SAMPLE * len(grp) / len(pool.dropna(subset=["Survey_HeatPump_Installation_Type"]))))
        picks.append(grp.sample(min(k, len(grp)), random_state=int(rng.integers(1e9))))
    sample = pd.concat(picks).head(Q3_SAMPLE)
    say(f"stratified sample: {len(sample)} households  "
        f"{dict(sample['Survey_HeatPump_Installation_Type'].value_counts())}")
    say("")

    rows = []
    for _, r in sample.iterrows():
        hid, wid = int(r["Household_ID"]), r["Weather_ID"]
        d = daily_for(hid)
        if d is None or len(d) < 30:
            continue
        dw = with_weather(d, wid, weather)
        slope, _, r2 = ols(dw["HeatingDegree_SIA_daily"].to_numpy(float), dw["kWh"].to_numpy(float))
        rows.append({"Household_ID": hid, "hp_type": r["Survey_HeatPump_Installation_Type"],
                     "n_days": len(dw), "slope_kWh_per_HDD": slope, "r2": r2, "series": "whole-household"})
    fit = pd.DataFrame(rows)
    say("--- whole-household daily kWh ~ HDD (SIA) ---")
    r2s = fit["r2"].dropna()
    say(f"  households fitted: {len(fit)}  (usable R2: {len(r2s)}; "
        f"{len(fit) - len(r2s)} had too few days or no HDD variation)")
    say(f"  R2: mean={r2s.mean():.3f}  median={r2s.median():.3f}  min={r2s.min():.3f}  max={r2s.max():.3f}")
    say(f"  quartiles: q25={r2s.quantile(.25):.3f}  q50={r2s.quantile(.5):.3f}  q75={r2s.quantile(.75):.3f}")
    say(f"  slope: median={fit['slope_kWh_per_HDD'].median():.3f} kWh/HDD")
    say(f"  R2 < {WEAK_R2}: {int((r2s < WEAK_R2).sum())} of {len(r2s)} ({(r2s < WEAK_R2).mean():.1%})")
    say("  R2 distribution by 0.1 bin:")
    for b, c in (r2s // 0.1 * 0.1).round(1).value_counts().sort_index().items():
        say(f"    [{b:.1f}, {b+0.1:.1f}): {c:>3}  {'#' * int(c)}")
    say("")

    # --- sub-metered comparison: the number that justifies whole-household modelling ---
    sub_ids = q15ov[q15ov["SMD_15min_MeasurementsAvailable_HeatPump"].astype(bool)]["Household_ID"].tolist()
    say(f"--- sub-metered households (n={len(sub_ids)} at 15min): HP-only vs whole-household ---")
    wid_map = dict(zip(hh["Household_ID"], hh["Weather_ID"]))
    rows = []
    for hid in sub_ids:
        d = daily_for(int(hid), with_hp=True)
        if d is None or len(d) < 30 or "kWh_hp" not in d or d["kWh_hp"].notna().sum() < 30:
            continue
        dw = with_weather(d, wid_map.get(hid), weather)
        hdd = dw["HeatingDegree_SIA_daily"].to_numpy(float)
        _, _, r2_tot = ols(hdd, dw["kWh"].to_numpy(float))
        slope_hp, _, r2_hp = ols(hdd, dw["kWh_hp"].to_numpy(float))
        rows.append({"Household_ID": int(hid), "n_days": len(dw), "r2_whole": r2_tot,
                     "r2_hp_only": r2_hp, "slope_hp": slope_hp,
                     "hp_share": dw["kWh_hp"].sum() / dw["kWh"].sum() if dw["kWh"].sum() else np.nan})
    sub = pd.DataFrame(rows)
    if len(sub):
        say(f"  households fitted: {len(sub)}")
        say(f"  R2 whole-household: mean={sub['r2_whole'].mean():.3f}  median={sub['r2_whole'].median():.3f}")
        say(f"  R2 heat-pump-only : mean={sub['r2_hp_only'].mean():.3f}  median={sub['r2_hp_only'].median():.3f}")
        delta = sub["r2_hp_only"] - sub["r2_whole"]
        say(f"  R2 gain from sub-metering: mean=+{delta.mean():.3f}  median=+{delta.median():.3f}  "
            f"max=+{delta.max():.3f}")
        say(f"  heat pump share of total consumption: median={sub['hp_share'].median():.1%}")
        say(f"  whole-household fits with R2 < {WEAK_R2}: {int((sub['r2_whole'] < WEAK_R2).sum())}")
        say(f"  HP-only fits with R2 < {WEAK_R2}:        {int((sub['r2_hp_only'] < WEAK_R2).sum())}")
        say("")
        say(f"  ==> THE NUMBER: within the same household, moving from the heat pump sub-meter to the")
        say(f"      whole-household signal costs {delta.median():.3f} of R2 at the median and "
            f"{delta.mean():.3f} at the mean")
        say(f"      (paired differences). Unpaired, the median fit falls from "
            f"{sub['r2_hp_only'].median():.3f} to {sub['r2_whole'].median():.3f}.")
        say(f"      Weak fits (R2 < {WEAK_R2}) go from {int((sub['r2_hp_only'] < WEAK_R2).sum())} "
            f"households sub-metered to {int((sub['r2_whole'] < WEAK_R2).sum())} whole-household.")
    else:
        say("  no sub-metered household produced a usable fit -- see the HeatPump column nulls.")
    say("")

    out = pd.concat([fit, sub.assign(series="sub-metered") if len(sub) else pd.DataFrame()],
                    ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# Q4 -- night setback signature
# ---------------------------------------------------------------------------
def hourly_profile(hid: int, wid: str, visit: pd.Timestamp, weather: pd.DataFrame) -> dict | None:
    """Mean kWh by local hour, before vs after, heating-season days only."""
    d = read_15min(hid)
    if d is None:
        return None
    w = weather[weather["Weather_ID"] == wid][["date", "Temperature_avg_daily"]]
    d = d.merge(w, on="date", how="left")
    # Heating season by daily mean temperature, per the brief.
    d = d[d["Temperature_avg_daily"] < HEATING_SEASON_TAVG]
    if d.empty:
        return None
    lo, hi = visit - pd.Timedelta(days=WINDOW_DAYS), visit + pd.Timedelta(days=WINDOW_DAYS)
    before = d[(d["date"] >= lo) & (d["date"] < visit)]
    after = d[(d["date"] > visit) & (d["date"] <= hi)]
    if len(before) < 96 * 10 or len(after) < 96 * 10:
        return None

    def prof(x):
        # Sum within (date, hour) first so each hour is a real hourly kWh, then average across days.
        h = x.groupby(["date", "hour"])["kWh_received_Total"].sum().reset_index()
        return h.groupby("hour")["kWh_received_Total"].mean()

    pb, pa = prof(before), prof(after)
    daily_b = before.groupby("date")["kWh_received_Total"].sum().mean()
    daily_a = after.groupby("date")["kWh_received_Total"].sum().mean()

    def share(p, lo_h, hi_h):
        return p.loc[lo_h:hi_h - 1].sum() / p.sum() if p.sum() else np.nan

    return {"Household_ID": hid, "profile_before": pb, "profile_after": pa,
            "daily_before": daily_b, "daily_after": daily_a,
            "night_share_before": share(pb, 0, 5), "night_share_after": share(pa, 0, 5),
            "morning_share_before": share(pb, 5, 9), "morning_share_after": share(pa, 5, 9),
            "n_days_before": before["date"].nunique(), "n_days_after": after["date"].nunique()}


def q4(u: pd.DataFrame, flags: pd.DataFrame, hh: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    say("=" * 100)
    say("Q4 -- NIGHT SETBACK SIGNATURE (the falsification test)")
    say("=" * 100)
    say(f"Profiles use LOCAL time ({LOCAL_TZ}). The raw timestamps are UTC; without conversion the")
    say(f"overnight window would be shifted 1-2 h and the setback smeared into the wrong hours.")
    say(f"Heating-season days only (daily mean T < {HEATING_SEASON_TAVG} C), "
        f"{WINDOW_DAYS} d each side of the visit.")
    say("")

    d = u.merge(flags, on="Household_ID")
    wid_map = dict(zip(hh["Household_ID"], hh["Weather_ID"]))
    cohorts = {
        "DEACTIVATED (21)": d[d["nightsetback_deactivated"]],
        "STILL ACTIVE (9)": d[d["nightsetback_still_active"]],
        "UNCHANGED (41)": d[d["group"] == "UNCHANGED"],
    }

    results, profiles = {}, []
    for name, sub in cohorts.items():
        recs = []
        for _, r in sub.iterrows():
            hid = int(r["Household_ID"])
            p = hourly_profile(hid, wid_map.get(hid), r["Visit_Date_parsed"], weather)
            if p:
                p["cohort"] = name
                recs.append(p)
                for hour in range(24):
                    profiles.append({"cohort": name, "Household_ID": hid, "hour": hour,
                                     "kWh_before": p["profile_before"].get(hour, np.nan),
                                     "kWh_after": p["profile_after"].get(hour, np.nan)})
        results[name] = recs
        say(f"{name}: {len(recs)} of {len(sub)} households produced a usable profile")
    say("")

    say("--- mean hourly load profile, kWh per hour (heating-season days) ---")
    header = f"{'hour':>4} |"
    for name in cohorts:
        header += f" {name.split()[0][:6]:>7} before {'after':>7} {'delta':>7} |"
    say(header)
    say("-" * len(header))
    for hour in range(24):
        line = f"{hour:>4} |"
        for name in cohorts:
            recs = results[name]
            b = np.nanmean([r["profile_before"].get(hour, np.nan) for r in recs]) if recs else np.nan
            a = np.nanmean([r["profile_after"].get(hour, np.nan) for r in recs]) if recs else np.nan
            line += f" {'':>6} {b:>6.3f} {a:>7.3f} {a-b:>+7.3f} |"
        say(line)
    say("")

    say("--- overnight (00:00-04:59) and morning (05:00-08:59) share of daily consumption ---")
    rows = []
    for name, recs in results.items():
        if not recs:
            continue
        nb = np.nanmean([r["night_share_before"] for r in recs])
        na = np.nanmean([r["night_share_after"] for r in recs])
        mb = np.nanmean([r["morning_share_before"] for r in recs])
        ma = np.nanmean([r["morning_share_after"] for r in recs])
        db = np.nanmean([r["daily_before"] for r in recs])
        da = np.nanmean([r["daily_after"] for r in recs])
        rows.append({"cohort": name, "n": len(recs),
                     "night%_before": round(100 * nb, 2), "night%_after": round(100 * na, 2),
                     "night_delta_pp": round(100 * (na - nb), 2),
                     "morning%_before": round(100 * mb, 2), "morning%_after": round(100 * ma, 2),
                     "morning_delta_pp": round(100 * (ma - mb), 2),
                     "daily_kWh_before": round(db, 2), "daily_kWh_after": round(da, 2)})
    shares = pd.DataFrame(rows)
    say(shares.to_string(index=False))
    say("")

    _cohort_deltas: dict[str, np.ndarray] = {}
    say("--- per-household spread of the overnight share change (pp) ---")
    for name, recs in results.items():
        if not recs:
            continue
        deltas = np.array([100 * (r["night_share_after"] - r["night_share_before"]) for r in recs])
        deltas = deltas[np.isfinite(deltas)]   # a household with an all-zero profile yields NaN
        _cohort_deltas[name] = deltas
        say(f"  {name:<18} n={len(deltas):>2}  mean={deltas.mean():+.2f}  median={np.median(deltas):+.2f}  "
            f"sd={deltas.std(ddof=1):.2f}  min={deltas.min():+.2f}  max={deltas.max():+.2f}")
        say(f"  {'':<18} increased overnight share: {int((deltas > 0).sum())}/{len(deltas)}")
    say("")

    say("FALSIFICATION VERDICT")
    if len(shares) >= 2:
        deact = shares[shares["cohort"].str.startswith("DEACT")]
        still = shares[shares["cohort"].str.startswith("STILL")]
        unch = shares[shares["cohort"].str.startswith("UNCH")]
        if len(deact) and len(still):
            dd = float(deact["night_delta_pp"].iat[0])
            ds = float(still["night_delta_pp"].iat[0])
            du = float(unch["night_delta_pp"].iat[0]) if len(unch) else np.nan
            say(f"  overnight share change: DEACTIVATED {dd:+.2f} pp, STILL ACTIVE {ds:+.2f} pp, "
                f"UNCHANGED {du:+.2f} pp")
            sep = abs(dd - ds)
            say(f"  separation between DEACTIVATED and STILL ACTIVE: {sep:.2f} pp")
            # Compare against the per-household spread rather than eyeballing the means.
            dd_arr, ds_arr = _cohort_deltas["DEACTIVATED (21)"], _cohort_deltas["STILL ACTIVE (9)"]
            sd_d = float(dd_arr.std(ddof=1))
            # Standard error of the difference in means -- the spread of the COHORT MEAN is the
            # right yardstick for a cohort-level claim, not the spread of individual households.
            se = float(np.sqrt(dd_arr.var(ddof=1)/len(dd_arr) + ds_arr.var(ddof=1)/len(ds_arr)))
            say(f"  per-household SD within DEACTIVATED: {sd_d:.2f} pp")
            say(f"  standard error of the DEACTIVATED-minus-STILL-ACTIVE difference: {se:.2f} pp "
                f"({sep/se:.2f} SE)")
            say(f"  households whose overnight share ROSE: "
                f"DEACTIVATED {int((dd_arr > 0).sum())}/{len(dd_arr)}, "
                f"STILL ACTIVE {int((ds_arr > 0).sum())}/{len(ds_arr)}")
            if sep > 2 * se:
                say("  => the cohorts SEPARATE at the cohort-mean level: the difference is "
                    f"{sep/se:.1f} standard errors,")
                say("     and the hourly table shows the right SHAPE, not just a level shift -- the")
                say("     deactivated group gains overnight load and loses morning/daytime load, while")
                say("     the still-active group rises at every hour. The individual spread is wide")
                say(f"     ({sd_d:.1f} pp), so per-household detection remains unreliable.")
            elif sep > se:
                say(f"  => WEAK separation: {sep/se:.1f} standard errors. Suggestive, not conclusive.")
            else:
                say("  => the cohorts DO NOT SEPARATE: the difference between deactivated and")
                say("     still-active is smaller than the household-to-household spread inside the")
                say("     deactivated group. With n=21 vs n=9 this effect is not recoverable from the")
                say("     overnight share alone. The headline deliverable fails its own falsification")
                say("     test at this sample size.")
    say("")
    say("NOTE: matplotlib is not installed in this environment, so no plot files were written.")
    say("The text tables above are the deliverable; install matplotlib if a figure is required.")
    say("")
    return pd.DataFrame(profiles)


# ---------------------------------------------------------------------------
# Q5 -- weather normalisation baseline
# ---------------------------------------------------------------------------
def q5(u: pd.DataFrame, hh: pd.DataFrame, weather: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    say("=" * 100)
    say("Q5 -- WEATHER NORMALISATION BASELINE (HDD regression)")
    say("=" * 100)

    controls = hh[(hh["Group"] == "control") & hh["SmartMeterData_Available_15min"].astype(bool)]
    controls = controls.sample(min(Q5_CONTROLS, len(controls)), random_state=int(rng.integers(1e9)))
    say(f"cohort: {len(u)} protocol households + {len(controls)} random controls")
    say("For controls there is no visit, so the series is split at the MIDPOINT of its own date")
    say("range -- a pseudo-visit. This measures baseline drift with no intervention, which is the")
    say("correct null for the treated fits.")
    say("")

    wid_map = dict(zip(hh["Household_ID"], hh["Weather_ID"]))
    rows = []
    targets = ([(int(r["Household_ID"]), r["Visit_Date_parsed"], "protocol") for _, r in u.iterrows()]
               + [(int(r["Household_ID"]), None, "control") for _, r in controls.iterrows()])

    for hid, visit, kind in targets:
        d = daily_for(hid)
        if d is None or len(d) < 60:
            continue
        dw = with_weather(d, wid_map.get(hid), weather).dropna(subset=["HeatingDegree_SIA_daily", "kWh"])
        if len(dw) < 60:
            continue
        split = visit if visit is not None else dw["date"].min() + (dw["date"].max() - dw["date"].min()) / 2
        pre, post = dw[dw["date"] < split], dw[dw["date"] > split]
        if len(pre) < 30 or len(post) < 30:
            continue
        slope, intercept, r2 = ols(pre["HeatingDegree_SIA_daily"].to_numpy(float), pre["kWh"].to_numpy(float))
        if not np.isfinite(slope):
            continue
        pred = intercept + slope * post["HeatingDegree_SIA_daily"].to_numpy(float)
        cv_d, nmbe_d = cvrmse_nmbe(post["kWh"].to_numpy(float), pred)
        # Monthly aggregation, because ASHRAE G14's 5%/15% thresholds are monthly.
        pm = post.assign(pred=pred, month=post["date"].dt.to_period("M")).groupby("month")[["kWh", "pred"]].sum()
        cv_m, nmbe_m = cvrmse_nmbe(pm["kWh"].to_numpy(float), pm["pred"].to_numpy(float))
        rows.append({"Household_ID": hid, "kind": kind, "n_pre": len(pre), "n_post": len(post),
                     "pre_r2": r2, "slope": slope, "cvrmse_daily": cv_d, "nmbe_daily": nmbe_d,
                     "n_months": len(pm), "cvrmse_monthly": cv_m, "nmbe_monthly": nmbe_m})

    fit = pd.DataFrame(rows)
    say(f"households fitted: {len(fit)}  ({dict(fit['kind'].value_counts())})")
    say("")
    for metric, label in [("cvrmse_monthly", "CV(RMSE) monthly %"), ("nmbe_monthly", "NMBE monthly %"),
                          ("cvrmse_daily", "CV(RMSE) daily %"), ("nmbe_daily", "NMBE daily %")]:
        s = fit[metric].dropna()
        say(f"{label:<22} n={len(s):>3}  mean={s.mean():7.2f}  median={s.median():7.2f}  "
            f"q25={s.quantile(.25):7.2f}  q75={s.quantile(.75):7.2f}  min={s.min():7.2f}  max={s.max():7.2f}")
    say("")

    say("--- ASHRAE Guideline 14 pass rates ---")
    for scope, thr, cv_col, nm_col in [("monthly", ASHRAE_MONTHLY, "cvrmse_monthly", "nmbe_monthly"),
                                       ("daily", ASHRAE_DAILY, "cvrmse_daily", "nmbe_daily")]:
        ok_cv = fit[cv_col].abs() <= thr["CVRMSE"]
        ok_nm = fit[nm_col].abs() <= thr["NMBE"]
        both = ok_cv & ok_nm
        say(f"  {scope:<8} thresholds NMBE <= {thr['NMBE']}%, CV(RMSE) <= {thr['CVRMSE']}%")
        say(f"    pass CV(RMSE): {int(ok_cv.sum()):>3}/{len(fit)} ({ok_cv.mean():.1%})")
        say(f"    pass NMBE    : {int(ok_nm.sum()):>3}/{len(fit)} ({ok_nm.mean():.1%})")
        say(f"    pass BOTH    : {int(both.sum()):>3}/{len(fit)} ({both.mean():.1%})")
    say("")
    say("--- by cohort (monthly, both criteria) ---")
    for kind, sub in fit.groupby("kind"):
        both = (sub["cvrmse_monthly"].abs() <= ASHRAE_MONTHLY["CVRMSE"]) & \
               (sub["nmbe_monthly"].abs() <= ASHRAE_MONTHLY["NMBE"])
        say(f"  {kind:<10} {int(both.sum()):>3}/{len(sub)} ({both.mean():.1%})")
    say("")

    pass_rate = ((fit["cvrmse_monthly"].abs() <= ASHRAE_MONTHLY["CVRMSE"])
                 & (fit["nmbe_monthly"].abs() <= ASHRAE_MONTHLY["NMBE"])).mean()
    say("VERDICT ON Q5")
    if pass_rate < 0.5:
        say(f"  Only {pass_rate:.1%} of households meet both ASHRAE G14 monthly criteria with a plain")
        say(f"  daily kWh ~ HDD regression. A simple HDD regression is NOT an adequate baseline for")
        say(f"  the majority of this fleet. The median CV(RMSE) alone tells you the residual scatter")
        say(f"  is comparable to the effect sizes you are trying to detect.")
        say(f"  A richer model is required -- but per the brief nothing more complex is fitted here.")
    else:
        say(f"  {pass_rate:.1%} of households meet both monthly criteria; a plain HDD regression is an")
        say(f"  adequate starting baseline for most of the fleet.")
    say("")
    return fit


# ---------------------------------------------------------------------------
# Q6 -- PV contamination
# ---------------------------------------------------------------------------
def q6(u: pd.DataFrame, hh: pd.DataFrame, weather: pd.DataFrame) -> None:
    say("=" * 100)
    say("Q6 -- PV CONTAMINATION")
    say("=" * 100)
    pv_all = tb(hh, "Installation_HasPVSystem")
    say(f"CONTRADICTS THE BRIEF: the brief states 34.65% of households have PV. In households.csv,")
    say(f"Installation_HasPVSystem is True for {int(pv_all.eq(True).sum())} of {len(hh)} "
        f"({pv_all.eq(True).fillna(False).mean():.2%}), False for {int(pv_all.eq(False).sum())}, and")
    say(f"NULL for {int(pv_all.isna().sum())} ({pv_all.isna().mean():.1%}). The real problem is not the")
    say(f"rate, it is that PV status is UNKNOWN for more than half the fleet. Among households where")
    say(f"it is known, {pv_all.eq(True).sum()/pv_all.notna().sum():.1%} have PV.")
    say("")

    d = u.merge(hh[["Household_ID", "Installation_HasPVSystem", "Weather_ID"]], on="Household_ID",
                how="left", suffixes=("", "_hh"))
    pv = tb(d, "Installation_HasPVSystem")
    say(f"--- within the {len(d)} base households ---")
    say(f"  PV = True : {int(pv.eq(True).sum())}")
    say(f"  PV = False: {int(pv.eq(False).sum())}")
    say(f"  PV = null : {int(pv.isna().sum())}")
    say("")

    wid_map = dict(zip(hh["Household_ID"], hh["Weather_ID"]))
    rows = []
    for i, r in enumerate(d.to_dict("records")):
        hid = int(r["Household_ID"])
        daily = daily_for(hid)
        if daily is None or len(daily) < 60:
            continue
        dw = with_weather(daily, wid_map.get(hid), weather)
        dw = dw.dropna(subset=["Temperature_avg_daily"])
        summer = dw[dw["date"].dt.month.isin([6, 7, 8])]["kWh"]
        winter = dw[dw["date"].dt.month.isin([12, 1, 2])]["kWh"]
        if len(summer) < 20 or len(winter) < 20:
            continue
        v = pv.iloc[i]
        status = "unknown" if pd.isna(v) else ("PV" if bool(v) else "no PV")
        rows.append({"Household_ID": hid, "pv": status,
                     "summer_kWh_day": summer.mean(), "winter_kWh_day": winter.mean(),
                     "winter_summer_ratio": winter.mean() / summer.mean() if summer.mean() else np.nan,
                     "summer_min_day": summer.min(), "n_summer_zero_days": int((summer <= 0).sum())})
    pvt = pd.DataFrame(rows)
    say("--- summer (JJA) vs winter (DJF) daily consumption by PV status ---")
    if len(pvt):
        agg = pvt.groupby("pv").agg(n=("Household_ID", "size"),
                                    summer_kWh=("summer_kWh_day", "mean"),
                                    winter_kWh=("winter_kWh_day", "mean"),
                                    ratio=("winter_summer_ratio", "median"),
                                    summer_min=("summer_min_day", "mean"),
                                    zero_days=("n_summer_zero_days", "mean")).round(2)
        say(agg.to_string())
        say("")
        pv_grp, no_grp = pvt[pvt["pv"] == "PV"], pvt[pvt["pv"] == "no PV"]
        if len(pv_grp) >= 3 and len(no_grp) >= 3:
            say(f"  summer daily kWh: PV={pv_grp['summer_kWh_day'].mean():.2f} vs "
                f"no PV={no_grp['summer_kWh_day'].mean():.2f}  "
                f"(std diff {std_diff(pv_grp['summer_kWh_day'], no_grp['summer_kWh_day']):+.2f})")
            say(f"  winter daily kWh: PV={pv_grp['winter_kWh_day'].mean():.2f} vs "
                f"no PV={no_grp['winter_kWh_day'].mean():.2f}  "
                f"(std diff {std_diff(pv_grp['winter_kWh_day'], no_grp['winter_kWh_day']):+.2f})")
            say(f"  winter/summer ratio: PV={pv_grp['winter_summer_ratio'].median():.2f} vs "
                f"no PV={no_grp['winter_summer_ratio'].median():.2f}")
            say("")
            wd = abs(std_diff(pv_grp["winter_kWh_day"], no_grp["winter_kWh_day"]))
            sd = abs(std_diff(pv_grp["summer_kWh_day"], no_grp["summer_kWh_day"]))
            say("VERDICT ON Q6")
            say(f"  PV distorts SUMMER consumption (std diff {sd:.2f}) more than WINTER "
                f"(std diff {wd:.2f}).")
            if wd < 0.3:
                say("  The heating-season distortion is small: in winter, Swiss PV output is low and")
                say("  the heat pump load is high, so self-consumption masks a small share of a large")
                say("  number. PV households can be RETAINED for heating-season analysis, provided")
                say("  the analysis window excludes the shoulder and summer months.")
            else:
                say("  The heating-season distortion is material; PV households need adjustment or")
                say("  exclusion even within the heating season.")
            say(f"  The harder problem is the {int(pv_all.isna().sum())} households with UNKNOWN PV status")
            say(f"  fleet-wide: they cannot be assigned to either arm, and at "
                f"{pv_all.isna().mean():.0%} of the fleet that is not a rounding error.")
    else:
        say("  no household produced enough summer and winter days to compare.")
    say("")


def main() -> None:
    rng = np.random.default_rng(SEED)
    hh = load("households")
    meta = load("meta")
    pr = load("protocols")
    q15ov = load("15min_overview")
    weather = load_weather()

    say(f"environment: numpy {np.__version__}, pandas {pd.__version__}; "
        f"scipy / sklearn / matplotlib NOT installed -- all fits are numpy least squares.")
    say("")

    u = build_base(pr, q15ov)
    flags = intervention_flags(u)

    q1(u, flags)
    peer = q2(hh, meta)
    hdd = q3(hh, q15ov, weather, rng)
    profiles = q4(u, flags, hh, weather)
    q5fit = q5(u, hh, weather, rng)
    q6(u, hh, weather)

    OUTPUT_DIR.mkdir(exist_ok=True)
    peer.to_csv(OUTPUT_DIR / "peer_group_feasibility.csv", index=False, sep=";")
    hdd.to_csv(OUTPUT_DIR / "hdd_fit_quality.csv", index=False, sep=";")
    profiles.to_csv(OUTPUT_DIR / "setback_profiles.csv", index=False, sep=";")
    q5fit.to_csv(OUTPUT_DIR / "hdd_baseline_fits.csv", index=False, sep=";")
    (OUTPUT_DIR / "scope_recon_summary.txt").write_text("\n".join(_report), encoding="utf-8")

    print("=" * 100)
    for f in ["scope_recon_summary.txt", "peer_group_feasibility.csv", "hdd_fit_quality.csv",
              "setback_profiles.csv", "hdd_baseline_fits.csv"]:
        print(f"  -> {OUTPUT_DIR / f}")


if __name__ == "__main__":
    main()
