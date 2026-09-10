from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from constants import (
    CATEGORICAL_FAULT_COLUMN,
    CATEGORICAL_FAULT_FLAG,
    DIRECT_FAULT_COLUMNS,
    INVERTED_FAULT_COLUMNS,
)


def build_fault_flags(protocols: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per fault indicator. True always means 'fault'.

    Some protocol columns are phrased the other way round (Okay=True means
    no fault), so those are inverted here and get an `_inv` suffix.
    """
    flags = pd.DataFrame(index=protocols.index)

    for col in DIRECT_FAULT_COLUMNS:
        if col in protocols.columns:
            flags[col] = protocols[col].fillna(False).astype(bool)

    for col in INVERTED_FAULT_COLUMNS:
        if col in protocols.columns:
            flags[col + "_inv"] = protocols[col] == False  # noqa: E712

    if CATEGORICAL_FAULT_COLUMN in protocols.columns:
        cat = protocols[CATEGORICAL_FAULT_COLUMN]
        flags[CATEGORICAL_FAULT_FLAG] = cat.notna() & (
            cat.astype(str).str.lower() != "okay"
        )

    return flags


def build_labels(protocols_first_visit: pd.DataFrame) -> pd.DataFrame:
    """Household-level label table: per-fault flags, fault count, any-fault."""
    flags = build_fault_flags(protocols_first_visit)
    labels = pd.DataFrame(
        {
            "Household_ID": protocols_first_visit["Household_ID"].values,
            "Visit_Date": protocols_first_visit["Visit_Date"].values,
            "n_faults": flags.sum(axis=1).values,
            "has_fault": (flags.sum(axis=1) > 0).astype(int).values,
        }
    )
    return pd.concat([labels, flags.astype(int).reset_index(drop=True)], axis=1)


def _col(group: pd.DataFrame, name: str) -> pd.Series:
    """Return a column, or an all-NaN series if the dataset lacks it."""
    if name in group.columns:
        return group[name]
    return pd.Series(np.nan, index=group.index, dtype="float64")


def extract_features(g: pd.DataFrame) -> pd.Series:
    """Turn one household's daily pre-visit history into a single feature row."""
    f: dict[str, float] = {}
    hp = _col(g, "kWh_received_HeatPump")
    tot = _col(g, "kWh_received_Total")
    hdd = _col(g, "HeatingDegree_SIA_daily")
    temp = _col(g, "Temperature_avg_daily")

    # Prefer the heat pump submeter; fall back to whole-household consumption.
    use_hp = hp.notna().mean() > 0.5
    energy = hp if use_hp else tot
    f["used_hp_submeter"] = int(use_hp)

    # Consumption level
    f["e_mean"] = energy.mean()
    f["e_std"] = energy.std()
    f["e_max"] = energy.max()
    f["e_cv"] = energy.std() / energy.mean() if energy.mean() else np.nan
    f["e_p90"] = energy.quantile(0.90)
    f["e_zero_frac"] = (energy.fillna(0) < 0.5).mean()
    f["hp_share"] = (
        (hp.sum() / tot.sum()) if tot.sum() and hp.notna().any() else np.nan
    )

    # Weather normalisation: energy against heating degree days
    m = energy.notna() & hdd.notna()
    if m.sum() > 30:
        slope, intercept, r, _, _ = stats.linregress(hdd[m], energy[m])
        f["hdd_slope"], f["hdd_intercept"], f["hdd_r2"] = slope, intercept, r**2
    else:
        f["hdd_slope"] = f["hdd_intercept"] = f["hdd_r2"] = np.nan
    f["e_per_hdd"] = energy.sum() / hdd.sum() if hdd.sum() else np.nan

    # Seasonal behaviour
    summer = energy[hdd.fillna(0) <= 0.1]
    winter = energy[temp < 5]
    f["summer_baseload"] = summer.mean() if len(summer) > 5 else np.nan
    f["winter_mean"] = winter.mean() if len(winter) > 5 else np.nan
    f["winter_summer_ratio"] = (
        f["winter_mean"] / f["summer_baseload"] if f["summer_baseload"] else np.nan
    )

    # Heating while it is mild outside points at a heating limit set too high
    mild = energy[(temp >= 15) & (temp < 20)]
    f["mild_weather_use"] = mild.mean() if len(mild) > 5 else np.nan
    f["mild_vs_summer"] = (
        f["mild_weather_use"] / f["summer_baseload"] if f["summer_baseload"] else np.nan
    )

    # Day-to-day pattern
    f["e_diff_std"] = energy.diff().abs().mean()
    weekday = g["Timestamp"].dt.dayofweek
    weekday_mean = energy[weekday < 5].mean()
    f["weekend_ratio"] = (
        energy[weekday >= 5].mean() / weekday_mean if weekday_mean else np.nan
    )

    # Context
    f["n_days"] = len(g)
    f["mean_temp"] = temp.mean()
    f["mean_hdd"] = hdd.mean()
    f["sunshine_mean"] = _col(g, "Sunshine_duration_daily").mean()
    return pd.Series(f)


def build_feature_table(df_pre: pd.DataFrame) -> pd.DataFrame:
    """Apply `extract_features` to every household in the pre-visit frame."""
    rows = []
    for household_id, group in df_pre.sort_values("Timestamp").groupby(
        "Household_ID", observed=True
    ):
        row = extract_features(group)
        row["Household_ID"] = household_id
        rows.append(row)
    features = pd.DataFrame(rows).reset_index(drop=True)
    cols = ["Household_ID"] + [c for c in features.columns if c != "Household_ID"]
    return features[cols]
