"""
Adds peer-relative versions of the dashboard's existing features, so the
Fault Prediction model gets the DHW-production-method and EV-ownership
confounds as something it can actually use, instead of two raw columns it
has to statistically rediscover on ~400 households.

Drop this next to pipeline.py / heapo_core.py in the dashboard repo and call
add_peer_relative_features() on the same dataframe build_dataset() already
produces (one row per household, static + feature columns already merged) --
no new data loading required.

Why this exists (see the group message / evidence charts for the numbers):
  - Two clean households in the SAME stratum, differing only in DHW method,
    had non-heating baselines 3.6x apart (610768 vs 376757).
  - The SAME target household's heating-curve ratio moved from 0.80 to 0.58
    depending only on whether the comparison peer owned an EV (610768 vs
    706817) -- the target didn't change, the peer choice did.
  A model that sees hdd_slope / hdd_intercept / summer_baseload as raw
  numbers, and DHW-method / EV as two unrelated columns, has no way to know
  those confounds are this large. A peer-relative feature bakes the
  correction in directly.

Usage:
    from peer_relative_features import add_peer_relative_features
    dataset = add_peer_relative_features(dataset)
    # dataset now has e.g. hdd_slope_vs_peer, summer_baseload_vs_peer, ...
    # feed those into hc.feature_matrix(...) alongside (or instead of) the
    # raw versions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Column names as they exist in THIS repo's dataset.parquet (constants.py /
# STATIC_COLUMNS). Change these four if the underlying schema changes --
# nothing else in this file needs to know about column names.
COL_HP_TYPE = "HeatPump_Installation_Type"
COL_RADIATOR = "HeatDistribution_System_Radiators"
COL_FLOOR = "HeatDistribution_System_FloorHeating"
COL_DHW_BY_HP = "DHW_Production_ByHeatPump"
COL_HAS_EV = "Building_ElectricVehicle_Available"
COL_HOUSEHOLD_ID = "Household_ID"
COL_HAS_FAULT = "has_fault"  # excluded from peer pools when present; safe to omit

# The 4 existing dashboard features most directly affected by the two
# confounds measured above (baseline-type features hit by DHW method;
# slope-type features hit by EV ownership). Add more column names here if
# useful -- anything numeric and already in the dataset works.
DEFAULT_FEATURE_COLS = ["hdd_slope", "hdd_intercept", "summer_baseload", "mild_weather_use"]


def assign_stratum(df: pd.DataFrame) -> pd.Series:
    """Same definition already used in fault_detection.py's assign_stratum():
    heat-pump type + distribution system. 'unknown' when distribution isn't
    recorded -- those households are excluded from peer averaging, not
    guessed into a stratum."""
    hp_type = df[COL_HP_TYPE]
    floor = df.get(COL_FLOOR)
    radiator = df.get(COL_RADIATOR)

    dist = pd.Series("unknown", index=df.index, dtype="object")
    both = (floor == True) & (radiator == True)  # noqa: E712
    dist[both] = "both"
    dist[(~both) & (floor == True)] = "floor"  # noqa: E712
    dist[(~both) & (floor != True) & (radiator == True)] = "radiator"  # noqa: E712

    stratum = hp_type.astype(str) + " + " + dist.astype(str)
    stratum[hp_type.isna() | (dist == "unknown")] = "unknown"
    return stratum


def add_peer_relative_features(
    dataset: pd.DataFrame,
    feature_cols: list[str] | None = None,
    min_peers: int = 3,
) -> pd.DataFrame:
    """
    For every household, finds its matched peer group (same stratum + same
    DHW production method + no EV, excluding itself and excluding any
    confirmed-fault peers when has_fault is present -- same rule as
    select_peer_group() in fault_detection.py), then adds one
    "<feature>_vs_peer" column per entry in feature_cols: the household's own
    value divided by its matched peers' mean value.

    A household with fewer than min_peers matched peers gets NaN in the new
    columns rather than a ratio computed on a handful of noisy peers -- see
    the printed coverage summary to know how often that happens on the real
    data. This mirrors pooled_peer_baseline()'s stance in fault_detection.py:
    an unstable single-peer (or near-single-peer) ratio is worse than no
    ratio at all.

    Does not mutate the input; returns a new dataframe with the extra
    columns appended.
    """
    feature_cols = feature_cols or DEFAULT_FEATURE_COLS
    missing = [c for c in feature_cols if c not in dataset.columns]
    if missing:
        raise KeyError(
            f"add_peer_relative_features: {missing} not found in the dataset -- "
            f"check they've been computed upstream (extract_features / build_dataset) "
            f"before calling this."
        )

    df = dataset.copy()
    df["_stratum"] = assign_stratum(df)

    dhw_col = COL_DHW_BY_HP if COL_DHW_BY_HP in df.columns else None
    ev_col = COL_HAS_EV if COL_HAS_EV in df.columns else None
    fault_col = COL_HAS_FAULT if COL_HAS_FAULT in df.columns else None

    if dhw_col is None:
        print(f"add_peer_relative_features: '{COL_DHW_BY_HP}' not found -- "
              f"peer groups will NOT be filtered by DHW method. Numbers will "
              f"be less reliable than the 3.6x-confound evidence suggests they should be.")
    if ev_col is None:
        print(f"add_peer_relative_features: '{COL_HAS_EV}' not found -- "
              f"peer groups will NOT exclude EV owners. Numbers will be less "
              f"reliable than the 0.80-vs-0.58 confound evidence suggests they should be.")

    n_total = len(df)

    for col in feature_cols:
        df[f"{col}_vs_peer"] = np.nan

    # Group by the criteria that actually define a valid peer pool, so this
    # is one groupby + merge per criterion combination rather than an
    # O(n^2) household-by-household loop -- matters once this runs on the
    # full ~1400-household population, not just the ~410 visited ones.
    group_keys = ["_stratum"]
    if dhw_col:
        group_keys.append(dhw_col)

    for keys, group in df.groupby(group_keys, dropna=False):
        if isinstance(keys, tuple):
            stratum_val = keys[0]
        else:
            stratum_val = keys
        if stratum_val == "unknown":
            continue

        pool = group
        if ev_col:
            pool = pool[pool[ev_col] != True]  # noqa: E712
        if fault_col:
            pool = pool[pool[fault_col] != 1]

        if len(pool) < min_peers + 1:  # +1 because a household isn't its own peer
            continue

        # every household in this group (not just the pool) can use the
        # pool's average, as long as it itself also qualifies for the pool
        # criteria (EV-free, non-fault) -- otherwise it's not being fairly
        # compared to begin with, so leave it NaN.
        eligible = group.index
        if ev_col:
            eligible = group.index[(group[ev_col] != True).values]  # noqa: E712
        if fault_col:
            eligible = eligible[(group.loc[eligible, fault_col] != 1).values]

        for col in feature_cols:
            # NaN-aware peer values in this pool -- most households in the
            # full population won't have this feature computed yet (no
            # smart-meter file pulled), and they must not silently count as
            # zeros or inflate the "peer count" while contributing nothing
            # to the sum. dropna() keeps sum/count/mean consistent with
            # each other by construction.
            valid_peer_vals = pool[col].dropna()
            for idx in eligible:
                own_val = df.loc[idx, col]
                if pd.isna(own_val):
                    continue
                others = valid_peer_vals.drop(index=idx, errors="ignore")
                if len(others) < min_peers:
                    continue
                peer_mean = others.mean()
                if peer_mean:
                    df.loc[idx, f"{col}_vs_peer"] = own_val / peer_mean

    df = df.drop(columns=["_stratum"])
    new_cols = [f"{c}_vs_peer" for c in feature_cols]
    coverage = df[new_cols[0]].notna().sum() if new_cols else 0
    print(f"add_peer_relative_features: {coverage}/{n_total} households got a "
          f"peer-matched comparison (min_peers={min_peers}). The rest are NaN in "
          f"the new columns -- too few valid peers to trust a ratio, not an error.")
    return df
