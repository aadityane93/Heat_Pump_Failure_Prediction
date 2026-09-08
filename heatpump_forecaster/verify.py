"""Self-check. Runs against the shipped data only; needs no other artefact.

    python verify.py

Every check should print 0 disagreements and FAILURES: 0. If one does not,
something in the package has drifted from the study it came from and the
numbers should not be trusted.

Coefficient differences of order 1e-7 are float32 storage precision in the
shipped parquet, not a logic difference; the tolerance is set accordingly.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import forecaster as F
import validity as V

DATA = os.path.join(HERE, "data")
ex = pd.read_parquet(os.path.join(DATA, "f6_dashboard_export_DRAFT.parquet"))
ex["local_date"] = pd.to_datetime(ex["local_date"])
facts = pd.read_parquet(os.path.join(DATA, "household_facts.parquet"))
coefs = pd.read_parquet(os.path.join(DATA, "w2_hl6_coefficients.parquet"))

failures = 0

# 1 --------------------------------------------------------------------------
print("1. predict() reproduces the export's stored predicted_kwh")
j = ex.merge(coefs[["Household_ID", "month_idx", "a", "b1", "b2"]],
             on=["Household_ID", "month_idx"], how="left")
mine = F.predict(j["a"], j["b1"], j["b2"], j["temp_mean"])
both = j["predicted_kwh"].notna() & pd.Series(mine).notna()
maxdiff = float(np.abs(mine[both] - j["predicted_kwh"][both]).max())
nulls = int((j["predicted_kwh"].isna() != pd.Series(mine).isna()).sum())
print("   rows with a stored prediction : %s" % format(int(both.sum()), ","))
print("   max abs difference            : %.3e kWh" % maxdiff)
print("   null pattern disagreements    : %d" % nulls)
failures += (maxdiff > 1e-9) + (nulls != 0)

# 2 --------------------------------------------------------------------------
print()
print("2. fit_expanding() reproduces the stored coefficients from this data")
rows = []
for hid in sorted(ex["Household_ID"].unique()):
    g = ex[(ex["Household_ID"] == hid) & ex["in_training_window"]]
    g = g[["local_date", "actual_kwh", "temp_mean"]]
    if len(g) < 400:
        continue
    try:
        m = F.fit_expanding(g, value_col="actual_kwh", temp_col="temp_mean",
                            date_col="local_date")
    except ValueError:
        continue
    k = m.merge(coefs[coefs["Household_ID"] == hid][["ym", "a", "b1", "b2"]],
                on="ym", suffixes=("_mine", "_ref"))
    if len(k):
        rows.append(max(float((k["a_mine"] - k["a_ref"]).abs().max()),
                        float((k["b1_mine"] - k["b1_ref"]).abs().max()),
                        float((k["b2_mine"] - k["b2_ref"]).abs().max())))
print("   households refitted           : %d" % len(rows))
print("   worst coefficient difference  : %.3e" % (max(rows) if rows else float("nan")))
# 1e-5 kWh, comfortably above float32 storage precision in the parquet and
# far below anything that could change a reading
failures += (max(rows) > 1e-5) if rows else 1

# 3 --------------------------------------------------------------------------
print()
print("3. validity ladder reproduces the recorded label counts")
labels = V.resolve(ex, facts)
got = V.summarise(labels.to_numpy())
expected = {"no_usable_measurement": 0, "pv_contaminated": 17032,
            "outside_fitted_window": 36537, "insufficient_training_data": 26129,
            "low_confidence_fit": 2755, "valid": 17311}
got["expected"] = got["label"].map(expected)
got["delta"] = got["n"] - got["expected"]
print(got[["label", "n", "expected", "delta", "share_pct"]].to_string(index=False))
bad = int((got["delta"] != 0).sum())
print("   labels disagreeing            : %d" % bad)
failures += bad

print()
print("FAILURES:", failures)
sys.exit(1 if failures else 0)
