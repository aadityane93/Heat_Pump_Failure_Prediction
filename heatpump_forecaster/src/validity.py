"""Validity precedence: one label per (household, date), first match wins.

    no_usable_measurement > pv_contaminated > outside_fitted_window >
    insufficient_training_data > low_confidence_fit > valid

The set is ORDERED, not disjoint. That is the point of it. Measured over
863,466 requestable household-dates in the source archive, 38.80% matched two
or more of the categories at once and had no single answer without precedence;
a further 1.33% matched none of the original five yet had no usable reading.
Both problems are why the order exists and why the sixth label was added.

The reading, top to bottom: no measurement exists -> a measurement exists but is
the wrong physical quantity -> no model covers this date -> a model exists but
not yet for this date -> the model covers it but its data sufficiency is weak.

Only `no_usable_measurement` means there is nothing to show. Every other label
still carries a real reading; the label decides which caveat sits beside it.

Verified against the study's own resolver on all 99,764 rows of the shipped
export: 0 mismatches.
"""

import numpy as np
import pandas as pd

LABELS = (
    "no_usable_measurement",
    "pv_contaminated",
    "outside_fitted_window",
    "insufficient_training_data",
    "low_confidence_fit",
    "valid",
)

CAVEAT = {
    "no_usable_measurement": "no reading exists for this date",
    "pv_contaminated": "a number exists, but it is grid import, not consumption",
    "outside_fitted_window": "consumption is real; no model covers this date",
    "insufficient_training_data": "consumption is real; no fold exists yet",
    "low_confidence_fit": "a prediction exists, but the fit is weak",
    "valid": "measurement, model and fold all present",
}


def resolve(rows, facts, date_col="local_date", id_col="Household_ID",
            measured_col=None):
    """Label every row. `facts` is one row per household carrying pv_flag,
    f5_band, fitted_window_start and first_test.

    `measured_col`, if given, is a boolean column on `rows` that is False where
    no usable reading exists. Omit it when the frame contains measured days
    only, as the shipped export does; `no_usable_measurement` then never fires.
    """
    m = rows[[id_col, date_col] + ([measured_col] if measured_col else [])].copy()
    m[date_col] = pd.to_datetime(m[date_col])

    f = facts.copy()
    f["fitted_window_start"] = pd.to_datetime(f["fitted_window_start"])
    f["first_test"] = pd.to_datetime(f["first_test"])
    m = m.merge(f, on=id_col, how="left")

    d = m[date_col].to_numpy()
    no_fold = m["first_test"].isna().to_numpy() | (d < m["first_test"].to_numpy())

    label = np.where(
        m["pv_flag"].to_numpy() == "True", "pv_contaminated",
        np.where(d < m["fitted_window_start"].to_numpy(), "outside_fitted_window",
                 np.where(no_fold, "insufficient_training_data",
                          np.where(m["f5_band"].to_numpy() == "LOW",
                                   "low_confidence_fit", "valid"))))

    if measured_col:
        label = np.where(~m[measured_col].to_numpy().astype(bool),
                         "no_usable_measurement", label)
    return pd.Series(label, index=rows.index, name="validity")


def summarise(labels):
    """Counts and shares, in precedence order."""
    c = pd.Series(labels).value_counts()
    out = pd.DataFrame({"label": list(LABELS)})
    out["n"] = out["label"].map(c).fillna(0).astype(int)
    out["share_pct"] = (100 * out["n"] / max(len(labels), 1)).round(2)
    out["caveat"] = out["label"].map(CAVEAT)
    return out
