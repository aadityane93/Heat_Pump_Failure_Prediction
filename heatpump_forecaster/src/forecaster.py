"""The daily consumption forecaster: specification S2_K15, window W2_hl6.

    predicted = a + b1 * max(0, K - T) + b2 * max(0, T - K)

Two straight lines meeting at a fixed knot K = 15 C. `a` is the predicted
consumption on a K-degree day, `b1` the heating slope below the knot, `b2` the
slope above it. Units are kWh/day and kWh/day per degree.

K is a nuisance parameter of this specification. It is NOT the degree-day base,
which is 12 C (SIA 381/3). Two numbers, two roles; do not feed one into the
other.

Coefficients are refitted every calendar month by weighted least squares on an
expanding window, with a 6-month half-life:

    w = 2 ** (-age_in_months / 6)

No day is ever discarded; the oldest ones simply stop mattering. Train on every
month before M, predict month M, step forward. Twelve months of training are
required before the first fold, so the first scored fold is month_idx 12.

numpy and pandas only.
"""

import numpy as np
import pandas as pd

KNOT = 15.0
HALF_LIFE_MONTHS = 6
MIN_TRAIN_MONTHS = 12


def design(temp_c, knot=KNOT):
    """The two hinge columns. Returns (x1, x2) = below-knot, above-knot."""
    t = np.asarray(temp_c, dtype=float)
    return np.clip(knot - t, 0, None), np.clip(t - knot, 0, None)


def predict(a, b1, b2, temp_c, knot=KNOT):
    x1, x2 = design(temp_c, knot)
    return a + b1 * x1 + b2 * x2


def _solve_weighted(x1, x2, y, w):
    """Weighted least squares for [1, x1, x2] -> y. Returns (a, b1, b2)."""
    X = np.column_stack([np.ones(len(y)), x1, x2])
    sw = np.sqrt(w)
    try:
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan
    return float(beta[0]), float(beta[1]), float(beta[2])


def fit_expanding(df, value_col="kwh", temp_col="temp_c", date_col=None,
                  knot=KNOT, half_life_months=HALF_LIFE_MONTHS,
                  min_train_months=MIN_TRAIN_MONTHS):
    """Fit one coefficient triple per calendar month, forward chaining.

    `df` needs a date (index or `date_col`), a consumption column and a
    temperature column. Returns one row per fold with a, b1, b2 and the count
    of training days behind it.

    A test month never contributes to its own coefficients.
    """
    d = df.copy()
    if date_col is not None:
        d = d.set_index(date_col)
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    d = d.dropna(subset=[value_col, temp_col])

    d["_ym"] = d.index.year * 12 + d.index.month   # matches the panel convention
    months = np.sort(d["_ym"].unique())
    if len(months) <= min_train_months:
        raise ValueError("need more than %d calendar months, got %d"
                         % (min_train_months, len(months)))

    x1_all, x2_all = design(d[temp_col], knot)
    y_all = d[value_col].to_numpy(float)
    ym_all = d["_ym"].to_numpy()

    rows = []
    for idx, m in enumerate(months[min_train_months:], start=min_train_months):
        mask = ym_all < m
        if mask.sum() < 30:
            continue
        # age measured from the month being predicted, so the most recent
        # training month carries the most weight
        age = (m - ym_all[mask]).astype(float)
        w = 2.0 ** (-age / half_life_months)
        a, b1, b2 = _solve_weighted(x1_all[mask], x2_all[mask], y_all[mask], w)
        rows.append({"month_idx": idx, "ym": int(m), "a": a, "b1": b1, "b2": b2,
                     "n_prior_days": int(mask.sum())})
    return pd.DataFrame(rows)


def apply_coefficients(df, coefs, value_col="kwh", temp_col="temp_c",
                       date_col=None, knot=KNOT):
    """Join each day to its own fold's coefficients and evaluate.

    This is a query, not a fit. Days before the first fold get NaN, which is
    correct: no model covered them yet.
    """
    d = df.copy()
    if date_col is not None:
        d = d.set_index(date_col)
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    d["ym"] = d.index.year * 12 + d.index.month

    out = d.merge(coefs[["ym", "a", "b1", "b2"]], on="ym", how="left")
    out.index = d.index
    out["predicted"] = predict(out["a"], out["b1"], out["b2"], out[temp_col], knot)
    out["deviation"] = out["predicted"] - out[value_col]
    # Undefined stays NaN. A zero deviation would read as a perfect prediction.
    with np.errstate(divide="ignore", invalid="ignore"):
        out["deviation_pct"] = np.where(out[value_col] > 0,
                                        100 * out["deviation"] / out[value_col],
                                        np.nan)
    return out


def cvrmse(actual, predicted):
    """RMSE as a percent of the mean. Comparable across series of any size."""
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    ok = np.isfinite(a) & np.isfinite(p)
    if not ok.any():
        return np.nan
    return 100 * np.sqrt(((p[ok] - a[ok]) ** 2).mean()) / a[ok].mean()


def nmbe(actual, predicted):
    """Normalised mean bias error. POSITIVE means over-prediction."""
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    ok = np.isfinite(a) & np.isfinite(p)
    if not ok.any():
        return np.nan
    return 100 * (p[ok] - a[ok]).sum() / a[ok].sum()


def score_by_fold(applied, value_col="kwh"):
    """Per-fold CV(RMSE) and NMBE, plus the medians the acceptance bar uses."""
    d = applied.dropna(subset=["predicted"]).copy()
    rows = []
    for ym, g in d.groupby("ym"):
        rows.append({"ym": int(ym), "n_days": len(g),
                     "mean_actual": g[value_col].mean(),
                     "cvrmse": cvrmse(g[value_col], g["predicted"]),
                     "nmbe": nmbe(g[value_col], g["predicted"])})
    folds = pd.DataFrame(rows)
    return {"folds": folds,
            "median_cvrmse": folds["cvrmse"].median() if len(folds) else np.nan,
            "median_abs_nmbe": folds["nmbe"].abs().median() if len(folds) else np.nan}


def noise_floor(df, value_col="kwh", temp_col="temp_c", date_col=None,
                temp_bin=0.5):
    """Empirical bound on what any weather-and-calendar model can achieve.

    Two days in the same month at the same temperature differ only by what the
    model cannot see. sigma = SD(differences) / sqrt(2), since each day carries
    independent noise. Returned as a CV(RMSE) percentage.

    Read every score against this, not against zero. Measured on the source
    archive it is 20-22% fleet-wide, and strongly seasonal: about 15% below
    0 C, rising to 29% between 15 and 20 C.
    """
    d = df.copy()
    if date_col is not None:
        d = d.set_index(date_col)
    d.index = pd.to_datetime(d.index)
    d = d.dropna(subset=[value_col, temp_col])
    d["_moy"] = d.index.month
    d["_bin"] = (d[temp_col] / temp_bin).round() * temp_bin

    diffs = []
    for _, cell in d.groupby(["_moy", "_bin"]):
        v = cell[value_col].to_numpy(float)
        for i in range(len(v) - 1):
            diffs.append(v[i + 1] - v[i])
    if len(diffs) < 30:
        return np.nan
    sigma = np.std(diffs, ddof=1) / np.sqrt(2)
    return 100 * sigma / d[value_col].mean()
