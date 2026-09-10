from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ID_COLS = ["Household_ID", "has_fault", "n_faults"]

# The four fault types with enough positives to model individually.
DEFAULT_FAULT_TARGETS = [
    "HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit",
    "HeatPump_NightSetbackSetting_Activated_BeforeVisit",
    "HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit",
    "DHW_Storage_LastDescaling_TooLongAgo",
]

PRETTY_FAULT_NAMES = {
    "HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit": "Heating curve set too high",
    "HeatPump_NightSetbackSetting_Activated_BeforeVisit": "Night setback active",
    "HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit": "Heating limit set too high",
    "DHW_Storage_LastDescaling_TooLongAgo": "Hot water tank overdue descaling",
    "HeatPump_AirSource_AirDuctsCleaningRequired": "Air ducts need cleaning",
    "HeatPump_TechnicallyOkay_inv": "Not technically okay",
    "HeatPump_BasicFunctionsOkay_inv": "Basic functions not okay",
    "HeatPump_Installation_CorrectlyPlanned_inv": "Installation badly planned",
    "HeatPump_AirSource_AirDuctsDistanceOkay_inv": "Air duct distance wrong",
    "HeatPump_AirSource_AirDuctsFree_inv": "Air ducts blocked",
    "HeatPump_AirSource_AirDuctsDrainOkay_inv": "Air duct drain faulty",
    "HeatPump_AirSource_EvaporatorClean_inv": "Evaporator dirty",
    "ExpansionTank_fault": "Expansion tank pressure off",
}


def pretty_fault(name: str) -> str:
    return PRETTY_FAULT_NAMES.get(name, name.replace("_BeforeVisit", "").replace("_", " "))


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

def _extract_one(g: pd.DataFrame) -> pd.Series:
    """Turn one household's daily series into a single feature row."""
    f = {}
    hp = g["kWh_received_HeatPump"]
    tot = g["kWh_received_Total"]
    hdd = g["HeatingDegree_SIA_daily"]
    temp = g["Temperature_avg_daily"]

    # Prefer the heat pump submeter when it exists, otherwise whole-house total.
    use_hp = hp.notna().mean() > 0.5
    energy = hp if use_hp else tot
    f["used_hp_submeter"] = int(use_hp)

    f["e_mean"] = energy.mean()
    f["e_std"] = energy.std()
    f["e_max"] = energy.max()
    f["e_cv"] = energy.std() / energy.mean() if energy.mean() else np.nan
    f["e_p90"] = energy.quantile(0.90)
    f["e_zero_frac"] = (energy.fillna(0) < 0.5).mean()

    f["hp_share"] = (hp.sum() / tot.sum()) if tot.sum() and hp.notna().any() else np.nan

    # Energy against heating degree days: slope is roughly the building's heat loss.
    m = energy.notna() & hdd.notna()
    if m.sum() > 30:
        sl, ic, r, _, _ = stats.linregress(hdd[m], energy[m])
        f["hdd_slope"], f["hdd_intercept"], f["hdd_r2"] = sl, ic, r ** 2
    else:
        f["hdd_slope"] = f["hdd_intercept"] = f["hdd_r2"] = np.nan

    f["e_per_hdd"] = energy.sum() / hdd.sum() if hdd.sum() else np.nan

    # No heating demand -> whatever is left is mostly hot water.
    summer = energy[hdd.fillna(0) <= 0.1]
    winter = energy[temp < 5]
    f["summer_baseload"] = summer.mean() if len(summer) > 5 else np.nan
    f["winter_mean"] = winter.mean() if len(winter) > 5 else np.nan
    f["winter_summer_ratio"] = (
        f["winter_mean"] / f["summer_baseload"] if f["summer_baseload"] else np.nan
    )

    # Still heating at 15-20 C suggests the heating limit is set too high.
    mild = energy[(temp >= 15) & (temp < 20)]
    f["mild_weather_use"] = mild.mean() if len(mild) > 5 else np.nan
    f["mild_vs_summer"] = (
        f["mild_weather_use"] / f["summer_baseload"] if f["summer_baseload"] else np.nan
    )

    f["e_diff_std"] = energy.diff().abs().mean()

    wd = g["Timestamp"].dt.dayofweek
    weekday_mean = energy[wd < 5].mean()
    f["weekend_ratio"] = energy[wd >= 5].mean() / weekday_mean if weekday_mean else np.nan

    f["n_days"] = len(g)
    f["mean_temp"] = temp.mean()
    f["mean_hdd"] = hdd.mean()
    f["sunshine_mean"] = g["Sunshine_duration_daily"].mean()
    return pd.Series(f)


def extract_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    """One feature row per household, from pre-visit daily readings."""
    grouped = df_daily.sort_values("Timestamp").groupby("Household_ID", observed=True)
    try:
        out = grouped.apply(_extract_one, include_groups=False)
    except TypeError:  # pandas < 2.2
        out = grouped.apply(_extract_one)
    return out.reset_index()


def build_dataset(
    features: pd.DataFrame,
    static: pd.DataFrame,
    labels: pd.DataFrame,
    min_days: int = 60,
) -> pd.DataFrame:
    """Features + protocol static fields + labels, one row per household."""
    label_cols = [c for c in labels.columns if c != "Visit_Date"]
    data = (
        features.merge(static, on="Household_ID", how="left")
        .merge(labels[label_cols], on="Household_ID", how="left")
    )
    return data[data["n_days"] >= min_days].copy()


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def make_models(seed: int = 42) -> dict:
    return {
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gbm": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=seed
        ),
    }


def split_columns(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def as_object(df):
    """Cast a categorical block to object dtype, keeping NaN as NaN.

    Several protocol fields land as pandas `bool` dtype, and SimpleImputer's
    most_frequent strategy rejects a bool block ("cannot use most_frequent
    strategy with non-numeric data"). Module level rather than a lambda so the
    pipeline stays picklable for n_jobs=-1.
    """
    return pd.DataFrame(df).astype(object)


def make_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    # sparse_threshold=0 forces a dense matrix. HistGradientBoosting rejects
    # sparse input, and whether the default threshold trips depends on how many
    # categorical columns survive the current filter, so pin it.
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("to_object", FunctionTransformer(as_object, feature_names_out="one-to-one")),
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                cat_cols,
            ),
        ],
        sparse_threshold=0,
    )


def make_pipeline(X: pd.DataFrame, model):
    num_cols, cat_cols = split_columns(X)
    return Pipeline([("prep", make_preprocessor(num_cols, cat_cols)), ("clf", model)])


def feature_matrix(data: pd.DataFrame, target: str) -> pd.DataFrame:
    drop = set(ID_COLS) | {target}
    # Every per-fault flag is part of the label, never an input.
    drop |= {c for c in data.columns if c in PRETTY_FAULT_NAMES}
    return data.drop(columns=[c for c in drop if c in data.columns])


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate_binary(y, proba, threshold: float = 0.5) -> dict:
    """Scores for one set of predictions.

    Read accuracy against `base rate`, which is what predicting "fault" for
    every household would score. ROC-AUC is left out: on a set this imbalanced
    it is dominated by the majority class.
    """
    y = np.asarray(y)
    pred = (proba > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "PR-AUC": average_precision_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
        "Accuracy": float((pred == y).mean()),
        "base rate": float(y.mean()),
        "Precision (fault)": tp / (tp + fp) if tp + fp else np.nan,
        "Recall (fault)": tp / (tp + fn) if tp + fn else np.nan,
    }
