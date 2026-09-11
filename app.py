"""
HEAPO heat pump fault dashboard.

    streamlit run app.py

Expects the folder written by prepare_data.py to sit next to this file:

    heapo_dashboard/
        app.py
        heapo_core.py
        artifacts/
            dataset.parquet       features + statics + labels, one row per household
            daily.parquet         trimmed pre-visit daily series
            households.parquet
            protocols.parquet
            smd_overview.parquet
            models.joblib         the fitted models, from train_models.py

Parquet is what prepare_data.py writes when pyarrow is installed; it falls back
to `<stem>.csv.gz`, so both are read here.

The app fits nothing. It loads the models train_models.py already fitted and
uses them to score houses.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from bootstrap import ensure_artifacts
import heapo_core as hc
from constants import LABEL_COLUMNS, STATIC_COLUMNS

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="HEAPO heat pump faults", layout="wide")

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def read_artifact(stem: str) -> pd.DataFrame | None:
    """Read `<stem>.parquet`, or the CSV prepare_data.py falls back to."""
    parquet = DATA_DIR / f"{stem}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    csv = DATA_DIR / f"{stem}.csv.gz"
    if csv.exists():
        return pd.read_csv(csv)
    return None


def as_id(s: pd.Series) -> pd.Series:
    """Household IDs arrive as str, float or int depending on the table."""
    return pd.to_numeric(s, errors="coerce").astype("Int64")


@st.cache_data(show_spinner="Reading data files...")
def load_data():
    dataset = read_artifact("dataset")
    daily = read_artifact("daily")
    if dataset is None or daily is None:
        raise FileNotFoundError(
            f"dataset and daily are both required in {DATA_DIR}. "
            "Rerun prepare_data.py without --skip-daily."
        )

    dataset["Household_ID"] = as_id(dataset["Household_ID"])
    daily["Household_ID"] = as_id(daily["Household_ID"])
    daily["Timestamp"] = pd.to_datetime(daily["Timestamp"])

    # prepare_data.py merges features, statics and labels into one table; the
    # pages want the label and static blocks separately.
    labels = dataset[
        ["Household_ID", *[c for c in LABEL_COLUMNS if c in dataset.columns]]
    ].copy()
    static = dataset[
        ["Household_ID", *[c for c in STATIC_COLUMNS if c in dataset.columns]]
    ].copy()

    households = read_artifact("households")
    protocols = read_artifact("protocols")
    overview = read_artifact("smd_overview")
    for frame in (households, protocols, overview):
        if frame is not None:
            frame["Household_ID"] = as_id(frame["Household_ID"])

    return daily, labels, static, households, protocols, overview


@st.cache_data(show_spinner="Recomputing features for this time window...")
def features_for_window(start: str, end: str) -> pd.DataFrame:
    """Features for every household, using only readings inside the window."""
    daily, *_ = load_data()
    window = daily[(daily["Timestamp"] >= start) & (daily["Timestamp"] <= end)]
    if window.empty:
        return pd.DataFrame()
    return hc.extract_features(window)


def to_bool(s: pd.Series) -> pd.Series:
    return s.map({True: True, False: False, "True": True, "False": False})


def n_true(s: pd.Series) -> int:
    return int((to_bool(s) == True).sum())  # noqa: E712


def label_bars(ax, bars, fmt="{:.0f}"):
    ax.bar_label(bars, fmt=fmt, padding=2, fontsize=8)


# --------------------------------------------------------------------------
# Page 1: data overview
# --------------------------------------------------------------------------

def page_overview(daily, labels, households, protocols, overview):
    st.title("What is in the HEAPO data")
    st.write(
        "Everything below describes the raw dataset, before any modelling. "
        "It is referenced from the analysis notebook, and gives a sense of what is available to work with."
    )

    cov = households.merge(overview, on="Household_ID", how="left")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Households", f"{len(households):,}")
    c2.metric("With a field visit", f"{n_true(households['Protocols_Available']):,}")
    c3.metric("Labelled and pre-visit data", f"{daily['Household_ID'].nunique():,}")
    c4.metric("Faulty at first visit", f"{int(labels['has_fault'].sum())} / {len(labels)}")

    st.divider()

    st.subheader("Household coverage")
    st.caption("Treatment versus control, what data each household has, and its weather station.")
    fig = plt.figure(figsize=(12, 7))
    a_group = fig.add_subplot(2, 2, 1)
    a_flags = fig.add_subplot(2, 2, 2)
    a_weather = fig.add_subplot(2, 1, 2)

    g = households["Group"].value_counts()
    b = a_group.bar(g.index, g.values, color=["#2a78d6", "#eb6834"])
    label_bars(a_group, b)
    a_group.set_title("Households by group")
    a_group.set_ylabel("households")

    flags = [
        "Protocols_Available",
        "Protocols_HasMultipleVisits",
        "MetaData_Available",
        "SmartMeterData_Available_15min",
        "SmartMeterData_Available_Daily",
        "SmartMeterData_Available_Monthly",
        "Installation_HasPVSystem",
    ]
    flags = [c for c in flags if c in households.columns]
    b = a_flags.barh(flags, [n_true(households[c]) for c in flags], color="#1baf7a")
    label_bars(a_flags, b)
    a_flags.set_title(f"Flag = True (of {len(households)} households)")
    a_flags.invert_yaxis()

    w = households["Weather_ID"].value_counts()
    b = a_weather.bar(w.index.astype(str), w.values, color="#4a3aa7")
    label_bars(a_weather, b)
    a_weather.set_title("Households per weather station")
    a_weather.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Field visits")
    st.caption("When the visits happened, how many households were seen twice, and the heat pump mix.")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    y = protocols["Visit_Year"].value_counts().sort_index()
    b = ax[0].bar(y.index.astype(int).astype(str), y.values, color="#2a78d6")
    label_bars(ax[0], b)
    ax[0].set_title("Visits per year")

    v = protocols["Household_ID"].value_counts().value_counts().sort_index()
    b = ax[1].bar(v.index.astype(str), v.values, color="#eb6834")
    label_bars(ax[1], b)
    ax[1].set_title("Visits per household")
    ax[1].set_xlabel("number of visits")

    t = protocols["HeatPump_Installation_Type"].value_counts(dropna=False)
    b = ax[2].bar(t.index.astype(str), t.values, color="#1baf7a")
    label_bars(ax[2], b)
    ax[2].set_title("Heat pump type")
    ax[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Time coverage")
    st.caption("How many days of readings each household has, split around its visit date.")
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    ax[0].hist(overview["SMD_daily_TimeAvailable_NumberDays"].dropna(), bins=40, color="#2a78d6")
    ax[0].set_title("Days of daily data")
    ax[0].set_xlabel("days")

    d = overview["SMD_daily_TimeAvailable_DaysBeforeVisit"].dropna()
    ax[1].hist(d, bins=40, color="#eb6834")
    ax[1].axvline(180, color="red", ls="--", lw=1.2, label="180 days")
    ax[1].set_title(f"Days before visit  (n={len(d)})")
    ax[1].set_xlabel("days")
    ax[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Heat pump metering and calendar coverage")
    st.caption(
        "Only a minority of households meter the heat pump separately, which is why the "
        "feature extraction falls back to whole-house consumption."
    )
    fig = plt.figure(figsize=(13, 5))
    a1 = fig.add_subplot(1, 3, 1)
    a2 = fig.add_subplot(1, 3, (2, 3))

    hp = overview["SMD_daily_MeasurementsAvailable_HeatPump"].value_counts(dropna=False)
    b = a1.bar(hp.index.astype(str), hp.values, color=["#eb6834", "#2a78d6"])
    label_bars(a1, b)
    a1.set_title("Separate heat pump channel")

    tl = cov.dropna(subset=["SMD_daily_TimeAvailable_EarliestTimestamp"]).copy()
    tl["start"] = pd.to_datetime(
        tl["SMD_daily_TimeAvailable_EarliestTimestamp"], utc=True
    ).dt.tz_localize(None)
    tl["end"] = pd.to_datetime(
        tl["SMD_daily_TimeAvailable_LatestTimestamp"], utc=True
    ).dt.tz_localize(None)
    tl = tl.sort_values("start").reset_index(drop=True)

    cmap = {"treatment": "#eb6834", "control": "#2a78d6"}
    a2.hlines(
        tl.index, tl["start"], tl["end"],
        colors=tl["Group"].map(cmap).fillna("grey"), linewidth=0.6,
    )
    a2.set_title("Data coverage per household (sorted by start)")
    a2.set_ylabel("household (sorted)")
    a2.legend(handles=[plt.Line2D([0], [0], color=c, lw=3, label=k) for k, c in cmap.items()],
              loc="lower right")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("From 1408 households to a trainable sample")
    st.caption("Each filter is a requirement the modelling pipeline imposes.")
    s1 = cov
    s2 = s1[to_bool(s1["Protocols_Available"]) == True]  # noqa: E712
    s3 = s2[to_bool(s2["SmartMeterData_Available_Daily"]) == True]  # noqa: E712
    s4 = s3[s3["SMD_daily_TimeAvailable_DaysBeforeVisit"] >= 180]
    s5 = s4[to_bool(s4["SMD_daily_MeasurementsAvailable_HeatPump"]) == True]  # noqa: E712

    steps = [
        "all households", "+ has protocol", "+ has daily readings",
        "+ 180 days before visit", "+ heat pump metered",
    ]
    counts = [len(s1), len(s2), len(s3), len(s4), len(s5)]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    b = ax.barh(steps[::-1], counts[::-1], color="#2a78d6")
    ax.bar_label(b, padding=3, fontsize=9)
    ax.set_xlim(0, max(counts) * 1.15)
    ax.set_xlabel("households")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Fault types found at the first visit"):
        flag_cols = [c for c in labels.columns if c in hc.PRETTY_FAULT_NAMES]
        counts = (
            labels[flag_cols].sum().sort_values(ascending=False)
            .rename("households").to_frame()
        )
        counts.index = [hc.pretty_fault(i) for i in counts.index]
        counts["share of labelled"] = (counts["households"] / len(labels)).round(3)
        st.dataframe(counts, use_container_width=True)


# --------------------------------------------------------------------------
# Page 2: scoring
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the trained models...")
def load_models():
    """The bundle written by train_models.py. Nothing is fitted in the app."""
    path = DATA_DIR / "models.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def time_window(daily, label: str, hint: str | None = None):
    """Date-range slider. Returns the two strings the feature cache wants."""
    tmin = daily["Timestamp"].min().date()
    tmax = daily["Timestamp"].max().date()
    start, end = st.sidebar.slider(
        label, min_value=tmin, max_value=tmax, value=(tmin, tmax),
        format="MMM YYYY", help=hint,
    )
    return str(start), str(end) + " 23:59:59"


def scoring_controls(daily, bundle):
    """Sidebar widgets for the scoring page."""
    st.sidebar.subheader("Houses")
    all_ids = sorted(daily["Household_ID"].unique().tolist())
    held_out = sorted(h for h in bundle["test_ids"] if h in set(all_ids))

    scope = st.sidebar.radio(
        "Which houses",
        ["All houses", "Pick individual houses"],
        label_visibility="collapsed",
    )
    if scope == "All houses":
        chosen = all_ids
        st.sidebar.caption(f"All {len(all_ids)} houses, training ones included.")
    else:
        chosen = st.sidebar.multiselect(
            "Household number", all_ids, default=held_out[:5],
            help="Type a household number to search. This works down to a single house.",
        )

    st.sidebar.subheader("Time window")
    start, end = time_window(
        daily, " ",
        hint=(
            "Features are rebuilt from the readings inside this window. The models "
            "were fitted on each household's full pre-visit history, so a narrow "
            "window moves the inputs away from what they were trained on."
        ),
    )

    st.sidebar.subheader("Decision threshold")
    threshold = st.sidebar.slider(
        "Probability above which a house is called faulty", 0.05, 0.95, 0.5, 0.05,
        label_visibility="collapsed",
    )

    return dict(start=start, end=end, chosen=chosen, threshold=threshold, scope=scope)


def model_label(names):
    """Numbered display names for the picker: model 1 - logistic, and so on.

    The selectbox still carries the real name, which is what indexes the bundle.
    """
    return {name: f"model {i} - {name}" for i, name in enumerate(names, 1)}.get


def plot_confusion(cm, threshold):
    """A readable confusion matrix, rather than a four-cell table."""
    fig, ax = plt.subplots(figsize=(3.25, 2.6))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
    ax.grid(False)

    names = ["no fault", "fault"]
    ax.set_xticks([0, 1], [f"predicted\n{s}" for s in names], fontsize=7)
    ax.set_yticks([0, 1], [f"actually\n{s}" for s in names], fontsize=7)
    ax.set_title(f"Confusion matrix at a threshold of {threshold:.2f}", fontsize=8, pad=7)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", fontsize=17,
                    color="white" if cm[i, j] > cm.max() / 2 else "#22303C")
    plt.tight_layout()
    return fig


def page_scoring(daily, labels, static):
    st.title("Fault Prediction")
    bundle = load_models()
    if bundle is None:
        st.error(
            f"No trained models at `{DATA_DIR / 'models.joblib'}`. Fit them once with\n\n"
            "```\npython train_models.py --artifacts artifacts\n```"
        )
        return

    st.caption(
        "Would a technician find something wrong at this house? Pick the houses in the "
        f"sidebar and the model, already trained on {bundle['n_train']} other households, "
        "gives each one a probability of having a fault."
    )

    cfg = scoring_controls(daily, bundle)
    if not cfg["chosen"]:
        st.info("Pick at least one house in the sidebar.")
        return

    features = features_for_window(cfg["start"], cfg["end"])
    if features.empty:
        st.warning("No readings fall inside that time window. Widen the slider.")
        return

    # min_days=0: a house you asked about is scored on whatever history it has.
    full = hc.build_dataset(features, static, labels, min_days=0)
    data = full[full["Household_ID"].isin(cfg["chosen"])].dropna(subset=["has_fault"]).copy()
    if data.empty:
        st.warning("None of those houses have readings inside this time window.")
        return

    missing = [h for h in cfg["chosen"] if h not in set(data["Household_ID"])]
    if missing:
        shown = ", ".join(str(h) for h in missing[:10])
        st.info(
            f"{len(missing)} of the houses you picked have no readings inside this time "
            f"window and are not scored: {shown}" + (" ..." if len(missing) > 10 else "")
        )

    # Reindex to the exact columns the pipelines were fitted on, so a column that
    # moved or disappeared fails here rather than quietly changing a score.
    X = hc.feature_matrix(data, "has_fault").reindex(columns=bundle["feature_columns"])
    y = data["has_fault"].astype(int).to_numpy()

    c1, c2 = st.columns(2)
    c1.metric("Houses scored", len(data))
    c2.metric("Faulty", f"{int(y.sum())} ({y.mean():.0%})")

    tab_binary, tab_multi = st.tabs(["Binary", "Multi-label"])

    # ---------------- binary ----------------
    with tab_binary:
        st.subheader("Binary Prediction: Faulty or Not Faulty")

        names = list(bundle["binary"].keys())
        model_name = st.selectbox(
            "Model", names, index=len(names) - 1, key="model_binary",
            format_func=model_label(names),
            help="All of them were fitted on the same households by train_models.py.",
        )

        proba = bundle["binary"][model_name].predict_proba(X)[:, 1]
        pred = (proba > cfg["threshold"]).astype(int)
        scores = hc.evaluate_binary(y, proba, cfg["threshold"])

        # PR-AUC is the only one of the three that survives the imbalance:
        # precision is near the base rate by construction, and recall alone is
        # maximised by calling every house faulty.
        base = scores["base rate"]
        m1, m2 = st.columns(2)
        m1.metric(
            "PR-AUC",
            "n/a" if pd.isna(scores["PR-AUC"]) else f"{scores['PR-AUC']:.3f}",
            help=(
                "How well the model ranks faulty houses above healthy ones. Guessing "
                f"scores {base:.3f} here."
            ),
        )
        m2.metric(
            "Accuracy", f"{scores['Accuracy']:.3f}",
            help=f"Share of houses called right. Always saying fault would score {base:.3f}.",
        )

        if len(np.unique(y)) > 1:
            cm = confusion_matrix(y, pred, labels=[0, 1])
            # width=content keeps the figure at its own size; the default
            # stretches it back to the column width, undoing the figsize.
            st.pyplot(plot_confusion(cm, cfg["threshold"]), width="content")
            plt.close("all")
            st.caption(
                f"It catches {cm[1, 1]} of the {cm[1].sum()} faulty houses, and lets "
                f"{cm[0, 1]} of the {cm[0].sum()} healthy ones through as false alarms."
            )
        else:
            st.info(
                "Every house here has the same outcome, so PR-AUC and the confusion "
                "matrix are undefined. The predictions themselves are below."
            )

        st.write("**Per House Prediction**")
        per_house = pd.DataFrame({
            "Household ID": data["Household_ID"].values,
            "Days Used": data["n_days"].astype(int).values,
            "P(Fault)": np.round(proba, 3),
            "Predicted": np.where(pred == 1, "fault", "no fault"),
            "Actual": np.where(y == 1, "fault", "no fault"),
            "Correct Prediction": pred == y,
        })
        st.dataframe(per_house, use_container_width=True, hide_index=True)
        st.download_button(
            "Download these predictions",
            per_house.to_csv(index=False).encode(),
            file_name="binary_predictions.csv",
            mime="text/csv",
        )

    # ---------------- multi-label ----------------
    with tab_multi:
        st.subheader("Multi-label Prediction: Which Fault")

        names = list(bundle["binary"].keys())
        ml_model = st.selectbox("Model", names, index=len(names) - 1, key="model_multi",
                                format_func=model_label(names))

        fitted = {t: p for t, p in bundle["multilabel"].get(ml_model, {}).items()
                  if t in data.columns}
        if not fitted:
            st.info(f"No per-fault models were fitted for {ml_model}.")
            return

        rows, proba_cols = [], {}
        for target, pipe in fitted.items():
            y_t = data[target].astype(int).to_numpy()
            p_t = pipe.predict_proba(X)[:, 1]
            proba_cols[hc.pretty_fault(target)] = np.round(p_t, 3)
            s = hc.evaluate_binary(y_t, p_t, cfg["threshold"])
            rows.append({
                "Fault": hc.pretty_fault(target),
                "No. of Faulty households": int(y_t.sum()),
                "PR-AUC (score)": s["PR-AUC"],
            })

        st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)
        st.caption(
            "Compare each PR-AUC against how common that fault is among the "
            "{n} houses below.".format(n=len(data))
        )

        st.write("**Predicted probability per house and fault type**")
        table = pd.DataFrame(proba_cols, index=data["Household_ID"].values)
        table.index.name = "Household_ID"
        shown = st.radio(
            "Show", ["Predicted probability", "Probability next to what was found"],
            horizontal=True, label_visibility="collapsed",
        )
        if shown.startswith("Predicted"):
            st.dataframe(
                table.style.background_gradient(cmap="Oranges", vmin=0, vmax=1),
                use_container_width=True,
            )
        else:
            truth = (
                data.set_index("Household_ID")[list(fitted)]
                .rename(columns=hc.pretty_fault)
                .add_suffix(" (actual)")
            )
            st.dataframe(table.join(truth), use_container_width=True)

        st.download_button(
            "Download these predictions",
            table.reset_index().to_csv(index=False).encode(),
            file_name="multilabel_predictions.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------

def main():
    with st.status("Preparing HEAPO data...", expanded=True) as setup_status:
        try:
            status = ensure_artifacts(
                DATA_DIR,
                progress=lambda message: setup_status.update(label=message),
            )
            setup_status.update(label=status, state="complete")
        except Exception as exc:
            setup_status.update(label="HEAPO setup failed", state="error")
            st.error(f"Could not prepare the HEAPO data automatically: {exc}")
            st.info(
                "Set HEAPO_DATA_DIR to an already-extracted data folder, or build "
                "artifacts locally with prepare_data.py and train_models.py."
            )
            return
    st.sidebar.caption(status)

    daily, labels, static, households, protocols, overview = load_data()

    st.sidebar.title("HEAPO")
    pages = ["Data overview", "Fault Prediction"]
    # The overview page describes the raw study; without those tables there is
    # nothing to describe.
    if any(f is None for f in (households, protocols, overview)):
        pages.remove("Data overview")
        st.sidebar.caption(
            "The data overview page needs households, protocols and smd_overview "
            "in the artifacts folder."
        )
    page = st.sidebar.radio("Page", pages, label_visibility="collapsed")
    st.sidebar.divider()

    if page == "Data overview":
        page_overview(daily, labels, households, protocols, overview)
    else:
        page_scoring(daily, labels, static)


if __name__ == "__main__":
    main()
