# HEAPO heat pump dashboard

A Streamlit dashboard over the HEAPO study: Swiss smart meter data scored against
on-site inspection protocols. It brings together separate workstreams that share
the same underlying archive.

| Page | What it answers | Workstream |
|---|---|---|
| **Fault prediction** | Is this household faulty, and which fault? | Aaditya |
| **Forecast cycle** | What did the meter record, what did the model predict, how big is the gap? | Miguel |
| **Peer comparison** | How does this household sit against its matched peers? | Abdul |
| **Physical baseline** | How far does this building sit above a code-compliant version of itself, and how much of its heating does the building not justify? | Miguel |
| **Stratum queue** | Which houses consume more than comparable houses, and how much energy does it involve? | Miguel |

Each page carries further views in the sidebar: the raw-data overview, the
forecaster walkthrough, the peer-matching validation analysis, the physical
baseline's building score and visit queue, the stratum summary, and each model's
data workflow and results.

## Running it

```bash
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt

python prepare_data.py --data-path heapo_data --out artifacts
python train_models.py --artifacts artifacts

streamlit run Fault_prediction.py
```

`prepare_data.py` downloads the HEAPO archive from Zenodo on first run if
`heapo_data/` is not already present. Point `HEAPO_DATA_DIR` at an existing copy
to skip the download.

The entry point is `Fault_prediction.py`, a three-line launcher that runs
`app.py` unchanged — Streamlit names the first nav entry after the file it is
given, and `app` was not a useful label.

## What has to be built, and what ships

`artifacts/` is generated, not committed. `prepare_data.py` writes the tables and
`train_models.py` fits the models into it; both must run once before the Fault
prediction and Peer comparison pages will work.

The Forecast cycle, Physical baseline and Stratum queue pages need no build step.
Their inputs ship with the repo.

`forecast_data/`:

| File | What it is |
|---|---|
| `f6_dashboard_export_DRAFT.parquet` | 99,764 rows, 95 households, one row per household-day: actual, predicted, deviation, temperature |
| `f6_household_features.parquet`, `f6_daily_features.parquet`, `f6_spine_flags.parquet` | the raw tables the validity labels are derived from |
| `f6_validity.py` | resolves one validity label per household-date |
| `R7_forecaster_walkthrough.ipynb` | the forecaster walkthrough, rendered in-page |

`physical_baseline/outputs/`:

| File | What it is |
|---|---|
| `export/physical_baseline_v1.parquet` | one row per audited household: building score and visit queue |
| `export/SCHEMA.md` | the export contract — every column, its unit, and how to display it |
| `deliverable/DELIVERABLE_physical_baseline.ipynb` | the model's data workflow and results, P0 to P13 |

`stratum_model/outputs/s12/`:

| File | What it is |
|---|---|
| `data/s12_dashboard_queue.parquet` | one row per household this page scores: category, peer group, excess energy, where the excess is |
| `S12_fleet_split.ipynb` | the model's data workflow and results |

The Physical baseline and Stratum queue pages read the live model directory when
it sits beside the dashboard, and the bundled snapshot otherwise. Override any
location with `F6_EXPORT`, `R7_NOTEBOOK`, `PHYS_BASELINE_DIR` or `STRATUM_DIR`.

## Layout

```
Fault_prediction.py          entry point
app.py                       Fault prediction page
heapo_core.py                features, model definitions, pipelines
pipeline.py, constants.py    labels, feature groups, palette
prepare_data.py              builds artifacts/ from heapo_data/
train_models.py              fits models.joblib into artifacts/
bootstrap.py                 fetches HEAPO data when missing

peer_relative_features.py    peer-matched feature ratios
theme.py                     palette and shared CSS
notebook_render.py           renders a notebook's markdown, figures and outputs
f6_walkthrough.py            forecaster walkthrough view

pages/
  1_Forecast_cycle.py        day readout + walkthrough
  2_Peer_comparison.py       peer ratios + validation analysis
  3_Physical_baseline.py     building score + visit queue + data workflow
  4_Stratum_queue.py         queue + summary + data workflow

forecast_data/               forecaster inputs
physical_baseline/           physical baseline export and notebook
stratum_model/               stratum queue and notebook
notebooks_and_codes/         analysis notebooks
```

## The models

Defined once in `heapo_core.make_models()` — logistic regression, random forest,
and histogram gradient boosting. `train_models.py` fits each against `has_fault`
and against four individual fault types, then writes all 15 fitted pipelines to
`artifacts/models.joblib`.

The dashboard fits nothing. Every prediction comes from that file, so a change to
a model only reaches the app after `train_models.py` is re-run. Adding a model to
`make_models()` is enough to make it appear in both dropdowns — the UI reads the
names from the bundle.

The Physical baseline page never loads these models. The fault probability it
shows beside each house is read from its own export.

## Reading the numbers

**The fault base rate is 0.854.** Of 119 labelled households, 102 have a
confirmed fault — they were visited because something was suspected. Predicting
"fault" for every household scores 85.4%, so accuracy at or below that is worth
nothing. `heapo_core.evaluate_binary()` reports the base rate alongside every
score for this reason, and omits ROC-AUC, which this imbalance dominates.

**The forecaster's specification was rejected.** The predictions on the Forecast
cycle page come from stored `W2_hl6` coefficients, a specification that failed
its acceptance bar at 4.646% median absolute NMBE against a 1.7% threshold. A
fit-quality band of HIGH means *among the better-fitted of a rejected
specification*, not accurate. Its export schema is also a draft and is not yet
agreed between the workstreams.

**Peer coverage is thin.** A peer must be fault-free and EV-free, which leaves 15
eligible households out of 119; only one stratum reaches the minimum pool size,
so 7 households get a ratio. That is a property of this labelled sample, not of
the peer method, which was written for the full fleet where most households are
never visited.

**The physical baseline confirms excess, never its absence.** It has no "normal"
state: a house with no finding is shown as such, not as fine. It ranks on the
lower bound of each household's band, within construction era, and keeps the
two kWh figures on its visit queue — what a technician could change and what
needs a builder — in separate columns that are never added together.

**The stratum queue's letters are its own.** A names a setting to check, B is
high consumption with the cause unknown, and C is nothing material — not a clean
bill of health. It shows one baseline at a time and ranks on excess energy. A
household is scored on either the physical baseline page or this one, never both.

## Data

HEAPO, 1,408 Swiss households, 15-minute and daily smart meter data with weather
and 410 inspection protocols. Downloaded from Zenodo by `bootstrap.py`.
`heapo_data/` is read-only and is never committed.
