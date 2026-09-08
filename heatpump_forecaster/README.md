# Heat pump daily consumption forecaster

The forecaster behind the F6 dashboard export, packaged so it can be refitted,
applied and checked outside the study repository.

```python
import sys; sys.path.insert(0, "src")
import forecaster as F, validity as V
import pandas as pd

df = pd.read_parquet("data/f6_dashboard_export_DRAFT.parquet")

coefs = F.fit_expanding(df[df.in_training_window & (df.Household_ID == 86109)],
                        value_col="actual_kwh", temp_col="temp_mean",
                        date_col="local_date")
```

Verify it against the shipped data before trusting anything:

```
python verify.py     # -> FAILURES: 0
```

| path | what |
|---|---|
| `src/forecaster.py` | the model — fit, apply, score, noise floor |
| `src/validity.py` | the six-label precedence ladder |
| `data/f6_dashboard_export_DRAFT.parquet` | 99,764 rows, 95 households, 20 columns |
| `data/w2_hl6_coefficients.parquet` | 843 fitted folds, the stored coefficients |
| `data/household_facts.parquet` | 95 rows, what the ladder needs per household |
| `notebooks/R7_forecaster_walkthrough.ipynb` | how the model was arrived at |
| `verify.py` | three self-checks against the shipped data |

Needs numpy and pandas. Nothing else — no sklearn, no xgboost.

---

## The model

```
predicted = a + b1 * max(0, 15 - T) + b2 * max(0, T - 15)
```

Two straight lines meeting at a knot fixed at 15 °C. `a` is the predicted
consumption on a 15 °C day, `b1` the heating slope below the knot, `b2` the slope
above it. Units are kWh/day and kWh/day per °C.

At exactly 15 °C both hinges are zero and the prediction collapses to `a`, which
is why `a` reads directly as "what this household uses on a mild day". It is
**not** baseload — it still contains hot water and appliances.

**The 15 °C knot is not the degree-day base.** That is 12 °C, from SIA 381/3 via
HEAPO. The knot is a nuisance parameter of this specification. Two numbers, two
roles; nothing here should be fed back as a degree-day base.

### How it is fitted

Weighted least squares, refitted every calendar month on an expanding window,
with a **6-month half-life**:

```
w = 2 ** (-age_in_months / 6)
```

A day six months old counts half as much as today, twelve months a quarter,
twenty-four months a sixteenth. No day is ever discarded — the oldest ones simply
stop mattering. That decay is the entire difference between this variant
(`W2_hl6`) and a plain expanding fit.

Folds are calendar months: train on everything before month M, predict month M,
step forward. **Twelve months of training are required before the first fold.** A
test month never contributes to its own coefficients.

There is **no trend term and no time index**. Seasonality enters only through
temperature. The level adapts because the model is refitted monthly under the
half-life, but nothing in it can project a trend forward.

### What it does not use

Only date, consumption and temperature. No occupancy, no metadata, no protocols,
no fault labels. Fault type deliberately does not gate inclusion — the
forecaster's job is to predict what the household actually does, faults included.

---

## The validity ladder

One label per household-date, first match wins:

```
no_usable_measurement > pv_contaminated > outside_fitted_window >
insufficient_training_data > low_confidence_fit > valid
```

The set is **ordered, not disjoint**, and that is the point. Over 863,466
requestable household-dates in the source archive, **38.80% matched two or more**
of the original five categories and had no single answer without precedence, and
a further **1.33% matched none** of them yet had no usable reading — which is why
the sixth label exists.

Reading top to bottom: no measurement exists → a measurement exists but is the
wrong physical quantity → no model covers this date → a model exists but not yet
for this date → the model covers it but its data sufficiency is weak.

Only `no_usable_measurement` means there is nothing to show. Every other label
still carries a real reading; the label decides which caveat sits beside it.

On the shipped export:

| label | rows | share |
|---|---|---|
| `pv_contaminated` | 17,032 | 17.07% |
| `outside_fitted_window` | 36,537 | 36.62% |
| `insufficient_training_data` | 26,129 | 26.19% |
| `low_confidence_fit` | 2,755 | 2.76% |
| `valid` | 17,311 | 17.35% |
| `no_usable_measurement` | 0 | 0% |

`no_usable_measurement` is zero because the export contains measured days only.
Pass `measured_col=` to `resolve()` if your frame includes unmeasured dates.

**`outside_fitted_window` is about the visit, not about age.** For a treatment
household the window opens the day after its inspection; every earlier date is
discarded, never trained on and never scored. Controls have no visit, so their
window opens at record start and they can never carry this label. Verified on 139
of 139 treatment households: `fitted_window_start >= Visit_Date`, never before.

---

## The data

95 households (75 scoring + 20 controls), 99,764 household-days, 2019-11-04 to
2024-02-27, from **HEAPO**, an open dataset of Swiss smart-meter data with on-site
heat pump inspection protocols (arXiv:2503.16993).

> ### The schema is a PROPOSAL, not agreed
>
> `f6_dashboard_export_DRAFT.parquet` carries `SCHEMA_STATUS` and `WARNING` in its
> own key-value metadata so the caveat travels with the file. **Column names,
> grain and coverage are not final** and were not agreed with the detector's
> consumer. Nothing downstream should be built against it as a stable contract.

Four conventions in this data are not obvious and will produce wrong results if
assumed away.

**`actual_kwh` is consumption, not a meter register.** It is the sum of interval
readings over the day. Do not difference it.

**The day is cut at Swiss local midnight** (Europe/Zurich), DST-aware. A fixed
UTC offset does not reproduce it.

**`temp_mean` is not the weather provider's own daily mean.** It averages the
Europe/Zurich day with each hourly stamp read as the hour's *beginning*, to match
the meter day. MeteoSwiss uses the UTC day with the stamp read as the hour's
*end*. The two agree on **0.72% of station-days** by design; the difference is
zero-mean (+0.006 °C, sd 0.320). Anyone joining to a MeteoSwiss column will see a
mismatch — this is the explanation, not a bug.

**`channel_used` must condition every cross-household statistic.** The archive
holds four measurement classes and they measure different physical quantities.
Pooling across them silently drops or misreads the HeatPump-only households.

### PV households are not forecastable against this target

PV export is not recorded at 15-minute resolution. For a household with solar the
meter reads **grid import, not consumption** — self-consumption is invisible and
is itself weather-driven, so it correlates with the regressor. This is a
**data-definition limit, not a fit-quality flag**: a PV household in the HIGH band
is not well modelled, it is well modelled against the wrong number. That is why
`pv_contaminated` sits second in the ladder, above every model-quality question.

`pv_flag` is three-level. `Unknown` is treated as usable because it was *measured*
indistinguishable from known non-PV (AUC 0.517, p 0.59), not assumed so.

---

## What the numbers are worth

The acceptance bar was fixed and dated before any weather model was fitted, and
asks two separate questions.

| | measured | daily bar | households meeting it |
|---|---|---|---|
| accuracy, median CV(RMSE) | **24.6%** | < 33.1% | 856 of 1,221 (70.1%) |
| bias, median absolute NMBE | **4.646%** | ≤ 1.7% | 45 of 1,221 (3.7%) |

**The shortfall is bias, not precision.** The model clears the accuracy bar on
seven households in ten. What it does not do is sit centred: each household tends
to run consistently high or low, and refitting monthly does not remove that
offset. A `fitquality_band` of HIGH means *among the better-fitted*, not
*accurate in absolute terms*.

### There is a floor, and it is close

Two days the same household lived through in the same month at the same
temperature still differ by σ ≈ 5 kWh/day. That bounds any model conditioned on
weather and calendar. `forecaster.noise_floor()` computes it for any series.

| | median CV(RMSE) |
|---|---|
| noise floor (30-day pairing) | **19.7%** |
| noise floor (any pairing) | 21.6% |
| this model | 24.6% |

Three to five points of headroom, fleet-wide. It is also strongly seasonal —
about **15%** below 0 °C rising to **29%** between 15 and 20 °C, because summer
consumption falls to roughly 12 kWh/day of behaviour that is not in the data at
all.

Measured on 30 households scored on these folds: a gradient-boosted model given
the same temperature column ties with this three-parameter regression (paired
median +0.4 points, better on 16 of 30). Adding a month-of-year term to the
residuals is worth about 8.6% off RMSE; a day-of-week term about 1.8%. Both are
in-sample upper bounds. **The feature mattered; the algorithm did not.**

---

## Verification

`verify.py` runs three checks against the shipped data:

| check | result |
|---|---|
| `predict()` vs the export's stored `predicted_kwh`, 24,879 rows | max difference **0.000e+00 kWh**, 0 null disagreements |
| `fit_expanding()` vs the stored coefficients, 78 households refit | worst difference **3.3e-07** (float32 storage precision) |
| validity ladder vs the recorded label counts | **0** labels disagreeing |

The ladder was additionally checked against the study's own resolver on all
99,764 rows: **0 mismatches**.

## Scope

This is a consumption forecaster, not a fault detector, and it has not been
validated as one. It is fitted per household with faults included, deliberately —
the opposite of what a residual model needs, which is a fault-free reference.

Cite the dataset (arXiv:2503.16993), not this repository. The 12 °C heating limit
comes from SIA 381/3, Swiss Association of Engineers and Architects, 1982.
