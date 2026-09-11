# `physical_baseline_v1` — dashboard export contract

Produced by `p11_export.py`. **214 rows × 47 columns**, one row per HEAPO protocol
household. Written as `.csv` (semicolon) and `.parquet`.

**The contract is the SCHEMA, not the numbers.** Column names, units and meanings
are frozen at `v1`. **The values behind them are expected to change** — the model
is still moving and has already revised its own totals twice. Bind the dashboard
to the columns, never to a number in a report.

**Breaking changes bump the version to `physical_baseline_v2` and both files ship
side by side for one cycle.** Additive columns do not bump the version.

---

## What this model answers, and what it does not

**It answers:** *how much more electricity does this house use than a
code-compliant version of itself would?* Computed from building geometry, era and
published Swiss standards. **It never reads the meter for this** — only for weather.

**It does not answer:** whether a heat pump is misconfigured. That is the separate
setting-fault axis, now populated from Aaditya's model as a suggestion (see *Visit axis*).

**Never merge with the peer model.** Its A/B/C means something different on a
different population. Every column here is prefixed `phys_` or `visit_` so a pasted table
cannot be misread. **Two methods, two dashboard sections, never one number.**

## Columns

### Identity
| column | type | meaning |
|---|---|---|
| `household_id` | int | HEAPO `Household_ID`. Unique, all 214 present. |
| `schema_version` | str | `physical_baseline_v1` |
| `generated_utc` | str | ISO 8601 build timestamp |

### Status — four states, and none of them says "normal"

| column | type | meaning |
|---|---|---|
| `phys_status` | enum | one of the four below |
| `phys_status_reason` | str | plain language, safe to show a user |
| `phys_no_result_reason` | str | non-empty only for the two no-result states |
| `phys_no_result_is_fixable` | bool | `True` = go collect the data; `False` = we looked |
| `phys_excess_confirmed` | bool | whole 95% band above zero |

| state | n | meaning | action |
|---|---:|---|---|
| **`HIGH_EXCESS`** | 41 | confirmed excess, worst 25% of its era | **act** |
| **`EXCESS_CONFIRMED`** | 114 | confirmed excess, below the action cut | real, not first |
| **`INCONCLUSIVE`** | 17 | computed, band straddles zero | **no claim** |
| **`INCOMPLETE_AUDIT`** | 42 | never computed, building data missing | **fixable — go record it** |

**There is no "normal" state and there never will be.** It would require a
household's whole band to sit at or below code, and **not one does**. We can
*confirm an excess* and never *confirm its absence*. The low end is the absence of
a finding, not a verdict of normality.

**"Excess", not "below code".** *Below* reads both ways — a user can take it as
*under the threshold*, i.e. good. *Excess* can only mean more than it should be.

**The two no-result states are separate because the actions differ.**
`INCOMPLETE_AUDIT` is fixable: 42 households where the audit did not record era,
per-storey geometry or renovation, and someone can go and record them.
`INCONCLUSIVE` is not: we computed, and the method cannot resolve it.
**`phys_no_result_is_fixable` carries this so the dashboard need not parse text.**

**Neither no-result state may render green.** 59 households have no finding. That
is not the same as being fine.

**No single letters on this axis.** Single letters read as grades, and an earlier
version collided with the since-removed `combined_category`, which used A/B/C/D for
the two-axis 2×2. Both axes now use words.

### Energy — the deliverable
| column | unit | meaning |
|---|---|---|
| `phys_deficiency_kwh_yr` | kWh/yr | point estimate |
| **`phys_deficiency_lo_kwh_yr`** | kWh/yr | **2.5th percentile — RANK AND DISPLAY THIS** |
| `phys_deficiency_hi_kwh_yr` | kWh/yr | 97.5th percentile |
| `phys_deficiency_band_kwh_yr` | kWh/yr | `hi − lo` |

**Show the lower bound.** It is the honest *"at least this much"*, and it
penalises poorly-documented houses instead of rewarding them for vagueness.

The point estimate uses each constant's **sourced central value**, which is
deliberately not always its band midpoint (`storey_height` 2.70 is the SIA gross
figure; `perimeter_factor` 1.00 is the minimum-perimeter square plan; `b_ground`
0.50 is the SIA factor). So the point is not the band centre by design.

Confirmed excess totals **220,959 kWh/yr on the lower bound**, of which
**`HIGH_EXCESS` is 111,115 kWh/yr across 41 households**. **Quote the lower bound.**

Also shipped: **`phys_deficiency_pct_of_reference`** (with `_lo` / `_hi`), which
reads where kWh cannot — *"+20% over code"* versus *"+208%"*. Fleet median
**+66%**, IQR 32–101%, range −4% to +233%.

### Decomposition
| column | share |
|---|---|
| `phys_term_envelope_kwh_yr` | ~75–80%, and the only term with independent validation (P9, ρ +0.536) |
| `phys_term_dhw_kwh_yr` | ~15–19% |
| `phys_term_emitter_kwh_yr` | ~5–6% |

The three sum to `phys_deficiency_kwh_yr` within rounding — asserted on write.

### Scenario — never summed into the headline
| column | meaning |
|---|---|
| `phys_scenario_capacity_lo/mid/hi_kwh_yr` | cost of oversizing at 5 / 10 / 15% derating |
| `phys_scenario_never_sum_into_total` | always `True` |

**Not a measurement.** Sources give a penalty range; nothing maps sizing ratio to
a penalty, so the shape is assumed. Display it as a separate, labelled line.
**Median household is zero** — half the fleet is at or below required capacity —
but a minority carry more here than in their envelope term. **Show per household,
never as a fleet share.**

### Context — not predictions
| column | meaning |
|---|---|
| `phys_context_e_as_built_kwh_yr` | modelled consumption, **excludes appliances** |
| `phys_context_e_reference_kwh_yr` | same house, code-compliant |
| `phys_level_is_not_a_prediction` | always `True` |

**Never compare these to a meter reading and never show them as an expected bill.**
They exclude lighting, cooking and appliances entirely. Only their *difference*
is meaningful.

### Data-quality flags — surface these next to the number
| column | meaning |
|---|---|
| `phys_completeness_tier` | `T1` tightest / `T2` / `T3` widest |
| `phys_jaz_derived` | `True` = ground-source with radiators, efficiency derived not published |
| `phys_geometry_reconciles` | `False` = recorded storey areas do not sum to the total |
| `phys_hdd_correction_extrapolated` | `True` = under 365 meter days, weather correction extrapolated |
| `phys_pv_level` | `PV` / `no PV` / `Unknown`. **`Unknown` is not `no PV`.** |
| `phys_n_meter_days` | usable days behind the weather term |

PV does **not** bias this model — the deficiency never reads the meter. It biases
any meter-based comparison, where PV households read ~2,300 kWh/yr low.

### Visit axis — POPULATED 2026-09-11 from P13

The combined product: **one order, two kWh columns, and the fault shown as
information.** Sort on the heating the building does not justify. The fault
model's suggestion rides along on every row and **never decides status or
rank** (Miguel 2026-09-11 - see `outputs/p13/reports/p13_filter_trap.md`).

| column | meaning |
|---|---|
| **`visit_status`** | one of the three states below |
| **`visit_technician_recoverable_kwh`** | **sort on this.** Slope excess `(β_meas − β_phys) × HDD × 365` |
| `visit_slope_excess_pct` | the same, as % of the physical slope |
| `visit_fault_probability` | logistic, max over heating curve + heating limit |
| `visit_named_fault` | the fault above threshold, or empty |
| `visit_fault_axis_available` | **`False` = the fault model never scored this house** |
| `visit_advisory_night_setback_prob` | **advisory only — never names the fault** |
| `visit_in_training_set` | **`True` = the fault model was fitted on this house** |
| `visit_rank_in_era` | rank within `SLOPE_EXCESS` by recoverable kWh, within era - **fault or no fault** |
| `visit_never_sum_with_phys_deficiency` | always `True` |
| `setting_fault_flag` | named fault present; **null where the model never ran** |
| `setting_fault_name` | as `visit_named_fault` |

| `visit_status` | n | action |
|---|---:|---|
| **`SLOPE_EXCESS`** | 125 | heating the building does not justify - **ranked**; the fault suggestion rides along, never gates |
| **`NO_SLOPE_EXCESS`** | 47 | heating response fits the building |
| **`NOT_ASSESSABLE`** | 42 | cannot compute |

**`combined_category` was REMOVED.** It was reserved as letters A/B/C/D; the design
settled on words, and nothing had consumed v1, so it was replaced by
`visit_status` at no migration cost.

### Two kWh columns — NEVER sum them

| column | what it is | who fixes it |
|---|---|---|
| `visit_technician_recoverable_kwh` | heating the building does not justify | **a technician**, by changing a setting |
| `phys_deficiency_lo_kwh_yr` | the building versus current code | **a builder** — insulation or replacement |

Summing them hides which number needs which trade. Example: household `7771161`
shows 1,862 recoverable against **4,472 not recoverable** — the setting fault is
real, and fixing it leaves most of the problem in the walls.

### Why the slope, not the level

`actual − E_as_built` (median 6,045 kWh/yr) is **dominated by appliance
electricity**. Appliances are weather-independent, so the **slope** comparison
cancels them (median 719). A heating curve set too high raises exactly the slope.

**~22% slope excess is BASELINE for everyone** (P9 measured the physics running
that far below the meter), not a fault. **Rank within era; never display the
absolute figure as recoverable energy.**

### The fault suggestion - shown, never gated

**It was a filter until 2026-09-11 and was removed.** Measured to pull AGAINST
the slope sort: the heating-curve probability correlates with slope excess at
rho **-0.39**, where physics predicts positive, and the recorded label inverts
the same way (AUC 0.441, replicating the S-series 0.386). The heating-limit
model's top weights are **blank-form-field flags** - fault rate 0.45 where the
buffer-tank field was filled in, 0.16 where it was left blank.

**Model `logistic`, read-only from Aaditya's bundle.** Chosen because a probability shown
to a technician has to spread — random forest and hist-GBM reach train AUC 1.000 and
collapse to ~0/1. **Night setback is advisory only** and never names the fault: logistic scores it
at held-out AUC **0.230**, i.e. inverted.

**The fault models do not survive held-out data** (P12). The suggestion is a
**plausibility hint, not a detector**, and applies only to the 97 households the
models were trained and tested on. **Mark `visit_in_training_set` rows on the
page** — their probabilities are optimistic.

**Aaditya's dashboard PR-AUC table (0.895 / 0.819 / 0.812) is computed on all 119
households including the 89 training rows. Do not quote it as performance.**

## Standing limitations to display, not hide

1. **Constants are second-hand.** Page-level citations exist; the standards
   themselves have not been read here. Not publication-grade.
2. **The era gradient is too steep** — newer houses under-predicted 24–53% (P9).
3. **Design-load offset 1.13–1.49** against SIA 384.201. Unexplained, untuned.
4. **Only the envelope term is validated** (ρ +0.536 per m², interval far from 0).
5. **Swiss constants.** The code ports; the constants do not.
6. **The fault suggestion pulls AGAINST the slope sort** (ρ −0.39) and the recorded
   label inverts the same way, replicated (AUC 0.441 / 0.386). It is shown, never
   gated. The heating-limit model partly learns which forms were filled in.
