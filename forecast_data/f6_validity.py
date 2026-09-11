"""F6 validity precedence, resolved on demand.

One label per (household, requested date), first match wins:

    no_usable_measurement > pv_contaminated > outside_fitted_window >
    insufficient_training_data > low_confidence_fit > valid

Confirmed in `outputs/decision_log.md`, 2026-09-03 row. Nothing here is
precomputed into the export: the label is derived at call time from the raw
tables written by `f6_raw_export.py`.
"""

import pandas as pd
from pathlib import Path

RES = Path("Forecast_consumption_cycle/outputs/f6/results")

LABELS = (
    "no_usable_measurement",
    "pv_contaminated",
    "outside_fitted_window",
    "insufficient_training_data",
    "low_confidence_fit",
    "valid",
)


class ValidityResolver:
    """Loads the raw export once, then answers per household-date."""

    def __init__(self, results_dir=RES):
        results_dir = Path(results_dir)
        hh = pd.read_parquet(results_dir / "f6_household_features.parquet")
        daily = pd.read_parquet(
            results_dir / "f6_daily_features.parquet",
            columns=["Household_ID", "local_date", "is_test"],
        )
        spine = pd.read_parquet(
            results_dir / "f6_spine_flags.parquet",
            columns=["Household_ID", "local_date", "slots_complete", "has_energy"],
        )

        hh["Household_ID"] = hh["Household_ID"].astype("int64")
        self._pv = dict(zip(hh["Household_ID"], hh["pv_flag"].astype(str)))
        self._band = dict(zip(hh["Household_ID"], hh["f5_band"].astype(str)))
        self._window_start = dict(
            zip(hh["Household_ID"], pd.to_datetime(hh["fitted_window_start"]))
        )

        # The first test fold is the protocol's own forward-chaining split
        # (clause 2). Derived here rather than exported as a label.
        ft = (
            daily[daily["is_test"].astype(bool)]
            .groupby("Household_ID")["local_date"]
            .min()
        )
        ft.index = ft.index.astype("int64")
        self._first_test = dict(zip(ft.index, pd.to_datetime(ft.values)))

        spine["Household_ID"] = spine["Household_ID"].astype("int64")
        key = list(zip(spine["Household_ID"], pd.to_datetime(spine["local_date"])))
        # Slot completeness is not value presence: a slot-complete day can be
        # ALL_NA, ALL_ZERO or PARTIAL_NA and carry no usable energy.
        self._measured = dict(
            zip(key, (spine["slots_complete"].to_numpy() & spine["has_energy"].to_numpy()))
        )

    def label(self, household_id, date):
        """Return exactly one label for this household on this date."""
        hid = int(household_id)
        d = pd.Timestamp(date).normalize()

        # 1. no measurement exists on the requested date. A date the spine does
        #    not carry at all (outside the record span) resolves here too.
        if not self._measured.get((hid, d), False):
            return "no_usable_measurement"

        # 2. the measurement exists but is the wrong physical quantity.
        if self._pv.get(hid) == "True":
            return "pv_contaminated"

        # 3. no fitted model covers this date.
        start = self._window_start.get(hid)
        if start is not None and pd.notna(start) and d < start:
            return "outside_fitted_window"

        # 4. a model exists for the household but not yet for this date.
        first_test = self._first_test.get(hid)
        if first_test is None or pd.isna(first_test) or d < first_test:
            return "insufficient_training_data"

        # 5. the fit covers the date but its data sufficiency is weak.
        if self._band.get(hid) == "LOW":
            return "low_confidence_fit"

        return "valid"


def label(household_id, date, _cache={}):
    """Convenience wrapper for one-off calls; shares a single resolver."""
    if "r" not in _cache:
        _cache["r"] = ValidityResolver()
    return _cache["r"].label(household_id, date)
