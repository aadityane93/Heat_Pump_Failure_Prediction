"""Fit the dashboard's models once, offline.

    python train_models.py --artifacts artifacts

The dashboard does not train anything. It loads what this script writes and
scores houses with it, so a prediction you see in the app comes from a model
that was fitted here, before the app started.

Households are split once, stratified on `has_fault`. The models are fitted on
the training side; the held-out side is recorded in the bundle so the dashboard
can offer those houses as the ones the model has genuinely not seen.

Writes `models.joblib` next to the other artifacts:

    binary          one fitted pipeline per model name
    multilabel      one fitted pipeline per model name and fault type
    feature_columns the exact input columns, in order, the pipelines expect
    train_ids       households the pipelines were fitted on
    test_ids        households held out for the dashboard
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

import heapo_core as hc

MIN_POSITIVES = 10


def load_dataset(artifacts: Path) -> pd.DataFrame:
    parquet = artifacts / "dataset.parquet"
    csv = artifacts / "dataset.csv.gz"
    if parquet.exists():
        data = pd.read_parquet(parquet)
    elif csv.exists():
        data = pd.read_csv(csv)
    else:
        raise SystemExit(f"no dataset.parquet or dataset.csv.gz in {artifacts}")
    data["Household_ID"] = pd.to_numeric(data["Household_ID"], errors="coerce").astype("Int64")
    # Group is study bookkeeping, not a feature, and the dashboard never has it
    # in the frame it scores. Drop it here so both sides carry the same columns.
    return data.drop(columns=["Group"], errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts", help="folder written by prepare_data.py")
    parser.add_argument("--test-size", type=float, default=0.25,
                        help="share of households held out for the dashboard")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    data = load_dataset(artifacts)
    data = data.dropna(subset=["has_fault"])
    print(f"Households with a label: {len(data)}  fault rate {data['has_fault'].mean():.3f}")

    train, test = train_test_split(
        data, test_size=args.test_size, stratify=data["has_fault"].astype(int),
        random_state=args.seed,
    )
    print(f"  training on {len(train)}, holding out {len(test)} for the dashboard")

    X = hc.feature_matrix(train, "has_fault")
    y = train["has_fault"].astype(int).to_numpy()

    bundle = {
        "binary": {},
        "multilabel": {},
        "feature_columns": list(X.columns),
        "train_ids": [int(h) for h in train["Household_ID"]],
        "test_ids": [int(h) for h in test["Household_ID"]],
        "targets": [],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(len(train)),
        "base_rate": float(y.mean()),
        "test_size": args.test_size,
        "seed": args.seed,
    }

    print("Fitting the binary models")
    for name, model in hc.make_models(seed=args.seed).items():
        bundle["binary"][name] = hc.make_pipeline(X, model).fit(X, y)
        print(f"  {name}")

    print("Fitting one model per fault type")
    for target in hc.DEFAULT_FAULT_TARGETS:
        if target not in train.columns:
            continue
        d = train.dropna(subset=[target])
        y_t = d[target].astype(int)
        if y_t.nunique() < 2 or y_t.sum() < MIN_POSITIVES:
            print(f"  {hc.pretty_fault(target)}: only {int(y_t.sum())} positives, skipped")
            continue
        X_t = hc.feature_matrix(d, target)[bundle["feature_columns"]]
        bundle["targets"].append(target)
        for name, model in hc.make_models(seed=args.seed).items():
            bundle["multilabel"].setdefault(name, {})[target] = (
                hc.make_pipeline(X_t, model).fit(X_t, y_t.to_numpy())
            )
        print(f"  {hc.pretty_fault(target)}: {int(y_t.sum())} positives")

    out = artifacts / "models.joblib"
    joblib.dump(bundle, out, compress=3)
    summary = {k: v for k, v in bundle.items()
               if k not in ("binary", "multilabel", "train_ids", "test_ids", "feature_columns")}
    summary["n_features"] = len(bundle["feature_columns"])
    summary["models"] = list(bundle["binary"])
    (artifacts / "models_meta.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("The dashboard reads this at startup. Rerun after rebuilding the artifacts.")


if __name__ == "__main__":
    main()
