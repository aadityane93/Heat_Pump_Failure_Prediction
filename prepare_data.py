from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bootstrap import ensure_heapo_loader, ensure_raw_data
from constants import LABEL_COLUMNS, STATIC_COLUMNS
from pipeline import build_feature_table, build_labels

DAILY_KEEP = [
    "Household_ID",
    "Timestamp",
    "Date",
    "kWh_received_Total",
    "kWh_received_HeatPump",
    "Temperature_avg_daily",
    "HeatingDegree_SIA_daily",
    "Sunshine_duration_daily",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def save_table(df: pd.DataFrame, out_dir: Path, stem: str) -> Path:
    """Write parquet when we can, compressed CSV otherwise."""
    target = out_dir / f"{stem}.parquet"
    try:
        df.to_parquet(target, index=False)
        return target
    except Exception as exc:  # pyarrow/fastparquet missing
        print(f"  parquet unavailable ({exc.__class__.__name__}), writing CSV instead")
        target = out_dir / f"{stem}.csv.gz"
        df.to_csv(target, index=False, compression="gzip")
        return target


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="float64").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include="int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def to_naive(series: pd.Series) -> pd.Series:
    """Drop the timezone if there is one, so comparisons never raise."""
    series = pd.to_datetime(series, errors="coerce")
    if getattr(series.dtype, "tz", None) is not None:
        series = series.dt.tz_localize(None)
    return series


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------


def load_static_tables(base: Path) -> dict[str, pd.DataFrame]:
    tables = {
        "protocols": base / "reports" / "protocols.csv",
        "meta_data": base / "meta_data" / "meta_data.csv",
        "households": base / "meta_data" / "households.csv",
        "smd_overview": base
        / "smart_meter_data"
        / "overview"
        / "smart_meter_data_daily_overview.csv",
    }
    out = {}
    for name, path in tables.items():
        if path.exists():
            out[name] = pd.read_csv(path, sep=";")
            print(f"  {name}: {out[name].shape}")
        else:
            print(f"  {name}: not found at {path}")
    return out


def load_daily_frames(heapo, households: pd.DataFrame):
    """Daily smart meter data for every household, plus weather per station."""
    all_households = heapo.get_all_households()
    print(f"  households reported by the loader: {len(all_households)}")

    smd_parts = []
    for i, h_id in enumerate(all_households, 1):
        if i % 100 == 0:
            print(f"    smart meter data {i}/{len(all_households)}")
        try:
            smd = heapo.load_smart_meter_data(household_id=h_id, resolution="daily")
            if smd is not None and not smd.empty:
                smd_parts.append(smd)
        except Exception:
            continue
    df_smd = pd.concat(smd_parts, ignore_index=True)
    del smd_parts
    gc.collect()
    print(f"  daily smart meter rows: {df_smd.shape}")

    weather_ids = households["Weather_ID"].dropna().unique()
    weather_parts = []
    for w_id in weather_ids:
        try:
            wea = heapo.load_weather_data(id=w_id, resolution="daily")
            if wea is not None and not wea.empty:
                weather_parts.append(wea)
        except Exception:
            continue
    df_weather = pd.concat(weather_parts, ignore_index=True)
    del weather_parts
    gc.collect()
    print(f"  daily weather rows: {df_weather.shape} from {len(weather_ids)} stations")

    return downcast(df_smd), downcast(df_weather)


def join_smd_and_weather(
    df_smd: pd.DataFrame, df_weather: pd.DataFrame, households: pd.DataFrame
) -> pd.DataFrame:
    df_smd["Timestamp"] = to_naive(df_smd["Timestamp"])
    df_smd["Date"] = df_smd["Timestamp"].dt.date
    df_weather["Timestamp"] = to_naive(df_weather["Timestamp"])
    df_weather["Date"] = df_weather["Timestamp"].dt.date
    df_weather = df_weather.drop_duplicates(subset=["Weather_ID", "Date"])

    weather_by_id = {w: g for w, g in df_weather.groupby("Weather_ID", observed=True)}
    station_of = households.set_index("Household_ID")["Weather_ID"].to_dict()

    chunks = []
    for h_id, group in df_smd.groupby("Household_ID", observed=True):
        w_id = station_of.get(h_id)
        weather_sub = weather_by_id.get(w_id)
        if weather_sub is None:
            continue
        chunks.append(
            group.merge(
                weather_sub.drop(columns=["Timestamp", "Weather_ID"], errors="ignore"),
                on="Date",
                how="left",
            )
        )
    joined = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    print(f"  joined daily frame: {joined.shape}")
    return joined


def first_visit_per_household(protocols: pd.DataFrame) -> pd.DataFrame:
    protocols = protocols.copy()
    protocols["Visit_Date"] = to_naive(protocols["Visit_Date"])
    return (
        protocols.sort_values("Visit_Date")
        .groupby("Household_ID", as_index=False)
        .first()
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default="/content/heapo_data",
        help="folder holding the unzipped HEAPO data",
    )
    parser.add_argument("--out", default="artifacts", help="where to write artifacts")
    parser.add_argument(
        "--min-days",
        type=int,
        default=60,
        help="drop households with fewer pre-visit days than this",
    )
    parser.add_argument(
        "--daily-all",
        action="store_true",
        help="keep every household in daily.parquet, not just the modelled ones",
    )
    parser.add_argument(
        "--skip-daily", action="store_true", help="do not write daily.parquet"
    )
    args = parser.parse_args()

    base = Path(args.data_path)
    ensure_raw_data(base)
    ensure_heapo_loader(base.parent)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading the static tables")
    static = load_static_tables(base)
    if "protocols" not in static or "households" not in static:
        raise SystemExit(
            f"protocols.csv and households.csv are required; looked under {base}"
        )
    protocols, households = static["protocols"], static["households"]

    print("Loading daily series through the HEAPO loader")
    try:
        from heapo import HEAPO  # imported here so the module stays importable without it
    except ModuleNotFoundError as exc:
        if exc.name == "heapo":
            raise SystemExit(
                "The HEAPO loader is not installed in this Python environment. "
                "Run `python -m pip install -r requirements.txt` and retry."
            ) from exc
        raise

    heapo = HEAPO(data_path=str(base) + "/", use_local_time=True, suppress_warning=True)
    df_smd, df_weather = load_daily_frames(heapo, households)

    print("Joining smart meter data with weather")
    daily = join_smd_and_weather(df_smd, df_weather, households)
    del df_smd, df_weather
    gc.collect()

    print("Building labels from the first field visit")
    protocols_first = first_visit_per_household(protocols)
    labels = build_labels(protocols_first)
    print(f"  labelled households: {len(labels)}")
    print(f"  with at least one fault: {int(labels['has_fault'].sum())}")

    print("Keeping only the days before each visit")
    visit_of = labels.set_index("Household_ID")["Visit_Date"].to_dict()
    daily["Visit_Date"] = daily["Household_ID"].map(visit_of)
    daily["Visit_Date"] = to_naive(daily["Visit_Date"])
    df_pre = daily[
        daily["Visit_Date"].notna() & (daily["Timestamp"] < daily["Visit_Date"])
    ].copy()
    print(f"  pre-visit rows: {df_pre.shape} across {df_pre['Household_ID'].nunique()} households")

    print("Extracting one feature row per household")
    features = build_feature_table(df_pre)
    print(f"  features: {features.shape}")

    statics = protocols_first[
        [c for c in ["Household_ID", *STATIC_COLUMNS] if c in protocols_first.columns]
    ]
    label_cols = [c for c in ["Household_ID", *LABEL_COLUMNS] if c in labels.columns]
    dataset = (
        features.merge(statics, on="Household_ID", how="left")
        .merge(labels[label_cols], on="Household_ID", how="left")
        .merge(
            households[["Household_ID", "Group"]]
            if "Group" in households.columns
            else households[["Household_ID"]],
            on="Household_ID",
            how="left",
        )
    )
    dataset = dataset[dataset["n_days"] >= args.min_days]
    print(f"  modelling table: {dataset.shape}")
    print(f"  fault rate: {dataset['has_fault'].mean():.3f}")

    print("Writing artifacts")
    save_table(dataset, out_dir, "dataset")
    save_table(households, out_dir, "households")
    save_table(protocols, out_dir, "protocols")
    if "smd_overview" in static:
        save_table(static["smd_overview"], out_dir, "smd_overview")

    if not args.skip_daily:
        keep = [c for c in DAILY_KEEP if c in daily.columns] + ["Visit_Date"]
        daily_out = daily[keep]
        if not args.daily_all:
            daily_out = daily_out[
                daily_out["Household_ID"].isin(dataset["Household_ID"])
            ]
        daily_out = daily_out.copy()
        daily_out["Household_ID"] = daily_out["Household_ID"].astype(str)
        save_table(downcast(daily_out), out_dir, "daily")
        print(f"  daily rows written: {len(daily_out)}")

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_path": str(base),
        "min_days": args.min_days,
        "n_households": int(len(dataset)),
        "fault_rate": float(dataset["has_fault"].mean()),
        "label_columns": [c for c in LABEL_COLUMNS if c in dataset.columns],
        "static_columns": [c for c in STATIC_COLUMNS if c in dataset.columns],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone. Artifacts are in {out_dir.resolve()}")

if __name__ == "__main__":
    main()
