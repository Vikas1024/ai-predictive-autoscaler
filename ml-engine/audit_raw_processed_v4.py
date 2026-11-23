#!/usr/bin/env python3

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

BATCH_DIR = Path("experiments/results/hpa_training/batch_20261123_141907")
DATA_DIR = Path("datasets/processed/multi_horizon_v4")

HORIZONS = (15, 30, 45, 60)
INTERVAL = 5.0

FEATURES = [
    "app_cpu_millicores",
    "app_memory_mib",
    "work_rps",
    "work_avg_latency_ms",
    "work_p95_latency_ms",
    "ready_replicas",
    "workload_active",
    "cpu_lag_1",
    "cpu_lag_2",
    "cpu_lag_3",
    "rps_lag_1",
    "rps_lag_2",
    "rps_lag_3",
]

RAW_NUMERIC = [
    "elapsed_seconds",
    "app_cpu_millicores",
    "app_memory_mib",
    "ready_replicas",
    "work_rps",
    "work_avg_latency_ms",
    "work_p95_latency_ms",
]

KEYS = ["trial_id", "elapsed_seconds"]


def fail(message):
    raise AssertionError(message)


def reconstruct_trial(meta_row):
    trial_id = meta_row["trial_id"]
    source_dir = Path(meta_row["source_directory"])
    metrics_path = source_dir / "metrics.csv"

    if not metrics_path.is_file():
        fail(f"{trial_id}: missing raw file: {metrics_path}")

    raw = pd.read_csv(metrics_path)

    for col in RAW_NUMERIC:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.sort_values("elapsed_seconds").reset_index(drop=True)

    if len(raw) != 50:
        fail(f"{trial_id}: expected 50 raw rows, found {len(raw)}")

    intervals = raw.elapsed_seconds.diff().dropna().to_numpy()
    if not np.allclose(intervals, INTERVAL, atol=0.25):
        fail(f"{trial_id}: invalid sampling interval")

    raw["trial_id"] = trial_id
    raw["profile"] = meta_row["profile"]
    raw["max_users"] = int(meta_row["max_users"])

    for lag in (1, 2, 3):
        raw[f"cpu_lag_{lag}"] = raw.app_cpu_millicores.shift(lag)
        raw[f"rps_lag_{lag}"] = raw.work_rps.shift(lag)

    raw["workload_active"] = np.where(
        raw.work_rps.notna(),
        (raw.work_rps > 0).astype(int),
        np.nan,
    )

    return raw


def compare_horizon(horizon, metadata):
    target = f"target_cpu_{horizon}s"
    steps = horizon // int(INTERVAL)

    reconstructed = []

    for _, meta_row in metadata.iterrows():
        trial = reconstruct_trial(meta_row)
        trial[target] = trial.app_cpu_millicores.shift(-steps)

        feature_ok = trial[FEATURES].notna().all(axis=1)
        target_ok = trial[target].notna()
        keep = feature_ok & target_ok

        expected = trial.loc[
            keep,
            FEATURES + [target, "trial_id", "profile", "max_users",
                        "elapsed_seconds"],
        ].copy()

        reconstructed.append(expected)

    expected = pd.concat(reconstructed, ignore_index=True)

    actual_path = DATA_DIR / f"cpu_prediction_{horizon}s_all.csv"
    if not actual_path.is_file():
        fail(f"{horizon}s: processed dataset missing: {actual_path}")

    actual = pd.read_csv(actual_path)

    columns = FEATURES + [
        target, "trial_id", "profile", "max_users", "elapsed_seconds"
    ]

    missing = set(columns) - set(actual.columns)
    if missing:
        fail(f"{horizon}s: processed file missing columns: {sorted(missing)}")

    expected = expected[columns].copy()
    actual = actual[columns].copy()

    if expected.duplicated(KEYS).any():
        fail(f"{horizon}s: duplicate keys in reconstructed data")

    if actual.duplicated(KEYS).any():
        fail(f"{horizon}s: duplicate keys in processed data")

    expected = expected.sort_values(KEYS).reset_index(drop=True)
    actual = actual.sort_values(KEYS).reset_index(drop=True)

    if len(expected) != len(actual):
        fail(
            f"{horizon}s: row count mismatch. "
            f"Reconstructed={len(expected)}, processed={len(actual)}"
        )

    for col in ["trial_id", "profile", "max_users"]:
        if not expected[col].equals(actual[col]):
            fail(f"{horizon}s: categorical mismatch in {col}")

    numeric_cols = FEATURES + [target, "elapsed_seconds"]

    for col in numeric_cols:
        left = expected[col].to_numpy(dtype=float)
        right = actual[col].to_numpy(dtype=float)

        if not np.isfinite(left).all() or not np.isfinite(right).all():
            fail(f"{horizon}s: non-finite values in {col}")

        matches = np.isclose(left, right, rtol=1e-9, atol=1e-8)
        if not matches.all():
            bad = np.where(~matches)[0]
            first = int(bad[0])
            fail(
                f"{horizon}s: mismatch in {col}; "
                f"{len(bad)} mismatches. "
                f"First row expected={left[first]}, actual={right[first]}"
            )

    train_path = DATA_DIR / f"cpu_prediction_{horizon}s_train.csv"
    test_path = DATA_DIR / f"cpu_prediction_{horizon}s_test.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if len(train) + len(test) != len(actual):
        fail(f"{horizon}s: train/test row accounting mismatch")

    if set(train.trial_id) & set(test.trial_id):
        fail(f"{horizon}s: train/test trial leakage")

    if set(train.trial_id) | set(test.trial_id) != set(actual.trial_id):
        fail(f"{horizon}s: train/test trial membership mismatch")

    print(
        f"PASS {horizon:2d}s | "
        f"rows={len(actual):3d} | "
        f"features={len(FEATURES)} | "
        f"train={len(train):3d} | test={len(test):3d} | "
        f"all values match"
    )

    return len(actual)


def main():
    manifest_path = DATA_DIR / "multi_horizon_v4_manifest.json"
    metadata_path = DATA_DIR / "trial_metadata.csv"

    if not BATCH_DIR.is_dir():
        fail(f"Source batch not found: {BATCH_DIR}")

    if not manifest_path.is_file() or not metadata_path.is_file():
        fail("Manifest or trial_metadata.csv missing")

    manifest = json.loads(manifest_path.read_text())
    metadata = pd.read_csv(metadata_path)

    if manifest.get("dataset_version") != "v4_multi_horizon":
        fail("Unexpected dataset version")

    if manifest.get("features") != FEATURES:
        fail("Feature schema differs from reviewed builder")

    if len(metadata) != 15:
        fail(f"Expected 15 selected trials, found {len(metadata)}")

    if metadata.trial_id.nunique() != 15:
        fail("Duplicate logical trial IDs in metadata")

    accepted = set(manifest["accepted_trials"])
    selected = set(metadata.trial_id)

    if accepted != selected:
        fail("Manifest and trial_metadata trial membership differ")

    print("V4 RAW-TO-PROCESSED RECONCILIATION")
    print(f"Source batch: {BATCH_DIR}")
    print(f"Selected logical trials: {len(metadata)}")
    print(f"Features checked: {len(FEATURES)}")
    print("-" * 72)

    total = 0
    for horizon in HORIZONS:
        rows = compare_horizon(horizon, metadata)
        total += rows

    print("-" * 72)
    print(f"TOTAL VERIFIED ROWS: {total}")
    print("V4 RAW-TO-PROCESSED RECONCILIATION PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nAUDIT FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
