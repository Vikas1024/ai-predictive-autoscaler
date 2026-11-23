#!/usr/bin/env python3
"""Build immutable-source v4 multi-horizon CPU prediction datasets."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BATCH_DIR = Path("experiments/results/hpa_training/batch_20261123_141907")
OUTPUT_DIR = Path("datasets/processed/multi_horizon_v4")
IMAGE = "predictive-autoscaler-app:v4"
EXPECTED_SAMPLES = 50
INTERVAL = 5.0
HORIZONS = (15, 30, 45, 60)
EXPECTED_TRIALS = {
    f"{profile}_{users:02d}_users"
    for profile in ("burst", "constant", "ramp_down", "ramp_up", "spike")
    for users in (8, 12, 16)
}
TEST_TRIALS = {
    "burst_08_users",
    "constant_12_users",
    "ramp_down_16_users",
    "spike_08_users",
}
FEATURES = [
    "app_cpu_millicores", "app_memory_mib", "work_rps",
    "work_avg_latency_ms", "work_p95_latency_ms", "ready_replicas",
    "workload_active", "cpu_lag_1", "cpu_lag_2", "cpu_lag_3",
    "rps_lag_1", "rps_lag_2", "rps_lag_3",
]
RAW_NUMERIC = [
    "elapsed_seconds", "app_cpu_millicores", "app_memory_mib",
    "ready_replicas", "work_rps", "work_avg_latency_ms",
    "work_p95_latency_ms",
]
TELEMETRY = ["work_rps", "work_avg_latency_ms", "work_p95_latency_ms"]
REQUIRED = RAW_NUMERIC + ["collection_status"]


def stop(message):
    raise SystemExit(f"ERROR: {message}")


def select_attempts():
    if not BATCH_DIR.is_dir():
        stop(f"Source batch not found: {BATCH_DIR}")
    candidates, ignored = {}, []
    for mp in sorted(BATCH_DIR.glob("*/experiment.json")):
        meta = json.loads(mp.read_text())
        logical_id = meta.get("trial_id")
        if not logical_id:
            stop(f"Missing trial_id in {mp}")
        if meta.get("status") != "completed":
            ignored.append({
                "directory": str(mp.parent),
                "trial_id": logical_id,
                "attempt": meta.get("attempt"),
                "status": meta.get("status"),
                "reason": "not_completed",
            })
            continue
        candidates.setdefault(logical_id, []).append(
            (int(meta.get("attempt", 1)), mp, meta)
        )
    selected = {}
    for logical_id, attempts in candidates.items():
        selected[logical_id] = sorted(attempts, key=lambda x: x[0])[-1]
    actual = set(selected)
    if actual != EXPECTED_TRIALS:
        stop(
            "Completed logical trial mismatch. "
            f"Missing={sorted(EXPECTED_TRIALS-actual)}; "
            f"unexpected={sorted(actual-EXPECTED_TRIALS)}"
        )
    train = actual - TEST_TRIALS
    if len(train) != 11 or len(TEST_TRIALS) != 4:
        stop("Unexpected train/test split size.")
    return selected, ignored, train


def load_trial(logical_id, metadata_path, meta):
    try:
        profile, users_text, _ = logical_id.rsplit("_", 2)
        users = int(users_text)
    except Exception:
        stop(f"Unexpected logical trial ID: {logical_id}")

    checks = {
        "image": meta.get("application_image") == IMAGE,
        "workers": meta.get("gunicorn_workers") == 2,
        "threads": meta.get("gunicorn_threads") == 4,
        "profile": meta.get("profile") == profile,
        "max_users": meta.get("max_users") == users,
        "duration": meta.get("duration_seconds") == 240,
        "interval": meta.get("collector_interval_seconds") == 5,
        "controller": meta.get("controller") == "Kubernetes HPA",
        "exit_code": meta.get("locust_exit_code") == 0,
        "failures": float(meta.get("failures", -1)) == 0.0,
        "samples": meta.get("metrics_samples") == EXPECTED_SAMPLES,
        "successful_samples": meta.get("successful_metrics_samples") == EXPECTED_SAMPLES,
    }
    bad = [k for k, v in checks.items() if not v]
    if bad:
        stop(f"{logical_id}: metadata validation failed: {bad}")

    csv_path = metadata_path.parent / "metrics.csv"
    if not csv_path.is_file():
        stop(f"{logical_id}: metrics.csv missing")
    df = pd.read_csv(csv_path)
    missing_cols = set(REQUIRED) - set(df.columns)
    if missing_cols:
        stop(f"{logical_id}: missing columns {sorted(missing_cols)}")
    if len(df) != EXPECTED_SAMPLES:
        stop(f"{logical_id}: expected 50 rows, got {len(df)}")
    if not (df["collection_status"] == "success").all():
        stop(f"{logical_id}: non-success collection_status")

    for col in RAW_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    critical = ["elapsed_seconds", "app_cpu_millicores", "app_memory_mib", "ready_replicas"]
    if df[critical].isna().any().any():
        stop(f"{logical_id}: missing critical CPU/memory/replica/time data")
    if np.isinf(df[RAW_NUMERIC].to_numpy(dtype=float)).any():
        stop(f"{logical_id}: infinite numeric value")
    if (df[["app_cpu_millicores", "app_memory_mib", "ready_replicas"]] < 0).any().any():
        stop(f"{logical_id}: negative CPU/memory/replica value")
    for col in TELEMETRY:
        if (df[col].dropna() < 0).any():
            stop(f"{logical_id}: negative {col} value")

    df = df.sort_values("elapsed_seconds").reset_index(drop=True)
    intervals = df.elapsed_seconds.diff().dropna().to_numpy()
    if not np.allclose(intervals, INTERVAL, atol=0.25):
        stop(f"{logical_id}: sampling interval is not consistently 5 seconds")

    df["trial_id"] = logical_id
    df["profile"] = meta["profile"]
    df["max_users"] = int(meta["max_users"])
    df["source_attempt"] = int(meta.get("attempt", 1))
    df["source_directory"] = str(metadata_path.parent)
    return df


def make_features(raw):
    df = raw.copy()
    for lag in (1, 2, 3):
        df[f"cpu_lag_{lag}"] = df.app_cpu_millicores.shift(lag)
        df[f"rps_lag_{lag}"] = df.work_rps.shift(lag)
    # Missing RPS remains missing; it is never interpreted as inactive.
    df["workload_active"] = np.where(
        df.work_rps.notna(), (df.work_rps > 0).astype(int), np.nan
    )
    return df


def main():
    selected, ignored, train_trials = select_attempts()
    if OUTPUT_DIR.exists():
        stop(f"Output directory already exists: {OUTPUT_DIR}. Nothing overwritten.")

    frames, metadata_rows, quality_rows = {}, [], []
    for logical_id in sorted(selected):
        attempt, mp, meta = selected[logical_id]
        df = make_features(load_trial(logical_id, mp, meta))
        frames[logical_id] = df
        metadata_rows.append({
            "trial_id": logical_id, "profile": meta["profile"],
            "max_users": meta["max_users"], "selected_attempt": attempt,
            "source_directory": str(mp.parent), "status": meta["status"],
            "requests": meta.get("requests"), "locust_failures": meta.get("failures"),
            "locust_average_response_ms": meta.get("avg_response_ms"),
            "locust_p95_response_ms": meta.get("p95_response_ms"),
            "locust_requests_per_second": meta.get("requests_per_second"),
            "metrics_samples": meta.get("metrics_samples"),
            "successful_metrics_samples": meta.get("successful_metrics_samples"),
        })
        quality_rows.append({
            "trial_id": logical_id, "profile": meta["profile"],
            "max_users": meta["max_users"], "source_rows": len(df),
            "missing_work_rps": int(df.work_rps.isna().sum()),
            "missing_work_avg_latency_ms": int(df.work_avg_latency_ms.isna().sum()),
            "missing_work_p95_latency_ms": int(df.work_p95_latency_ms.isna().sum()),
            "missing_app_cpu_millicores": int(df.app_cpu_millicores.isna().sum()),
            "missing_app_memory_mib": int(df.app_memory_mib.isna().sum()),
            "source_attempt": attempt, "source_directory": str(mp.parent),
        })

    generated, sample_audit = {}, []
    for horizon in HORIZONS:
        steps = horizon // int(INTERVAL)
        if steps * INTERVAL != horizon:
            stop(f"Horizon {horizon} is not aligned to sampling interval.")
        pieces = []
        for trial_id, base in frames.items():
            df = base.copy()
            target = f"target_cpu_{horizon}s"
            df[target] = df.app_cpu_millicores.shift(-steps)
            feature_ok = df[FEATURES].notna().all(axis=1)
            target_ok = df[target].notna()
            keep = feature_ok & target_ok
            kept = df.loc[keep].copy()
            if kept.empty:
                stop(f"No usable rows for {trial_id}, horizon {horizon}s.")
            sample_audit.append({
                "trial_id": trial_id, "profile": kept.profile.iloc[0],
                "max_users": int(kept.max_users.iloc[0]),
                "horizon_seconds": horizon, "source_rows": len(df),
                "excluded_missing_features": int((~feature_ok).sum()),
                "excluded_future_target": int((feature_ok & ~target_ok).sum()),
                "included_rows": len(kept),
                "missing_work_rps_source_rows": int(df.work_rps.isna().sum()),
                "missing_avg_latency_source_rows": int(df.work_avg_latency_ms.isna().sum()),
                "missing_p95_latency_source_rows": int(df.work_p95_latency_ms.isna().sum()),
                "source_attempt": int(kept.source_attempt.iloc[0]),
                "source_directory": kept.source_directory.iloc[0],
            })
            pieces.append(kept)

        full = pd.concat(pieces, ignore_index=True)
        cols = FEATURES + [target, "trial_id", "profile", "max_users", "elapsed_seconds"]
        full = full[cols].copy()
        if full.isna().any().any():
            stop(f"Missing values remain in {horizon}s dataset.")
        if not np.isfinite(full.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all():
            stop(f"Non-finite values in {horizon}s dataset.")

        train = full[full.trial_id.isin(train_trials)].copy()
        test = full[full.trial_id.isin(TEST_TRIALS)].copy()
        if set(train.trial_id) != train_trials or set(test.trial_id) != TEST_TRIALS:
            stop(f"Trial membership mismatch at {horizon}s.")
        if set(train.trial_id) & set(test.trial_id) or len(train) + len(test) != len(full):
            stop(f"Train/test leakage or row accounting error at {horizon}s.")
        generated[horizon] = (full, train, test, steps)

    manifest = {
        "dataset_version": "v4_multi_horizon",
        "source_batch": str(BATCH_DIR),
        "application_image": IMAGE,
        "controller": "Kubernetes HPA",
        "sampling_interval_seconds": INTERVAL,
        "expected_samples_per_trial": EXPECTED_SAMPLES,
        "logical_trial_count": len(selected),
        "accepted_trials": sorted(selected),
        "ignored_incomplete_attempts": ignored,
        "train_trials": sorted(train_trials),
        "test_trials": sorted(TEST_TRIALS),
        "split_strategy": "Fixed trial-level holdout: 11 train trials, 4 test trials. No trial is shared between partitions.",
        "test_split_limitation": "Four held-out trials cover four of five workload profiles; interpret test metrics cautiously.",
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "missing_data_policy": {
            "raw_data_modified": False,
            "imputation": "None",
            "zero_filled_missing_telemetry": False,
            "sample_rule": "Exclude rows with missing current RPS/latency, any of three historical RPS lag values, or future CPU target.",
            "workload_active": "Derived only from observed current RPS; missing RPS is not treated as inactive.",
            "audit_files": ["trial_quality.csv", "horizon_sample_audit.csv"],
        },
        "horizons": {},
    }

    OUTPUT_DIR.mkdir(parents=True)
    pd.DataFrame(metadata_rows).to_csv(OUTPUT_DIR / "trial_metadata.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(OUTPUT_DIR / "trial_quality.csv", index=False)
    pd.DataFrame(sample_audit).to_csv(OUTPUT_DIR / "horizon_sample_audit.csv", index=False)

    for horizon, (full, train, test, steps) in generated.items():
        train_path = OUTPUT_DIR / f"cpu_prediction_{horizon}s_train.csv"
        test_path = OUTPUT_DIR / f"cpu_prediction_{horizon}s_test.csv"
        all_path = OUTPUT_DIR / f"cpu_prediction_{horizon}s_all.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        full.to_csv(all_path, index=False)
        manifest["horizons"][f"{horizon}s"] = {
            "prediction_horizon_seconds": horizon,
            "target_column": f"target_cpu_{horizon}s",
            "shift_steps": steps, "total_rows": len(full),
            "train_rows": len(train), "test_rows": len(test),
            "train_trials": sorted(train.trial_id.unique()),
            "test_trials": sorted(test.trial_id.unique()),
            "train_file": str(train_path), "test_file": str(test_path),
            "all_file": str(all_path),
        }

    manifest_path = OUTPUT_DIR / "multi_horizon_v4_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n========== V4 DATASET SUMMARY ==========")
    print(f"Accepted logical trials: {len(selected)}")
    print(f"Training trials: {len(train_trials)}")
    print(f"Testing trials: {len(TEST_TRIALS)}")
    print(f"Features: {len(FEATURES)}")
    for horizon in HORIZONS:
        full, train, test, _ = generated[horizon]
        print(f"{horizon}s: total={len(full)} train={len(train)} test={len(test)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")
    print("Dataset generation complete.")


if __name__ == "__main__":
    main()
