from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BATCH_DIR = Path(
    "experiments/results/hpa_training/batch_20261121_155504"
)
OUTPUT_DIR = Path("datasets/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE = "predictive-autoscaler-app:v3"
HORIZON_STEPS = 6
EXPECTED_SAMPLES = 50
EXPECTED_TRIALS = 13

FEATURE_COLUMNS = [
    "app_cpu_millicores",
    "app_memory_mib",
    "work_rps",
    "work_rps_missing",
    "work_avg_latency_ms",
    "work_p95_latency_ms",
    "work_avg_latency_ms_missing",
    "work_p95_latency_ms_missing",
    "ready_replicas",
    "workload_active",
    "cpu_lag_1",
    "cpu_lag_2",
    "cpu_lag_3",
    "rps_lag_1",
    "rps_lag_2",
    "rps_lag_3",
]

REQUIRED_RAW_COLUMNS = [
    "elapsed_seconds",
    "app_cpu_millicores",
    "app_memory_mib",
    "ready_replicas",
    "work_rps",
    "work_avg_latency_ms",
    "work_p95_latency_ms",
    "collection_status",
]

all_frames = []
accepted = []
rejected = []

for metadata_path in sorted(BATCH_DIR.glob("*/experiment.json")):
    trial_dir = metadata_path.parent
    trial_id = trial_dir.name

    try:
        metadata = json.loads(metadata_path.read_text())

        # Strict v3 eligibility checks.
        if metadata.get("status") != "completed":
            raise ValueError("Experiment status is not completed")

        if metadata.get("application_image") != IMAGE:
            raise ValueError("Not a confirmed v3 image")

        if metadata.get("gunicorn_workers") != 2:
            raise ValueError("Unexpected Gunicorn worker count")

        if metadata.get("gunicorn_threads") != 4:
            raise ValueError("Unexpected Gunicorn thread count")

        if float(metadata.get("failures", -1)) != 0:
            raise ValueError("Locust failures are non-zero or missing")

        if metadata.get("metrics_samples") != EXPECTED_SAMPLES:
            raise ValueError("Unexpected metric sample count")

        if metadata.get("successful_metrics_samples") != EXPECTED_SAMPLES:
            raise ValueError("Not all metric samples succeeded")

        csv_path = trial_dir / "metrics.csv"
        if not csv_path.exists():
            raise ValueError("metrics.csv is missing")

        df = pd.read_csv(csv_path)

        missing_columns = set(REQUIRED_RAW_COLUMNS) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")

        if len(df) != EXPECTED_SAMPLES:
            raise ValueError(f"Expected 50 rows, found {len(df)}")

        if not (df["collection_status"] == "success").all():
            raise ValueError("CSV contains unsuccessful metric samples")

        df = df.sort_values("elapsed_seconds").reset_index(drop=True)

        # Confirm approximately five-second sampling intervals.
        intervals = df["elapsed_seconds"].diff().dropna()
        if not np.allclose(intervals, 5.0, atol=0.25):
            raise ValueError("Sampling intervals are not consistently 5 seconds")

        numeric_columns = [
            "elapsed_seconds",
            "app_cpu_millicores",
            "app_memory_mib",
            "ready_replicas",
            "work_rps",
            "work_avg_latency_ms",
            "work_p95_latency_ms",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        # Preserve missing-value indicators before imputation.
        for column in [
            "work_rps",
            "work_avg_latency_ms",
            "work_p95_latency_ms",
        ]:
            df[f"{column}_missing"] = df[column].isna().astype(int)
            df[column] = df[column].fillna(0)

        df["workload_active"] = (df["work_rps"] > 0).astype(int)

        # All lag and target operations are confined to this trial.
        for lag in (1, 2, 3):
            df[f"cpu_lag_{lag}"] = df["app_cpu_millicores"].shift(lag)
            df[f"rps_lag_{lag}"] = df["work_rps"].shift(lag)

        df["target_cpu_30s"] = (
            df["app_cpu_millicores"].shift(-HORIZON_STEPS)
        )

        df["trial_id"] = trial_id
        df["profile"] = metadata["profile"]
        df["max_users"] = metadata["max_users"]

        df = df.dropna(
            subset=FEATURE_COLUMNS + ["target_cpu_30s"]
        ).copy()

        if df.empty:
            raise ValueError("No usable rows after feature engineering")

        all_frames.append(df)
        accepted.append(trial_id)

    except Exception as exc:
        rejected.append((trial_id, str(exc)))


print("\n========== TRIAL AUDIT ==========")
print(f"Accepted trials: {len(accepted)}")
for trial in accepted:
    print(f"  ACCEPTED: {trial}")

print(f"\nRejected trials: {len(rejected)}")
for trial, reason in rejected:
    print(f"  REJECTED: {trial} — {reason}")

if len(accepted) != EXPECTED_TRIALS:
    raise SystemExit(
        f"Expected {EXPECTED_TRIALS} valid v3 trials; found {len(accepted)}. "
        "Dataset generation stopped."
    )

dataset = pd.concat(all_frames, ignore_index=True)

OUTPUT_COLUMNS = (
    FEATURE_COLUMNS
    + ["target_cpu_30s", "trial_id", "profile", "max_users"]
)

dataset = dataset[OUTPUT_COLUMNS].copy()

if dataset.isna().any().any():
    raise SystemExit("Dataset contains missing values. Generation stopped.")

if not np.isfinite(dataset.select_dtypes(include=[np.number])).all().all():
    raise SystemExit("Dataset contains infinite numeric values.")

# Fixed trial-level holdout for reproducible evaluation.
test_trials_expected = {
    "burst_08_users",
    "ramp_down_08_users",
    "ramp_up_16_users",
    "spike_12_users",
}

available_trials = set(dataset["trial_id"].unique())

if not test_trials_expected.issubset(available_trials):
    raise SystemExit("One or more required test trials are missing.")

test_mask = dataset["trial_id"].isin(test_trials_expected)

train = dataset.loc[~test_mask].copy()
test = dataset.loc[test_mask].copy()

train_trials = set(train["trial_id"])
test_trials = set(test["trial_id"])

if train_trials.intersection(test_trials):
    raise SystemExit("Train/test trial leakage detected.")

# Save full dataset and split files.
full_path = OUTPUT_DIR / "hpa_v3_cpu_prediction_dataset.csv"
train_path = OUTPUT_DIR / "hpa_v3_cpu_prediction_train.csv"
test_path = OUTPUT_DIR / "hpa_v3_cpu_prediction_test.csv"
manifest_path = OUTPUT_DIR / "hpa_v3_dataset_manifest.json"

dataset.to_csv(full_path, index=False)
train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)

manifest = {
    "dataset_version": "v3",
    "application_image": IMAGE,
    "gunicorn_workers": 2,
    "gunicorn_threads": 4,
    "prediction_horizon_seconds": HORIZON_STEPS * 5,
    "expected_trials": EXPECTED_TRIALS,
    "accepted_trials": accepted,
    "rejected_trials": [
        {"trial_id": trial, "reason": reason}
        for trial, reason in rejected
    ],
    "total_rows": len(dataset),
    "feature_count": len(FEATURE_COLUMNS),
    "train_rows": len(train),
    "test_rows": len(test),
    "train_trials": sorted(train_trials),
    "test_trials": sorted(test_trials),
    "random_state": 42,
}

manifest_path.write_text(json.dumps(manifest, indent=2))

print("\n========== DATASET SUMMARY ==========")
print(f"Full dataset: {full_path}")
print(f"Training set: {train_path}")
print(f"Test set: {test_path}")
print(f"Manifest: {manifest_path}")
print(f"Total rows: {len(dataset)}")
print(f"Features: {len(FEATURE_COLUMNS)}")
print(f"Training rows: {len(train)}")
print(f"Test rows: {len(test)}")
print(f"Training trials: {len(train_trials)}")
print(f"Test trials: {len(test_trials)}")
print(f"Train/test overlap: {train_trials.intersection(test_trials)}")

print("\nRows by workload profile:")
print(dataset.groupby("profile").size())

print("\nRows by trial:")
print(dataset.groupby("trial_id").size())

print("\nTarget CPU distribution:")
print(dataset["target_cpu_30s"].describe())

print("\nMissing values:", dataset.isna().sum().sum())
print("\nDataset generation complete.")
