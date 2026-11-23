#!/usr/bin/env python3

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "datasets/processed/multi_horizon_v4"
MODEL_DIR = ROOT / "models/multi_horizon/v4_run_001"
REPLAY_DIR = ROOT / "reports/multi_horizon/v4_run_001"

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

HORIZONS = (30, 60)
TARGET_CPU_PER_REPLICA = 60.0
MIN_REPLICAS = 1
MAX_REPLICAS = 5

REPLAY_CASES = [
    {
        "name": "heldout",
        "source": DATA_DIR / "cpu_prediction_30s_test.csv",
        "replay": REPLAY_DIR / "advisory_replay_heldout.csv",
        "expected_rows": 148,
    },
    {
        "name": "60s_test",
        "source": DATA_DIR / "cpu_prediction_60s_test.csv",
        "replay": REPLAY_DIR / "advisory_replay_60s_test.csv",
        "expected_rows": 135,
    },
]


def fail(message):
    raise AssertionError(message)


def close_check(label, expected, actual):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if expected.shape != actual.shape:
        fail(f"{label}: shape mismatch")

    if not np.isfinite(expected).all():
        fail(f"{label}: expected values contain non-finite values")

    if not np.isfinite(actual).all():
        fail(f"{label}: actual values contain non-finite values")

    if not np.allclose(expected, actual, rtol=1e-9, atol=1e-8):
        diff = np.abs(expected - actual)
        idx = int(np.argmax(diff))
        fail(
            f"{label}: mismatch at row {idx}; "
            f"expected={expected[idx]}, actual={actual[idx]}, "
            f"max_difference={diff[idx]}"
        )


def recommend_replicas(predicted_cpu):
    raw = int(np.ceil(max(0.0, float(predicted_cpu)) / TARGET_CPU_PER_REPLICA))
    return min(MAX_REPLICAS, max(MIN_REPLICAS, raw))


def audit_case(case, models):
    source_path = case["source"]
    replay_path = case["replay"]

    if not source_path.is_file():
        fail(f"Missing source dataset: {source_path}")

    if not replay_path.is_file():
        fail(f"Missing replay file: {replay_path}")

    source = pd.read_csv(source_path)
    replay = pd.read_csv(replay_path)

    if len(source) != case["expected_rows"]:
        fail(
            f"{case['name']}: source row count changed; "
            f"expected={case['expected_rows']}, actual={len(source)}"
        )

    if len(replay) != case["expected_rows"]:
        fail(
            f"{case['name']}: replay row count changed; "
            f"expected={case['expected_rows']}, actual={len(replay)}"
        )

    # The advisor replay should preserve every original source column/value.
    for col in source.columns:
        if col not in replay.columns:
            fail(f"{case['name']}: replay missing source column {col}")

        left = source[col]
        right = replay[col]

        if pd.api.types.is_numeric_dtype(left):
            close_check(
                f"{case['name']} source column {col}",
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
            )
        elif not left.fillna("<NA>").equals(right.fillna("<NA>")):
            fail(f"{case['name']}: source column changed: {col}")

    if not set(FEATURES).issubset(replay.columns):
        fail(f"{case['name']}: one or more model features are missing")

    X = replay[FEATURES].astype(float)

    for horizon in HORIZONS:
        prediction_col = f"predicted_cpu_{horizon}s_millicores"
        recommendation_col = f"recommended_replicas_{horizon}s"

        if prediction_col not in replay.columns:
            fail(f"{case['name']}: missing {prediction_col}")

        if recommendation_col not in replay.columns:
            fail(f"{case['name']}: missing {recommendation_col}")

        regenerated = np.maximum(0.0, models[horizon].predict(X))

        close_check(
            f"{case['name']} {horizon}s predictions",
            regenerated,
            replay[prediction_col].to_numpy(dtype=float),
        )

        expected_recommendations = np.array(
            [recommend_replicas(value) for value in regenerated],
            dtype=int,
        )

        actual_recommendations = replay[recommendation_col].to_numpy(dtype=int)

        if not np.array_equal(expected_recommendations, actual_recommendations):
            mismatch = np.where(
                expected_recommendations != actual_recommendations
            )[0]
            idx = int(mismatch[0])
            fail(
                f"{case['name']} {horizon}s recommendation mismatch at row "
                f"{idx}: expected={expected_recommendations[idx]}, "
                f"actual={actual_recommendations[idx]}"
            )

        if (
            (actual_recommendations < MIN_REPLICAS).any()
            or (actual_recommendations > MAX_REPLICAS).any()
        ):
            fail(f"{case['name']} {horizon}s recommendation outside 1–5")

        print(
            f"PASS {case['name']:8s} | {horizon:2d}s | "
            f"predictions match | recommendations match | "
            f"rows={len(replay)}"
        )


def main():
    models = {}

    for horizon in HORIZONS:
        model_path = MODEL_DIR / f"extratrees_{horizon}s.joblib"
        if not model_path.is_file():
            fail(f"Missing model: {model_path}")
        models[horizon] = joblib.load(model_path)

    print("V4 PREDICTIVE SCALING ADVISOR AUDIT")
    print(f"Target CPU per replica: {TARGET_CPU_PER_REPLICA}m")
    print(f"Replica bounds: {MIN_REPLICAS}–{MAX_REPLICAS}")
    print("-" * 76)

    for case in REPLAY_CASES:
        audit_case(case, models)

    print("-" * 76)
    print("Replay source data: unchanged")
    print("Model predictions: independently reproduced")
    print("Replica formula: independently verified")
    print("V4 ADVISOR REPLAY AUDIT PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nAUDIT FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
