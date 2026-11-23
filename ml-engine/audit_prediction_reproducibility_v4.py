#!/usr/bin/env python3

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path("datasets/processed/multi_horizon_v4")
MODEL_DIR = Path("models/multi_horizon/v4_run_001")
REPORT_DIR = Path("reports/multi_horizon/v4_run_001")

HORIZONS = (15, 30, 45, 60)

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

MODELS = {
    "RandomForest": "randomforest",
    "ExtraTrees": "extratrees",
}


def fail(message):
    raise AssertionError(message)


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def assert_close(label, expected, actual, atol=1e-8):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if expected.shape != actual.shape:
        fail(f"{label}: shape mismatch {expected.shape} != {actual.shape}")

    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        fail(f"{label}: non-finite values detected")

    if not np.allclose(expected, actual, rtol=1e-9, atol=atol):
        diff = np.abs(expected - actual)
        idx = int(np.argmax(diff))
        fail(
            f"{label}: mismatch. Max difference={diff[idx]:.12g}; "
            f"index={idx}; expected={expected[idx]}; actual={actual[idx]}"
        )


def main():
    heldout_path = REPORT_DIR / "heldout_metrics.csv"
    if not heldout_path.is_file():
        fail(f"Missing heldout metrics: {heldout_path}")

    heldout = pd.read_csv(heldout_path)
    verified_models = 0

    print("V4 SAVED-MODEL PREDICTION REPRODUCIBILITY")
    print("-" * 76)

    for horizon in HORIZONS:
        test_path = DATA_DIR / f"cpu_prediction_{horizon}s_test.csv"
        prediction_path = REPORT_DIR / f"predictions_{horizon}s.csv"

        if not test_path.is_file() or not prediction_path.is_file():
            fail(f"{horizon}s: missing test or prediction CSV")

        test = pd.read_csv(test_path)
        report = pd.read_csv(prediction_path)

        if len(test) != len(report):
            fail(f"{horizon}s: test/report row count mismatch")

        keys = ["trial_id", "profile", "max_users", "elapsed_seconds"]
        if not test[keys].equals(report[keys]):
            fail(f"{horizon}s: prediction report row alignment mismatch")

        target = f"target_cpu_{horizon}s"
        actual = test[target].to_numpy(dtype=float)

        assert_close(
            f"{horizon}s actual CPU",
            actual,
            report["actual_cpu"].to_numpy(dtype=float),
        )

        persistence = test["app_cpu_millicores"].to_numpy(dtype=float)
        assert_close(
            f"{horizon}s persistence",
            persistence,
            report["persistence_prediction"].to_numpy(dtype=float),
        )

        # Validate the saved persistence error column too.
        assert_close(
            f"{horizon}s persistence absolute error",
            np.abs(actual - persistence),
            report["persistence_absolute_error"].to_numpy(dtype=float),
        )

        for model_name, filename_prefix in MODELS.items():
            model_path = MODEL_DIR / f"{filename_prefix}_{horizon}s.joblib"
            if not model_path.is_file():
                fail(f"Missing saved model: {model_path}")

            model = joblib.load(model_path)
            regenerated = model.predict(test[FEATURES])
            prediction_col = f"{filename_prefix}_prediction"
            error_col = f"{filename_prefix}_absolute_error"

            assert_close(
                f"{horizon}s {model_name} predictions",
                regenerated,
                report[prediction_col].to_numpy(dtype=float),
            )

            assert_close(
                f"{horizon}s {model_name} absolute errors",
                np.abs(actual - regenerated),
                report[error_col].to_numpy(dtype=float),
            )

            metric_row = heldout[
                (heldout.horizon_seconds == horizon)
                & (heldout.model == model_name)
            ]

            if len(metric_row) != 1:
                fail(
                    f"{horizon}s {model_name}: expected one heldout "
                    f"metric row, found {len(metric_row)}"
                )

            recomputed = calculate_metrics(actual, regenerated)

            for metric_name, value in recomputed.items():
                saved_value = float(metric_row.iloc[0][metric_name])
                if not np.isclose(value, saved_value, rtol=1e-9, atol=1e-8):
                    fail(
                        f"{horizon}s {model_name} {metric_name} mismatch: "
                        f"recomputed={value}, saved={saved_value}"
                    )

            verified_models += 1
            print(
                f"PASS {horizon:2d}s | {model_name:12s} | "
                f"predictions, absolute errors and metrics match"
            )

        # Independently verify persistence held-out metrics.
        persistence_row = heldout[
            (heldout.horizon_seconds == horizon)
            & (heldout.model == "Persistence")
        ]

        if len(persistence_row) != 1:
            fail(f"{horizon}s: persistence heldout metric row missing")

        persistence_metrics = calculate_metrics(actual, persistence)

        for metric_name, value in persistence_metrics.items():
            saved_value = float(persistence_row.iloc[0][metric_name])
            if not np.isclose(value, saved_value, rtol=1e-9, atol=1e-8):
                fail(
                    f"{horizon}s persistence {metric_name} mismatch: "
                    f"recomputed={value}, saved={saved_value}"
                )

        print(f"PASS {horizon:2d}s | Persistence  | held-out metrics match")

    print("-" * 76)
    print(f"Saved ML models verified: {verified_models}")
    print("V4 PREDICTION REPRODUCIBILITY PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nAUDIT FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
