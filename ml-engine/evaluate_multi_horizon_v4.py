#!/usr/bin/env python3
"""Evaluate v4 multi-horizon CPU predictors without trial leakage."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

DATA_DIR = Path("datasets/processed/multi_horizon_v4")
MANIFEST_PATH = DATA_DIR / "multi_horizon_v4_manifest.json"
MODEL_DIR = Path("models/multi_horizon/v4_run_001")
REPORT_DIR = Path("reports/multi_horizon/v4_run_001")

HORIZONS = (15, 30, 45, 60)
RANDOM_STATE = 42
N_SPLITS = 5

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


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def make_models():
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
    }


def validate_manifest(manifest):
    if manifest.get("dataset_version") != "v4_multi_horizon":
        raise SystemExit("Unexpected dataset version.")
    if manifest.get("features") != FEATURES:
        raise SystemExit("Feature schema mismatch.")
    if manifest.get("feature_count") != len(FEATURES):
        raise SystemExit("Feature count mismatch.")

    train_trials = set(manifest["train_trials"])
    test_trials = set(manifest["test_trials"])
    if len(train_trials) != 11 or len(test_trials) != 4:
        raise SystemExit("Unexpected train/test trial count.")
    if train_trials & test_trials:
        raise SystemExit("Manifest contains trial leakage.")
    if len(train_trials | test_trials) != 15:
        raise SystemExit("Manifest does not describe 15 unique trials.")
    return train_trials, test_trials


def validate_frame(df, name, target, required_trials):
    required = FEATURES + [
        target, "trial_id", "profile", "max_users", "elapsed_seconds"
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise SystemExit(f"{name}: missing columns {sorted(missing)}")
    if df[required].isna().any().any():
        raise SystemExit(f"{name}: missing values found.")
    numeric = df[FEATURES + [target, "elapsed_seconds"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise SystemExit(f"{name}: non-finite numeric values.")
    if set(df["trial_id"]) != required_trials:
        raise SystemExit(f"{name}: trial membership mismatch.")
    if (df["app_cpu_millicores"] < 0).any():
        raise SystemExit(f"{name}: negative CPU values.")
    if (df[target] < 0).any():
        raise SystemExit(f"{name}: negative target CPU values.")


def main():
    if not MANIFEST_PATH.is_file():
        raise SystemExit(f"Manifest missing: {MANIFEST_PATH}")
    if MODEL_DIR.exists() or REPORT_DIR.exists():
        raise SystemExit(
            "v4_run_001 output already exists. Nothing will be overwritten."
        )

    manifest = json.loads(MANIFEST_PATH.read_text())
    train_trials, test_trials = validate_manifest(manifest)

    # Check all inputs before creating output directories.
    datasets = {}
    for horizon in HORIZONS:
        target = f"target_cpu_{horizon}s"
        train_path = DATA_DIR / f"cpu_prediction_{horizon}s_train.csv"
        test_path = DATA_DIR / f"cpu_prediction_{horizon}s_test.csv"
        if not train_path.is_file() or not test_path.is_file():
            raise SystemExit(f"Missing input CSV for {horizon}s.")
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        validate_frame(train, f"{horizon}s training", target, train_trials)
        validate_frame(test, f"{horizon}s testing", target, test_trials)
        if set(train.trial_id) & set(test.trial_id):
            raise SystemExit(f"{horizon}s train/test leakage.")
        datasets[horizon] = (train, test)

    cv_rows, test_rows, trial_rows = [], [], []
    predictions_by_horizon = {}
    models_by_horizon = {}
    summary = {
        "evaluation_version": "v4_multi_horizon_evaluation",
        "dataset_manifest": str(MANIFEST_PATH),
        "random_state": RANDOM_STATE,
        "cv_strategy": "GroupKFold",
        "cv_splits": N_SPLITS,
        "group_column": "trial_id",
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "train_trials": sorted(train_trials),
        "test_trials": sorted(test_trials),
        "model_selection_policy": (
            "Model comparison uses training-only grouped out-of-fold "
            "predictions. Held-out test results are reported separately."
        ),
        "horizons": {},
    }

    for horizon in HORIZONS:
        print(f"\n{'=' * 65}\nEVALUATING {horizon}-SECOND HORIZON\n{'=' * 65}")
        train, test = datasets[horizon]
        target = f"target_cpu_{horizon}s"

        X_train, y_train = train[FEATURES], train[target]
        X_test, y_test = test[FEATURES], test[target]
        groups = train["trial_id"]

        if groups.nunique() < N_SPLITS:
            raise SystemExit(f"Insufficient training groups for {horizon}s.")

        print(
            f"Training rows={len(train)} ({groups.nunique()} trials); "
            f"test rows={len(test)} ({test.trial_id.nunique()} trials)"
        )

        # Persistence has no fitted parameters. Its current-CPU prediction
        # is evaluated on training rows as a reference and on held-out rows.
        persistence_cv_pred = train["app_cpu_millicores"].to_numpy()
        persistence_test_pred = test["app_cpu_millicores"].to_numpy()
        persistence_cv_metrics = metrics(y_train, persistence_cv_pred)
        persistence_test_metrics = metrics(y_test, persistence_test_pred)

        cv_rows.append({
            "horizon_seconds": horizon,
            "model": "Persistence",
            "evaluation_scope": "training_reference_not_model_cv",
            "training_rows": len(train),
            **persistence_cv_metrics,
        })
        test_rows.append({
            "horizon_seconds": horizon,
            "model": "Persistence",
            "training_rows": len(train),
            "test_rows": len(test),
            **persistence_test_metrics,
        })

        print("Persistence training reference:", persistence_cv_metrics)
        print("Persistence held-out test:", persistence_test_metrics)

        model_predictions = {}
        horizon_models = {}

        for model_name, model in make_models().items():
            print(f"\nTraining-only grouped CV: {model_name}")
            oof = cross_val_predict(
                model,
                X_train,
                y_train,
                cv=GroupKFold(n_splits=N_SPLITS),
                groups=groups,
                n_jobs=1,
            )
            cv_metrics = metrics(y_train, oof)
            cv_rows.append({
                "horizon_seconds": horizon,
                "model": model_name,
                "evaluation_scope": "grouped_out_of_fold_training",
                "training_rows": len(train),
                **cv_metrics,
            })

            # Final model is fitted on all training trials only.
            model.fit(X_train, y_train)
            test_pred = model.predict(X_test)
            test_metrics = metrics(y_test, test_pred)
            test_rows.append({
                "horizon_seconds": horizon,
                "model": model_name,
                "training_rows": len(train),
                "test_rows": len(test),
                **test_metrics,
            })

            model_predictions[model_name] = test_pred
            horizon_models[model_name] = model
            print("Grouped CV:", cv_metrics)
            print("Held-out test:", test_metrics)

        pred_df = test[
            ["trial_id", "profile", "max_users", "elapsed_seconds"]
        ].copy()
        pred_df["actual_cpu"] = y_test.to_numpy()
        pred_df["persistence_prediction"] = persistence_test_pred
        pred_df["persistence_absolute_error"] = np.abs(
            y_test.to_numpy() - persistence_test_pred
        )

        for model_name, pred in model_predictions.items():
            key = model_name.lower()
            pred_df[f"{key}_prediction"] = pred
            pred_df[f"{key}_absolute_error"] = np.abs(
                y_test.to_numpy() - pred
            )

        for trial_id, group in pred_df.groupby("trial_id"):
            for model_name, pred_col in [
                ("Persistence", "persistence_prediction"),
                ("RandomForest", "randomforest_prediction"),
                ("ExtraTrees", "extratrees_prediction"),
            ]:
                trial_metric = metrics(group["actual_cpu"], group[pred_col])
                trial_rows.append({
                    "horizon_seconds": horizon,
                    "trial_id": trial_id,
                    "profile": group["profile"].iloc[0],
                    "max_users": int(group["max_users"].iloc[0]),
                    "model": model_name,
                    "samples": len(group),
                    **trial_metric,
                })

        predictions_by_horizon[horizon] = pred_df
        models_by_horizon[horizon] = horizon_models
        summary["horizons"][f"{horizon}s"] = {
            "training_rows": len(train),
            "test_rows": len(test),
            "training_trials": sorted(train.trial_id.unique()),
            "test_trials": sorted(test.trial_id.unique()),
            "target_column": target,
        }

    # Create output directories only after all evaluations succeed.
    MODEL_DIR.mkdir(parents=True)
    REPORT_DIR.mkdir(parents=True)

    for horizon in HORIZONS:
        predictions_by_horizon[horizon].to_csv(
            REPORT_DIR / f"predictions_{horizon}s.csv", index=False
        )
        for model_name, model in models_by_horizon[horizon].items():
            joblib.dump(
                model,
                MODEL_DIR / f"{model_name.lower()}_{horizon}s.joblib",
            )

    cv_df = pd.DataFrame(cv_rows)
    test_df = pd.DataFrame(test_rows)
    trial_df = pd.DataFrame(trial_rows)

    cv_df.to_csv(REPORT_DIR / "cross_validation_metrics.csv", index=False)
    test_df.to_csv(REPORT_DIR / "heldout_metrics.csv", index=False)
    trial_df.to_csv(REPORT_DIR / "per_trial_metrics.csv", index=False)

    summary["cross_validation_metrics"] = cv_rows
    summary["heldout_metrics"] = test_rows
    summary["per_trial_metrics"] = trial_rows
    summary_path = REPORT_DIR / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 75)
    print("HELD-OUT TEST COMPARISON")
    print("=" * 75)
    print(test_df[["horizon_seconds", "model", "MAE", "RMSE", "R2"]].to_string(index=False))

    print("\nTRAINING-ONLY GROUPED CV")
    print(cv_df[["horizon_seconds", "model", "evaluation_scope", "MAE", "RMSE", "R2"]].to_string(index=False))

    print(f"\nModels: {MODEL_DIR}")
    print(f"Reports: {REPORT_DIR}")
    print(f"Summary: {summary_path}")
    print("V4 evaluation complete.")


if __name__ == "__main__":
    main()
