from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("datasets/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/model_evaluation")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "hpa_v3_cpu_prediction_train.csv"
TEST_PATH = DATA_DIR / "hpa_v3_cpu_prediction_test.csv"

MODEL_PATH = MODEL_DIR / "random_forest_cpu_v1.joblib"

FEATURES = [
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

TARGET = "target_cpu_30s"
RANDOM_STATE = 42

# --------------------------------------------------
# Load and validate data
# --------------------------------------------------

print("\n========== LOADING DATA ==========")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

required = FEATURES + [TARGET, "trial_id", "profile", "max_users"]

for name, df in [("Training", train), ("Testing", test)]:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} dataset missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError(f"{name} dataset contains missing values")

    if not np.isfinite(df[FEATURES + [TARGET]].to_numpy()).all():
        raise ValueError(f"{name} dataset contains non-finite values")

train_trials = set(train["trial_id"])
test_trials = set(test["trial_id"])

if train_trials.intersection(test_trials):
    raise ValueError("Train/test trial leakage detected")

if train["trial_id"].nunique() < 5:
    raise ValueError("At least five training trials are required")

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

print(f"Training rows: {len(train)}")
print(f"Testing rows: {len(test)}")
print(f"Training trials: {len(train_trials)}")
print(f"Testing trials: {len(test_trials)}")
print(f"Features: {len(FEATURES)}")

# --------------------------------------------------
# Group cross-validation
# --------------------------------------------------

print("\n========== GROUP CROSS-VALIDATION ==========")

cv = GroupKFold(n_splits=5)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    max_features=1.0,
    random_state=RANDOM_STATE,
    n_jobs=2,
)

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    groups=train["trial_id"],
    cv=cv,
    scoring={
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    },
    n_jobs=1,
    return_train_score=False,
)

cv_report = pd.DataFrame({
    "fold": range(1, 6),
    "mae": -cv_results["test_mae"],
    "rmse": -cv_results["test_rmse"],
    "r2": cv_results["test_r2"],
})

cv_report.to_csv(REPORT_DIR / "cross_validation_metrics.csv", index=False)

print(cv_report.round(3).to_string(index=False))

cv_summary = {
    "mae_mean": float(cv_report["mae"].mean()),
    "mae_std": float(cv_report["mae"].std()),
    "rmse_mean": float(cv_report["rmse"].mean()),
    "rmse_std": float(cv_report["rmse"].std()),
    "r2_mean": float(cv_report["r2"].mean()),
    "r2_std": float(cv_report["r2"].std()),
}

print("\nCross-validation summary:")
print(json.dumps(cv_summary, indent=2))

# --------------------------------------------------
# Fit final model on all training data
# --------------------------------------------------

print("\n========== FINAL MODEL TRAINING ==========")

model.fit(X_train, y_train)

# --------------------------------------------------
# Independent test evaluation
# --------------------------------------------------

print("\n========== HELD-OUT TEST EVALUATION ==========")

predictions = model.predict(X_test)

# Mean-prediction baseline.
baseline_prediction = np.full(
    len(y_test),
    y_train.mean(),
)

def calculate_metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }

model_metrics = calculate_metrics(y_test, predictions)
baseline_metrics = calculate_metrics(y_test, baseline_prediction)

print("Random Forest:")
print(json.dumps(model_metrics, indent=2))

print("\nMean baseline:")
print(json.dumps(baseline_metrics, indent=2))

mae_improvement = (
    (baseline_metrics["mae"] - model_metrics["mae"])
    / baseline_metrics["mae"] * 100
    if baseline_metrics["mae"] > 0 else None
)

print(f"\nMAE improvement over baseline: {mae_improvement:.2f}%"
      if mae_improvement is not None
      else "\nMAE improvement unavailable")

# --------------------------------------------------
# Save predictions
# --------------------------------------------------

prediction_output = test[
    ["trial_id", "profile", "max_users"]
].copy()

prediction_output["actual_cpu_30s"] = y_test.to_numpy()
prediction_output["predicted_cpu_30s"] = predictions
prediction_output["baseline_cpu_30s"] = baseline_prediction
prediction_output["absolute_error"] = np.abs(
    y_test.to_numpy() - predictions
)

prediction_output["sample_order"] = (
    prediction_output.groupby("trial_id").cumcount() + 1
)

prediction_output.to_csv(
    REPORT_DIR / "predictions.csv",
    index=False,
)

# --------------------------------------------------
# Per-trial metrics
# --------------------------------------------------

trial_metrics = []

for trial_id, group in prediction_output.groupby("trial_id"):
    trial_metrics.append({
        "trial_id": trial_id,
        "profile": group["profile"].iloc[0],
        "max_users": int(group["max_users"].iloc[0]),
        **calculate_metrics(
            group["actual_cpu_30s"],
            group["predicted_cpu_30s"],
        ),
    })

trial_report = pd.DataFrame(trial_metrics)
trial_report.to_csv(
    REPORT_DIR / "per_trial_metrics.csv",
    index=False,
)

print("\nPer-trial test metrics:")
print(trial_report.round(3).to_string(index=False))

# --------------------------------------------------
# Actual vs predicted chart
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.65,
)

lower = min(y_test.min(), predictions.min())
upper = max(y_test.max(), predictions.max())

plt.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--",
)

plt.xlabel("Actual CPU 30 seconds ahead (millicores)")
plt.ylabel("Predicted CPU (millicores)")
plt.title("Random Forest: Actual vs Predicted CPU")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "actual_vs_predicted.png",
    dpi=150,
)
plt.close()

# --------------------------------------------------
# Prediction timeline by held-out trial
# --------------------------------------------------

for trial_id, group in prediction_output.groupby("trial_id"):
    plt.figure(figsize=(9, 4))

    plt.plot(
        group["sample_order"],
        group["actual_cpu_30s"],
        label="Actual CPU",
        marker="o",
        markersize=3,
    )

    plt.plot(
        group["sample_order"],
        group["predicted_cpu_30s"],
        label="Predicted CPU",
        marker="x",
        markersize=3,
    )

    plt.xlabel("Sample order within trial")
    plt.ylabel("CPU (millicores)")
    plt.title(f"CPU Prediction — {trial_id}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_name = trial_id.replace("/", "_")

    plt.savefig(
        REPORT_DIR / f"timeline_{safe_name}.png",
        dpi=150,
    )
    plt.close()

# --------------------------------------------------
# Feature importance
# --------------------------------------------------

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

importance.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False,
)

print("\nTop feature importances:")
print(importance.head(10).round(4).to_string(index=False))

# --------------------------------------------------
# Save model and evaluation report
# --------------------------------------------------

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "prediction_horizon_seconds": 30,
        "model_type": "RandomForestRegressor",
        "random_state": RANDOM_STATE,
    },
    MODEL_PATH,
)

evaluation = {
    "model": "RandomForestRegressor",
    "model_parameters": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": 1.0,
        "random_state": RANDOM_STATE,
    },
    "prediction_horizon_seconds": 30,
    "training_rows": len(train),
    "testing_rows": len(test),
    "training_trials": sorted(train_trials),
    "testing_trials": sorted(test_trials),
    "feature_count": len(FEATURES),
    "cross_validation": cv_summary,
    "test_metrics": model_metrics,
    "baseline_metrics": baseline_metrics,
    "mae_improvement_over_baseline_percent": mae_improvement,
}

(REPORT_DIR / "metrics.json").write_text(
    json.dumps(evaluation, indent=2)
)

print("\n========== TRAINING COMPLETE ==========")
print(f"Model saved: {MODEL_PATH}")
print(f"Reports saved: {REPORT_DIR}")
