import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)

TRAIN = "datasets/processed/hpa_v3_cpu_prediction_train.csv"
TEST = "datasets/processed/hpa_v3_cpu_prediction_test.csv"

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

target = "target_cpu_30s"
features = [
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

X_train = train[features]
y_train = train[target]
groups = train["trial_id"]

X_test = test[features]
y_test = test[target]

models = {
    "RandomForest_v1": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=42,
        n_jobs=2,
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=42,
        n_jobs=2,
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=200,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42,
    ),
}

def metrics(actual, predicted):
    return {
        "MAE": round(mean_absolute_error(actual, predicted), 2),
        "RMSE": round(root_mean_squared_error(actual, predicted), 2),
        "R2": round(r2_score(actual, predicted), 3),
    }

cv = GroupKFold(n_splits=5)

print("\n========== TRAINING-ONLY GROUP CV ==========")

results = {}

persistence_cv = train["app_cpu_millicores"].to_numpy()
results["Persistence"] = metrics(y_train, persistence_cv)

print("Persistence:", results["Persistence"])

for name, model in models.items():
    predictions = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        groups=groups,
        n_jobs=1,
    )

    results[name] = metrics(y_train, predictions)
    print(f"{name}: {results[name]}")

# Select the best ML model by training-only grouped CV MAE.
best_name = min(
    models,
    key=lambda name: results[name]["MAE"]
)

print(f"\nSelected model by CV MAE: {best_name}")

# Fit selected model on all training data.
best_model = models[best_name]
best_model.fit(X_train, y_train)

test_predictions = best_model.predict(X_test)
persistence_test = test["app_cpu_millicores"].to_numpy()

print("\n========== HELD-OUT TEST ==========")
print("Selected model:", metrics(y_test, test_predictions))
print("Persistence:", metrics(y_test, persistence_test))

print("\n========== TEST MAE IMPROVEMENT ==========")
rf_mae = metrics(y_test, test_predictions)["MAE"]
persistence_mae = metrics(y_test, persistence_test)["MAE"]

improvement = (
    (persistence_mae - rf_mae) / persistence_mae
) * 100

print(f"Selected model vs persistence: {improvement:.2f}%")

print("\nBenchmark complete.")
