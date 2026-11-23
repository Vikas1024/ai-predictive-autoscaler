#!/usr/bin/env python3
"""
Predictive scaling advisor (read-only).

Loads the validated v4 Extra Trees models and converts predicted aggregate
application CPU into advisory replica counts. It never calls kubectl and
never changes Kubernetes resources.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "multi_horizon" / "v4_run_001"
MANIFEST_PATH = ROOT / "datasets" / "processed" / "multi_horizon_v4" / "multi_horizon_v4_manifest.json"

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

CPU_REQUEST_MILLICORES = 100.0
HPA_TARGET_UTILIZATION_PERCENT = 60.0
MIN_REPLICAS = 1
MAX_REPLICAS = 5
HORIZONS = (30, 60)


def load_manifest():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != "v4_multi_horizon":
        raise ValueError("Unexpected dataset manifest version.")
    if manifest.get("features") != FEATURES:
        raise ValueError("Model feature schema does not match the v4 manifest.")
    return manifest


def load_models():
    models = {}
    for horizon in HORIZONS:
        path = MODEL_DIR / f"extratrees_{horizon}s.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Trained model not found: {path}")
        models[horizon] = joblib.load(path)
    return models


def validate_features(payload):
    missing = [name for name in FEATURES if name not in payload]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")

    values = {}
    for name in FEATURES:
        try:
            value = float(payload[name])
        except (TypeError, ValueError):
            raise ValueError(f"Feature '{name}' must be numeric.") from None
        if not math.isfinite(value):
            raise ValueError(f"Feature '{name}' must be finite.")
        values[name] = value

    nonnegative = [
        "app_cpu_millicores", "app_memory_mib", "work_rps",
        "work_avg_latency_ms", "work_p95_latency_ms",
        "ready_replicas", "cpu_lag_1", "cpu_lag_2", "cpu_lag_3",
        "rps_lag_1", "rps_lag_2", "rps_lag_3",
    ]
    for name in nonnegative:
        if values[name] < 0:
            raise ValueError(f"Feature '{name}' cannot be negative.")
    if values["workload_active"] not in (0.0, 1.0):
        raise ValueError("workload_active must be 0 or 1.")
    if values["ready_replicas"] < 1:
        raise ValueError("ready_replicas must be at least 1 for inference.")
    return values


def recommend_replicas(predicted_cpu_millicores):
    target_cpu_per_replica = (
        CPU_REQUEST_MILLICORES * HPA_TARGET_UTILIZATION_PERCENT / 100.0
    )
    raw = math.ceil(max(0.0, predicted_cpu_millicores) / target_cpu_per_replica)
    return min(MAX_REPLICAS, max(MIN_REPLICAS, raw))


def advise(payload, models):
    features = validate_features(payload)
    X = pd.DataFrame([[features[name] for name in FEATURES]], columns=FEATURES)

    results = {}
    for horizon, model in models.items():
        prediction = float(model.predict(X)[0])
        if not math.isfinite(prediction):
            raise ValueError(f"{horizon}s model returned a non-finite prediction.")
        prediction = max(0.0, prediction)
        recommendation = recommend_replicas(prediction)
        results[f"{horizon}s"] = {
            "predicted_aggregate_cpu_millicores": round(prediction, 2),
            "target_cpu_per_replica_millicores": 60.0,
            "recommended_replicas": recommendation,
        }

    return {
        "mode": "advisory_only",
        "controller": "Kubernetes HPA (unchanged)",
        "model_family": "ExtraTreesRegressor",
        "current_ready_replicas": int(features["ready_replicas"]),
        "replica_bounds": {"minimum": MIN_REPLICAS, "maximum": MAX_REPLICAS},
        "hpa_reference": {
            "cpu_request_per_replica_millicores": CPU_REQUEST_MILLICORES,
            "target_utilization_percent": HPA_TARGET_UTILIZATION_PERCENT,
            "target_cpu_per_replica_millicores": 60.0,
        },
        "recommendations": results,
        "safety_note": (
            "Recommendations are estimates only. No Kubernetes API or kubectl "
            "write operation is performed. HPA remains the active controller. "
            "HPA tolerance, stabilization and rate limits are not simulated."
        ),
    }


def replay_csv(input_path, output_path, models):
    df = pd.read_csv(input_path)
    missing = [name for name in FEATURES if name not in df.columns]
    if missing:
        raise ValueError(f"Replay CSV missing features: {', '.join(missing)}")

    output = df.copy()
    for horizon, model in models.items():
        X = output[FEATURES].astype(float)
        predicted = np.maximum(0.0, model.predict(X))
        output[f"predicted_cpu_{horizon}s_millicores"] = predicted
        output[f"recommended_replicas_{horizon}s"] = [
            recommend_replicas(value) for value in predicted
        ]

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return len(output)


def main():
    parser = argparse.ArgumentParser(
        description="Read-only predictive scaling advisor using v4 Extra Trees models."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input-json", help="Path to one JSON object containing all 13 features.")
    mode.add_argument("--replay-csv", help="Replay a CSV containing the 13 features.")
    parser.add_argument(
        "--output",
        help="Output CSV path for replay mode. Refuses to overwrite.",
    )
    args = parser.parse_args()

    load_manifest()
    models = load_models()

    if args.input_json:
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        result = advise(payload, models)
        print(json.dumps(result, indent=2))
        return

    if not args.output:
        parser.error("--output is required with --replay-csv")

    count = replay_csv(args.replay_csv, args.output, models)
    print(f"Replay complete. Rows: {count}")
    print(f"Saved advisory predictions to: {args.output}")
    print("No Kubernetes resources were modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
