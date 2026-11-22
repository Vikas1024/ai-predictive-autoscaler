import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


APP_LABEL = "app=predictive-autoscaler-app"
DEPLOYMENT = "predictive-autoscaler-app"


def run_kubectl(args):
    result = subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return result.stdout.strip()


def parse_cpu(value):
    if value.endswith("n"):
        return float(value[:-1]) / 1_000_000
    if value.endswith("u"):
        return float(value[:-1]) / 1_000
    if value.endswith("m"):
        return float(value[:-1])
    return float(value) * 1000


def parse_memory(value):
    units = {
        "Ki": 1 / 1024,
        "Mi": 1,
        "Gi": 1024,
        "Ti": 1024 * 1024,
        "K": 1000 / (1024 * 1024),
        "M": 1000 / 1024,
        "G": 1000 * 1000 / 1024,
    }

    for suffix, multiplier in units.items():
        if value.endswith(suffix):
            return float(value[:-len(suffix)]) * multiplier

    return float(value) / (1024 * 1024)


def get_pod_metrics():
    output = run_kubectl([
        "top", "pods",
        "-l", APP_LABEL,
        "--no-headers",
    ])

    total_cpu = 0.0
    total_memory = 0.0

    for line in output.splitlines():
        fields = line.split()

        if len(fields) >= 3:
            total_cpu += parse_cpu(fields[1])
            total_memory += parse_memory(fields[2])

    return total_cpu, total_memory


def get_replica_metrics():
    output = run_kubectl([
        "get", "deployment", DEPLOYMENT, "-o", "json"
    ])

    deployment = json.loads(output)
    status = deployment.get("status", {})

    return {
        "desired_replicas": deployment["spec"].get("replicas", 0),
        "ready_replicas": status.get("readyReplicas", 0),
        "available_replicas": status.get("availableReplicas", 0),
    }


def prometheus_query(base_url, query):
    params = urlencode({"query": query})
    url = f"{base_url.rstrip('/')}/api/v1/query?{params}"

    with urlopen(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("status") != "success":
        raise RuntimeError("Prometheus query failed")

    results = payload["data"]["result"]

    if not results:
        return None

    return float(results[0]["value"][1])


def get_prometheus_metrics(base_url):
    work_rps = prometheus_query(
        base_url,
        'sum(rate(http_requests_total{endpoint="work"}[1m]))'
    )

    average_latency = prometheus_query(
        base_url,
        """
        sum(rate(http_request_duration_seconds_sum{endpoint="work"}[1m]))
        /
        sum(rate(http_request_duration_seconds_count{endpoint="work"}[1m]))
        """
    )

    p95_latency = prometheus_query(
        base_url,
        """
        histogram_quantile(
            0.95,
            sum(rate(http_request_duration_seconds_bucket{endpoint="work"}[1m]))
            by (le)
        )
        """
    )

    return {
        "work_rps": work_rps,
        "work_avg_latency_ms": (
            average_latency * 1000 if average_latency is not None else None
        ),
        "work_p95_latency_ms": (
            p95_latency * 1000 if p95_latency is not None else None
        ),
    }


def collect_sample(prometheus_url):
    sample = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "app_cpu_millicores": None,
        "app_memory_mib": None,
        "desired_replicas": None,
        "ready_replicas": None,
        "available_replicas": None,
        "work_rps": None,
        "work_avg_latency_ms": None,
        "work_p95_latency_ms": None,
        "collection_status": "success",
        "error": "",
    }

    errors = []

    try:
        cpu, memory = get_pod_metrics()
        sample["app_cpu_millicores"] = round(cpu, 2)
        sample["app_memory_mib"] = round(memory, 2)
    except Exception as exc:
        errors.append(f"Kubernetes metrics: {exc}")

    try:
        sample.update(get_replica_metrics())
    except Exception as exc:
        errors.append(f"Replica metrics: {exc}")

    try:
        sample.update(get_prometheus_metrics(prometheus_url))
    except Exception as exc:
        errors.append(f"Prometheus metrics: {exc}")

    if errors:
        sample["collection_status"] = "partial"
        sample["error"] = " | ".join(errors)

    return sample


def main():
    parser = argparse.ArgumentParser(
        description="Collect synchronized autoscaler experiment metrics"
    )

    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument(
        "--output",
        default="experiments/results/hpa_collector_test.csv",
    )
    parser.add_argument(
        "--prometheus",
        default="http://192.168.49.2:30090",
    )

    args = parser.parse_args()

    if args.duration <= 0 or args.interval <= 0:
        parser.error("Duration and interval must be positive")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "timestamp_utc",
        "elapsed_seconds",
        "app_cpu_millicores",
        "app_memory_mib",
        "desired_replicas",
        "ready_replicas",
        "available_replicas",
        "work_rps",
        "work_avg_latency_ms",
        "work_p95_latency_ms",
        "collection_status",
        "error",
    ]

    print(f"Output: {output_path}")
    print(f"Duration: {args.duration}s | Interval: {args.interval}s")
    print("Starting metric collection...")

    start = time.monotonic()
    next_sample = start
    sample_count = 0

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()

        while time.monotonic() - start < args.duration:
            elapsed = round(time.monotonic() - start, 2)
            sample = collect_sample(args.prometheus)
            sample["elapsed_seconds"] = elapsed

            writer.writerow(sample)
            csv_file.flush()
            sample_count += 1

            print(
                f"[{elapsed:>6.1f}s] "
                f"CPU={sample['app_cpu_millicores']}m | "
                f"Memory={sample['app_memory_mib']}MiB | "
                f"Replicas={sample['ready_replicas']}/"
                f"{sample['desired_replicas']} | "
                f"RPS={sample['work_rps']} | "
                f"Status={sample['collection_status']}"
            )

            next_sample += args.interval
            sleep_time = next_sample - time.monotonic()

            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f"\nCollection complete. Samples: {sample_count}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
