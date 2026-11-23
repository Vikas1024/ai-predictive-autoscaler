import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

APP_DEPLOYMENT = "predictive-autoscaler-app"
HPA_NAME = "predictive-autoscaler-hpa"

DURATION = 240
COLLECTOR_INTERVAL = 5
COOLDOWN_TIMEOUT = 360
POLL_INTERVAL = 5

PROFILES = ["constant", "ramp_up", "spike", "burst", "ramp_down"]
USER_LEVELS = [8, 12, 16]

TRIALS = [
    {
        "profile": profile,
        "users": users,
        "trial_id": f"{profile}_{users:02d}_users",
    }
    for profile in PROFILES
    for users in USER_LEVELS
]


def now():
    return datetime.now(timezone.utc).isoformat()


def run_command(command, check=True, capture=True):
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n"
            f"{result.stderr or result.stdout}"
        )

    return result


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def kubectl_json(*args):
    result = run_command(["kubectl", *args])
    return json.loads(result.stdout)


def http_check(url, expected_text=None):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = response.read().decode("utf-8", errors="replace")
            return (
                response.status == 200
                and (expected_text is None or expected_text in body)
            )
    except Exception:
        return False


def get_hpa_state():
    hpa = kubectl_json("get", "hpa", HPA_NAME, "-o", "json")
    deployment = kubectl_json(
        "get", "deployment", APP_DEPLOYMENT, "-o", "json"
    )

    cpu_percent = None

    for metric in hpa.get("status", {}).get("currentMetrics", []):
        resource = metric.get("resource", {})
        if resource.get("name") == "cpu":
            cpu_percent = resource.get(
                "current", {}
            ).get("averageUtilization")
            break

    return {
        "cpu_percent": cpu_percent,
        "current_replicas": hpa.get("status", {}).get("currentReplicas", 0),
        "desired_replicas": hpa.get("status", {}).get("desiredReplicas", 0),
        "ready_replicas": deployment.get("status", {}).get("readyReplicas", 0),
    }


def get_urls():
    ip = run_command(["minikube", "ip"]).stdout.strip()
    return f"http://{ip}:30080", f"http://{ip}:30090"


def check_environment():
    print("\n[ENVIRONMENT] Checking Kubernetes node...", flush=True)
    run_command(["kubectl", "get", "nodes"])

    print("[ENVIRONMENT] Checking Metrics API...", flush=True)
    api = kubectl_json(
        "get", "apiservice", "v1beta1.metrics.k8s.io", "-o", "json"
    )

    available = any(
        c.get("type") == "Available" and c.get("status") == "True"
        for c in api.get("status", {}).get("conditions", [])
    )
    if not available:
        raise RuntimeError("Metrics API is unavailable.")

    print("[ENVIRONMENT] Checking pod metrics...", flush=True)
    run_command(["kubectl", "top", "pods"])

    state = get_hpa_state()
    if state["cpu_percent"] is None:
        raise RuntimeError("HPA CPU metric is unavailable.")

    app_url, prometheus_url = get_urls()

    if not http_check(f"{app_url}/health"):
        raise RuntimeError("Application health endpoint failed.")

    if not http_check(
        f"{prometheus_url}/-/ready", "Prometheus Server is Ready"
    ):
        raise RuntimeError("Prometheus is not ready.")

    print("[ENVIRONMENT] All checks passed.", flush=True)
    return app_url, prometheus_url


def check_runtime_health(app_url, prometheus_url):
    if not http_check(f"{app_url}/health"):
        raise RuntimeError("Application health check failed.")

    if not http_check(
        f"{prometheus_url}/-/ready", "Prometheus Server is Ready"
    ):
        raise RuntimeError("Prometheus readiness check failed.")

    state = get_hpa_state()
    if state["cpu_percent"] is None:
        raise RuntimeError("HPA CPU metrics are unavailable.")

    return state


def wait_for_hpa_recovery(timeout=COOLDOWN_TIMEOUT):
    print(
        "\n[COOLDOWN] Waiting for CPU <=20% and one healthy replica.",
        flush=True,
    )

    started = time.monotonic()
    stable_checks = 0

    while time.monotonic() - started < timeout:
        elapsed = int(time.monotonic() - started)

        try:
            state = get_hpa_state()

            recovered = (
                state["cpu_percent"] is not None
                and state["cpu_percent"] <= 20
                and state["current_replicas"] == 1
                and state["desired_replicas"] == 1
                and state["ready_replicas"] == 1
            )

            stable_checks = stable_checks + 1 if recovered else 0

            print(
                f"[COOLDOWN] {elapsed:03d}s | "
                f"CPU={state['cpu_percent']}% | "
                f"current={state['current_replicas']} | "
                f"desired={state['desired_replicas']} | "
                f"ready={state['ready_replicas']} | "
                f"stable={stable_checks}/3",
                flush=True,
            )

            if stable_checks >= 3:
                print("[COOLDOWN] Recovery confirmed.", flush=True)
                return

        except Exception as exc:
            stable_checks = 0
            print(f"[COOLDOWN] Check error: {exc}", flush=True)

        time.sleep(10)

    raise TimeoutError(
        f"HPA did not recover within {timeout} seconds."
    )


def tail_file(path, lines=15):
    path = Path(path)
    if not path.exists():
        return "Log file not found."

    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def prepare_trial_directory(batch_dir, trial_id, attempt):
    if attempt == 1:
        trial_dir = batch_dir / trial_id
    else:
        trial_dir = batch_dir / f"{trial_id}_retry_{attempt - 1:02d}"

    trial_dir.mkdir(parents=True, exist_ok=False)
    return trial_dir


def move_locust_outputs(prefix, trial_dir):
    suffixes = [
        "_stats.csv",
        "_failures.csv",
        "_stats_history.csv",
        "_exceptions.csv",
    ]

    for suffix in suffixes:
        generated = Path(str(prefix) + suffix)
        if generated.exists():
            generated.rename(trial_dir / ("locust" + suffix))


def read_locust_summary(trial_dir):
    path = trial_dir / "locust_stats.csv"

    if not path.exists():
        return {
            "requests": None,
            "failures": None,
            "avg_response_ms": None,
            "p95_response_ms": None,
            "requests_per_second": None,
        }

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    aggregate = next(
        (r for r in rows if r.get("Name") == "Aggregated"), None
    )

    if not aggregate:
        return {
            "requests": None,
            "failures": None,
            "avg_response_ms": None,
            "p95_response_ms": None,
            "requests_per_second": None,
        }

    def number(field):
        value = aggregate.get(field)
        try:
            return float(value) if value not in ("", None) else None
        except (ValueError, TypeError):
            return None

    return {
        "requests": number("Request Count"),
        "failures": number("Failure Count"),
        "avg_response_ms": number("Average Response Time"),
        "p95_response_ms": number("95%"),
        "requests_per_second": number("Requests/s"),
    }


def update_progress(batch_dir, batch_data, active=None):
    statuses = []

    for trial in TRIALS[:batch_data["requested_trials"]]:
        trial_id = trial["trial_id"]
        matching = []

        for folder in batch_dir.glob(f"{trial_id}*"):
            metadata_path = folder / "experiment.json"
            if metadata_path.exists():
                try:
                    metadata = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                    matching.append(metadata)
                except Exception:
                    pass

        if matching:
            matching.sort(key=lambda x: x.get("attempt", 1))
            statuses.append({
                "trial_id": trial_id,
                "status": matching[-1].get("status", "unknown"),
            })
        else:
            statuses.append({"trial_id": trial_id, "status": "pending"})

    counts = {
        status: sum(t["status"] == status for t in statuses)
        for status in ("completed", "failed", "pending", "running")
    }

    progress = {
        "updated_at_utc": now(),
        "requested_trials": batch_data["requested_trials"],
        "completed_trials": counts["completed"],
        "failed_trials": counts["failed"],
        "pending_trials": counts["pending"],
        "active_trial": active,
        "trials": statuses,
    }

    write_json(batch_dir / "progress.json", progress)
    return progress


def run_trial(
    trial,
    batch_dir,
    app_url,
    prometheus_url,
    duration,
    trial_number,
    total_trials,
    attempt=1,
    batch_data=None,
):
    trial_id = trial["trial_id"]
    profile = trial["profile"]
    users = trial["users"]

    trial_dir = prepare_trial_directory(batch_dir, trial_id, attempt)

    metrics_path = trial_dir / "metrics.csv"
    locust_prefix = trial_dir / "locust"
    locust_log = trial_dir / "locust.log"
    collector_log = trial_dir / "collector.log"
    metadata_path = trial_dir / "experiment.json"

    metadata = {
        "trial_id": trial_id,
        "attempt": attempt,
        "profile": profile,
        "max_users": users,
        "duration_seconds": duration,
        "collector_interval_seconds": COLLECTOR_INTERVAL,
        "prediction_horizon_seconds": 30,
        "controller": "Kubernetes HPA",
        "application_image": "predictive-autoscaler-app:v4",
        "gunicorn_workers": 2,
        "gunicorn_threads": 4,
        "application_url": app_url,
        "prometheus_url": prometheus_url,
        "started_at_utc": now(),
        "status": "running",
    }
    write_json(metadata_path, metadata)

    print("\n" + "=" * 76, flush=True)
    print(
        f"[TRIAL {trial_number:02d}/{total_trials:02d}] "
        f"STARTING: {trial_id} | Attempt {attempt}",
        flush=True,
    )
    print(
        f"Profile={profile} | Max users={users} | Duration={duration}s",
        flush=True,
    )
    print(f"Output: {trial_dir}", flush=True)
    print("=" * 76, flush=True)

    collector_command = [
        sys.executable,
        "experiments/metrics_collector.py",
        "--duration", str(duration + 10),
        "--interval", str(COLLECTOR_INTERVAL),
        "--output", str(metrics_path),
        "--prometheus", prometheus_url,
    ]

    collector_process = None
    locust_process = None
    locust_return_code = None
    trial_error = None
    started = time.monotonic()

    try:
        with collector_log.open("w") as collector_output:
            collector_process = subprocess.Popen(
                collector_command,
                cwd=ROOT,
                stdout=collector_output,
                stderr=subprocess.STDOUT,
            )

            print("[COLLECTOR] Started. Initializing...", flush=True)
            time.sleep(5)

            if collector_process.poll() is not None:
                raise RuntimeError(
                    "Metrics collector exited before Locust started."
                )

            environment = os.environ.copy()
            environment["TRAINING_PROFILE"] = profile
            environment["TRAINING_MAX_USERS"] = str(users)

            locust_command = [
                sys.executable,
                "-m", "locust",
                "-f", "workloads/training_locustfile.py",
                "--headless",
                "--host", app_url,
                "--users", str(users),
                "--spawn-rate", "4",
                "--run-time", f"{duration}s",
                "--csv", str(locust_prefix),
                "--csv-full-history",
            ]

            with locust_log.open("w") as locust_output:
                locust_process = subprocess.Popen(
                    locust_command,
                    cwd=ROOT,
                    env=environment,
                    stdout=locust_output,
                    stderr=subprocess.STDOUT,
                )

                print("[LOCUST] Workload started.", flush=True)
                last_hpa_report = 0

                while locust_process.poll() is None:
                    elapsed = int(time.monotonic() - started)

                    if collector_process.poll() is not None:
                        raise RuntimeError(
                            "Metrics collector exited while Locust was running."
                        )

                    print(
                        f"[TRIAL {trial_number:02d}/{total_trials:02d}] "
                        f"Elapsed={elapsed}s/{duration}s | "
                        f"Locust=RUNNING | "
                        f"Collector=RUNNING",
                        flush=True,
                    )

                    if elapsed - last_hpa_report >= 15:
                        try:
                            state = get_hpa_state()
                            print(
                                f"[HPA] CPU={state['cpu_percent']}% | "
                                f"replicas={state['current_replicas']}/"
                                f"{state['desired_replicas']} | "
                                f"ready={state['ready_replicas']}",
                                flush=True,
                            )
                        except Exception as exc:
                            print(f"[HPA] Status unavailable: {exc}", flush=True)

                        last_hpa_report = elapsed

                    if batch_data is not None:
                        update_progress(
                            batch_dir,
                            batch_data,
                            active={
                                "trial_id": trial_id,
                                "stage": "locust_running",
                                "elapsed_seconds": elapsed,
                            },
                        )

                    time.sleep(POLL_INTERVAL)

                locust_return_code = locust_process.returncode

            print(
                f"[LOCUST] Finished with exit code {locust_return_code}.",
                flush=True,
            )

            collector_return_code = collector_process.wait(timeout=90)

            print(
                f"[COLLECTOR] Finished with exit code "
                f"{collector_return_code}.",
                flush=True,
            )

            if collector_return_code != 0:
                raise RuntimeError(
                    f"Metrics collector exited with code {collector_return_code}."
                )

        move_locust_outputs(locust_prefix, trial_dir)

        if not metrics_path.exists():
            raise RuntimeError("Metrics CSV was not created.")

        with metrics_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        successful_rows = sum(
            row.get("collection_status") == "success" for row in rows
        )

        if len(rows) < 10:
            raise RuntimeError(
                f"Only {len(rows)} metrics samples collected."
            )

        if successful_rows < len(rows) * 0.90:
            raise RuntimeError(
                "More than 10% of metrics samples failed collection."
            )

        summary = read_locust_summary(trial_dir)
        metadata.update(summary)
        metadata["metrics_samples"] = len(rows)
        metadata["successful_metrics_samples"] = successful_rows
        metadata["locust_exit_code"] = locust_return_code
        metadata["ended_at_utc"] = now()

        if locust_return_code != 0:
            metadata["status"] = "failed"
            metadata["error"] = (
                f"Locust exited with code {locust_return_code}."
            )
            metadata["locust_log_tail"] = tail_file(locust_log, 20)
        elif summary["requests"] is None:
            metadata["status"] = "failed"
            metadata["error"] = "Locust aggregate statistics are missing."
        elif summary["failures"] and summary["failures"] > 0:
            metadata["status"] = "failed"
            metadata["error"] = (
                f"Locust recorded {summary['failures']} failed requests."
            )
        else:
            metadata["status"] = "completed"

        write_json(metadata_path, metadata)

        print("\n[TRIAL SUMMARY]", flush=True)
        print(f"Status: {metadata['status']}", flush=True)
        print(f"Requests: {summary['requests']}", flush=True)
        print(f"Failures: {summary['failures']}", flush=True)
        print(f"Average latency: {summary['avg_response_ms']} ms", flush=True)
        print(f"P95 latency: {summary['p95_response_ms']} ms", flush=True)
        print(f"Metrics samples: {len(rows)} ({successful_rows} successful)", flush=True)

        if metadata["status"] == "failed":
            print(f"Reason: {metadata['error']}", flush=True)
            print("\n[LOCUST LOG - LAST 20 LINES]", flush=True)
            print(tail_file(locust_log, 20), flush=True)

        return metadata

    except Exception as exc:
        trial_error = str(exc)

        if locust_process and locust_process.poll() is None:
            locust_process.terminate()
            try:
                locust_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                locust_process.kill()

        if collector_process and collector_process.poll() is None:
            collector_process.terminate()
            try:
                collector_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                collector_process.kill()

        move_locust_outputs(locust_prefix, trial_dir)

        metadata["status"] = "failed"
        metadata["error"] = trial_error
        metadata["ended_at_utc"] = now()

        if locust_return_code is not None:
            metadata["locust_exit_code"] = locust_return_code

        metadata["locust_log_tail"] = tail_file(locust_log, 20)
        metadata["collector_log_tail"] = tail_file(collector_log, 15)

        write_json(metadata_path, metadata)

        print(f"\n[TRIAL FAILED] {trial_id}", flush=True)
        print(f"Reason: {trial_error}", flush=True)
        print("\n[LOCUST LOG]", flush=True)
        print(tail_file(locust_log, 20), flush=True)
        print("\n[COLLECTOR LOG]", flush=True)
        print(tail_file(collector_log, 15), flush=True)

        return metadata


def get_latest_trial_metadata(batch_dir, trial_id):
    candidates = []

    for folder in batch_dir.glob(f"{trial_id}*"):
        path = folder / "experiment.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                candidates.append(data)
            except Exception:
                pass

    if not candidates:
        return None

    return max(candidates, key=lambda x: x.get("attempt", 1))


def main():
    parser = argparse.ArgumentParser(
        description="Resumable HPA training-data collection runner"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of planned trials for a new batch. Default: 1.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DURATION,
        help="Duration of each trial in seconds.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Existing batch directory to resume.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed trials in a resumed batch using new folders.",
    )
    parser.add_argument(
        "--skip-initial-recovery",
        action="store_true",
        help="Skip initial HPA recovery wait for a new batch.",
    )

    args = parser.parse_args()

    if args.duration < 120:
        parser.error("--duration must be at least 120 seconds.")

    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume.")

    if args.resume:
        batch_dir = Path(args.resume).expanduser()
        if not batch_dir.is_absolute():
            batch_dir = ROOT / batch_dir
        batch_dir = batch_dir.resolve()

        batch_path = batch_dir / "batch.json"
        if not batch_path.exists():
            parser.error(f"Batch metadata not found: {batch_path}")

        batch_data = json.loads(batch_path.read_text(encoding="utf-8"))
        total_trials = batch_data["requested_trials"]
        duration = batch_data["duration_seconds"]

        print(f"\n[RESUME] Batch: {batch_data['batch_id']}", flush=True)
        print(f"[RESUME] Directory: {batch_dir}", flush=True)
        print(f"[RESUME] Planned trials: {total_trials}", flush=True)
        print(f"[RESUME] Duration per trial: {duration}s", flush=True)
    else:
        if args.limit < 1 or args.limit > len(TRIALS):
            parser.error(f"--limit must be between 1 and {len(TRIALS)}")

        total_trials = args.limit
        duration = args.duration

        batch_id = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
        batch_dir = (
            ROOT / "experiments" / "results" / "hpa_training" / batch_id
        )
        batch_dir.mkdir(parents=True, exist_ok=False)

        batch_data = {
            "batch_id": batch_id,
            "created_at_utc": now(),
            "requested_trials": total_trials,
            "duration_seconds": duration,
            "profiles": PROFILES,
            "user_levels": USER_LEVELS,
            "controller": "Kubernetes HPA",
            "application_image": "predictive-autoscaler-app:v4",
            "status": "running",
        }

        write_json(batch_dir / "batch.json", batch_data)

    print("\nPredictive Autoscaler - Training Data Collection", flush=True)
    print(f"Trials planned: {total_trials}", flush=True)
    print(f"Duration per trial: {duration}s", flush=True)

    app_url, prometheus_url = check_environment()

    if not args.resume and not args.skip_initial_recovery:
        wait_for_hpa_recovery()

    completed = 0
    failed = 0
    skipped = 0
    stopped_reason = None

    update_progress(batch_dir, batch_data)

    try:
        for index, trial in enumerate(TRIALS[:total_trials], start=1):
            previous = get_latest_trial_metadata(
                batch_dir, trial["trial_id"]
            )

            if previous and previous.get("status") == "completed":
                skipped += 1
                print(
                    f"\n[SKIP {index:02d}/{total_trials:02d}] "
                    f"{trial['trial_id']} already completed.",
                    flush=True,
                )
                continue

            if previous and previous.get("status") == "failed":
                if not args.retry_failed:
                    skipped += 1
                    print(
                        f"\n[SKIP {index:02d}/{total_trials:02d}] "
                        f"{trial['trial_id']} previously failed. "
                        f"Use --retry-failed to retry.",
                        flush=True,
                    )
                    continue

            attempt = 1
            if previous:
                attempt = previous.get("attempt", 1) + 1

            print(
                f"\n[PLAN] Trial {index}/{total_trials}: "
                f"{trial['trial_id']}",
                flush=True,
            )

            try:
                wait_for_hpa_recovery()
                check_runtime_health(app_url, prometheus_url)
            except Exception as exc:
                stopped_reason = (
                    f"Environment or HPA recovery failure before "
                    f"{trial['trial_id']}: {exc}"
                )
                print(f"\n[BATCH STOPPED] {stopped_reason}", flush=True)
                break

            result = run_trial(
                trial,
                batch_dir,
                app_url,
                prometheus_url,
                duration,
                index,
                total_trials,
                attempt,
                batch_data,
            )

            if result["status"] == "completed":
                completed += 1
            else:
                failed += 1

            update_progress(batch_dir, batch_data)

            print(
                f"\n[BATCH PROGRESS] "
                f"Completed={completed} | Failed={failed} | "
                f"Skipped={skipped} | "
                f"Remaining={total_trials - index}",
                flush=True,
            )

            try:
                wait_for_hpa_recovery()
                check_runtime_health(app_url, prometheus_url)
            except Exception as exc:
                stopped_reason = (
                    f"Environment did not recover after "
                    f"{trial['trial_id']}: {exc}"
                )
                print(f"\n[BATCH STOPPED] {stopped_reason}", flush=True)
                break

    except KeyboardInterrupt:
        stopped_reason = "Interrupted manually by user."
        print("\n[BATCH INTERRUPTED] Progress has been saved.", flush=True)

    finally:
        progress = update_progress(batch_dir, batch_data)

        batch_data["completed_trials"] = progress["completed_trials"]
        batch_data["failed_trials"] = progress["failed_trials"]
        batch_data["pending_trials"] = progress["pending_trials"]
        batch_data["finished_at_utc"] = now()

        if stopped_reason:
            batch_data["status"] = "interrupted"
            batch_data["stop_reason"] = stopped_reason
        elif progress["pending_trials"] == 0:
            batch_data["status"] = (
                "completed" if progress["failed_trials"] == 0
                else "completed_with_failures"
            )
        else:
            batch_data["status"] = "incomplete"

        write_json(batch_dir / "batch.json", batch_data)

    print("\n" + "=" * 76, flush=True)
    print("EXPERIMENT BATCH SUMMARY", flush=True)
    print("=" * 76, flush=True)
    print(f"Batch: {batch_data['batch_id']}", flush=True)
    print(f"Completed: {batch_data['completed_trials']}", flush=True)
    print(f"Failed: {batch_data['failed_trials']}", flush=True)
    print(f"Pending: {batch_data['pending_trials']}", flush=True)
    print(f"Status: {batch_data['status']}", flush=True)
    if stopped_reason:
        print(f"Stop reason: {stopped_reason}", flush=True)
    print(f"Results: {batch_dir}", flush=True)
    print(f"Progress: {batch_dir / 'progress.json'}", flush=True)
    print("=" * 76, flush=True)


if __name__ == "__main__":
    main()
