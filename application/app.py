from flask import Flask, jsonify, Response, request
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    CONTENT_TYPE_LATEST,
    generate_latest,
    multiprocess,
)
import os
import socket
import time

app = Flask(__name__)

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request processing duration",
    ["method", "endpoint"]
)

WORKLOAD_DURATION = Histogram(
    "cpu_workload_duration_seconds",
    "CPU-intensive workload execution duration"
)


@app.before_request
def start_timer():
    request.start_time = time.time()


@app.after_request
def record_metrics(response):
    endpoint = request.endpoint or "unknown"

    HTTP_REQUESTS.labels(
        method=request.method,
        endpoint=endpoint,
        status=str(response.status_code)
    ).inc()

    duration = time.time() - getattr(request, "start_time", time.time())

    HTTP_REQUEST_DURATION.labels(
        method=request.method,
        endpoint=endpoint
    ).observe(duration)

    return response


@app.route("/")
def home():
    return jsonify({
        "project": "AI-Driven Predictive Auto-Scaling Framework",
        "application": "Cloud Workload Simulator",
        "status": "running",
        "hostname": socket.gethostname()
    })


@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/work")
def work():
    start = time.time()

    result = 0
    for i in range(1, 500000):
        result += i * i

    duration = time.time() - start
    WORKLOAD_DURATION.observe(duration)

    return jsonify({
        "workload": "CPU intensive",
        "processing_time_seconds": round(duration, 4),
        "result": result,
        "hostname": socket.gethostname()
    })


@app.route("/metrics")
def metrics():
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        metrics_data = generate_latest(registry)
    else:
        # Supports local single-process development.
        metrics_data = generate_latest()

    return Response(
        metrics_data,
        mimetype=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
