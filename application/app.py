from flask import Flask, jsonify, Response, request
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
import socket

app = Flask(__name__)

# Total HTTP requests, classified by method, endpoint and status
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

# HTTP response latency in seconds
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request processing duration",
    ["method", "endpoint"]
)

# CPU workload execution duration
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
    method = request.method
    status = str(response.status_code)

    HTTP_REQUESTS.labels(
        method=method,
        endpoint=endpoint,
        status=status
    ).inc()

    duration = time.time() - getattr(request, "start_time", time.time())

    HTTP_REQUEST_DURATION.labels(
        method=method,
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
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
