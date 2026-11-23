#!/bin/sh
set -eu

: "${PROMETHEUS_MULTIPROC_DIR:=/tmp/prometheus_multiproc}"
export PROMETHEUS_MULTIPROC_DIR

mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

# Remove stale metrics files from a previous container process.
find "$PROMETHEUS_MULTIPROC_DIR" -mindepth 1 -maxdepth 1 -type f -delete

exec gunicorn \
    --config /app/gunicorn.conf.py \
    --bind 0.0.0.0:5000 \
    --workers 2 \
    --threads 4 \
    --timeout 60 \
    app:app
