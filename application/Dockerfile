FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY gunicorn.conf.py .
COPY entrypoint.sh .

RUN chmod +x /app/entrypoint.sh

ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc

EXPOSE 5000

CMD ["/app/entrypoint.sh"]
