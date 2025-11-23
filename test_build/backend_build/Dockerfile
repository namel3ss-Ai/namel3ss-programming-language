# syntax=docker/dockerfile:1
    FROM python:3.11-slim AS builder
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1
    WORKDIR /app
    COPY requirements.txt ./
    RUN python -m pip install --upgrade pip && \
        python -m pip install --no-cache-dir --prefix=/install -r requirements.txt

    FROM python:3.11-slim AS runtime
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        NAMEL3SS_ENV=production \
        NAMEL3SS_APP_NAME="Simple Test" \
        PORT=8000
    WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
    COPY --from=builder /install /usr/local
    COPY . /app

    # Run as a non-root user for better container isolation.
    RUN addgroup --system namel3ss && adduser --system --ingroup namel3ss namel3ss
    USER namel3ss

    EXPOSE 8000
    HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
        CMD curl -fsS http://localhost:8000/healthz || exit 1

    # Respect NAMEL3SS_* env vars for secrets and connectors.
    CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--access-logfile", "-", "--log-level", "info", "--worker-tmp-dir", "/tmp/simple-test"]
