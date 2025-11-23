# CI/CD Pipeline and Deployment Guide

This guide documents the production CI/CD pipeline for Namel3ss components using GitHub Actions, Docker, and Kubernetes. It targets engineers responsible for building and releasing the generated [Runtime](docs/reference/GLOSSARY.md#runtime) for [Applications](docs/reference/GLOSSARY.md#application) compiled from `.n3` sources.

## Overview
- Enforces linting and formatting for Python, TypeScript, Markdown, and YAML.
- Runs unit, integration, and end-to-end tests before container builds.
- Produces multi-platform container images (backend, frontend, YJS server).
- Scans images with Trivy and publishes SARIF to GitHub Security.
- Deploys to staging on `develop`; deploys to production on release tags.
- Performs health checks, smoke tests, and rollbacks on failure.

> **Best Practice**  
> Treat staging as a production rehearsal: run the full test suite, verify observability, and confirm secrets via environment variables rather than embedding them in manifests.

## Pipeline Architecture
```
Push/PR
 ├─ Lint & Format (Ruff, ESLint/TypeScript, markdownlint, yamllint)
 ├─ Unit & Integration Tests (pytest matrix + PostgreSQL, Vitest)
 ├─ E2E Tests (Playwright: Chromium/Firefox/WebKit)
 ├─ Build Docker Images (backend, frontend, YJS)
 ├─ Security Scan (Trivy, SARIF upload)
 ├─ Deploy to Staging (develop branch) with smoke tests
 └─ Deploy to Production (tags v*) with backup and rollback
```

## Workflows

### Lint and Format (`.github/workflows/lint.yml`)
- Jobs: Ruff, ESLint + TypeScript type check, markdownlint-cli2, yamllint.
- Triggers: push/PR to `main` and `develop`, manual dispatch.

### Unit and Integration Tests (`.github/workflows/unit-tests.yml`)
- Python: matrix on 3.10/3.11/3.12, PostgreSQL service, pytest + coverage → Codecov.
- Frontend: Node 18, Vitest + coverage → Codecov.
- Triggers: push/PR to `main` and `develop`, manual dispatch.

### End-to-End Tests (`.github/workflows/e2e-tests.yml`)
- Playwright matrix across Chromium, Firefox, WebKit.
- Uses Docker Compose services (PostgreSQL, backend, YJS).
- Uploads screenshots, traces, and videos on failure.

### Build and Push Images (`.github/workflows/build.yml`)
- Builds backend (FastAPI), frontend (React + Vite), and YJS server.
- Multi-platform (amd64, arm64) with caching.
- Tags with semantic versioning; runs Trivy and uploads SARIF.

### Deploy to Staging (`.github/workflows/deploy-staging.yml`)
- Trigger: push to `develop`, manual dispatch.
- Actions: configure Kubernetes context, apply manifests in `k8s/staging/`, wait for rollout, run smoke tests, notify via Slack.

### Deploy to Production (`.github/workflows/deploy-production.yml`)
- Trigger: release tags `v*`, manual dispatch with version input.
- Actions: database backup, apply manifests in `k8s/production/`, health checks with 10m timeout, smoke tests, GitHub Release creation, Slack notifications, rollback on failure.

## Environments

### Staging
- Namespace: `n3-staging`
- Replicas: backend 2, frontend 2, YJS 2; autoscale 2–10
- Database: PostgreSQL 16 (10Gi)
- TLS: cert-manager with Let’s Encrypt
- URL: https://staging.n3-graph-editor.com

### Production
- Namespace: `n3-production`
- Replicas: backend 3, frontend 3, YJS 3; autoscale backend 3–20, frontend/YJS 3–15
- Database: PostgreSQL 16 StatefulSet (50Gi)
- Pod disruption budgets and anti-affinity enabled
- URL: https://n3-graph-editor.com
- Strategy: RollingUpdate (maxSurge 1, maxUnavailable 0), automatic rollback on failure

> **Warning**  
> Always back up the production database before applying migrations or changes to stateful components.

## Prerequisites
- GitHub secrets: `KUBE_CONFIG_STAGING`, `KUBE_CONFIG_PRODUCTION`, `STAGING_DATABASE_URL`, `PRODUCTION_DATABASE_URL`, JWT secrets, `SLACK_WEBHOOK`, smoke test credentials, `CODECOV_TOKEN`.
- GHCR permissions: `GITHUB_TOKEN` with `packages:write`.
- Kubernetes: version 1.27+, NGINX Ingress, cert-manager, storage class for PVCs.

## Deployment Process
1. Merge to `develop` → staging deploy runs automatically. Verify via health endpoint and smoke tests.
2. Tag release `vX.Y.Z` → production deploy runs: tests, build, DB backup, deploy, health checks, smoke tests, GitHub Release.
3. Monitor rollout via `kubectl rollout status` and logs; rollback on failure is automatic in production workflows.

## Monitoring and Rollback
- Health checks: `/api/health` for backend, `/` for frontend, WebSocket connect for YJS.
- Observability: Prometheus, Grafana, Jaeger dashboards per environment.
- Rollback: `kubectl rollout undo deployment/<component> -n <namespace>` or automatic rollback on failed health/smoke tests in production workflow.

## Troubleshooting
- Image pull failures: verify GHCR credentials and secret presence in namespace.
- Pod crashes: inspect `kubectl logs` and `kubectl describe` for events; confirm resource limits.
- Database connectivity: test from a running pod using the configured `DATABASE_URL`.
- Ingress/TLS: check ingress objects, controller logs, cert-manager status, and DNS resolution.

> **See Also:** [Style Guide](docs/STYLE_GUIDE.md), [Glossary](docs/reference/GLOSSARY.md)
