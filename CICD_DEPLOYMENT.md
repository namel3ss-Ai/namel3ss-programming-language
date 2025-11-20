# CI/CD Pipeline & Deployment Guide

Complete CI/CD pipeline for the N3 Graph Editor using GitHub Actions, Docker, and Kubernetes.

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Workflows](#workflows)
- [Deployment Environments](#deployment-environments)
- [Setup Instructions](#setup-instructions)
- [Deployment Process](#deployment-process)
- [Monitoring & Rollback](#monitoring--rollback)
- [Troubleshooting](#troubleshooting)

## Overview

This CI/CD pipeline automates:
- **Code Quality**: Linting with Ruff (Python) and ESLint (TypeScript)
- **Testing**: Unit tests, integration tests, and E2E tests
- **Building**: Docker images for Backend, Frontend, and YJS Server
- **Security**: Vulnerability scanning with Trivy
- **Deployment**: Automated deployments to staging and production
- **Monitoring**: Health checks and smoke tests

### Technology Stack

- **CI/CD**: GitHub Actions
- **Containerization**: Docker + Docker Compose
- **Container Registry**: GitHub Container Registry (GHCR)
- **Orchestration**: Kubernetes
- **Ingress**: NGINX Ingress Controller
- **TLS**: cert-manager with Let's Encrypt
- **Monitoring**: Prometheus + Grafana + Jaeger

## Pipeline Architecture

```
┌─────────────┐
│   Push/PR   │
└──────┬──────┘
       │
       ├─────► Lint & Format Check
       │       ├─ Python (Ruff)
       │       ├─ TypeScript (ESLint)
       │       ├─ Markdown
       │       └─ YAML
       │
       ├─────► Unit & Integration Tests
       │       ├─ Python Tests (pytest)
       │       ├─ Frontend Tests (Vitest)
       │       └─ Coverage Upload (Codecov)
       │
       ├─────► E2E Tests (Playwright)
       │       ├─ Chromium
       │       ├─ Firefox
       │       └─ WebKit
       │
       ├─────► Build Docker Images
       │       ├─ Backend
       │       ├─ Frontend
       │       └─ YJS Server
       │
       ├─────► Security Scan (Trivy)
       │       └─ Upload to GitHub Security
       │
       ├─────► Deploy to Staging (develop branch)
       │       ├─ Kubernetes Deployment
       │       ├─ Health Checks
       │       └─ Smoke Tests
       │
       └─────► Deploy to Production (tags: v*)
               ├─ Backup Database
               ├─ Kubernetes Deployment
               ├─ Health Checks
               ├─ Smoke Tests
               ├─ Create GitHub Release
               └─ Rollback on Failure
```

## Workflows

### 1. Lint & Format (`.github/workflows/lint.yml`)

Runs on every push and PR to `main` and `develop` branches.

**Jobs**:
- **Python Linting**: Ruff linter and formatter
- **Frontend Linting**: ESLint + TypeScript type check
- **Markdown Linting**: markdownlint-cli2
- **YAML Linting**: yamllint

**Triggers**:
- Push to `main` or `develop`
- Pull requests to `main` or `develop`
- Manual dispatch

### 2. Unit & Integration Tests (`.github/workflows/unit-tests.yml`)

Comprehensive testing of Python backend and React frontend.

**Jobs**:
- **Python Tests**: 
  - Matrix: Python 3.10, 3.11, 3.12
  - PostgreSQL service container
  - Coverage reports to Codecov
  - pytest with coverage

- **Frontend Tests**:
  - Node.js 18
  - Vitest unit tests
  - Coverage reports to Codecov

**Triggers**:
- Push to `main` or `develop`
- Pull requests to `main` or `develop`
- Manual dispatch

### 3. E2E Tests (`.github/workflows/e2e-tests.yml`)

End-to-end testing with Playwright across multiple browsers.

**Jobs**:
- **Matrix**: Chromium, Firefox, WebKit
- **Services**: Docker Compose (PostgreSQL, Backend, YJS Server)
- **Tests**: 67 comprehensive E2E tests
- **Artifacts**: Reports, screenshots, traces, videos

**Triggers**:
- Push to `main` or `develop`
- Pull requests to `main` or `develop`
- Manual dispatch

### 4. Build Docker Images (`.github/workflows/build.yml`)

Build and push multi-platform Docker images to GHCR.

**Jobs**:
- **Build Backend**: FastAPI application
- **Build Frontend**: React + Vite application
- **Build YJS Server**: WebSocket collaboration server
- **Security Scan**: Trivy vulnerability scanning

**Features**:
- Multi-platform builds (amd64, arm64)
- Layer caching for faster builds
- Semantic versioning tags
- Security scanning with Trivy
- SARIF upload to GitHub Security

**Triggers**:
- Push to `main` or `develop`
- Version tags (`v*`)
- Pull requests to `main`
- Manual dispatch

### 5. Deploy to Staging (`.github/workflows/deploy-staging.yml`)

Automated deployment to staging environment.

**Jobs**:
- Configure Kubernetes context
- Create/update ConfigMap and Secrets
- Deploy to Kubernetes (`k8s/staging/`)
- Wait for rollout completion
- Run smoke tests
- Slack notification

**Environment**: `staging`
**URL**: https://staging.n3-graph-editor.com

**Triggers**:
- Push to `develop` branch
- Manual dispatch

### 6. Deploy to Production (`.github/workflows/deploy-production.yml`)

Controlled deployment to production with safeguards.

**Jobs**:
- Configure Kubernetes context
- Backup database before deployment
- Deploy to Kubernetes (`k8s/production/`)
- Wait for rollout completion (10 min timeout)
- Run smoke tests (health, auth, API)
- Create GitHub Release
- Rollback on failure
- Slack notifications

**Environment**: `production`
**URL**: https://n3-graph-editor.com

**Triggers**:
- Version tags (`v*`)
- Manual dispatch with version input

## Deployment Environments

### Staging Environment

**Purpose**: Pre-production testing and validation

**Configuration**:
- **Namespace**: `n3-staging`
- **Replicas**: 
  - Backend: 2
  - Frontend: 2
  - YJS Server: 2
- **Resources**: Moderate (512Mi-1Gi RAM, 500m-1000m CPU)
- **Autoscaling**: 2-10 pods
- **Database**: PostgreSQL 16 (10Gi storage)
- **URL**: https://staging.n3-graph-editor.com

**Deployment Strategy**:
- RollingUpdate (default)
- Automatic on push to `develop`

### Production Environment

**Purpose**: Live production system

**Configuration**:
- **Namespace**: `n3-production`
- **Replicas**: 
  - Backend: 3
  - Frontend: 3
  - YJS Server: 3
- **Resources**: High (1Gi-2Gi RAM, 1000m-2000m CPU)
- **Autoscaling**: 3-20 pods (backend), 3-15 (frontend/yjs)
- **Database**: PostgreSQL 16 StatefulSet (50Gi storage)
- **URL**: https://n3-graph-editor.com
- **Pod Disruption Budgets**: Minimum 2 available pods
- **Anti-affinity**: Pods spread across nodes

**Deployment Strategy**:
- RollingUpdate (maxSurge: 1, maxUnavailable: 0)
- Manual approval required
- Automatic rollback on failure
- Database backup before deployment

## Setup Instructions

### Prerequisites

1. **GitHub Repository Secrets**:
   ```
   KUBE_CONFIG_STAGING              # Kubernetes config for staging
   KUBE_CONFIG_PRODUCTION           # Kubernetes config for production
   STAGING_DATABASE_URL             # Staging database connection
   PRODUCTION_DATABASE_URL          # Production database connection
   STAGING_SECRET_KEY               # JWT secret for staging
   PRODUCTION_SECRET_KEY            # JWT secret for production
   SLACK_WEBHOOK                    # Slack notifications
   SMOKE_TEST_USER                  # Test user for smoke tests
   SMOKE_TEST_PASSWORD              # Test user password
   CODECOV_TOKEN                    # Codecov upload token
   ```

2. **GitHub Container Registry**:
   - Enable GHCR in repository settings
   - Ensure `GITHUB_TOKEN` has `packages:write` permission

3. **Kubernetes Cluster**:
   - Kubernetes 1.27+
   - NGINX Ingress Controller
   - cert-manager for TLS certificates
   - StorageClass for persistent volumes

### Initial Setup

#### 1. Configure Kubernetes Cluster

```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@n3-graph-editor.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

#### 2. Generate Kubernetes Config

```bash
# Get kubeconfig
kubectl config view --raw > kubeconfig.yaml

# Base64 encode for GitHub Secrets
cat kubeconfig.yaml | base64

# Add to GitHub Secrets as KUBE_CONFIG_STAGING or KUBE_CONFIG_PRODUCTION
```

#### 3. Generate Secret Keys

```bash
# Generate JWT secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Or using OpenSSL
openssl rand -base64 32

# Add to GitHub Secrets as STAGING_SECRET_KEY or PRODUCTION_SECRET_KEY
```

#### 4. Setup Database

```bash
# Create PostgreSQL database
createdb n3_graphs

# Create user
createuser -P n3  # Enter password when prompted

# Grant privileges
psql -c "GRANT ALL PRIVILEGES ON DATABASE n3_graphs TO n3;"

# Connection string format:
# postgresql+asyncpg://n3:PASSWORD@HOST:5432/n3_graphs
```

#### 5. Deploy Initial Version

```bash
# Deploy to staging
git checkout develop
git push origin develop

# Deploy to production (after testing)
git checkout main
git tag v1.0.0
git push origin v1.0.0
```

## Deployment Process

### Staging Deployment

```bash
# 1. Merge feature branch to develop
git checkout develop
git merge feature/my-feature
git push origin develop

# 2. GitHub Actions automatically:
#    - Runs linting
#    - Runs tests
#    - Builds Docker images
#    - Deploys to staging
#    - Runs smoke tests

# 3. Verify deployment
curl https://staging.n3-graph-editor.com/api/health

# 4. Test in staging environment
open https://staging.n3-graph-editor.com
```

### Production Deployment

```bash
# 1. Merge develop to main
git checkout main
git merge develop
git push origin main

# 2. Create release tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# 3. GitHub Actions automatically:
#    - Runs all tests
#    - Builds production images
#    - Backs up database
#    - Deploys to production
#    - Runs smoke tests
#    - Creates GitHub Release

# 4. Monitor deployment
kubectl get pods -n n3-production -w

# 5. Verify production
curl https://n3-graph-editor.com/api/health
```

### Manual Deployment

```bash
# Trigger manual deployment via GitHub UI
# Go to: Actions → Deploy to Production → Run workflow
# Enter version: v1.0.0
```

## Monitoring & Rollback

### Health Checks

```bash
# Backend health
curl https://n3-graph-editor.com/api/health

# Frontend health
curl https://n3-graph-editor.com/

# YJS Server health (WebSocket)
wscat -c wss://n3-graph-editor.com/yjs
```

### Monitoring Deployment

```bash
# Watch rollout status
kubectl rollout status deployment/backend -n n3-production

# View pods
kubectl get pods -n n3-production

# Check logs
kubectl logs -f deployment/backend -n n3-production

# View events
kubectl get events -n n3-production --sort-by='.lastTimestamp'
```

### Manual Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/backend -n n3-production
kubectl rollout undo deployment/frontend -n n3-production
kubectl rollout undo deployment/yjs-server -n n3-production

# Rollback to specific revision
kubectl rollout undo deployment/backend -n n3-production --to-revision=2

# View rollout history
kubectl rollout history deployment/backend -n n3-production
```

### Automatic Rollback

The production deployment workflow automatically rolls back on failure:
- Failed health checks
- Failed smoke tests
- Deployment timeout (10 minutes)

## Troubleshooting

### Common Issues

#### 1. Image Pull Errors

```bash
# Verify image exists
docker pull ghcr.io/ssebowadisan/namel3ss-programming-language-backend:v1.0.0

# Check secret
kubectl get secret ghcr-secret -n n3-production -o yaml

# Recreate secret
kubectl delete secret ghcr-secret -n n3-production
# Then redeploy
```

#### 2. Pod Crashes

```bash
# View pod logs
kubectl logs -f <pod-name> -n n3-production

# Describe pod for events
kubectl describe pod <pod-name> -n n3-production

# Check resource usage
kubectl top pods -n n3-production
```

#### 3. Database Connection Issues

```bash
# Test database connection
kubectl exec -it deployment/backend -n n3-production -- \
  python -c "import asyncpg; asyncpg.connect('$DATABASE_URL')"

# Check PostgreSQL pod
kubectl logs -f deployment/postgres -n n3-production
```

#### 4. Ingress Not Working

```bash
# Check ingress
kubectl get ingress -n n3-production
kubectl describe ingress n3-ingress -n n3-production

# Check ingress controller
kubectl logs -f -n ingress-nginx deployment/ingress-nginx-controller

# Verify DNS
dig n3-graph-editor.com
```

#### 5. TLS Certificate Issues

```bash
# Check certificate
kubectl get certificate -n n3-production
kubectl describe certificate n3-production-tls -n n3-production

# Check cert-manager logs
kubectl logs -f -n cert-manager deployment/cert-manager
```

### Debug Commands

```bash
# Shell into pod
kubectl exec -it deployment/backend -n n3-production -- /bin/bash

# Port forward for local access
kubectl port-forward deployment/backend 8000:8000 -n n3-production

# View resource usage
kubectl top nodes
kubectl top pods -n n3-production

# Check persistent volumes
kubectl get pv
kubectl get pvc -n n3-production
```

### Performance Tuning

```bash
# Scale manually
kubectl scale deployment/backend --replicas=5 -n n3-production

# Update resource limits
kubectl set resources deployment/backend \
  --limits=cpu=2000m,memory=2Gi \
  --requests=cpu=1000m,memory=1Gi \
  -n n3-production

# View HPA status
kubectl get hpa -n n3-production
kubectl describe hpa backend-hpa -n n3-production
```

## Best Practices

1. **Always test in staging first**
   - Deploy to staging before production
   - Run full E2E test suite
   - Verify all features work

2. **Use semantic versioning**
   - Format: `vMAJOR.MINOR.PATCH`
   - Example: `v1.0.0`, `v1.1.0`, `v2.0.0`

3. **Monitor deployments**
   - Watch rollout status
   - Check application logs
   - Verify metrics in Prometheus

4. **Database migrations**
   - Run migrations before deployment
   - Test on staging first
   - Always backup before production migrations

5. **Blue-green deployments**
   - Use RollingUpdate strategy
   - Set `maxUnavailable: 0` for zero downtime
   - Verify new pods before terminating old ones

6. **Security**
   - Rotate secrets regularly
   - Use strong passwords
   - Enable pod security policies
   - Keep images updated

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/)
- [cert-manager Documentation](https://cert-manager.io/docs/)

---

**Pipeline Status**: ✅ Production-ready CI/CD pipeline with automated testing, building, and deployment.

For issues or questions, see the [main README](../README.md) or file an issue.
