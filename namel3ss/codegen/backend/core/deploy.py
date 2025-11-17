"""Helpers for emitting backend deployment artefacts."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Sequence

from ..state import BackendState

__all__ = ["emit_deployment_artifacts"]


def _slugify(value: str | None, fallback: str = "namel3ss-app") -> str:
    if not value:
        return fallback
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("- ").lower()
    return slug or fallback


def _render_backend_requirements() -> str:
    return textwrap.dedent(
        """
        fastapi>=0.110,<1.0
        uvicorn[standard]>=0.30,<0.31
        gunicorn>=21.2,<22.0
        httpx>=0.28,<0.29
        pydantic>=2.7,<3.0
        sqlalchemy>=2.0,<3.0
        """
    ).strip() + "\n"


def _render_dockerfile(app_name: str, slug: str) -> str:
    return textwrap.dedent(
        f"""
        # syntax=docker/dockerfile:1
        FROM python:3.11-slim AS builder
        ENV PYTHONDONTWRITEBYTECODE=1 \\
            PYTHONUNBUFFERED=1
        WORKDIR /app
        COPY requirements.txt ./
        RUN python -m pip install --upgrade pip && \\
            python -m pip install --no-cache-dir --prefix=/install -r requirements.txt

        FROM python:3.11-slim AS runtime
        ENV PYTHONDONTWRITEBYTECODE=1 \\
            PYTHONUNBUFFERED=1 \\
            NAMEL3SS_ENV=production \\
            NAMEL3SS_APP_NAME="{app_name}" \\
            PORT=8000
        WORKDIR /app
    RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
        COPY --from=builder /install /usr/local
        COPY . /app

        # Run as a non-root user for better container isolation.
        RUN addgroup --system namel3ss && adduser --system --ingroup namel3ss namel3ss
        USER namel3ss

        EXPOSE 8000
        HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \\
            CMD curl -fsS http://localhost:8000/healthz || exit 1

        # Respect NAMEL3SS_* env vars for secrets and connectors.
        CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--access-logfile", "-", "--log-level", "info", "--worker-tmp-dir", "/tmp/{slug}"]
        """
    ).strip() + "\n"


def _render_dockerignore() -> str:
    return textwrap.dedent(
        """
        __pycache__
        *.py[cod]
        *.log
        .git
        .github
        .mypy_cache
        .pytest_cache
        .ruff_cache
        .venv
        node_modules
        frontend
        deploy/terraform/.terraform
        tmp_*
        tests
        */__pycache__
        """
    ).strip() + "\n"


def _render_nginx_config(app_name: str) -> str:
    return textwrap.dedent(
        f"""
        # Example reverse-proxy for {app_name}
        events {{}}

        http {{
            upstream namel3ss_backend {{
                server backend:8000;
            }}

            server {{
                listen 80;
                server_name example.com;

                # Serve the static frontend bundle (optional)
                location / {{
                    root /usr/share/nginx/html;
                    try_files $uri $uri/ /index.html;
                }}

                location /api/ {{
                    proxy_pass http://namel3ss_backend;
                    proxy_set_header Host $host;
                    proxy_set_header X-Real-IP $remote_addr;
                    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                    proxy_set_header X-Forwarded-Proto $scheme;
                }}

                location /ws/ {{
                    proxy_pass http://namel3ss_backend;
                    proxy_http_version 1.1;
                    proxy_set_header Upgrade $http_upgrade;
                    proxy_set_header Connection "Upgrade";
                    proxy_set_header Host $host;
                }}
            }}
        }}
        """
    ).strip() + "\n"


def _render_caddyfile(app_name: str) -> str:
  template = """
    # Example Caddyfile for __APP_NAME__
    example.com {
      encode gzip
      log

      @api {
        path /api/* /metrics /healthz /readyz
      }
      reverse_proxy @api backend:8000

      @ws {
        path /ws/*
      }
      reverse_proxy @ws backend:8000 {
        header_up Host {host}
        header_up X-Forwarded-Proto {scheme}
      }

      handle /* {
        root * /srv/frontend
        file_server
      }
    }
    """
  return textwrap.dedent(template).replace("__APP_NAME__", app_name).strip() + "\n"


def _render_ci_workflow(slug: str) -> str:
    return textwrap.dedent(
        f"""
        name: ci

        on:
          push:
            branches: [ main ]
          pull_request:
            branches: [ main ]

        jobs:
          tests:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v4
              - uses: actions/setup-python@v5
                with:
                  python-version: '3.11'
              - name: Install dependencies
                run: |
                  python -m pip install --upgrade pip
                  pip install -r requirements.txt
              - name: Run pytest
                run: python -m pytest -q

          docker:
            needs: tests
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v4
              - name: Build image
                run: docker build -t ghcr.io/your-org/{slug}:$GITHUB_SHA .
              # Uncomment to push to GHCR
              #- name: Push image
              #  env:
              #    GHCR_TOKEN: ${{{{ secrets.GITHUB_TOKEN }}}}
              #  run: |
              #    echo $GHCR_TOKEN | docker login ghcr.io -u ${{{{ github.actor }}}} --password-stdin
              #    docker push ghcr.io/your-org/{slug}:$GITHUB_SHA
        """
    ).strip() + "\n"


def _render_terraform_main(app_name: str, slug: str) -> str:
    return textwrap.dedent(
        f"""
        terraform {{
          required_providers {{
            kubernetes = {{
              source  = "hashicorp/kubernetes"
              version = "~> 2.29"
            }}
          }}
        }}

        provider "kubernetes" {{
          host                   = var.kube_host
          token                  = var.kube_token
          cluster_ca_certificate = base64decode(var.kube_ca)
        }}

        resource "kubernetes_namespace" "this" {{
          metadata {{
            name = var.namespace
          }}
        }}

        resource "kubernetes_deployment" "backend" {{
          metadata {{
            name      = "{slug}"
            namespace = kubernetes_namespace.this.metadata[0].name
            labels = {{
              app = "{slug}"
            }}
          }}
          spec {{
            replicas = var.replicas
            selector {{
              match_labels = {{ app = "{slug}" }}
            }}
            template {{
              metadata {{
                labels = {{ app = "{slug}" }}
              }}
              spec {{
                container {{
                  name  = "backend"
                  image = var.backend_image
                  port {{
                    container_port = 8000
                  }}
                  env {{
                    name  = "NAMEL3SS_ENV"
                    value = var.environment
                  }}
                  env_from {{
                    secret_ref {{ name = kubernetes_secret.env.metadata[0].name }}
                  }}
                  resources {{
                    limits = {{
                      cpu    = "500m"
                      memory = "512Mi"
                    }}
                    requests = {{
                      cpu    = "100m"
                      memory = "256Mi"
                    }}
                  }}
                }}
              }}
            }}
          }}
        }}

        resource "kubernetes_service" "backend" {{
          metadata {{
            name      = "{slug}"
            namespace = kubernetes_namespace.this.metadata[0].name
          }}
          spec {{
            selector = {{ app = "{slug}" }}
            port {{
              port        = 80
              target_port = 8000
            }}
          }}
        }}

        resource "kubernetes_secret" "env" {{
          metadata {{
            name      = "{slug}-env"
            namespace = kubernetes_namespace.this.metadata[0].name
          }}
          data = {{
            NAMEL3SS_ENV = base64encode(var.environment)
          }}
        }}
        """
    ).strip() + "\n"


def _render_terraform_variables() -> str:
    return textwrap.dedent(
        """
        variable "kube_host" {
          description = "Kubernetes API server URL"
          type        = string
        }

        variable "kube_token" {
          description = "ServiceAccount token"
          type        = string
          sensitive   = true
        }

        variable "kube_ca" {
          description = "Base64 encoded CA cert"
          type        = string
          sensitive   = true
        }

        variable "namespace" {
          description = "Namespace for the backend"
          type        = string
          default     = "namel3ss"
        }

        variable "backend_image" {
          description = "Container image for the backend"
          type        = string
        }

        variable "replicas" {
          description = "Number of backend replicas"
          type        = number
          default     = 2
        }

        variable "environment" {
          description = "Environment name (dev|stage|prod)"
          type        = string
          default     = "prod"
        }
        """
    ).strip() + "\n"


def _render_terraform_outputs() -> str:
    return textwrap.dedent(
        """
        output "service_name" {
          description = "Service backing the Namel3ss backend"
          value       = kubernetes_service.backend.metadata[0].name
        }

        output "namespace" {
          value = kubernetes_namespace.this.metadata[0].name
        }
        """
    ).strip() + "\n"


def _render_deploy_readme(app_name: str) -> str:
    return textwrap.dedent(
        f"""
        # Deployment artefacts for {app_name}

        This folder is generated by Namel3ss to provide production-ready starting points:

        - `Dockerfile` and `.dockerignore` support container builds with Gunicorn + Uvicorn workers.
        - `deploy/nginx.conf` and `deploy/Caddyfile` show how to terminate TLS and forward /api + /ws traffic.
        - `.github/workflows/ci.yml` is a GitHub Actions template that runs pytest and optionally builds/pushes an image.
        - `deploy/terraform` contains a lightweight Kubernetes deployment using the official provider.
        - `deploy/helm` packages the same manifests into a Helm chart for more flexible rollouts.
        - `deploy/env.example` lists the environment variables referenced in this app so you can template them in CI.

        These artefacts are templates—review them carefully and adapt to your infrastructure (cloud, secrets manager,
        image registry, etc.).
        """
    ).strip() + "\n"


def _render_helm_chart(app_name: str) -> str:
    return textwrap.dedent(
        f"""
        apiVersion: v2
        name: namel3ss-backend
        description: Helm chart for {app_name}
        type: application
        version: 0.1.0
        appVersion: "0.1.0"
        """
    ).strip() + "\n"


def _render_helm_values(slug: str) -> str:
  template = """
        replicaCount: 2

        image:
      repository: __IMAGE_REPO__
          tag: "latest"
          pullPolicy: IfNotPresent

        service:
          type: ClusterIP
          port: 80

        ingress:
          enabled: false
          className: ""
          hosts:
            - host: example.com
              paths:
                - path: /
                  pathType: Prefix
          tls: []

        env:
          NAMEL3SS_ENV: production
    """
  return textwrap.dedent(template).replace("__IMAGE_REPO__", f"ghcr.io/your-org/{slug}").strip() + "\n"


def _render_helm_deployment() -> str:
  return textwrap.dedent(
    """
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: {{ include "namel3ss-backend.fullname" . }}
          labels:
            app.kubernetes.io/name: {{ include "namel3ss-backend.name" . }}
        spec:
          replicas: {{ .Values.replicaCount }}
          selector:
            matchLabels:
              app.kubernetes.io/name: {{ include "namel3ss-backend.name" . }}
          template:
            metadata:
              labels:
                app.kubernetes.io/name: {{ include "namel3ss-backend.name" . }}
            spec:
              containers:
                - name: backend
                  image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
                  imagePullPolicy: {{ .Values.image.pullPolicy }}
                  ports:
                    - containerPort: 8000
                  env:
                    {{- range $key, $value := .Values.env }}
                    - name: {{ $key }}
                      value: "{{ $value }}"
                    {{- end }}
    """
  ).strip() + "\n"


def _render_helm_service() -> str:
  return textwrap.dedent(
    """
        apiVersion: v1
        kind: Service
        metadata:
          name: {{ include "namel3ss-backend.fullname" . }}
        spec:
          type: {{ .Values.service.type }}
          ports:
            - port: {{ .Values.service.port }}
              targetPort: 8000
          selector:
            app.kubernetes.io/name: {{ include "namel3ss-backend.name" . }}
        """
    ).strip() + "\n"


def _render_helm_ingress() -> str:
  return textwrap.dedent(
    """
        {{- if .Values.ingress.enabled }}
        apiVersion: networking.k8s.io/v1
        kind: Ingress
        metadata:
          name: {{ include "namel3ss-backend.fullname" . }}
          annotations:
            kubernetes.io/ingress.class: {{ .Values.ingress.className | default "nginx" }}
        spec:
          rules:
            {{- range .Values.ingress.hosts }}
            - host: {{ .host }}
              http:
                paths:
                  {{- range .paths }}
                  - path: {{ .path }}
                    pathType: {{ .pathType }}
                    backend:
                      service:
                        name: {{ include "namel3ss-backend.fullname" $ }}
                        port:
                          number: {{ $.Values.service.port }}
                  {{- end }}
            {{- end }}
          tls:
            {{- range .Values.ingress.tls }}
            - hosts: {{ .hosts }}
              secretName: {{ .secretName }}
            {{- end }}
        {{- end }}
        """
    ).strip() + "\n"


def _render_env_example(env_keys: Sequence[str]) -> str:
    if env_keys:
        body = "\n".join(f"{key}=changeme" for key in env_keys)
    else:
        body = "NAMEL3SS_ENV=production"
    header = "# Environment variables used by the generated backend\n"
    return header + body + "\n"


def emit_deployment_artifacts(out_dir: Path, state: BackendState) -> None:
    """Write deployment helpers into *out_dir* for the provided backend state."""

    app_name = str(state.app.get("name") or "Namel3ss App")
    slug = _slugify(app_name)

    (out_dir / "requirements.txt").write_text(_render_backend_requirements(), encoding="utf-8")
    (out_dir / "Dockerfile").write_text(_render_dockerfile(app_name, slug), encoding="utf-8")
    (out_dir / ".dockerignore").write_text(_render_dockerignore(), encoding="utf-8")

    deploy_dir = out_dir / "deploy"
    terraform_dir = deploy_dir / "terraform"
    helm_dir = deploy_dir / "helm"
    helm_templates = helm_dir / "templates"

    helm_templates.mkdir(parents=True, exist_ok=True)
    terraform_dir.mkdir(parents=True, exist_ok=True)
    deploy_dir.mkdir(parents=True, exist_ok=True)

    (deploy_dir / "README.md").write_text(_render_deploy_readme(app_name), encoding="utf-8")
    (deploy_dir / "nginx.conf").write_text(_render_nginx_config(app_name), encoding="utf-8")
    (deploy_dir / "Caddyfile").write_text(_render_caddyfile(app_name), encoding="utf-8")
    (deploy_dir / "env.example").write_text(_render_env_example(state.env_keys), encoding="utf-8")

    (terraform_dir / "main.tf").write_text(_render_terraform_main(app_name, slug), encoding="utf-8")
    (terraform_dir / "variables.tf").write_text(_render_terraform_variables(), encoding="utf-8")
    (terraform_dir / "outputs.tf").write_text(_render_terraform_outputs(), encoding="utf-8")

    (helm_dir / "Chart.yaml").write_text(_render_helm_chart(app_name), encoding="utf-8")
    (helm_dir / "values.yaml").write_text(_render_helm_values(slug), encoding="utf-8")
    (helm_templates / "deployment.yaml").write_text(_render_helm_deployment(), encoding="utf-8")
    (helm_templates / "service.yaml").write_text(_render_helm_service(), encoding="utf-8")
    (helm_templates / "ingress.yaml").write_text(_render_helm_ingress(), encoding="utf-8")

    workflows_dir = out_dir / ".github" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    (workflows_dir / "ci.yml").write_text(_render_ci_workflow(slug), encoding="utf-8")