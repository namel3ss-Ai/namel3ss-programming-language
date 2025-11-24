# Namel3ss Deploy Runtime

Docker, Kubernetes, and cloud deployment runtime for the Namel3ss programming language.

## Overview

This package provides deployment adapters for Namel3ss applications. It generates Dockerfiles, Kubernetes manifests, cloud platform configurations (AWS, GCP, Azure), and orchestration files, demonstrating that **Namel3ss is a language that targets multiple deployment platforms**.

## Installation

```bash
pip install namel3ss-runtime-deploy
```

Or install with optional features:

```bash
# With cloud platform support
pip install namel3ss-runtime-deploy[aws,gcp,azure]

# With Kubernetes tools
pip install namel3ss-runtime-deploy[k8s]

# Development dependencies
pip install namel3ss-runtime-deploy[dev]
```

## Usage

### Generate Docker Configuration

```python
from namel3ss import Parser, build_backend_ir
from namel3ss_runtime_deploy import generate_docker

# Parse .ai source
source = '''
app "MyApp" connects to postgres "DB".

prompt "Greet" {
    model: "gpt-4o-mini"
    template: "Say hello to {{name}}."
}
'''

parser = Parser(source)
module = parser.parse()
app = module.body[0]

# Build IR
ir = build_backend_ir(app)

# Generate Docker configuration
generate_docker(
    ir=ir,
    output_dir="deploy/",
    base_image="python:3.11-slim",
    multi_stage=True,
)
```

### Generate Kubernetes Manifests

```python
from namel3ss_runtime_deploy import generate_kubernetes

generate_kubernetes(
    ir=ir,
    output_dir="k8s/",
    replicas=3,
    enable_hpa=True,           # Horizontal Pod Autoscaler
    enable_ingress=True,
    domain="myapp.example.com",
)
```

### Generate Cloud Platform Configuration

```python
from namel3ss_runtime_deploy import generate_aws_config

# AWS Elastic Beanstalk / ECS
generate_aws_config(
    ir=ir,
    output_dir="aws/",
    deployment_type="ecs",      # ecs, lambda, beanstalk
    region="us-east-1",
)
```

## What Gets Generated

### Docker Output

```
deploy/
├── Dockerfile                 # Multi-stage build
├── Dockerfile.backend         # Backend service
├── Dockerfile.frontend        # Frontend static files
├── docker-compose.yml         # Local development
├── docker-compose.prod.yml    # Production config
├── .dockerignore
└── scripts/
    ├── build.sh
    ├── push.sh
    └── deploy.sh
```

### Kubernetes Output

```
k8s/
├── namespace.yaml
├── backend/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   └── hpa.yaml               # Horizontal Pod Autoscaler
├── frontend/
│   ├── deployment.yaml
│   └── service.yaml
├── database/
│   ├── statefulset.yaml       # PostgreSQL
│   ├── service.yaml
│   └── pvc.yaml               # Persistent volume claim
├── ingress.yaml               # Nginx ingress
├── kustomization.yaml
└── scripts/
    ├── apply.sh
    └── rollback.sh
```

### AWS Output

```
aws/
├── task-definition.json       # ECS task definition
├── service.json               # ECS service
├── ecs-params.yml
├── cloudformation.yml         # Infrastructure as code
├── buildspec.yml              # CodeBuild
├── appspec.yml                # CodeDeploy
└── .ebextensions/             # Elastic Beanstalk (if used)
    └── app.config
```

## Features

### ✅ Docker Support
- Multi-stage builds (optimized size)
- Development and production configurations
- Docker Compose orchestration
- Health checks
- Volume management

### ✅ Kubernetes Support
- Deployment manifests
- Service definitions
- ConfigMaps and Secrets
- Horizontal Pod Autoscaling
- Ingress configuration
- StatefulSets for databases
- Kustomize overlays

### ✅ Cloud Platforms
- **AWS:** ECS, Lambda, Elastic Beanstalk, App Runner
- **GCP:** Cloud Run, GKE, App Engine
- **Azure:** Container Instances, AKS, App Service
- **DigitalOcean:** App Platform, Kubernetes

### ✅ CI/CD Integration
- GitHub Actions workflows
- GitLab CI/CD pipelines
- CircleCI configuration
- Jenkins pipelines
- Azure DevOps

### ✅ Infrastructure as Code
- CloudFormation (AWS)
- Terraform modules
- Pulumi templates
- ARM templates (Azure)

## Configuration

### Docker Generation

```python
from namel3ss_runtime_deploy import generate_docker

generate_docker(
    ir=ir,
    output_dir="deploy/",
    base_image="python:3.11-slim",
    multi_stage=True,
    include_frontend=True,
    optimize_size=True,          # Use Alpine, strip binaries
    health_check={
        "endpoint": "/health",
        "interval": "30s",
        "timeout": "3s",
        "retries": 3,
    },
)
```

### Kubernetes Generation

```python
from namel3ss_runtime_deploy import generate_kubernetes

generate_kubernetes(
    ir=ir,
    output_dir="k8s/",
    namespace="namel3ss-prod",
    replicas=3,
    enable_hpa=True,
    hpa_config={
        "min_replicas": 2,
        "max_replicas": 10,
        "cpu_threshold": 70,
        "memory_threshold": 80,
    },
    enable_ingress=True,
    ingress_class="nginx",
    domain="myapp.example.com",
    tls_enabled=True,
    resource_limits={
        "cpu": "1000m",
        "memory": "1Gi",
    },
    resource_requests={
        "cpu": "100m",
        "memory": "256Mi",
    },
)
```

### AWS ECS Generation

```python
from namel3ss_runtime_deploy import generate_aws_config

generate_aws_config(
    ir=ir,
    output_dir="aws/",
    deployment_type="ecs",
    region="us-east-1",
    cluster_name="namel3ss-cluster",
    vpc_id="vpc-xxxxx",
    subnets=["subnet-xxxxx", "subnet-yyyyy"],
    load_balancer_type="application",
    auto_scaling={
        "min_capacity": 2,
        "max_capacity": 10,
        "target_cpu": 70,
    },
)
```

## Architecture

### Deployment Strategies

The deploy runtime supports multiple strategies:

| Strategy | Use Case | Complexity |
|----------|----------|------------|
| **Docker Compose** | Local dev, small deployments | Low |
| **Kubernetes** | Production, scalable apps | High |
| **AWS ECS** | AWS-native, managed containers | Medium |
| **AWS Lambda** | Serverless, event-driven | Low |
| **Cloud Run** | GCP-native, auto-scaling | Low |

### Multi-Environment Support

```
deploy/
├── dev/                       # Development environment
│   ├── docker-compose.yml
│   └── .env.dev
├── staging/                   # Staging environment
│   ├── k8s/
│   └── .env.staging
└── production/                # Production environment
    ├── k8s/
    ├── cloudformation.yml
    └── .env.production
```

## Advanced Usage

### Custom Dockerfile

The generated Dockerfile can be extended:

```dockerfile
# deploy/Dockerfile.custom
FROM namel3ss-generated-image:latest

# Add custom tools
RUN apt-get update && apt-get install -y \
    custom-tool \
    another-dependency

# Custom configuration
COPY custom-config.yml /app/config/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Helm Charts

For advanced Kubernetes deployments:

```python
from namel3ss_runtime_deploy import generate_helm_chart

generate_helm_chart(
    ir=ir,
    output_dir="helm/namel3ss-app",
    chart_version="1.0.0",
    app_version="0.5.0",
)
```

Generated structure:

```
helm/namel3ss-app/
├── Chart.yaml
├── values.yaml
├── values-dev.yaml
├── values-prod.yaml
└── templates/
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml
    └── _helpers.tpl
```

### CI/CD Pipelines

```python
from namel3ss_runtime_deploy import generate_ci_pipeline

generate_ci_pipeline(
    ir=ir,
    output_dir=".github/workflows/",
    platform="github-actions",
    stages=["test", "build", "deploy"],
    deploy_target="aws-ecs",
)
```

Generated workflow:

```yaml
# .github/workflows/deploy.yml
name: Deploy to AWS ECS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t myapp:latest .
      - name: Push to ECR
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
          docker push myapp:latest
      - name: Deploy to ECS
        run: aws ecs update-service --cluster namel3ss --service myapp --force-new-deployment
```

## Deployment Examples

### Local Docker Compose

```bash
cd deploy
docker-compose up -d
```

### Kubernetes Deployment

```bash
cd k8s

# Apply all manifests
kubectl apply -f .

# Or use the script
./scripts/apply.sh

# Check status
kubectl get pods -n namel3ss-prod

# View logs
kubectl logs -f deployment/backend -n namel3ss-prod
```

### AWS ECS Deployment

```bash
cd aws

# Build and push Docker image
./scripts/build.sh
./scripts/push.sh

# Deploy to ECS
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs update-service --cluster namel3ss-cluster --service myapp --task-definition myapp:latest
```

### Serverless (AWS Lambda)

```python
from namel3ss_runtime_deploy import generate_serverless_config

generate_serverless_config(
    ir=ir,
    output_dir="serverless/",
    provider="aws",
    runtime="python3.11",
    functions={
        "greet": {
            "handler": "handler.greet",
            "events": [{"http": {"path": "greet", "method": "post"}}],
        }
    },
)
```

Deploy:

```bash
cd serverless
npm install -g serverless
serverless deploy --stage prod --region us-east-1
```

## Monitoring & Observability

Generated configurations include:

- **Health checks:** `/health`, `/ready` endpoints
- **Metrics:** Prometheus-compatible `/metrics`
- **Logging:** Structured JSON logs
- **Tracing:** OpenTelemetry integration

### Example Prometheus Configuration

```yaml
# k8s/monitoring/prometheus.yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: namel3ss-backend
spec:
  selector:
    matchLabels:
      app: backend
  endpoints:
    - port: metrics
      interval: 30s
```

## Relationship to Core

The deploy runtime **depends on** the Namel3ss language core:

```
namel3ss (core)          ← Provides IR types, parser
    ↑
    | imports
    |
namel3ss-runtime-deploy  ← Consumes IR, generates deployment configs
```

**Dependency rules:**
- ✅ Deploy runtime can import from `namel3ss` core
- ❌ Core CANNOT import from deploy runtime
- ✅ Deploy runtime is independent of other runtimes

## Alternative Runtimes

Namel3ss supports multiple runtime targets:

- **namel3ss-runtime-deploy** (this package) - Docker, K8s, cloud platforms
- **namel3ss-runtime-http** - FastAPI/HTTP backends
- **namel3ss-runtime-frontend** - Static sites, React apps
- **Custom runtimes** - Build your own!

See [docs/RUNTIME_GUIDE.md](../../docs/RUNTIME_GUIDE.md) for creating custom runtimes.

## Development

### Run Tests

```bash
cd runtimes/deploy
pytest
```

### Build Package

```bash
python -m build
```

### Install Locally

```bash
pip install -e .
```

## License

MIT License - see LICENSE file for details.

## Links

- **Repository:** https://github.com/SsebowaDisan/namel3ss-programming-language
- **Documentation:** https://github.com/SsebowaDisan/namel3ss-programming-language/tree/main/runtimes/deploy
- **Issues:** https://github.com/SsebowaDisan/namel3ss-programming-language/issues
- **Language Core:** https://github.com/SsebowaDisan/namel3ss-programming-language/tree/main/namel3ss
