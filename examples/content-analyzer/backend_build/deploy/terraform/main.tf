terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.29"
    }
  }
}

provider "kubernetes" {
  host                   = var.kube_host
  token                  = var.kube_token
  cluster_ca_certificate = base64decode(var.kube_ca)
}

resource "kubernetes_namespace" "this" {
  metadata {
    name = var.namespace
  }
}

resource "kubernetes_deployment" "backend" {
  metadata {
    name      = "content-analyzer"
    namespace = kubernetes_namespace.this.metadata[0].name
    labels = {
      app = "content-analyzer"
    }
  }
  spec {
    replicas = var.replicas
    selector {
      match_labels = { app = "content-analyzer" }
    }
    template {
      metadata {
        labels = { app = "content-analyzer" }
      }
      spec {
        container {
          name  = "backend"
          image = var.backend_image
          port {
            container_port = 8000
          }
          env {
            name  = "NAMEL3SS_ENV"
            value = var.environment
          }
          env_from {
            secret_ref { name = kubernetes_secret.env.metadata[0].name }
          }
          resources {
            limits = {
              cpu    = "500m"
              memory = "512Mi"
            }
            requests = {
              cpu    = "100m"
              memory = "256Mi"
            }
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "backend" {
  metadata {
    name      = "content-analyzer"
    namespace = kubernetes_namespace.this.metadata[0].name
  }
  spec {
    selector = { app = "content-analyzer" }
    port {
      port        = 80
      target_port = 8000
    }
  }
}

resource "kubernetes_secret" "env" {
  metadata {
    name      = "content-analyzer-env"
    namespace = kubernetes_namespace.this.metadata[0].name
  }
  data = {
    NAMEL3SS_ENV = base64encode(var.environment)
  }
}
