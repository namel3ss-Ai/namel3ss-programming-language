output "service_name" {
  description = "Service backing the Namel3ss backend"
  value       = kubernetes_service.backend.metadata[0].name
}

output "namespace" {
  value = kubernetes_namespace.this.metadata[0].name
}
