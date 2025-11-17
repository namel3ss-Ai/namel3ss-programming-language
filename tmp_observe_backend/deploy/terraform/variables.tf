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
