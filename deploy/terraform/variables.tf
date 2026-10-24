variable "region" {
  type    = string
  default = "us-east-1"
}

variable "name" {
  description = "Cluster / resource name prefix"
  type        = string
  default     = "skillforge"
}

variable "kubernetes_version" {
  type    = string
  default = "1.31"
}

variable "node_instance_types" {
  type    = list(string)
  default = ["t3.large"]
}

variable "node_count" {
  description = "Desired worker nodes (the stack needs ~3 vCPU / 6 GiB; 2 x t3.large is comfortable)"
  type        = number
  default     = 2
}

variable "vpc_cidr" {
  type    = string
  default = "10.42.0.0/16"
}

variable "api_allowed_cidrs" {
  description = "CIDRs allowed to reach the public Kubernetes API endpoint. Restrict this to your own IP."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
