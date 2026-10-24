output "cluster_name" {
  value = module.eks.cluster_name
}

output "kubeconfig_command" {
  value = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}"
}

output "ecr_registry" {
  description = "Use as image.registry (with a trailing slash) when installing the Helm chart"
  value       = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.region}.amazonaws.com/"
}

output "ecr_repositories" {
  value = { for k, r in aws_ecr_repository.images : k => r.repository_url }
}

data "aws_caller_identity" "current" {}
