resource "aws_cloudwatch_log_group" "this" {
  name              = "/aws/eks/${var.cluster_name}/cluster"
  retention_in_days = var.cluster_log_retention_in_days
  tags              = var.tags
}

resource "aws_eks_cluster" "this" {
  name                      = var.cluster_name
  enabled_cluster_log_types = ["api", "audit", "scheduler"]
  role_arn                  = var.cluster_role_arn
  version                   = var.cluster_version
  tags                      = var.tags


  depends_on = [
    aws_cloudwatch_log_group.this
  ]
}