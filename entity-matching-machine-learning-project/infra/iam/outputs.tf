output "cluster_role_arn" {
  value = aws_iam_role.cluster.arn
}

output "worker_node_arn" {
  value = aws_iam_role.workers.arn
}

output "emr_iam_role_name" {
  value = aws_iam_role.emr_job.name
}

output "argo_iam_role_name" {
  value = aws_iam_role.argo.name
}

output "argo_iam_role_arn" {
  value = aws_iam_role.argo.arn
}