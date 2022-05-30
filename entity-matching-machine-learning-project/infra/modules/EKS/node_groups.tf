resource "aws_eks_node_group" "primary" {
  node_group_name = "${var.cluster_name}-primary"

  cluster_name  = var.cluster_name
  node_role_arn = aws_iam_role.workers.arn
  subnet_ids    = var.private_subnets
}

resource "aws_eks_node_group" "executor" {
  node_group_name = "${var.cluster_name}-executor"
  cluster_name    = var.cluster_name
  node_role_arn   = aws_iam_role.workers.arn
  subnet_ids      = var.private_subnets
}
