output "cluster_id" {
  description = "The name/id of the EKS cluster. Will block on cluster creation until the cluster is really ready"
  value       = element(concat(aws_eks_cluster.this.*.id, [""]), 0)
  depends_on = [data.http.wait_for_cluster]
}
