resource "kubernetes_namespace" "data_science_emr" {
  metadata {
    name = "data-science-emr"
  }
  depends_on = [
    aws_iam_role.emr_job
  ]
}

resource "kubernetes_role" "emr-container" {
  metadata {
    name      = "emr-containers"
    namespace = kubernetes_namespace.data_science_emr.id
  }