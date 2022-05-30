resource "kubernetes_namespace" "argo" {
  metadata {
    name = "argo-run"
  }
  depends_on = [
    aws_iam_role.argo
  ]
}

resource "kubernetes_role" "argo-workflow" {
  metadata {
    name      = "workflow"
    namespace = kubernetes_namespace.argo.id
  }

  rule {
    api_groups = [""]
    resources  = ["pods"]
    verbs      = ["get", "watch", "patch"]
  }
  rule {
    api_groups = [""]
    resources  = ["pods/log"]
    verbs      = ["get", "watch"]
  }
}

resource "kubernetes_service_account" "argo" {
  metadata {
    name = "workflow"
    namespace = kubernetes_namespace.argo.id
  }
}

resource "kubernetes_role_binding" "argo" {
  metadata {
    name      = "workflow"
    namespace = kubernetes_namespace.argo.id
  }
  role_ref {
    kind      = "Role"
    name      = kubernetes_role.argo-workflow.id
  }
  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.argo.id
  }
}
