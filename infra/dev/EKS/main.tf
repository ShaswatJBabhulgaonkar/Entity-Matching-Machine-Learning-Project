module "eks" {
  source          = "../../modules/EKS"
  cluster_name    = local.cluster_name
  cluster_version = "1.21"

  tags = {
    Environment = "dev"
    Owner       = "data-platform"
  }