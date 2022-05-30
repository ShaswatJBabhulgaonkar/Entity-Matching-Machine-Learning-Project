module "datascience-dev-s3-bucket" {
  source = "../../modules/s3"

  bucket_name = "${var.bucket_name}-${var.environment}"
  acl         = var.acl
  environment = var.environment
  owner       = var.owner
  cost_center = var.cost_center
}