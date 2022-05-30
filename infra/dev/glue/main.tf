module "datascience-dev-glue-database" {
  source = "../../modules/glue"
  environment = "ci"
}