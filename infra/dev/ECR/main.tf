module "ct-aact-ingestor" {
  source = "../../modules/ecr"
  name = "h1-data-science-ml-em"

  tags = {
    Owner       = "data-science"
  }
}