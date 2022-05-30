module "ml-em" {
  source = "../../modules/ecr"
  name = "data-science-ml-em"

  tags = {
    Owner       = "data-science"
  }
}
