terraform {
  backend "glue" {
    bucket         = "h1-data-science-non-prod-tf-state"
    key            = "data-science/dev/glue/tf.state"
    region         = "us-east-2"
    dynamodb_table = "h1datascience-dev-tf-infra"
  }
}