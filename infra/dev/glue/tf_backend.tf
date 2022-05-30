terraform {
  backend "glue" {
    bucket         = "data-science-tf-state"
    key            = "data-science/dev/glue/tf.state"
    region         = "us-east-2"
    dynamodb_table = "datascience-dev-tf-infra"
  }
}
