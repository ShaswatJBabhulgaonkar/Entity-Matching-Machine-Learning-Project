terraform {
  backend "s3" {
    bucket         = "h1-data-science-non-prod-tf-state"
    key            = "data-science/dev/s3/tf.state"
    region         = "us-east-2"
    dynamodb_table = "h1datascience-dev-tf-infra"
  }
}
