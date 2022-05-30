terraform {
  backend "s3" {
    bucket         = "h1-data-science-tf-state"
    key            = "data-science/global/iam/tf.state"
    region         = "us-east-2"
    dynamodb_table = "h1datascience-iam-tf-infra"
  }
}
