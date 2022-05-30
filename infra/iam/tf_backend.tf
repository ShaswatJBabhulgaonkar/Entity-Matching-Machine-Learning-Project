terraform {
  backend "s3" {
    bucket         = "data-science-tf-state"
    key            = "data-science/global/iam/tf.state"
    region         = "us-east-2"
    dynamodb_table = "datascience-iam-tf-infra"
  }
}
