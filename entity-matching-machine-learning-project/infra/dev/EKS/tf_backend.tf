terraform {
  backend "s3" {
    bucket = "h1-data-science-non-prod-tf-state"
    key    = "data-science/dev/ecr/tf.state"
    region = "us-east-2"
    profile = "h1-data-science-non-prod"
  }
}