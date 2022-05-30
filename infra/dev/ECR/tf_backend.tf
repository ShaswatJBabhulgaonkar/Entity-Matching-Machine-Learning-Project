terraform {
  backend "s3" {
    bucket = "data-science-tf-state"
    key    = "data-science/dev/ecr/tf.state"
    region = "us-east-2"
    profile = "data-science"
  }
}
