variable "aws_region" {
  description = "The AWS region to use to create resources."
  default     = "us-east-2"
}

variable "environment" {
  type        = string
  description = "Environment name"
}