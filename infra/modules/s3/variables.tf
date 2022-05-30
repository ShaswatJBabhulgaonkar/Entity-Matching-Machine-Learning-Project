ariable "aws_region" {
  description = "The AWS region to use to create resources."
  default     = "us-east-2"
}

variable "tags" {
    type        = map
    description = "(Optional) A mapping of tags to assign to the bucket."
    default     = {
        environment = "ci"
        terraform   = "true"
    }
}

variable "versioning" {
    type        = bool
    description = "(Optional) A state of versioning."
    default     = true
}

variable "acl" {
    type        = string
    description = " Defaults to private "
    default     = "private"
}

variable "bucket_name" {
    type        = string
    description = "Bucket Name"
}

variable "environment" {
    type = string
    description = "Environment Name"
}

variable "owner" {
    type = string
    description = "Owner"
}

variable "cost_center" {
    type = string
    description = "Cost Center"
}
