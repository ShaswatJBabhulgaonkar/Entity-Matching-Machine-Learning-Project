variable "region" {
  type = map(string)
  default = {
    "N_Virginia"   = "us-east-1"
    "Ohio"         = "us-east-2"
    "N_California" = "us-west-1"
    "Oregon"       = "us-west-2"
    "Mumbai"       = "ap-south-1"
  }
}

variable "bucket_name" {
    type        = string
    description = "Bucket Name"
    default     = "ml-em"
}

variable "acl" {
    type        = string
    description = " Defaults to private "
    default     = "private"
}

variable "environment" {
    type        = string
    description = "Environment Name"
    default     = "dev"
}

variable "owner" {
    type        = string
    description = "Owner"
    default     = "Data-Science"
}

variable "cost_center" {
    type        = string
    description = "Cost Center"
    default     = "Data-Science"
}