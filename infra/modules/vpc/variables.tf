variable "create_vpc" {
  description = "Controls if VPC should be created (it affects almost all resources)"
  type        = bool
  default     = true
}

variable "name" {
  description = "Name to be used on all the resources as identifier"
  type        = string
  default     = ""
}

variable "cidr" {
  description = "The CIDR block for the VPC. Default value is a valid CIDR, but not acceptable by AWS and should be overridden"
  type        = string
  default     = "0.0.0.0/0"
}

variable "public_subnet_suffix" {
  description = "Suffix to append to public subnets name"
  type        = string
  default     = "public"
}

variable "private_subnet_suffix" {
  description = "Suffix to append to private subnets name"
  type        = string
  default     = "private"
}

variable "public_subnets" {
  description = "A list of public subnets inside the VPC"
  type        = list(string)
  default     = []
}

variable "private_subnets" {
  description = "A list of private subnets inside the VPC"
  type        = list(string)
  default     = []
}

variable "enable_s3_endpoint" {
  description = "Should be true if you want to provision an S3 endpoint to the VPC"
  type        = bool
  default     = false
}

variable "s3_endpoint_type" {
  description = "S3 VPC endpoint type. Note - S3 Interface type support is only available on AWS provider 3.10 and later"
  type        = string
  default     = "Gateway"
}

variable "s3_endpoint_security_group_ids" {
  description = "The ID of one or more security groups to associate with the network interface for S3 interface endpoint"
  type        = list(string)
  default     = []
}

variable "s3_endpoint_subnet_ids" {
  description = "The ID of one or more subnets in which to create a network interface for S3 interface endpoint. Only a single subnet within an AZ is supported. If omitted, private subnets will be used."
  type        = list(string)
  default     = []
}

variable "s3_endpoint_private_dns_enabled" {
  description = "Whether or not to associate a private hosted zone with the specified VPC for S3 interface endpoint"
  type        = bool
  default     = false
}

variable "s3_endpoint_policy" {
  description = "A policy to attach to the endpoint that controls access to the service. Defaults to full access"
  type        = string
  default     = null
}

variable "azs" {
  description = "A list of availability zones names or ids in the region"
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "A map of tags to add to all resources"
  type        = map(string)
  default     = {}
}

variable "vpc_tags" {
  description = "Additional tags for the VPC"
  type        = map(string)
  default     = {}
}
