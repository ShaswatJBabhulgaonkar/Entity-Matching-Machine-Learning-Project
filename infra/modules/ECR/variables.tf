variable "name" {
  type        = string
  description = "Name of the repository."
}

variable "immutable" {
  type        = bool
  description = "Set to true to prevent tags from being overwritten."
  default     = false
}

variable "tags" {
  type        = map(string)
  description = "Map of tags to assign to the resource."
}

variable "scan_on_push" {
  type        = bool
  description = "Indicates whether images should be scanned when pushed. Default to true."
  default     = true
}