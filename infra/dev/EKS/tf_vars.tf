variable "region" {
  type = map(string)
  default = {
    "N_Virginia"   = "us-east-1"
    "Ohio"         = "us-east-2"
    "N_California" = "us-west-1"
    "Oregon"       = "us-west-2"
  }
}
