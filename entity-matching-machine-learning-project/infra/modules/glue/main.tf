resource "aws_glue_catalog_database" "my_glue_catalog_database" {
  name        = "${var.environment}_ml_em"
  description = "${var.environment} Environment ml em glue database"
}