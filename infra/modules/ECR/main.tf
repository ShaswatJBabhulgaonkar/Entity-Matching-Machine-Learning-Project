resource "aws_ecr_repository" "repository" {
  name                 = var.name
  image_tag_mutability = var.immutable ? "IMMUTABLE" : "MUTABLE"

  image_scanning_configuration {
    scan_on_push = var.scan_on_push
  }

  tags = var.tags

  depends_on = [var.module_depends_on]
}