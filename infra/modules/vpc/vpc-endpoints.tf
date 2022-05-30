data "aws_vpc_endpoint_service" "s3" {
  service = "s3"
  service_type = "Gateway"
}

resource "aws_vpc_endpoint" "s3" {
  vpc_id            = local.vpc_id
  service_name      = data.aws_vpc_endpoint_service.s3.service_name
  tags = var.vpc_tags
}

resource "aws_vpc_endpoint_route_table_association" "private_s3" {
  count = local.nat_gateway_count
  vpc_endpoint_id = aws_vpc_endpoint.s3.id
  route_table_id  = element(aws_route_table.private.*.id, count.index)
}
