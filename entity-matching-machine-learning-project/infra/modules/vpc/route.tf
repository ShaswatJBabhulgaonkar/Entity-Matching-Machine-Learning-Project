locals {
  nat_gateway_count = length(var.azs)
  vpc_id            = aws_vpc.this.id
}

resource "aws_internet_gateway" "this" {
  vpc_id = local.vpc_id

  tags = merge(
    {
      "Name" = format("%s", var.name)
    },
    var.tags,
  )
}

resource "aws_route_table" "public" {

  vpc_id = local.vpc_id

  tags = merge(
    {
      "Name" = format("%s-${var.public_subnet_suffix}", var.name)
    },
    var.tags,
  )
}

resource "aws_route" "public_internet_gateway" {

  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.this.id

  timeouts {
    create = "5m"
  }
}

resource "aws_route_table" "private" {
  count = local.nat_gateway_count

  vpc_id = local.vpc_id

  tags = merge(
    {
      "Name" = format(
        "%s-${var.private_subnet_suffix}-%s",
        var.name,
        element(var.azs, count.index),
      )
    },
    var.tags,
  )
}

resource "aws_eip" "nat" {
  count = local.nat_gateway_count

  vpc = true

  tags = merge(
    {
      "Name" = format(
        "%s-%s",
        var.name,
        element(var.azs, count.index),
      )
    },
    var.tags,
  )
}

locals {
  nat_gateway_ips = split(
    ",",
    join(",", aws_eip.nat.*.id),
  )
}

resource "aws_nat_gateway" "this" {
  count = local.nat_gateway_count

  allocation_id = element(
    local.nat_gateway_ips,
    count.index,
  )
  subnet_id = element(
    aws_subnet.public.*.id,
    count.index,
  )

  tags = merge(
    {
      "Name" = format(
        "%s-%s",
        var.name,
        element(var.azs, count.index),
      )
    },
    var.tags,
  )

  depends_on = [aws_internet_gateway.this]
}

resource "aws_route" "private_nat_gateway" {
  count = local.nat_gateway_count

  route_table_id         = element(aws_route_table.private.*.id, count.index)
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = element(aws_nat_gateway.this.*.id, count.index)

  timeouts {
    create = "5m"
  }
}

resource "aws_route_table_association" "private" {
  count = length(var.private_subnets)

  subnet_id = element(aws_subnet.private.*.id, count.index)
  route_table_id = element(aws_route_table.private.*.id, count.index,
  )
}

resource "aws_route_table_association" "public" {
  count = length(var.public_subnets)

  subnet_id      = element(aws_subnet.public.*.id, count.index)
  route_table_id = aws_route_table.public.id
}
