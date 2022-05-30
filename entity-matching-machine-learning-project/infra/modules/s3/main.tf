resource "aws_s3_bucket" "my_s3_bucket" {
  bucket = var.bucket_name
  acl = var.acl
  
   versioning {
    enabled = var.versioning
  }
  
  tags = {
    "Environment" = var.environment
    "Owner"       = var.owner
    "Cost-Center" = var.cost_center
  }
}

resource "aws_s3_bucket_public_access_block" "my_s3_bucket_public_acces_block" {
  bucket = aws_s3_bucket.my_s3_bucket.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
