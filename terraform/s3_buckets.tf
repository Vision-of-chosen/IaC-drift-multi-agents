# Existing S3 buckets detected during drift analysis

resource "aws_s3_bucket" "cloudtrail_logs" {
  bucket = "aws-cloudtrail-logs-hkt"
}

resource "aws_s3_bucket" "bedrock_kb" {
  bucket = "datmh-bedrock-kb"
}

resource "aws_s3_bucket" "drift_logs" {
  bucket = "genifast-drift-logs"
}

# Apply standard security settings to all buckets
resource "aws_s3_bucket_public_access_block" "block_public_access" {
  for_each = toset(["aws-cloudtrail-logs-hkt", "datmh-bedrock-kb", "genifast-drift-logs"])

  bucket = each.key

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable versioning for all buckets
resource "aws_s3_bucket_versioning" "enable_versioning" {
  for_each = toset(["aws-cloudtrail-logs-hkt", "datmh-bedrock-kb", "genifast-drift-logs"])

  bucket = each.key
  versioning_configuration {
    status = "Enabled"
  }
}

# Add default encryption for all buckets
resource "aws_s3_bucket_server_side_encryption_configuration" "default_encryption" {
  for_each = toset(["aws-cloudtrail-logs-hkt", "datmh-bedrock-kb", "genifast-drift-logs"])

  bucket = each.key

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}