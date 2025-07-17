# Example Terraform configuration for drift detection testing
# This creates basic AWS resources that can be monitored for drift

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"

# EC2 Instance for testing CloudWatch logs
resource "aws_instance" "test_instance" {
  ami           = "ami-0c55b159cbfafe1f0"  # Amazon Linux 2 AMI (HVM), SSD Volume Type
  instance_type = "t2.micro"
  
  iam_instance_profile = aws_iam_instance_profile.test_profile.name

  user_data = <<-EOF
              #!/bin/bash
              echo "Hello from EC2" | tee /var/log/test.log
              aws logs create-log-stream --log-group-name ${aws_cloudwatch_log_group.test_log_group.name} --log-stream-name ec2-instance-logs
              aws logs put-log-events --log-group-name ${aws_cloudwatch_log_group.test_log_group.name} --log-stream-name ec2-instance-logs --log-events timestamp=$(date +%s%N | cut -b1-13),message="Hello from EC2"
              EOF

  tags = {
    Name = "terraform-drift-test-instance"
  }
}

# IAM instance profile for the EC2 instance
resource "aws_iam_instance_profile" "test_profile" {
  name = "terraform-drift-test-profile"
  role = aws_iam_role.test_role.name
}
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
}

# Note: A default VPC (vpc-0913b9969b0533ea1) exists in the account.
# This is standard for AWS accounts and does not represent a significant drift.
# If a custom VPC is required, it should be defined here.
# Configure the AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = "terraform-drift-detection"
      ManagedBy   = "terraform"
      CreatedBy   = "RemediateAgent"
      LastUpdated = formatdate("YYYY-MM-DD", timestamp())
    }
  }
}

# Random suffix for unique resource names
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# Note: The following S3 buckets were detected but are not managed by Terraform:
# - aws-cloudtrail-logs-hkt (Created: 2023-07-01)
# - datmh-bedrock-kb (Created: 2023-07-12)
# - genifast-drift-logs (Created: 2023-07-01)
# These buckets should be reviewed for potential import into Terraform or exclusion from drift detection.

# S3 Bucket for testing drift detection
resource "aws_s3_bucket" "test_bucket" {
  bucket = "terraform-drift-test-s49q9z8j"

  tags = {
    Name        = "terraform-drift-test-bucket"
    Environment = var.environment
    ManagedBy   = "terraform"
  }

  lifecycle_rule {
    enabled = true

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    expiration {
      days = 90
    }
  }
}

# S3 Bucket versioning configuration
resource "aws_s3_bucket_versioning" "test_bucket_versioning" {
  bucket = aws_s3_bucket.test_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Bucket encryption configuration
resource "aws_s3_bucket_server_side_encryption_configuration" "test_bucket_encryption" {
  bucket = aws_s3_bucket.test_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 Bucket public access block
resource "aws_s3_bucket_public_access_block" "test_bucket_pab" {
  bucket = aws_s3_bucket.test_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Existing default VPC Security Group
resource "aws_security_group" "default" {
  name        = "default"
  description = "default VPC security group"
  vpc_id      = "vpc-0913b9969b0533ea1"

  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "default"
  }

  lifecycle {
    ignore_changes = [
      name,
      description,
      ingress,
      egress,
    ]
  }
}

# IAM Role for testing
resource "aws_iam_role" "test_role" {
  name        = "terraform-drift-test-role-s49q9z8j"
  description = "IAM role for drift detection testing"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "terraform-drift-test-role"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# IAM Policy for the test role
resource "aws_iam_role_policy" "test_policy" {
  name = "terraform-drift-test-policy"
  role = aws_iam_role.test_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "${aws_s3_bucket.test_bucket.arn}",
          "${aws_s3_bucket.test_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${var.aws_region}:*:log-group:${aws_cloudwatch_log_group.test_log_group.name}:*"
      }
    ]
  })
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "test_log_group" {
  name              = "/aws/terraform-drift-test/s49q9z8j"
  retention_in_days = 14  # Set a default retention period

  tags = {
    Name = "terraform-drift-test-logs"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
} 