# Example Terraform configuration for drift detection testing
# This creates basic AWS resources that can be monitored for drift

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
}

# Configure the AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = "terraform-drift-detection"
      ManagedBy   = "terraform"
    }
  }
}

# Random suffix for unique resource names
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# S3 Bucket for testing drift detection
resource "aws_s3_bucket" "test_bucket" {
  bucket = "${var.bucket_prefix}-${random_string.suffix.result}"
}

# S3 Bucket versioning configuration
resource "aws_s3_bucket_versioning" "test_bucket_versioning" {
  bucket = aws_s3_bucket.test_bucket.id
  versioning_configuration {
    status = var.bucket_versioning_enabled ? "Enabled" : "Suspended"
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

# EC2 Security Group for testing
resource "aws_security_group" "test_sg" {
  name_prefix = "terraform-drift-test-"
  description = "Security group for drift detection testing"
  
  # Allow SSH access (potential drift target)
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }
  
  # Allow HTTP access (potential drift target)  
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "terraform-drift-test-sg"
  }
}

# IAM Role for testing
resource "aws_iam_role" "test_role" {
  name = "terraform-drift-test-role-${random_string.suffix.result}"

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
    Name = "terraform-drift-test-role"
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
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.test_bucket.arn}/*"
      }
    ]
  })
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "test_log_group" {
  name              = "/aws/terraform-drift-test/${random_string.suffix.result}"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "terraform-drift-test-logs"
  }
} 