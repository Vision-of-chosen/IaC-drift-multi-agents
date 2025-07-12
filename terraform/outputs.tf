# Outputs for Terraform drift detection testing

output "s3_bucket_name" {
  description = "Name of the created S3 bucket"
  value       = aws_s3_bucket.test_bucket.id
}

output "s3_bucket_arn" {
  description = "ARN of the created S3 bucket"
  value       = aws_s3_bucket.test_bucket.arn
}

output "security_group_id" {
  description = "ID of the created security group"
  value       = aws_security_group.test_sg.id
}

output "security_group_arn" {
  description = "ARN of the created security group"
  value       = aws_security_group.test_sg.arn
}

output "iam_role_name" {
  description = "Name of the created IAM role"
  value       = aws_iam_role.test_role.name
}

output "iam_role_arn" {
  description = "ARN of the created IAM role"
  value       = aws_iam_role.test_role.arn
}

output "cloudwatch_log_group_name" {
  description = "Name of the created CloudWatch log group"
  value       = aws_cloudwatch_log_group.test_log_group.name
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the created CloudWatch log group"
  value       = aws_cloudwatch_log_group.test_log_group.arn
}

output "random_suffix" {
  description = "Random suffix used for resource naming"
  value       = random_string.suffix.result
} 