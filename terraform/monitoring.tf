# CloudWatch Alarm for S3 bucket changes
resource "aws_cloudwatch_metric_alarm" "s3_bucket_changes" {
  alarm_name          = "s3-bucket-changes-${aws_s3_bucket.test_bucket.id}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "NumberOfObjectsChanged"
  namespace           = "AWS/S3"
  period              = "300"
  statistic           = "Sum"
  threshold           = "0"
  alarm_description   = "This alarm monitors changes to objects in the S3 bucket"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    BucketName = aws_s3_bucket.test_bucket.id
  }
}

# SNS Topic for alerts
resource "aws_sns_topic" "alerts" {
  name = "s3-change-alerts-${random_string.suffix.result}"
}

# SNS Topic Policy
resource "aws_sns_topic_policy" "alerts" {
  arn = aws_sns_topic.alerts.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "Allow CloudWatch Alarms"
        Effect    = "Allow"
        Principal = { Service = "cloudwatch.amazonaws.com" }
        Action    = "SNS:Publish"
        Resource  = aws_sns_topic.alerts.arn
      }
    ]
  })
}