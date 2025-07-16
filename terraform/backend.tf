terraform {
  backend "s3" {
    bucket         = "terraform-state-bucket-s49q9z8j"
    key            = "terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock-s49q9z8j"
  }
}