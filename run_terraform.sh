#!/bin/bash
set -e

cd ./terraform

echo "Initializing Terraform..."
terraform init

echo "Generating Terraform plan..."
terraform plan -out=tfplan

echo "Applying Terraform changes..."
terraform apply -auto-approve tfplan

echo "Terraform apply complete."