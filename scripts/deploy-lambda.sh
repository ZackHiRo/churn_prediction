#!/bin/bash
# Quick deployment script for AWS Lambda
# Usage: ./scripts/deploy-lambda.sh [image-tag]

set -e

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
ECR_REPOSITORY=${ECR_REPOSITORY:-churn-prediction-lambda}
LAMBDA_FUNCTION_NAME=${LAMBDA_FUNCTION_NAME:-churn-prediction-api}
IMAGE_TAG=${1:-latest}

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo "🚀 Deploying to AWS Lambda"
echo "=========================="
echo "Region: $AWS_REGION"
echo "ECR Repository: $ECR_REPOSITORY"
echo "Image Tag: $IMAGE_TAG"
echo "Lambda Function: $LAMBDA_FUNCTION_NAME"
echo ""

# Step 1: Login to ECR
echo "📦 Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $ECR_REGISTRY

# Step 2: Build Docker image
echo "🔨 Building Docker image..."
docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .

# Step 3: Push to ECR
echo "⬆️  Pushing image to ECR..."
docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

# Step 4: Update Lambda function
echo "🔄 Updating Lambda function..."
aws lambda update-function-code \
    --function-name $LAMBDA_FUNCTION_NAME \
    --image-uri $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG \
    --region $AWS_REGION

# Step 5: Wait for update to complete
echo "⏳ Waiting for Lambda update to complete..."
aws lambda wait function-updated \
    --function-name $LAMBDA_FUNCTION_NAME \
    --region $AWS_REGION

echo ""
echo "✅ Deployment complete!"
echo ""
echo "To test the function, run:"
echo "  aws lambda invoke \\"
echo "    --function-name $LAMBDA_FUNCTION_NAME \\"
echo "    --payload '{\"httpMethod\": \"POST\", \"path\": \"/predict\", \"body\": \"{}\"}' \\"
echo "    response.json"
echo ""
echo "Or view logs:"
echo "  aws logs tail /aws/lambda/$LAMBDA_FUNCTION_NAME --follow"

