#!/bin/bash
# Test Lambda function locally using Docker and RIE (Runtime Interface Emulator)
# Usage: ./scripts/test-lambda-local.sh

set -e

# Configuration
IMAGE_NAME=${IMAGE_NAME:-churn-prediction-lambda}
CONTAINER_NAME=${CONTAINER_NAME:-churn-lambda-test}
PORT=${PORT:-9000}

echo "🧪 Testing Lambda function locally"
echo "=================================="
echo ""

# Check if RIE is installed
RIE_PATH="$HOME/.aws-lambda-rie/aws-lambda-rie"
if [ ! -f "$RIE_PATH" ]; then
    echo "📥 Downloading Lambda Runtime Interface Emulator..."
    mkdir -p ~/.aws-lambda-rie
    curl -Lo $RIE_PATH \
        https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie
    chmod +x $RIE_PATH
    echo "✅ RIE downloaded"
fi

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME .

# Stop and remove existing container if running
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Run container with RIE
echo "🚀 Starting container with RIE on port $PORT..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8080 \
    -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-} \
    -e DAGSHUB_TOKEN=${DAGSHUB_TOKEN:-} \
    -e MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME:-churn_prediction} \
    -e MLFLOW_MODEL_NAME=${MLFLOW_MODEL_NAME:-ChurnModel} \
    -e MLFLOW_MODEL_STAGE=${MLFLOW_MODEL_STAGE:-Production} \
    --entrypoint /aws-lambda-rie \
    $IMAGE_NAME \
    /usr/local/bin/python -m awslambdaric src.app.handler

# Wait for container to start
echo "⏳ Waiting for container to start..."
sleep 3

# Test the endpoint
echo ""
echo "🧪 Testing /predict endpoint..."
echo ""

# Test with a sample payload
curl -X POST "http://localhost:$PORT/2015-03-31/functions/function/invocations" \
    -H "Content-Type: application/json" \
    -d '{
        "httpMethod": "POST",
        "path": "/predict",
        "headers": {"Content-Type": "application/json"},
        "body": "{\"tenure\": 12, \"MonthlyCharges\": 70.5, \"TotalCharges\": 846, \"PhoneService\": \"Yes\", \"MultipleLines\": \"No\", \"InternetService\": \"DSL\", \"OnlineSecurity\": \"No\", \"OnlineBackup\": \"No\", \"DeviceProtection\": \"No\", \"TechSupport\": \"No\", \"StreamingTV\": \"No\", \"StreamingMovies\": \"No\", \"Contract\": \"Month-to-month\", \"PaperlessBilling\": \"Yes\", \"PaymentMethod\": \"Electronic check\", \"SeniorCitizen\": 0, \"Partner\": \"Yes\", \"Dependents\": \"No\", \"gender\": \"Male\"}"
    }' | jq '.' || echo "Response received (jq not installed, raw output above)"

echo ""
echo "✅ Test complete!"
echo ""
echo "To view logs:"
echo "  docker logs -f $CONTAINER_NAME"
echo ""
echo "To stop the container:"
echo "  docker stop $CONTAINER_NAME"
echo "  docker rm $CONTAINER_NAME"

