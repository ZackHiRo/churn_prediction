FROM public.ecr.aws/lambda/python:3.10 AS base

# Install system dependencies if needed (kept minimal for Lambda)

# Copy dependency files
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy application source
COPY src ${LAMBDA_TASK_ROOT}/src

# Set the CMD to your handler
CMD ["src.app.handler"]


