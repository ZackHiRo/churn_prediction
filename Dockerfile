FROM public.ecr.aws/lambda/python:3.10 AS base

# Install system dependencies needed for building Python packages (e.g., xgboost)
# Install build tools, newer GCC (xgboost requires GCC >= 8.1), and newer cmake (xgboost requires cmake >= 3.18)
RUN yum install -y gcc gcc-c++ make curl tar gzip && \
    curl -L https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.tar.gz -o /tmp/cmake.tar.gz && \
    tar -xzf /tmp/cmake.tar.gz -C /tmp && \
    cp -r /tmp/cmake-3.28.0-linux-x86_64/* /usr/local/ && \
    ln -sf /usr/local/bin/cmake /usr/bin/cmake && \
    rm -rf /tmp/cmake.tar.gz /tmp/cmake-3.28.0-linux-x86_64 && \
    # Try to install GCC 8+ - Amazon Linux 2 may have gcc8 available
    (yum install -y gcc8 gcc8-c++ && \
     ln -sf /usr/bin/gcc8 /usr/bin/gcc && \
     ln -sf /usr/bin/g++8 /usr/bin/g++ || \
     echo "GCC 8+ not available in repos, will try with system GCC") && \
    yum clean all && \
    rm -rf /var/cache/yum

# Copy dependency files
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

# Upgrade pip and install Python dependencies
# Force xgboost to use pre-built wheels only (fail if not available, then try older version)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir $(grep -v xgboost ${LAMBDA_TASK_ROOT}/requirements.txt) && \
    (pip install --no-cache-dir --only-binary :all: xgboost || \
     (echo "No pre-built wheel available for xgboost 3.x, trying xgboost 2.x..." && \
      pip install --no-cache-dir --only-binary :all: "xgboost<3.0.0,>=2.0.0"))

# Copy application source
COPY src ${LAMBDA_TASK_ROOT}/src

# Set the CMD to your handler
CMD ["src.app.handler"]


