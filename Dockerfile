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
# xgboost will use pre-built wheels (preferred) or build from source if needed
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy application source
COPY src ${LAMBDA_TASK_ROOT}/src

# Set the CMD to your handler
CMD ["src.app.handler"]


