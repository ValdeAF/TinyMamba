# Use a lighter, CPU-only Python base
FROM python:3.10-slim

# prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install System Dependencies & C Build Tools
# Added tar, curl, and ca-certificates which VS Code needs to install its server
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    tar \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up the working directory
WORKDIR /app

# 3. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# 4. Environment configuration
ENV PYTHONPATH="/app"

# Default to bash
CMD ["/bin/bash"]
