# Multi-stage Dockerfile for GAN Cyber Range Simulator
FROM python:3.10-slim as base

# Security: Create non-root user
RUN groupadd -r cyberrange && useradd --no-log-init -r -g cyberrange cyberrange

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
RUN pip install -e ".[dev,security,ml]"
COPY . .
USER cyberrange
EXPOSE 8000
CMD ["python", "-m", "gan_cyber_range.api"]

# Production stage
FROM base as production
COPY src/ ./src/
RUN pip install --no-cache-dir .
USER cyberrange
EXPOSE 8000
CMD ["python", "-m", "gan_cyber_range.api"]