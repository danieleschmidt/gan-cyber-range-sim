# Multi-stage Dockerfile for GAN Cyber Range Simulator
FROM python:3.12-slim-bookworm AS builder

# Build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create build user
RUN useradd --create-home --shell /bin/bash builder
USER builder
WORKDIR /home/builder

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim-bookworm AS production

# Security updates and runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser

# Copy Python packages from builder
COPY --from=builder /home/builder/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Create application directories
RUN mkdir -p /app /app/logs /app/data \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

# Expose ports
EXPOSE 8080 9090

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHON_DISABLE_MODULES=yaml,pickle

# Default command
CMD ["python3", "simple_cli.py", "simulate", "--duration", "1.0"]

# Labels for metadata
LABEL maintainer="GAN Cyber Range Team"
LABEL version="1.0.0"
LABEL description="GAN Cyber Range Simulator - Adversarial Security Training"
LABEL org.opencontainers.image.source="https://github.com/yourusername/gan-cyber-range"
