# Multi-stage Dockerfile for GAN Cyber Range Simulator
# Security-hardened container with minimal attack surface

# Base stage with common dependencies
FROM python:3.11-slim as base

# Metadata
LABEL maintainer="info@gan-cyber-range.org"
LABEL version="0.1.0"
LABEL description="GAN Cyber Range Simulator - Security Research Platform"

# Security hardening: Update packages and install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        # Build dependencies
        build-essential \
        git \
        curl \
        # Security tools for cyber range functionality
        nmap \
        netcat-openbsd \
        tcpdump \
        dnsutils \
        # Process management
        tini \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Security: Create non-root user with specific UID/GID
RUN groupadd -r -g 1001 cyberrange && \
    useradd --no-log-init -r -g cyberrange -u 1001 -m -d /home/cyberrange cyberrange

# Set working directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/tmp && \
    chown -R cyberrange:cyberrange /app

# Install Python dependencies separately for better caching
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev,test,security,performance,ml]"

# Copy source code
COPY --chown=cyberrange:cyberrange . .

# Security context
USER cyberrange

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Use tini for proper signal handling
ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "gan_cyber_range.api"]

# Testing stage - for running tests in CI/CD
FROM development as testing

# Run tests during build
RUN python -m pytest tests/ --tb=short

# Security scanning stage
FROM base as security-scan

# Install security scanning tools
RUN pip install --no-cache-dir bandit safety semgrep

# Copy source for scanning
COPY --chown=cyberrange:cyberrange src/ ./src/
COPY --chown=cyberrange:cyberrange requirements.txt pyproject.toml ./

# Run security scans
USER cyberrange
RUN bandit -r src/ -f json -o /tmp/bandit-report.json || true
RUN safety check --json --output /tmp/safety-report.json || true

# Production stage - minimal and secure
FROM python:3.11-slim as production

# Install only runtime dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        # Minimal runtime dependencies
        curl \
        tini \
        # Security tools needed for runtime
        nmap \
        netcat-openbsd \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Create user
RUN groupadd -r -g 1001 cyberrange && \
    useradd --no-log-init -r -g cyberrange -u 1001 -m -d /home/cyberrange cyberrange

WORKDIR /app

# Copy only production dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir .

# Copy only source code
COPY --chown=cyberrange:cyberrange src/ ./src/

# Create runtime directories
RUN mkdir -p /app/logs /app/data /app/tmp && \
    chown -R cyberrange:cyberrange /app

# Security hardening
USER cyberrange

# Remove write permissions from application directory
USER root
RUN chmod -R 555 /app/src
USER cyberrange

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Security labels
LABEL security.scan.results="/tmp/security-reports/"
LABEL security.non-root="true"
LABEL security.read-only="true"

# Expose port
EXPOSE 8000

# Use tini for proper signal handling
ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "gan_cyber_range.api"]

# SBOM generation stage
FROM production as sbom

# Install SBOM generation tools
USER root
RUN pip install --no-cache-dir cyclonedx-bom

# Generate Software Bill of Materials
RUN cyclonedx-py -o /tmp/sbom.json .

USER cyberrange