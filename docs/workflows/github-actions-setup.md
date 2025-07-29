# GitHub Actions Setup Guide

This document provides templates and guidance for implementing GitHub Actions workflows for the GAN Cyber Range Simulator.

## Overview

The following workflows should be implemented to achieve full CI/CD automation:

1. **Continuous Integration** - Testing, linting, security scanning
2. **Security Scanning** - SAST, dependency scanning, container scanning  
3. **Release Automation** - Automated releases and changelog generation
4. **Container Build** - Docker image building and publishing
5. **Documentation** - Automated documentation generation and deployment

## Required Secrets

Configure these secrets in GitHub repository settings:

```bash
# Required for basic CI/CD
CODECOV_TOKEN          # Code coverage reporting
SONAR_TOKEN           # SonarCloud integration (optional)

# Required for container publishing
DOCKER_USERNAME       # Docker Hub username
DOCKER_PASSWORD       # Docker Hub password or token
GHCR_TOKEN           # GitHub Container Registry token

# Required for security scanning
SNYK_TOKEN           # Snyk vulnerability scanning
SEMGREP_API_TOKEN    # Semgrep security scanning

# Required for automated releases
RELEASE_TOKEN        # GitHub token with release permissions
```

## Workflow Templates

### 1. Continuous Integration (`ci.yml`)

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
        
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .[dev,security]
        
    - name: Run pre-commit hooks
      uses: pre-commit/action@v3.0.0
      
    - name: Run tests with coverage
      run: |
        pytest --cov=gan_cyber_range --cov-report=xml --cov-report=term-missing
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        token: ${{ secrets.CODECOV_TOKEN }}
        fail_ci_if_error: true

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Bandit security scan
      run: |
        pip install bandit[toml]
        bandit -r src/ -f json -o bandit-report.json
        
    - name: Run Safety vulnerability check
      run: |
        pip install safety
        safety check --json --output safety-report.json
        
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
```

### 2. Security Scanning (`security.yml`)

```yaml
name: Security Scanning

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2 AM

jobs:
  sast:
    name: Static Application Security Testing
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
        
    - name: Run Semgrep
      uses: returntocorp/semgrep-action@v1
      with:
        config: >-
          p/security-audit
          p/secrets
          p/ci
          p/python
      env:
        SEMGREP_API_TOKEN: ${{ secrets.SEMGREP_API_TOKEN }}

  dependency-scan:
    name: Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Snyk to check for vulnerabilities
      uses: snyk/actions/python@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      with:
        args: --severity-threshold=high

  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: docker build -t gan-cyber-range:test .
      
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'gan-cyber-range:test'
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

### 3. Release Automation (`release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
        
    - name: Build package
      run: python -m build
      
    - name: Generate changelog
      id: changelog
      uses: mikepenz/release-changelog-builder-action@v4
      with:
        configuration: ".github/changelog-config.json"
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        body: ${{ steps.changelog.outputs.changelog }}
        files: |
          dist/*.whl
          dist/*.tar.gz
      env:
        GITHUB_TOKEN: ${{ secrets.RELEASE_TOKEN }}
        
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: |
        twine upload dist/*
```

### 4. Container Build (`docker.yml`)

```yaml
name: Docker Build & Publish

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
      
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Log in to Container Registry
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

## Implementation Steps

1. **Create `.github/workflows/` directory**
2. **Add the workflow files** using the templates above
3. **Configure repository secrets** as listed in the Required Secrets section
4. **Test workflows** with a test commit/PR
5. **Monitor workflow runs** and adjust as needed

## Security Considerations

- All workflows use pinned action versions for security
- Secrets are properly scoped and encrypted
- Container images are scanned for vulnerabilities
- SARIF results are uploaded to GitHub Security tab
- Dependencies are automatically updated via Dependabot

## Performance Optimization

- Matrix builds for multiple Python versions
- Docker layer caching enabled
- Dependency caching for faster builds
- Concurrent workflow execution with proper cancellation
- Conditional job execution to save compute resources

## Monitoring and Alerting

Configure GitHub repository settings for:
- Failed workflow notifications
- Security alert notifications  
- Dependency vulnerability alerts
- Code scanning alerts

## Troubleshooting

Common issues and solutions:

1. **Build failures**: Check Python version compatibility
2. **Security scan failures**: Review and fix identified vulnerabilities
3. **Permission errors**: Verify GitHub token permissions
4. **Container build failures**: Check Dockerfile syntax and dependencies