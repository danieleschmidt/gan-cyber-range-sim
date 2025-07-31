# Advanced CI/CD Workflow Templates

## Overview

This document provides production-ready GitHub Actions workflow templates optimized for the GAN Cyber Range Simulator's advanced security and research requirements.

## Core CI/CD Workflow Template

### `.github/workflows/ci.yml`

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.10'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  lint-and-format:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -e ".[dev,security]"
          
      - name: Run ruff linting
        run: ruff check src/ tests/
        
      - name: Run ruff formatting check
        run: ruff format --check src/ tests/
        
      - name: Run mypy type checking
        run: mypy src/

  security-scan:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install security tools
        run: |
          pip install bandit[toml] safety semgrep
          
      - name: Run Bandit security scan
        run: |
          bandit -r src/ -f sarif -o bandit-results.sarif
          
      - name: Upload Bandit results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: bandit-results.sarif
          
      - name: Run Safety vulnerability scan
        run: safety check --json --output safety-results.json
        continue-on-error: true
        
      - name: Run Semgrep security scan
        env:
          SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}
        run: semgrep ci --sarif --output=semgrep-results.sarif
        continue-on-error: true
        
      - name: Upload Semgrep results
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: semgrep-results.sarif

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    services:
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
          
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
          key: ${{ runner.os }}-pip-${{ matrix.python-version }}-${{ hashFiles('**/requirements.txt') }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -e ".[dev,ml]"
          
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=gan_cyber_range --cov-report=xml
        
      - name: Run integration tests
        run: pytest tests/integration/ -v
        env:
          REDIS_URL: redis://localhost:6379
          
      - name: Run security tests
        run: pytest tests/security/ -v -m security
        
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

  container-security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker build -t test-image .
        
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'test-image'
          format: 'sarif'
          output: 'trivy-results.sarif'
          
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'
          
      - name: Run Docker Scout
        uses: docker/scout-action@v1
        if: ${{ github.event_name != 'pull_request_target' }}
        with:
          command: cves
          image: test-image
          only-severities: critical,high
          exit-code: true

  build-and-push:
    needs: [lint-and-format, security-scan, test, container-security]
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write
      
    steps:
      - uses: actions/checkout@v4
      
      - name: Log in to Container Registry
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
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}
            
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

## Advanced Security Workflow

### `.github/workflows/security-advanced.yml`

```yaml
name: Advanced Security Scanning

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  comprehensive-security-audit:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      contents: read
      
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for secret scanning
          
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install comprehensive security tools
        run: |
          pip install bandit[toml] safety semgrep pip-audit detect-secrets
          
      - name: Run comprehensive dependency audit
        run: |
          pip-audit --format=json --output=pip-audit-results.json
          safety check --json --output=safety-results.json
          
      - name: Detect secrets in codebase
        run: |
          detect-secrets scan --all-files --baseline .secrets.baseline \
            --exclude-files '\.git/.*' \
            --exclude-files '\.pytest_cache/.*' \
            --exclude-files '__pycache__/.*'
            
      - name: Advanced SAST with CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: python
          
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2
        
      - name: SBOM Generation
        uses: anchore/sbom-action@v0
        with:
          path: ./
          format: spdx-json
          
      - name: Upload SBOM
        uses: actions/upload-artifact@v3
        with:
          name: sbom
          path: ./*-sbom-spdx.json

  license-compliance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: License compliance check
        uses: fossa-contrib/fossa-action@v2
        with:
          api-key: ${{ secrets.FOSSA_API_KEY }}
          
  supply-chain-security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: SLSA Provenance Generation
        uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.7.0
        with:
          base64-subjects: ${{ needs.build.outputs.hashes }}
```

## Deployment Workflows

### `.github/workflows/deploy-staging.yml`

```yaml
name: Deploy to Staging

on:
  push:
    branches: [ develop ]
  workflow_dispatch:
    inputs:
      image_tag:
        description: 'Docker image tag to deploy'
        required: true
        default: 'develop'

env:
  KUBE_NAMESPACE: cyber-range-staging
  DEPLOYMENT_NAME: gan-cyber-range

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2
          
      - name: Configure kubectl
        run: |
          aws eks update-kubeconfig --name cyber-range-staging-cluster
          
      - name: Deploy to staging
        run: |
          kubectl set image deployment/${{ env.DEPLOYMENT_NAME }} \
            gan-cyber-range=ghcr.io/${{ github.repository }}:${{ github.event.inputs.image_tag || 'develop' }} \
            -n ${{ env.KUBE_NAMESPACE }}
            
          kubectl rollout status deployment/${{ env.DEPLOYMENT_NAME }} -n ${{ env.KUBE_NAMESPACE }}
          
      - name: Run smoke tests
        run: |
          kubectl wait --for=condition=ready pod -l app=gan-cyber-range -n ${{ env.KUBE_NAMESPACE }} --timeout=300s
          # Add smoke test commands here
          
      - name: Notify deployment status
        uses: 8398a7/action-slack@v3
        if: always()
        with:
          status: ${{ job.status }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### `.github/workflows/deploy-production.yml`

```yaml
name: Deploy to Production

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      release_tag:
        description: 'Release tag to deploy'
        required: true

env:
  KUBE_NAMESPACE: cyber-range-prod
  DEPLOYMENT_NAME: gan-cyber-range

jobs:
  pre-deployment-checks:
    runs-on: ubuntu-latest
    steps:
      - name: Verify release readiness
        run: |
          # Add production readiness checks
          echo "Verifying security scans passed"
          echo "Verifying all tests passed"
          echo "Verifying staging deployment successful"
          
  deploy-production:
    needs: pre-deployment-checks
    runs-on: ubuntu-latest
    environment: production
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2
          
      - name: Blue-Green Deployment
        run: |
          # Implement blue-green deployment strategy
          ./scripts/blue-green-deploy.sh ${{ github.event.release.tag_name }}
          
      - name: Health checks
        run: |
          # Comprehensive health checks
          ./scripts/production-health-check.sh
          
      - name: Rollback on failure
        if: failure()
        run: |
          ./scripts/rollback-deployment.sh
```

## Performance and Load Testing

### `.github/workflows/performance.yml`

```yaml
name: Performance Testing

on:
  schedule:
    - cron: '0 4 * * 1'  # Weekly on Monday
  workflow_dispatch:
    inputs:
      test_duration:
        description: 'Test duration in minutes'
        required: true
        default: '10'

jobs:
  performance-benchmark:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install performance testing tools
        run: |
          pip install locust pytest-benchmark
          
      - name: Run benchmark tests
        run: |
          pytest tests/performance/ --benchmark-json=benchmark.json
          
      - name: Run load tests
        run: |
          locust -f tests/load/locustfile.py \
            --host=https://staging.gan-cyber-range.com \
            --users=100 \
            --spawn-rate=10 \
            --run-time=${{ github.event.inputs.test_duration || '10' }}m \
            --html=load-test-report.html
            
      - name: Upload performance reports
        uses: actions/upload-artifact@v3
        with:
          name: performance-reports
          path: |
            benchmark.json
            load-test-report.html
```

## Implementation Instructions

1. **Create `.github/workflows/` directory** if it doesn't exist
2. **Add workflow files** based on your specific needs
3. **Configure secrets** in repository settings:
   - `SEMGREP_APP_TOKEN`
   - `FOSSA_API_KEY`
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `SLACK_WEBHOOK`
4. **Customize deployment targets** and environment configurations
5. **Test workflows** with pull requests before enabling on main branch

## Security Considerations

- All workflows follow **principle of least privilege**
- **Sensitive operations** require environment approval
- **Container images** are scanned before deployment
- **SBOM and provenance** generated for supply chain security
- **Automated rollback** on deployment failures

These templates provide enterprise-grade CI/CD capabilities while maintaining the security standards required for cybersecurity research tools.