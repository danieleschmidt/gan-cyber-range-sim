# 🔧 Manual Setup Requirements

## Overview

Due to GitHub App permission limitations, some configurations require manual setup by repository maintainers. This document outlines all manual steps needed to complete the SDLC implementation.

## GitHub Repository Settings

### 1. Repository Configuration

**Required Repository Settings:**
```bash
# Set repository description
curl -X PATCH \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/danieleschmidt/gan-cyber-range-sim \
  -d '{
    "description": "A generative adversarial cyber-range where attacker LLMs spin up exploits while defender LLMs patch in real time",
    "homepage": "https://gan-cyber-range.org",
    "topics": ["cybersecurity", "ai", "machine-learning", "kubernetes", "docker", "llm", "red-team", "blue-team", "cyber-range"],
    "has_issues": true,
    "has_projects": true,
    "has_wiki": false,
    "has_discussions": true,
    "allow_squash_merge": true,
    "allow_merge_commit": false,
    "allow_rebase_merge": true,
    "delete_branch_on_merge": true,
    "security_and_analysis": {
      "secret_scanning": {"status": "enabled"},
      "secret_scanning_push_protection": {"status": "enabled"}
    }
  }'
```

### 2. Branch Protection Rules

**Main Branch Protection:**
```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "ci/tests",
      "ci/security-scan",
      "ci/build"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "required_approving_review_count": 2
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
```

### 3. Required GitHub Actions Workflows

Create these workflow files in `.github/workflows/`:

#### CI Workflow (`.github/workflows/ci.yml`)
```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e ".[dev,test]"
    
    - name: Run tests
      run: |
        pytest tests/ --cov=gan_cyber_range --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.10
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety semgrep detect-secrets
    
    - name: Run security checks
      run: |
        bandit -r src/
        safety check
        semgrep --config=auto src/
        detect-secrets scan --baseline .secrets.baseline

  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Build Docker image
      run: |
        docker build -t gan-cyber-range:${{ github.sha }} .
        docker build -t gan-cyber-range:latest .
```

#### Security Scanning (`.github/workflows/security-scan.yml`)
```yaml
name: Security Scan

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run dependency scan
      uses: pypa/gh-action-pip-audit@v1.0.8
      with:
        inputs: requirements.txt

  container-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Build image
      run: docker build -t test-image .
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'test-image'
        format: 'sarif'
        output: 'trivy-results.sarif'
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  codeql:
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
    steps:
    - uses: actions/checkout@v4
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: python
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
```

#### Deployment (`.github/workflows/cd.yml`)
```yaml
name: CD

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: staging
    steps:
    - uses: actions/checkout@v4
    - name: Deploy to staging
      run: |
        echo "Deploy to staging environment"
        # Add staging deployment steps

  deploy-production:
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    environment: production
    needs: [test-staging]
    steps:
    - uses: actions/checkout@v4
    - name: Deploy to production
      run: |
        echo "Deploy to production environment"
        # Add production deployment steps
```

## Security Configuration

### 1. Secret Scanning Setup

Enable GitHub secret scanning:
1. Go to repository Settings → Security & analysis
2. Enable "Secret scanning"
3. Enable "Push protection"
4. Configure custom patterns if needed

### 2. Dependabot Configuration

The `.github/dependabot.yml` file is already configured. Ensure:
1. Dependabot is enabled in repository settings
2. Security updates are enabled
3. Review team assignments are correct

### 3. Code Scanning Setup

1. Enable CodeQL analysis in repository settings
2. Configure SARIF upload permissions
3. Set up custom security policies if needed

## Environment Configuration

### 1. Environment Secrets

Configure these secrets in repository settings:

**Required Secrets:**
- `OPENAI_API_KEY`: OpenAI API key for LLM agents
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude agents
- `DOCKER_REGISTRY_URL`: Container registry URL
- `DOCKER_REGISTRY_USERNAME`: Registry username
- `DOCKER_REGISTRY_PASSWORD`: Registry password
- `KUBERNETES_CONFIG`: Base64 encoded kubeconfig for deployment

**Optional Secrets:**
- `CODECOV_TOKEN`: For code coverage reporting
- `SLACK_WEBHOOK_URL`: For deployment notifications
- `DATADOG_API_KEY`: For monitoring integration

### 2. Environment Variables

Set these in your deployment environment:
```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://redis:6379
PROMETHEUS_URL=http://prometheus:9090
```

## Compliance Setup

### 1. Audit Logging

Configure audit logging for compliance:
1. Enable GitHub audit log streaming
2. Set up log retention policies
3. Configure compliance reporting

### 2. Access Control

1. Review and update team permissions
2. Configure SAML/SSO if required
3. Set up branch protection rules
4. Configure required reviews

## Monitoring Setup

### 1. Health Checks

Configure external monitoring:
- Set up uptime monitoring for public endpoints
- Configure SSL certificate monitoring
- Set up DNS monitoring

### 2. Alerting

Configure alert channels:
1. Set up Slack/Teams integration
2. Configure PagerDuty/OpsGenie
3. Set up email notifications
4. Configure escalation policies

## Deployment Environments

### 1. Staging Environment

Requirements:
- Kubernetes cluster access
- Container registry access
- Database setup
- Monitoring stack deployment

### 2. Production Environment

Additional requirements:
- High availability setup
- Backup configuration
- Disaster recovery procedures
- Security hardening

## Verification Checklist

After manual setup, verify:

- [ ] All GitHub Actions workflows are running successfully
- [ ] Security scanning is active and reporting
- [ ] Branch protection rules are enforced
- [ ] Dependabot is creating update PRs
- [ ] Secret scanning is blocking commits with secrets
- [ ] Code owners are being automatically requested for reviews
- [ ] Deployment pipelines are functional
- [ ] Monitoring and alerting are operational
- [ ] Backup procedures are tested
- [ ] Compliance controls are documented and verified

## Support

For assistance with manual setup:
- **Infrastructure**: infrastructure@gan-cyber-range.org
- **Security**: security@gan-cyber-range.org
- **Development**: dev@gan-cyber-range.org

## Next Steps

1. Complete manual GitHub configuration
2. Set up CI/CD workflows
3. Configure monitoring and alerting
4. Test deployment procedures
5. Validate security controls
6. Document any deviations or customizations