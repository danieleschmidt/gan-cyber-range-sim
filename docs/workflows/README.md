# GitHub Actions Workflows

This directory contains documentation and templates for GitHub Actions workflows. Due to security considerations, actual workflow files should be created manually by repository maintainers.

## Required Workflows

### 1. Continuous Integration (CI)

**File**: `.github/workflows/ci.yml`

```yaml
name: Continuous Integration

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
        python-version: [3.10, 3.11, 3.12]
    
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
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest tests/ --cov=gan_cyber_range --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### 2. Security Scanning

**File**: `.github/workflows/security.yml`

```yaml
name: Security Scanning

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM

jobs:
  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety semgrep
    
    - name: Run Bandit security scan
      run: bandit -r src/ -f json -o bandit-report.json
    
    - name: Run Safety check
      run: safety check --json --output safety-report.json
    
    - name: Run Semgrep
      run: semgrep --config=auto src/ --json --output=semgrep-report.json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
          semgrep-report.json
```

### 3. Code Quality

**File**: `.github/workflows/quality.yml`

```yaml
name: Code Quality

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  quality:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install black isort flake8 mypy
    
    - name: Check code formatting with Black
      run: black --check src/ tests/
    
    - name: Check import sorting with isort
      run: isort --check-only src/ tests/
    
    - name: Lint with flake8
      run: flake8 src/ tests/
    
    - name: Type check with mypy
      run: mypy src/
```

### 4. Container Security

**File**: `.github/workflows/container-scan.yml`

```yaml
name: Container Security Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  container-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: docker build -t gan-cyber-range:${{ github.sha }} .
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'gan-cyber-range:${{ github.sha }}'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

### 5. Documentation

**File**: `.github/workflows/docs.yml`

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install mkdocs mkdocs-material
    
    - name: Build documentation
      run: mkdocs build
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./site
```

## Deployment Workflows

### Production Deployment

**File**: `.github/workflows/deploy-prod.yml`

```yaml
name: Production Deployment

on:
  release:
    types: [published]

jobs:
  deploy:
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
    
    - name: Deploy to EKS
      run: |
        aws eks update-kubeconfig --name production-cluster
        kubectl apply -f deployments/k8s/production/
```

## Security Considerations

### Secrets Management
- Use GitHub Secrets for API keys and credentials
- Never commit secrets to the repository
- Rotate secrets regularly
- Use least privilege access

### Environment Protection
- Enable required reviews for production deployments
- Use environment-specific secrets
- Implement deployment gates and approvals

### Vulnerability Management
- Schedule regular security scans
- Set up automated alerts for high-severity vulnerabilities
- Establish SLAs for vulnerability remediation

## Workflow Best Practices

1. **Branch Protection**: Enable branch protection rules for main branch
2. **Required Checks**: Make CI/CD checks required before merging
3. **Signed Commits**: Require signed commits for security
4. **Dependency Updates**: Use Dependabot for automated dependency updates
5. **License Scanning**: Include license compliance checks

## Manual Setup Required

After creating the workflow files, ensure:

1. **Secrets Configuration**: Add required secrets in repository settings
2. **Environment Setup**: Configure production/staging environments
3. **Branch Rules**: Set up branch protection rules
4. **Notifications**: Configure Slack/email notifications for failures
5. **Monitoring**: Set up dashboards for workflow metrics

## Troubleshooting

### Common Issues
- **Permission Errors**: Check repository and organization permissions
- **Secret Access**: Verify secrets are correctly configured
- **Resource Limits**: Monitor GitHub Actions usage and limits
- **Dependency Conflicts**: Pin dependency versions for consistency

### Debug Steps
1. Check workflow logs for specific error messages
2. Verify environment variables and secrets
3. Test workflows in fork repository first
4. Use workflow debugging features

Remember: These workflows enhance security and code quality while maintaining the defensive research focus of the project.