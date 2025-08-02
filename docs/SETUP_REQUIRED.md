# Manual Setup Requirements

This document outlines manual configuration steps required to complete the SDLC implementation due to GitHub App permission limitations.

## GitHub Actions Workflows

The repository includes comprehensive workflow documentation and templates in `docs/workflows/examples/`. These need to be manually created in `.github/workflows/` directory.

### Required Workflows

1. **CI Pipeline** (`ci.yml`)
   - Copy from: `docs/workflows/examples/ci.yml`
   - Purpose: Pull request validation, testing, security scanning
   - Required secrets: None (uses GitHub tokens)

2. **CD Pipeline** (`cd.yml`)
   - Copy from: `docs/workflows/examples/cd.yml`
   - Purpose: Automated deployment to staging/production
   - Required secrets: `DEPLOY_TOKEN`, `STAGING_URL`, `PROD_URL`

3. **Security Scanning** (`security-scan.yml`)
   - Copy from: `docs/workflows/examples/security-scan.yml`
   - Purpose: Comprehensive security vulnerability scanning
   - Required secrets: `SNYK_TOKEN` (optional)

4. **Dependency Updates** (`dependency-update.yml`)
   - Copy from: `docs/workflows/examples/dependency-update.yml`
   - Purpose: Automated dependency management
   - Required secrets: None

### Setup Instructions

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/

# Commit workflows
git add .github/workflows/
git commit -m "feat: add GitHub Actions workflows"
```

## Repository Settings

### Branch Protection Rules

Configure branch protection for `main` branch:

1. Navigate to: Settings → Branches → Add rule
2. Branch name pattern: `main`
3. Configure:
   - [x] Require a pull request before merging
   - [x] Require approvals (minimum 1)
   - [x] Dismiss stale PR approvals when new commits are pushed
   - [x] Require review from code owners
   - [x] Require status checks to pass before merging
   - [x] Require branches to be up to date before merging
   - [x] Require conversation resolution before merging
   - [x] Include administrators

### Required Status Checks

Add these status checks (once workflows are active):
- `test` (from CI workflow)
- `lint` (from CI workflow)
- `security-scan` (from security workflow)
- `type-check` (from CI workflow)

### Repository Security Settings

1. **Security & Analysis**:
   - [x] Dependency graph
   - [x] Dependabot alerts
   - [x] Dependabot security updates
   - [x] Code scanning alerts
   - [x] Secret scanning alerts

2. **Secrets Configuration**:
   ```
   # Repository secrets (if needed)
   DEPLOY_TOKEN=<deployment_token>
   SNYK_TOKEN=<snyk_api_token>
   SLACK_WEBHOOK=<slack_webhook_url>
   ```

## Environment Setup

### Development Environment

1. **Install Pre-commit Hooks**:
   ```bash
   pre-commit install
   pre-commit install --hook-type commit-msg
   ```

2. **Setup Development Dependencies**:
   ```bash
   ./scripts/setup-dev.sh
   ```

3. **Verify Installation**:
   ```bash
   make test
   make lint
   make security-check
   ```

### Container Environment

1. **Build and Test Container**:
   ```bash
   docker-compose up --build
   ```

2. **Run Security Scan**:
   ```bash
   docker run --rm -v $(pwd):/app -w /app securecodewarrior/docker-slim build --include-shell gan-cyber-range:latest
   ```

## External Integrations

### Security Tools

1. **Snyk Integration**:
   - Sign up at: https://snyk.io
   - Generate API token
   - Add to repository secrets as `SNYK_TOKEN`

2. **SLSA Verification**:
   - Install SLSA verifier: https://github.com/slsa-framework/slsa-verifier
   - Configure build attestation generation

### Monitoring & Observability

1. **Prometheus Setup**:
   ```yaml
   # Add to monitoring stack
   global:
     scrape_interval: 15s
   scrape_configs:
     - job_name: 'gan-cyber-range'
       static_configs:
         - targets: ['localhost:8080']
   ```

2. **Grafana Dashboards**:
   - Import dashboards from `monitoring/grafana-dashboards.json`
   - Configure data sources for Prometheus and Elasticsearch

## Compliance Configuration

### NIST Cybersecurity Framework

1. **Policy Implementation**:
   - Review `docs/COMPLIANCE.md`
   - Implement required controls
   - Document control implementation

2. **Audit Trail**:
   ```bash
   # Enable audit logging
   git config --global log.showSignature true
   git config --global user.signingkey <GPG_KEY_ID>
   ```

### Data Protection

1. **Secrets Management**:
   - Implement HashiCorp Vault or similar
   - Configure secret rotation policies
   - Update deployment scripts to use secure secret management

2. **Data Classification**:
   - Implement data labeling system
   - Configure data retention policies
   - Setup data backup and recovery procedures

## Testing & Validation

### Security Testing

1. **Penetration Testing**:
   ```bash
   # Run security tests
   python -m pytest tests/security/ -v
   ```

2. **Vulnerability Assessment**:
   ```bash
   # Full security scan
   ./scripts/security-scan-automation.sh
   ```

### Performance Testing

1. **Load Testing**:
   ```bash
   # Run performance benchmarks
   python scripts/performance-benchmark.py
   ```

2. **Resource Monitoring**:
   ```bash
   # Monitor resource usage
   docker stats
   kubectl top pods
   ```

## Maintenance Tasks

### Daily Tasks

- [ ] Review security alerts
- [ ] Check build status
- [ ] Monitor system metrics
- [ ] Review dependency updates

### Weekly Tasks

- [ ] Run comprehensive security scan
- [ ] Review and update documentation
- [ ] Analyze performance metrics
- [ ] Update threat model if needed

### Monthly Tasks

- [ ] Conduct security audit
- [ ] Review and update compliance documentation
- [ ] Analyze SDLC metrics and trends
- [ ] Plan infrastructure improvements

## Troubleshooting

### Common Issues

1. **Workflow Failures**:
   - Check GitHub Actions logs
   - Verify required secrets are configured
   - Ensure branch protection rules are properly set

2. **Security Scan Failures**:
   - Update security scanning tools
   - Review and address flagged vulnerabilities
   - Check for new CVE databases

3. **Build Issues**:
   - Verify Docker environment
   - Check dependency conflicts
   - Review container resource limits

### Support Resources

- **Documentation**: `docs/` directory
- **Issues**: GitHub Issues with appropriate templates
- **Security**: Email security@gan-cyber-range.org
- **Community**: GitHub Discussions

## Validation Checklist

### Phase 1: Basic Setup (Week 1)
- [ ] GitHub workflows created and functioning
- [ ] Branch protection rules configured
- [ ] Pre-commit hooks installed and working
- [ ] Container builds successfully
- [ ] Basic security scans passing

### Phase 2: Integration (Week 2-3)
- [ ] CI/CD pipeline fully operational
- [ ] Security tools integrated
- [ ] Monitoring dashboards configured
- [ ] Compliance controls implemented
- [ ] Documentation updated

### Phase 3: Validation (Week 4)
- [ ] End-to-end testing completed
- [ ] Security audit passed
- [ ] Performance benchmarks met
- [ ] Team training completed
- [ ] Production readiness confirmed

---

**Note**: This setup process should be completed by repository maintainers with appropriate permissions. Contact the development team if you need assistance with any of these steps.