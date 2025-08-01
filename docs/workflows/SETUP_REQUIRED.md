# Manual GitHub Workflows Setup Required

Due to GitHub App permission limitations, the workflow files in this repository could not be automatically created in the `.github/workflows/` directory. This document contains the manual setup steps required to implement the complete CI/CD pipeline.

## 🚨 Important Notice

The checkpointed SDLC implementation has created comprehensive workflow templates, but **manual action is required** to activate them. Repository maintainers must manually copy the workflow files and configure the necessary secrets and permissions.

## Required Manual Actions

### 1. Copy Workflow Files

Copy all workflow template files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Create the workflows directory
mkdir -p .github/workflows

# Copy all workflow templates
cp docs/workflows/examples/*.yml .github/workflows/

# Commit the workflow files
git add .github/workflows/
git commit -m "ci: add comprehensive CI/CD workflows"
git push
```

### 2. Required GitHub Secrets

Configure the following secrets in your GitHub repository settings:

#### **Core Application Secrets**
```bash
# AI/ML API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here

# Database and Infrastructure
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port/db
SECRET_KEY=your_32_character_secret_key_here
ENCRYPTION_KEY=your_32_character_encryption_key
JWT_SECRET=your_jwt_secret_key_here
```

#### **Deployment Secrets**
```bash
# Kubernetes Configuration (base64 encoded kubeconfig files)
STAGING_KUBECONFIG=base64_encoded_staging_kubeconfig
PRODUCTION_KUBECONFIG=base64_encoded_production_kubeconfig

# Container Registry
GHCR_TOKEN=your_github_container_registry_token

# Cloud Provider (if using AWS/Azure/GCP)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AZURE_CREDENTIALS=your_azure_service_principal_json
GCP_SERVICE_ACCOUNT_KEY=your_gcp_service_account_json
```

#### **Security and Monitoring**
```bash
# Security Tools
SNYK_TOKEN=your_snyk_api_token
CODACY_PROJECT_TOKEN=your_codacy_token
SONARCLOUD_TOKEN=your_sonarcloud_token

# Monitoring and Notifications
SLACK_WEBHOOK=your_slack_webhook_url
SECURITY_SLACK_WEBHOOK=your_security_slack_webhook
PAGERDUTY_INTEGRATION_KEY=your_pagerduty_key

# Dependency Management
DEPENDABOT_TOKEN=your_dependabot_token_with_write_permissions
```

### 3. Repository Settings Configuration

Configure the following repository settings:

#### **Branch Protection Rules**
```yaml
# For main branch
Branch name pattern: main
Required status checks:
  - CI / Code Quality & Security
  - CI / Test Suite
  - CI / Build & Scan Container
  - CI / Kubernetes Validation
  - CI / Documentation Build
Require pull request reviews: 2
Require review from code owners: true
Dismiss stale reviews: true
Require status checks to pass: true
Require branches to be up to date: true
Restrict pushes: true
Allow force pushes: false
Allow deletions: false
```

#### **Repository Permissions**
- Enable **Actions** with read and write permissions
- Enable **Packages** for container registry
- Enable **Security alerts** and **Dependabot**
- Configure **Code scanning** alerts
- Enable **Secret scanning**

### 4. GitHub Environments Setup

Create the following environments in repository settings:

#### **Staging Environment**
```yaml
Environment name: staging
Required reviewers: []  # No approval required for staging
Environment secrets:
  - STAGING_KUBECONFIG
  - STAGING_DATABASE_URL
  - STAGING_REDIS_URL
Deployment branches: main, develop, staging/*
```

#### **Production Environment**
```yaml
Environment name: production
Required reviewers: 
  - security-team
  - lead-developers
  - devops-team
Wait timer: 5 minutes
Environment secrets:
  - PRODUCTION_KUBECONFIG
  - PRODUCTION_DATABASE_URL
  - PRODUCTION_REDIS_URL
Deployment branches: main, v*
```

#### **Rollback Approval Environment**
```yaml
Environment name: rollback-approval
Required reviewers:
  - platform-team
  - incident-commander
Wait timer: 0 minutes
Purpose: Emergency rollback approvals
```

### 5. Required Repository Files

Create these additional files that the workflows expect:

#### **Container Structure Test Configuration**
```yaml
# container-structure-test.yaml
schemaVersion: '2.0.0'
commandTests:
  - name: 'Python version check'
    command: 'python'
    args: ['--version']
    expectedOutput: ['Python 3.1*']
fileExistenceTests:
  - name: 'Application files'
    path: '/app'
    shouldExist: true
    isDirectory: true
  - name: 'Requirements file'
    path: '/app/requirements.txt'
    shouldExist: true
fileContentTests:
  - name: 'Non-root user'
    path: '/etc/passwd'
    expectedContents: ['app:.*']
metadataTest:
  exposedPorts: ['8000', '8080']
  cmd: ['/app/start.sh']
```

#### **Security Policy Configuration**
```yaml
# security-gates.yaml
security_thresholds:
  overall_score: 85
  static_analysis: 80
  dependencies: 80
  container: 80
  infrastructure: 80
  secrets: 95
  compliance: 90

critical_vulnerabilities:
  max_allowed: 0
  
high_vulnerabilities:
  max_allowed: 5

medium_vulnerabilities:
  max_allowed: 20

fail_on_policy_violations: true
```

#### **Performance Baseline**
```json
// performance-baseline.json
{
  "agent_response_time": {
    "p50": 1.2,
    "p95": 3.4,
    "p99": 8.1,
    "max": 15.0
  },
  "simulation_throughput": {
    "scenarios_per_hour": 12,
    "concurrent_agents": 8
  },
  "resource_limits": {
    "memory_usage_mb": 2048,
    "cpu_usage_percent": 80
  }
}
```

### 6. Required Scripts

The workflows reference several scripts that need to be created:

#### **Core Scripts to Create**
```bash
# Security and validation scripts
scripts/check-secrets.py
scripts/validate_security_policies.py
scripts/check-insecure-deps.py
scripts/check-performance-regression.py

# Deployment scripts
scripts/smoke-tests.py
scripts/production-health-check.py
scripts/load-test.py
scripts/configure-alerts.py

# Dependency management scripts
scripts/filter-security-updates.py
scripts/filter-minor-updates.py
scripts/generate-dependency-report.py
scripts/check-base-image-updates.py
scripts/update-dockerfile.py

# Reporting scripts
scripts/generate-security-report.py
scripts/calculate-security-score.py
scripts/check-security-gates.py
scripts/generate-deployment-report.py
scripts/extended-monitoring.py
```

### 7. Monitoring and Alerting Setup

Configure external monitoring and alerting:

#### **Prometheus Configuration**
```yaml
# config/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  external_labels:
    cluster: 'gan-cyber-range'
    environment: 'production'

scrape_configs:
  - job_name: 'gan-cyber-range'
    static_configs:
      - targets: ['gan-cyber-range:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
```

#### **Grafana Dashboards**
```yaml
# config/grafana/provisioning/dashboards/gan-cyber-range.yaml
apiVersion: 1
providers:
  - name: 'gan-cyber-range'
    orgId: 1
    folder: 'GAN Cyber Range'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
```

### 8. RBAC and Service Accounts

Create required Kubernetes RBAC configurations:

#### **Service Account for CI/CD**
```yaml
# deployments/k8s/cicd-rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: gan-cyber-range-cicd
  namespace: gan-cyber-range-production
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: gan-cyber-range-cicd
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "create", "update", "patch", "delete"]
- apiGroups: ["networking.k8s.io"]
  resources: ["networkpolicies", "ingresses"]
  verbs: ["get", "list", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: gan-cyber-range-cicd
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: gan-cyber-range-cicd
subjects:
- kind: ServiceAccount
  name: gan-cyber-range-cicd
  namespace: gan-cyber-range-production
```

## 🔍 Verification Steps

After completing the manual setup:

### 1. Test Workflow Triggers
```bash
# Create a test PR to trigger CI
git checkout -b test-workflows
echo "# Test workflow" >> test-file.md
git add test-file.md
git commit -m "test: trigger workflow validation"
git push -u origin test-workflows
# Create PR through GitHub UI
```

### 2. Verify Secret Configuration
```bash
# Check if secrets are properly configured
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/actions/secrets
```

### 3. Validate Security Gates
```bash
# Run security checks locally first
python scripts/check-security-gates.py --config security-gates.yaml
```

### 4. Test Deployment Pipeline
```bash
# Create a deployment test
git checkout main
git tag v0.1.0-test
git push origin v0.1.0-test
# Monitor deployment workflows
```

## 📞 Support and Troubleshooting

### Common Issues

1. **Workflow Permission Errors**
   - Ensure GitHub App has necessary permissions
   - Check repository settings allow Actions
   - Verify secret access permissions

2. **Kubernetes Deployment Failures**
   - Validate kubeconfig files are correct
   - Test cluster connectivity
   - Check RBAC permissions

3. **Container Registry Issues**
   - Verify GHCR_TOKEN has write permissions
   - Check package visibility settings
   - Ensure proper registry authentication

4. **Security Scan Failures**
   - Update security tool configurations
   - Check API token validity
   - Review security thresholds

### Getting Help

- **Internal Documentation**: Check `docs/workflows/` for detailed guides
- **GitHub Issues**: Create issue with `workflow` and `support` labels
- **Security Issues**: Use private security contact for sensitive matters
- **Community Support**: Join Discord channel for community help

## ✅ Setup Checklist

- [ ] Copy workflow files to `.github/workflows/`
- [ ] Configure all required GitHub secrets
- [ ] Set up branch protection rules  
- [ ] Create staging and production environments
- [ ] Configure repository permissions
- [ ] Create required configuration files
- [ ] Implement required scripts
- [ ] Set up monitoring and alerting
- [ ] Configure Kubernetes RBAC
- [ ] Test workflow triggers
- [ ] Verify security gates
- [ ] Validate deployment pipeline
- [ ] Document any customizations

---

**Once all steps are completed**, the repository will have a fully functional, enterprise-grade CI/CD pipeline with comprehensive security scanning, automated testing, and deployment automation.

**Estimated Setup Time**: 4-6 hours for experienced DevOps engineer
**Required Skills**: GitHub Actions, Kubernetes, Security tooling, CI/CD pipelines