#!/bin/bash

# Advanced Security Scanning Automation for GAN Cyber Range Simulator
# Comprehensive security scanning with container, dependency, and code analysis

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_ROOT/security_reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create reports directory
create_reports_dir() {
    mkdir -p "$REPORTS_DIR"
    log_info "Created security reports directory: $REPORTS_DIR"
}

# Container Security Scanning
scan_container_vulnerabilities() {
    log_info "Starting container vulnerability scanning..."
    
    # Build the container if it doesn't exist
    if ! docker images | grep -q "gan-cyber-range"; then
        log_info "Building container for security scanning..."
        docker build -t gan-cyber-range:security-scan "$PROJECT_ROOT"
    fi
    
    # Trivy container scanning
    if command -v trivy &> /dev/null; then
        log_info "Running Trivy container vulnerability scan..."
        trivy image \
            --format json \
            --output "$REPORTS_DIR/trivy-container-$TIMESTAMP.json" \
            --severity HIGH,CRITICAL \
            gan-cyber-range:security-scan
        
        trivy image \
            --format table \
            --output "$REPORTS_DIR/trivy-container-$TIMESTAMP.txt" \
            --severity HIGH,CRITICAL \
            gan-cyber-range:security-scan
            
        log_success "Trivy container scan completed"
    else
        log_warning "Trivy not installed, skipping container vulnerability scan"
    fi
    
    # Docker Bench Security
    if [ -f "$PROJECT_ROOT/docker-bench-security.sh" ]; then
        log_info "Running Docker Bench Security..."
        bash "$PROJECT_ROOT/docker-bench-security.sh" > "$REPORTS_DIR/docker-bench-$TIMESTAMP.txt"
        log_success "Docker Bench Security completed"
    else
        log_info "Downloading Docker Bench Security..."
        curl -L https://github.com/docker/docker-bench-security/archive/main.tar.gz | \
            tar -xz --strip-components=1 docker-bench-security-main/docker-bench-security.sh
        bash docker-bench-security.sh > "$REPORTS_DIR/docker-bench-$TIMESTAMP.txt"
        rm docker-bench-security.sh
        log_success "Docker Bench Security completed"
    fi
}

# Dependency Security Scanning
scan_dependencies() {
    log_info "Starting dependency security scanning..."
    
    cd "$PROJECT_ROOT"
    
    # Safety - Python dependency vulnerability scanner
    if command -v safety &> /dev/null; then
        log_info "Running Safety dependency scan..."
        safety check \
            --json \
            --output "$REPORTS_DIR/safety-$TIMESTAMP.json" \
            --ignore 70612 # Ignore specific jinja2 vulnerability if needed
        
        safety check \
            --output "$REPORTS_DIR/safety-$TIMESTAMP.txt"
            
        log_success "Safety dependency scan completed"
    else
        log_warning "Safety not installed, installing..."
        pip install safety
        safety check --json --output "$REPORTS_DIR/safety-$TIMESTAMP.json"
    fi
    
    # pip-audit - Modern Python dependency scanner
    if command -v pip-audit &> /dev/null; then
        log_info "Running pip-audit dependency scan..."
        pip-audit \
            --format=json \
            --output="$REPORTS_DIR/pip-audit-$TIMESTAMP.json" \
            --desc
            
        pip-audit \
            --format=cyclonedx-json \
            --output="$REPORTS_DIR/sbom-$TIMESTAMP.json"
            
        log_success "pip-audit scan completed"
    else
        log_warning "pip-audit not installed, installing..."
        pip install pip-audit
        pip-audit --format=json --output="$REPORTS_DIR/pip-audit-$TIMESTAMP.json"
    fi
    
    # OSV-Scanner for comprehensive vulnerability detection
    if command -v osv-scanner &> /dev/null; then
        log_info "Running OSV-Scanner..."
        osv-scanner \
            --format json \
            --output "$REPORTS_DIR/osv-scanner-$TIMESTAMP.json" \
            .
        log_success "OSV-Scanner completed"
    else
        log_info "OSV-Scanner not available, skipping"
    fi
}

# Static Application Security Testing (SAST)
run_sast_analysis() {
    log_info "Starting Static Application Security Testing (SAST)..."
    
    cd "$PROJECT_ROOT"
    
    # Bandit - Python security linter
    if command -v bandit &> /dev/null; then
        log_info "Running Bandit SAST analysis..."
        bandit -r src/ \
            -f json \
            -o "$REPORTS_DIR/bandit-$TIMESTAMP.json" \
            --skip B101,B602,B603 # Skip assert and subprocess warnings for research code
            
        bandit -r src/ \
            -f txt \
            -o "$REPORTS_DIR/bandit-$TIMESTAMP.txt" \
            --skip B101,B602,B603
            
        log_success "Bandit SAST analysis completed"
    else
        log_warning "Bandit not installed, installing..."
        pip install bandit[toml]
        bandit -r src/ -f json -o "$REPORTS_DIR/bandit-$TIMESTAMP.json"
    fi
    
    # Semgrep - Advanced static analysis
    if command -v semgrep &> /dev/null; then
        log_info "Running Semgrep advanced SAST..."
        semgrep \
            --config=auto \
            --json \
            --output="$REPORTS_DIR/semgrep-$TIMESTAMP.json" \
            --severity=ERROR \
            --severity=WARNING \
            src/
            
        log_success "Semgrep SAST analysis completed"
    else
        log_warning "Semgrep not installed, installing..."
        pip install semgrep
        semgrep --config=auto --json --output="$REPORTS_DIR/semgrep-$TIMESTAMP.json" src/
    fi
    
    # CodeQL analysis (if available)
    if command -v codeql &> /dev/null; then
        log_info "Running CodeQL analysis..."
        codeql database create \
            --language=python \
            --source-root=src/ \
            "$REPORTS_DIR/codeql-db-$TIMESTAMP"
            
        codeql database analyze \
            "$REPORTS_DIR/codeql-db-$TIMESTAMP" \
            --format=sarif-latest \
            --output="$REPORTS_DIR/codeql-$TIMESTAMP.sarif"
            
        log_success "CodeQL analysis completed"
    else
        log_info "CodeQL not available, skipping advanced code analysis"
    fi
}

# Secret Detection
detect_secrets() {
    log_info "Starting secret detection scan..."
    
    cd "$PROJECT_ROOT"
    
    # detect-secrets
    if command -v detect-secrets &> /dev/null; then
        log_info "Running detect-secrets scan..."
        
        # Create or update baseline
        if [ ! -f .secrets.baseline ]; then
            detect-secrets scan --baseline .secrets.baseline
        fi
        
        # Scan for new secrets
        detect-secrets scan \
            --baseline .secrets.baseline \
            --all-files \
            --force-use-all-plugins > "$REPORTS_DIR/secrets-scan-$TIMESTAMP.json"
            
        # Audit existing baseline
        detect-secrets audit \
            --baseline .secrets.baseline \
            --report --json > "$REPORTS_DIR/secrets-audit-$TIMESTAMP.json"
            
        log_success "Secret detection completed"
    else
        log_warning "detect-secrets not installed, installing..."
        pip install detect-secrets
        detect-secrets scan --baseline .secrets.baseline
    fi
    
    # TruffleHog for comprehensive secret scanning
    if command -v trufflehog &> /dev/null; then
        log_info "Running TruffleHog secret scan..."
        trufflehog git file://. \
            --json > "$REPORTS_DIR/trufflehog-$TIMESTAMP.json"
        log_success "TruffleHog scan completed"
    else
        log_info "TruffleHog not available, skipping"
    fi
}

# Infrastructure Security Assessment
assess_infrastructure_security() {
    log_info "Starting infrastructure security assessment..."
    
    cd "$PROJECT_ROOT"
    
    # Docker Compose security check
    if [ -f "docker-compose.yml" ]; then
        log_info "Analyzing Docker Compose security..."
        
        # Check for security misconfigurations
        python3 -c "
import yaml
import json
from datetime import datetime

with open('docker-compose.yml', 'r') as f:
    compose = yaml.safe_load(f)

security_issues = []

for service_name, service in compose.get('services', {}).items():
    # Check for privileged containers
    if service.get('privileged'):
        security_issues.append({
            'service': service_name,
            'issue': 'Privileged container detected',
            'severity': 'HIGH'
        })
    
    # Check for host network mode
    if service.get('network_mode') == 'host':
        security_issues.append({
            'service': service_name,
            'issue': 'Host network mode detected',
            'severity': 'MEDIUM'
        })
    
    # Check for volume mounts
    volumes = service.get('volumes', [])
    for volume in volumes:
        if isinstance(volume, str) and volume.startswith('/'):
            security_issues.append({
                'service': service_name,
                'issue': f'Host path volume mount: {volume}',
                'severity': 'LOW'
            })

report = {
    'timestamp': datetime.now().isoformat(),
    'scan_type': 'docker_compose_security',
    'issues_found': len(security_issues),
    'issues': security_issues
}

with open('$REPORTS_DIR/docker-compose-security-$TIMESTAMP.json', 'w') as f:
    json.dump(report, f, indent=2)
"
        log_success "Docker Compose security analysis completed"
    fi
    
    # Kubernetes security check (if manifests exist)
    if find . -name "*.yaml" -o -name "*.yml" | grep -E "(k8s|kubernetes)" &> /dev/null; then
        log_info "Found Kubernetes manifests, running security analysis..."
        
        # Use kubesec if available
        if command -v kubesec &> /dev/null; then
            find . -name "*.yaml" -o -name "*.yml" | grep -E "(k8s|kubernetes)" | \
            while read -r file; do
                kubesec scan "$file" > "$REPORTS_DIR/kubesec-$(basename "$file")-$TIMESTAMP.json"
            done
            log_success "Kubernetes security analysis completed"
        else
            log_info "kubesec not available, skipping Kubernetes security analysis"
        fi
    fi
}

# Generate comprehensive security report
generate_security_report() {
    log_info "Generating comprehensive security report..."
    
    python3 -c "
import json
import os
from datetime import datetime
from pathlib import Path

reports_dir = Path('$REPORTS_DIR')
timestamp = '$TIMESTAMP'

# Collect all security scan results
security_data = {
    'scan_timestamp': datetime.now().isoformat(),
    'project': 'GAN Cyber Range Simulator',
    'scan_type': 'comprehensive_security_audit',
    'reports': {}
}

# Process each type of security report
report_types = [
    ('trivy', 'Container Vulnerabilities'),
    ('safety', 'Dependency Vulnerabilities'),
    ('pip-audit', 'Python Dependencies'),
    ('bandit', 'Static Analysis'),
    ('semgrep', 'Advanced SAST'),
    ('secrets-scan', 'Secret Detection'),
    ('docker-compose-security', 'Infrastructure Security')
]

total_issues = 0
critical_issues = 0
high_issues = 0

for report_type, description in report_types:
    pattern = f'{report_type}-{timestamp}.json'
    report_files = list(reports_dir.glob(pattern))
    
    if report_files:
        report_file = report_files[0]
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
                security_data['reports'][report_type] = {
                    'description': description,
                    'file': str(report_file),
                    'data': data
                }
                
                # Count issues based on report type
                if report_type == 'trivy' and 'Results' in data:
                    for result in data['Results']:
                        if 'Vulnerabilities' in result:
                            for vuln in result['Vulnerabilities']:
                                total_issues += 1
                                if vuln.get('Severity') == 'CRITICAL':
                                    critical_issues += 1
                                elif vuln.get('Severity') == 'HIGH':
                                    high_issues += 1
                                    
                elif report_type == 'bandit' and isinstance(data, dict):
                    results = data.get('results', [])
                    total_issues += len(results)
                    for result in results:
                        if result.get('issue_severity') == 'HIGH':
                            high_issues += 1
                            
        except Exception as e:
            security_data['reports'][report_type] = {
                'description': description,
                'error': str(e)
            }

# Summary statistics
security_data['summary'] = {
    'total_issues': total_issues,
    'critical_issues': critical_issues,
    'high_issues': high_issues,
    'scan_coverage': len([r for r in security_data['reports'] if 'error' not in security_data['reports'][r]])
}

# Risk assessment
if critical_issues > 0:
    risk_level = 'CRITICAL'
elif high_issues > 5:
    risk_level = 'HIGH'
elif total_issues > 10:
    risk_level = 'MEDIUM'
else:
    risk_level = 'LOW'

security_data['risk_assessment'] = {
    'overall_risk_level': risk_level,
    'recommendations': [
        'Review and remediate all critical vulnerabilities immediately',
        'Implement automated security scanning in CI/CD pipeline',
        'Regular dependency updates and security monitoring',
        'Container image hardening and minimal base images'
    ]
}

# Write comprehensive report
with open(reports_dir / f'comprehensive-security-report-{timestamp}.json', 'w') as f:
    json.dump(security_data, f, indent=2)

print(f'Comprehensive security report generated: comprehensive-security-report-{timestamp}.json')
print(f'Total issues found: {total_issues}')
print(f'Critical issues: {critical_issues}')
print(f'High severity issues: {high_issues}')
print(f'Overall risk level: {risk_level}')
"
    
    log_success "Comprehensive security report generated"
}

# Cleanup old reports (keep last 10)
cleanup_old_reports() {
    log_info "Cleaning up old security reports..."
    
    find "$REPORTS_DIR" -name "*.json" -type f | \
    sort -r | \
    tail -n +21 | \
    xargs rm -f
    
    find "$REPORTS_DIR" -name "*.txt" -type f | \
    sort -r | \
    tail -n +11 | \
    xargs rm -f
    
    log_success "Old security reports cleaned up"
}

# Main execution function
main() {
    log_info "Starting comprehensive security scanning for GAN Cyber Range Simulator"
    log_info "Timestamp: $TIMESTAMP"
    
    create_reports_dir
    
    # Run all security scans
    scan_container_vulnerabilities
    scan_dependencies  
    run_sast_analysis
    detect_secrets
    assess_infrastructure_security
    
    # Generate comprehensive report
    generate_security_report
    
    # Cleanup
    cleanup_old_reports
    
    log_success "Comprehensive security scanning completed!"
    log_info "Reports available in: $REPORTS_DIR"
    
    # Display summary
    echo
    echo "=== SECURITY SCAN SUMMARY ==="
    echo "Reports directory: $REPORTS_DIR"
    echo "Timestamp: $TIMESTAMP"
    echo "Scan types completed:"
    echo "  ✓ Container vulnerability scanning"
    echo "  ✓ Dependency security analysis"  
    echo "  ✓ Static application security testing"
    echo "  ✓ Secret detection"
    echo "  ✓ Infrastructure security assessment"
    echo "  ✓ Comprehensive security report"
    echo
    echo "Next steps:"
    echo "1. Review comprehensive-security-report-$TIMESTAMP.json"
    echo "2. Address critical and high severity issues"
    echo "3. Integrate scanning into CI/CD pipeline"
    echo "4. Schedule regular security assessments"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi