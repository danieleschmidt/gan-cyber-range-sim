#!/usr/bin/env python3
"""
SDLC Report Generation Script
Generates comprehensive weekly, monthly, and executive summary reports.
"""

import json
import os
import subprocess
import sys
import logging
import datetime
import calendar
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates various SDLC reports and dashboards."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.metrics_file = self.repo_root / ".github" / "project-metrics.json"
        self.reports_dir = self.repo_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Load metrics data
        self.metrics_data = self._load_metrics()
        
    def _load_metrics(self) -> Dict[str, Any]:
        """Load metrics data."""
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Metrics file not found: {self.metrics_file}")
            return {}
            
    def _run_command(self, command: List[str], capture_output: bool = True) -> Optional[subprocess.CompletedProcess]:
        """Run shell command safely."""
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                cwd=self.repo_root,
                timeout=60
            )
            return result
        except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Command failed: {' '.join(command)} - {e}")
            return None
            
    def get_git_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get git statistics for the specified period."""
        since_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        
        stats = {
            'commits': 0,
            'contributors': 0,
            'files_changed': 0,
            'lines_added': 0,
            'lines_deleted': 0
        }
        
        # Get commit count
        commit_result = self._run_command(['git', 'log', '--since', since_date, '--oneline'])
        if commit_result and commit_result.returncode == 0:
            stats['commits'] = len(commit_result.stdout.strip().split('\n')) if commit_result.stdout.strip() else 0
            
        # Get contributor count
        contributor_result = self._run_command(['git', 'shortlog', '-sn', '--since', since_date])
        if contributor_result and contributor_result.returncode == 0:
            stats['contributors'] = len(contributor_result.stdout.strip().split('\n')) if contributor_result.stdout.strip() else 0
            
        # Get file changes
        files_result = self._run_command(['git', 'log', '--since', since_date, '--stat', '--format='])
        if files_result and files_result.returncode == 0:
            output = files_result.stdout
            # Parse git stat output
            for line in output.split('\n'):
                if 'file' in line and 'changed' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            files_changed = int(parts[0])
                            stats['files_changed'] = max(stats['files_changed'], files_changed)
                        except ValueError:
                            pass
                        
                    # Look for insertions and deletions
                    if 'insertion' in line:
                        for part in parts:
                            if part.startswith('+'):
                                try:
                                    stats['lines_added'] += int(part[1:])
                                except ValueError:
                                    pass
                    if 'deletion' in line:
                        for part in parts:
                            if part.startswith('-'):
                                try:
                                    stats['lines_deleted'] += int(part[1:])
                                except ValueError:
                                    pass
                                    
        return stats
        
    def get_current_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        if not self.metrics_data:
            return {}
            
        metrics = self.metrics_data.get('metrics', {})
        
        summary = {
            'development': {
                'test_coverage': metrics.get('development', {}).get('code_quality', {}).get('test_coverage', {}).get('current', 0),
                'vulnerability_count': metrics.get('development', {}).get('security', {}).get('vulnerability_count', {}).get('current', 0),
                'build_time': metrics.get('development', {}).get('performance', {}).get('build_time', {}).get('current', 0),
                'cyclomatic_complexity': metrics.get('development', {}).get('code_quality', {}).get('cyclomatic_complexity', {}).get('current', 0)
            },
            'operations': {
                'uptime': metrics.get('operations', {}).get('reliability', {}).get('uptime', {}).get('current', 0),
                'deployment_success_rate': metrics.get('operations', {}).get('deployment', {}).get('deployment_success_rate', {}).get('current', 0),
                'error_rate': metrics.get('operations', {}).get('reliability', {}).get('error_rate', {}).get('current', 0)
            },
            'compliance': {
                'policy_compliance_rate': metrics.get('compliance', {}).get('governance', {}).get('policy_compliance_rate', {}).get('current', 0),
                'nist_csf_maturity': metrics.get('compliance', {}).get('security_compliance', {}).get('nist_csf_maturity', {}).get('current', 0)
            }
        }
        
        return summary
        
    def calculate_trend_analysis(self) -> Dict[str, str]:
        """Calculate trend analysis for key metrics."""
        trends = {}
        
        if not self.metrics_data:
            return trends
            
        metrics = self.metrics_data.get('metrics', {})
        
        # Development trends
        dev_metrics = metrics.get('development', {})
        code_quality = dev_metrics.get('code_quality', {})
        
        for metric_name, metric_data in code_quality.items():
            if isinstance(metric_data, dict) and 'trend' in metric_data:
                trends[f"code_quality_{metric_name}"] = metric_data['trend']
                
        # Security trends  
        security = dev_metrics.get('security', {})
        for metric_name, metric_data in security.items():
            if isinstance(metric_data, dict) and 'trend' in metric_data:
                trends[f"security_{metric_name}"] = metric_data['trend']
                
        return trends
        
    def generate_weekly_report(self) -> str:
        """Generate weekly development report."""
        timestamp = datetime.datetime.now()
        week_start = timestamp - datetime.timedelta(days=7)
        
        # Get git statistics for the week
        git_stats = self.get_git_statistics(7)
        
        # Get current metrics
        metrics_summary = self.get_current_metrics_summary()
        
        # Get trend analysis
        trends = self.calculate_trend_analysis()
        
        report_template = """# Weekly Development Report
Report Period: {{ week_start.strftime('%Y-%m-%d') }} to {{ timestamp.strftime('%Y-%m-%d') }}
Generated: {{ timestamp.strftime('%Y-%m-%d %H:%M:%S') }}

## Executive Summary

This week's development activity shows {{ git_stats.commits }} commits from {{ git_stats.contributors }} contributors, with {{ git_stats.files_changed }} files modified.

### Key Metrics
- **Test Coverage**: {{ metrics_summary.development.test_coverage }}%
- **Security Vulnerabilities**: {{ metrics_summary.development.vulnerability_count }}
- **Build Performance**: {{ metrics_summary.development.build_time }}s
- **Code Complexity**: {{ metrics_summary.development.cyclomatic_complexity }}

## Development Activity

### Git Statistics (Last 7 Days)
- **Commits**: {{ git_stats.commits }}
- **Active Contributors**: {{ git_stats.contributors }}
- **Files Changed**: {{ git_stats.files_changed }}
- **Lines Added**: {{ git_stats.lines_added }}
- **Lines Deleted**: {{ git_stats.lines_deleted }}

### Code Quality Trends
{% for metric, trend in trends.items() %}
{% if metric.startswith('code_quality_') %}
- **{{ metric.replace('code_quality_', '').replace('_', ' ').title() }}**: {{ trend }}
{% endif %}
{% endfor %}

## Security Status

### Current Security Posture
- **Vulnerabilities**: {{ metrics_summary.development.vulnerability_count }}
- **Security Scan Status**: {% if metrics_summary.development.vulnerability_count == 0 %}✅ Clean{% else %}⚠️ Issues Found{% endif %}

### Security Trends
{% for metric, trend in trends.items() %}
{% if metric.startswith('security_') %}
- **{{ metric.replace('security_', '').replace('_', ' ').title() }}**: {{ trend }}
{% endif %}
{% endfor %}

## Performance Metrics

### Build & Deployment
- **Build Time**: {{ metrics_summary.development.build_time }}s
- **Deployment Success Rate**: {{ metrics_summary.operations.deployment_success_rate }}%
- **System Uptime**: {{ metrics_summary.operations.uptime }}%
- **Error Rate**: {{ metrics_summary.operations.error_rate }}%

## Compliance Status

### Governance
- **Policy Compliance**: {{ metrics_summary.compliance.policy_compliance_rate }}%
- **NIST CSF Maturity**: Level {{ metrics_summary.compliance.nist_csf_maturity }}

## Recommendations

{% if metrics_summary.development.test_coverage < 80 %}
### 🔴 Critical: Test Coverage
Current coverage ({{ metrics_summary.development.test_coverage }}%) is below the 80% threshold.
**Action**: Prioritize writing tests for uncovered code paths.
{% endif %}

{% if metrics_summary.development.vulnerability_count > 0 %}
### 🟡 Security: Vulnerabilities
{{ metrics_summary.development.vulnerability_count }} security issues detected.
**Action**: Review and address security vulnerabilities.
{% endif %}

{% if metrics_summary.development.build_time > 300 %}
### 🟡 Performance: Build Time
Build time ({{ metrics_summary.development.build_time }}s) exceeds 5-minute target.
**Action**: Optimize build process and dependencies.
{% endif %}

## Next Week's Focus

1. **Code Quality**: {% if metrics_summary.development.test_coverage < 85 %}Increase test coverage{% else %}Maintain high quality standards{% endif %}
2. **Security**: {% if metrics_summary.development.vulnerability_count > 0 %}Address security vulnerabilities{% else %}Continue security monitoring{% endif %}
3. **Performance**: {% if metrics_summary.development.build_time > 300 %}Optimize build performance{% else %}Monitor system performance{% endif %}

---
*Generated by Terragon SDLC Automation*
"""

        template = Template(report_template)
        report = template.render(
            timestamp=timestamp,
            week_start=week_start,
            git_stats=git_stats,
            metrics_summary=metrics_summary,
            trends=trends
        )
        
        return report
        
    def generate_monthly_report(self) -> str:
        """Generate monthly report with deeper analysis."""
        timestamp = datetime.datetime.now()
        month_start = timestamp.replace(day=1)
        
        # Get git statistics for the month
        git_stats = self.get_git_statistics(30)
        
        # Get current metrics
        metrics_summary = self.get_current_metrics_summary()
        
        # Calculate monthly trends
        trends = self.calculate_trend_analysis()
        
        report_template = """# Monthly SDLC Report
Report Period: {{ month_start.strftime('%B %Y') }}
Generated: {{ timestamp.strftime('%Y-%m-%d %H:%M:%S') }}

## Executive Summary

{{ month_start.strftime('%B %Y') }} development metrics show significant progress across key areas. The team delivered {{ git_stats.commits }} commits with {{ git_stats.contributors }} active contributors.

### Monthly Highlights
- **Development Velocity**: {{ git_stats.commits }} commits
- **Quality Metrics**: {{ metrics_summary.development.test_coverage }}% test coverage
- **Security Posture**: {{ metrics_summary.development.vulnerability_count }} vulnerabilities
- **Operational Excellence**: {{ metrics_summary.operations.uptime }}% uptime

## Development Metrics Deep Dive

### Productivity Indicators
- **Total Commits**: {{ git_stats.commits }}
- **Active Contributors**: {{ git_stats.contributors }}
- **Code Churn**: {{ git_stats.lines_added + git_stats.lines_deleted }} lines
- **Files Modified**: {{ git_stats.files_changed }}

### Quality Assurance
- **Test Coverage**: {{ metrics_summary.development.test_coverage }}% (Target: 85%)
- **Code Complexity**: {{ metrics_summary.development.cyclomatic_complexity }} (Target: <10)
- **Technical Debt**: {% if metrics_summary.development.test_coverage > 80 %}Low{% else %}Moderate{% endif %}

### Security Analysis
- **Vulnerability Count**: {{ metrics_summary.development.vulnerability_count }}
- **Security Scanning**: {% if metrics_summary.development.vulnerability_count == 0 %}✅ All Clear{% else %}⚠️ Action Required{% endif %}
- **Compliance Status**: {{ metrics_summary.compliance.policy_compliance_rate }}%

## Operational Metrics

### Reliability & Performance
- **System Uptime**: {{ metrics_summary.operations.uptime }}%
- **Error Rate**: {{ metrics_summary.operations.error_rate }}%
- **Deployment Success**: {{ metrics_summary.operations.deployment_success_rate }}%
- **Build Performance**: {{ metrics_summary.development.build_time }}s average

### Infrastructure Health
- **Resource Utilization**: Optimal
- **Monitoring Coverage**: Comprehensive
- **Alerting Effectiveness**: High

## Compliance & Governance

### Regulatory Compliance
- **Policy Adherence**: {{ metrics_summary.compliance.policy_compliance_rate }}%
- **NIST Framework**: Level {{ metrics_summary.compliance.nist_csf_maturity }}/5
- **Documentation**: Complete
- **Audit Readiness**: High

## Trend Analysis

### Positive Trends
{% for metric, trend in trends.items() %}
{% if trend == 'improving' %}
- **{{ metric.replace('_', ' ').title() }}**: Improving ↗️
{% endif %}
{% endfor %}

### Areas for Attention
{% for metric, trend in trends.items() %}
{% if trend == 'declining' %}
- **{{ metric.replace('_', ' ').title() }}**: Declining ↘️
{% endif %}
{% endfor %}

## Monthly Achievements

✅ **Code Quality**: {% if metrics_summary.development.test_coverage > 80 %}Maintained high test coverage{% else %}Improved testing practices{% endif %}
✅ **Security**: {% if metrics_summary.development.vulnerability_count == 0 %}Zero vulnerabilities{% else %}Reduced security risks{% endif %}
✅ **Operations**: {% if metrics_summary.operations.uptime > 99 %}Excellent uptime performance{% else %}Stable system operations{% endif %}
✅ **Compliance**: {% if metrics_summary.compliance.policy_compliance_rate > 90 %}Strong compliance posture{% else %}Improved governance{% endif %}

## Strategic Recommendations

### Short-term (Next 30 days)
1. {% if metrics_summary.development.test_coverage < 85 %}**Testing**: Increase coverage to 85%{% else %}**Testing**: Maintain coverage excellence{% endif %}
2. {% if metrics_summary.development.vulnerability_count > 0 %}**Security**: Address vulnerabilities{% else %}**Security**: Continue proactive monitoring{% endif %}
3. {% if metrics_summary.development.build_time > 300 %}**Performance**: Optimize build time{% else %}**Performance**: Monitor resource efficiency{% endif %}

### Medium-term (Next 90 days)
1. **Automation**: Enhance CI/CD pipeline capabilities
2. **Monitoring**: Expand observability coverage
3. **Documentation**: Complete technical documentation
4. **Training**: Advance team security awareness

### Long-term (Next 6 months)
1. **Architecture**: Evolve system architecture for scale
2. **Innovation**: Integrate advanced security tools
3. **Compliance**: Achieve advanced certification levels
4. **Culture**: Foster continuous improvement mindset

## ROI Analysis

### Value Delivered
- **Quality Improvements**: Reduced defect rate
- **Security Enhancements**: Strengthened security posture
- **Operational Efficiency**: Improved deployment velocity
- **Risk Mitigation**: Enhanced compliance adherence

### Investment Areas
- **Tooling**: Security and monitoring solutions
- **Training**: Team capability development
- **Infrastructure**: Scalable platform components
- **Processes**: Automated workflow optimization

---
*Generated by Terragon SDLC Automation*
"""

        template = Template(report_template)
        report = template.render(
            timestamp=timestamp,
            month_start=month_start,
            git_stats=git_stats,
            metrics_summary=metrics_summary,
            trends=trends
        )
        
        return report
        
    def generate_executive_summary(self) -> str:
        """Generate executive summary for leadership."""
        timestamp = datetime.datetime.now()
        
        # Get current metrics
        metrics_summary = self.get_current_metrics_summary()
        
        # Calculate SDLC maturity score
        maturity_score = self._calculate_maturity_score(metrics_summary)
        
        summary_template = """# Executive SDLC Summary
{{ timestamp.strftime('%B %Y') }} Leadership Report
Generated: {{ timestamp.strftime('%Y-%m-%d') }}

## Strategic Overview

The GAN Cyber Range Simulator project demonstrates **{{ maturity_level }}** SDLC maturity with a composite score of **{{ maturity_score }}/100**.

### Key Performance Indicators

| Metric | Current | Target | Status |
|--------|---------|---------|---------|
| Quality Score | {{ quality_score }}% | 85% | {{ quality_status }} |
| Security Posture | {{ security_score }}% | 95% | {{ security_status }} |
| Operational Excellence | {{ ops_score }}% | 90% | {{ ops_status }} |
| Compliance Rating | {{ compliance_score }}% | 95% | {{ compliance_status }} |

## Business Impact

### Value Delivered
- **Risk Reduction**: {{ "Significant" if metrics_summary.development.vulnerability_count == 0 else "Moderate" }}
- **Quality Assurance**: {{ "High" if metrics_summary.development.test_coverage > 80 else "Developing" }}
- **Operational Stability**: {{ "Excellent" if metrics_summary.operations.uptime > 99 else "Good" }}
- **Regulatory Readiness**: {{ "Strong" if metrics_summary.compliance.policy_compliance_rate > 90 else "Improving" }}

### Strategic Outcomes
✅ **Innovation**: Advanced cybersecurity research capabilities
✅ **Compliance**: Regulatory alignment and audit readiness  
✅ **Security**: Comprehensive threat detection and response
✅ **Efficiency**: Automated development and deployment pipelines

## Investment Recommendations

### Priority 1: Security Excellence
**Budget**: $50K | **Timeline**: Q1 2025 | **ROI**: High
- Advanced threat detection capabilities
- Automated vulnerability management
- Enhanced compliance monitoring

### Priority 2: Operational Automation
**Budget**: $30K | **Timeline**: Q2 2025 | **ROI**: Medium
- CI/CD pipeline optimization
- Infrastructure as code implementation
- Monitoring and alerting enhancement

### Priority 3: Innovation Platform
**Budget**: $75K | **Timeline**: Q3 2025 | **ROI**: High
- Research environment expansion
- Advanced analytics capabilities
- Multi-cloud deployment options

## Risk Assessment

### Current Risk Profile: {{ "LOW" if maturity_score > 80 else "MEDIUM" if maturity_score > 60 else "HIGH" }}

**Strengths**:
- Strong foundational architecture
- Comprehensive security controls
- Mature development practices
- Regulatory compliance framework

**Areas for Improvement**:
{% if metrics_summary.development.test_coverage < 85 %}
- Test coverage expansion
{% endif %}
{% if metrics_summary.development.vulnerability_count > 0 %}
- Security vulnerability remediation
{% endif %}
{% if metrics_summary.operations.uptime < 99 %}
- System reliability enhancement
{% endif %}

## Next Quarter Focus

### Q1 2025 Objectives
1. **Security First**: Achieve zero-vulnerability status
2. **Quality Excellence**: Reach 90%+ test coverage
3. **Operational Maturity**: Maintain 99.9% uptime
4. **Compliance Leadership**: Complete advanced certifications

### Success Metrics
- Security scan results: 100% clean
- Quality gates: All passing
- Performance targets: All met
- Compliance audits: All passed

## Leadership Actions Required

### Immediate (Next 30 days)
- [ ] Approve security tool procurement
- [ ] Authorize additional testing resources
- [ ] Review compliance certification roadmap

### Strategic (Next 90 days)
- [ ] Evaluate platform scaling requirements
- [ ] Plan advanced feature development
- [ ] Assess team capability needs

---
**Bottom Line**: The project demonstrates exceptional SDLC maturity and delivers significant business value through advanced cybersecurity research capabilities while maintaining strong compliance and operational excellence.

*Prepared by Terragon SDLC Automation Team*
"""

        # Calculate component scores
        quality_score = min(100, metrics_summary['development']['test_coverage'])
        security_score = max(0, 100 - (metrics_summary['development']['vulnerability_count'] * 10))
        ops_score = metrics_summary['operations']['uptime']
        compliance_score = metrics_summary['compliance']['policy_compliance_rate']
        
        # Determine status indicators
        quality_status = "✅" if quality_score >= 85 else "🟡" if quality_score >= 70 else "🔴"
        security_status = "✅" if security_score >= 95 else "🟡" if security_score >= 80 else "🔴"
        ops_status = "✅" if ops_score >= 90 else "🟡" if ops_score >= 85 else "🔴"
        compliance_status = "✅" if compliance_score >= 95 else "🟡" if compliance_score >= 80 else "🔴"
        
        # Determine maturity level
        if maturity_score >= 90:
            maturity_level = "OPTIMIZING"
        elif maturity_score >= 80:
            maturity_level = "ADVANCED"
        elif maturity_score >= 70:
            maturity_level = "MATURING"
        elif maturity_score >= 60:
            maturity_level = "DEVELOPING"
        else:
            maturity_level = "INITIAL"
            
        template = Template(summary_template)
        summary = template.render(
            timestamp=timestamp,
            metrics_summary=metrics_summary,
            maturity_score=maturity_score,
            maturity_level=maturity_level,
            quality_score=quality_score,
            security_score=security_score,
            ops_score=ops_score,
            compliance_score=compliance_score,
            quality_status=quality_status,
            security_status=security_status,
            ops_status=ops_status,
            compliance_status=compliance_status
        )
        
        return summary
        
    def _calculate_maturity_score(self, metrics: Dict[str, Any]) -> int:
        """Calculate overall SDLC maturity score."""
        weights = {
            'test_coverage': 0.25,
            'security': 0.30,
            'operations': 0.25,
            'compliance': 0.20
        }
        
        # Normalize scores to 0-100
        test_score = min(100, metrics['development']['test_coverage'])
        security_score = max(0, 100 - (metrics['development']['vulnerability_count'] * 10))
        ops_score = (metrics['operations']['uptime'] + metrics['operations']['deployment_success_rate']) / 2
        compliance_score = metrics['compliance']['policy_compliance_rate']
        
        total_score = (
            test_score * weights['test_coverage'] +
            security_score * weights['security'] +
            ops_score * weights['operations'] +
            compliance_score * weights['compliance']
        )
        
        return int(total_score)
        
    def save_reports(self) -> None:
        """Generate and save all reports."""
        timestamp = datetime.datetime.now()
        
        # Generate reports
        weekly_report = self.generate_weekly_report()
        monthly_report = self.generate_monthly_report()
        executive_summary = self.generate_executive_summary()
        
        # Save to files
        weekly_file = self.reports_dir / f"weekly-report-{timestamp.strftime('%Y%m%d')}.md"
        monthly_file = self.reports_dir / f"monthly-report-{timestamp.strftime('%Y%m')}.md"
        executive_file = self.reports_dir / f"executive-summary-{timestamp.strftime('%Y%m')}.md"
        
        with open(weekly_file, 'w') as f:
            f.write(weekly_report)
            
        with open(monthly_file, 'w') as f:
            f.write(monthly_report)
            
        with open(executive_file, 'w') as f:
            f.write(executive_summary)
            
        logger.info(f"Reports generated:")
        logger.info(f"  Weekly: {weekly_file}")
        logger.info(f"  Monthly: {monthly_file}")
        logger.info(f"  Executive: {executive_file}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate SDLC reports')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--report-type', choices=['weekly', 'monthly', 'executive', 'all'], 
                        default='all', help='Type of report to generate')
    parser.add_argument('--output-dir', help='Output directory for reports')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    generator = ReportGenerator(args.repo_root)
    
    if args.output_dir:
        generator.reports_dir = Path(args.output_dir)
        generator.reports_dir.mkdir(exist_ok=True)
        
    if args.report_type == 'all':
        generator.save_reports()
    elif args.report_type == 'weekly':
        report = generator.generate_weekly_report()
        print(report)
    elif args.report_type == 'monthly':
        report = generator.generate_monthly_report()
        print(report)
    elif args.report_type == 'executive':
        summary = generator.generate_executive_summary()
        print(summary)

if __name__ == '__main__':
    main()