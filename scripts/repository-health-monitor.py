#!/usr/bin/env python3
"""
Repository Health Monitoring Script
Continuously monitors repository health metrics and alerts on issues.
"""

import json
import os
import subprocess
import sys
import logging
import datetime
import smtplib
from pathlib import Path
from typing import Dict, List, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RepositoryHealthMonitor:
    """Monitors repository health and sends alerts."""
    
    def __init__(self, repo_root: str = ".", config_file: str = None):
        self.repo_root = Path(repo_root)
        self.config = self._load_config(config_file)
        self.metrics_file = self.repo_root / ".github" / "project-metrics.json"
        self.health_thresholds = self._load_thresholds()
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load monitoring configuration."""
        default_config = {
            "monitoring": {
                "enabled": True,
                "check_interval_hours": 24,
                "alert_channels": ["console"],  # email, slack, webhook
                "severity_levels": ["critical", "warning", "info"]
            },
            "email": {
                "smtp_server": "localhost",
                "smtp_port": 587,
                "sender": "repo-monitor@company.com",
                "recipients": ["team@company.com"]
            },
            "slack": {
                "webhook_url": "",
                "channel": "#dev-alerts"
            },
            "webhook": {
                "url": "",
                "headers": {}
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
                
        return default_config
        
    def _load_thresholds(self) -> Dict[str, Any]:
        """Load health check thresholds."""
        return {
            "critical": {
                "test_coverage": 60,  # Below this is critical
                "vulnerability_count": 5,
                "build_failure_rate": 20,
                "days_since_last_commit": 14
            },
            "warning": {
                "test_coverage": 75,
                "vulnerability_count": 2,
                "build_failure_rate": 10,
                "days_since_last_commit": 7
            },
            "performance": {
                "build_time_seconds": 600,  # 10 minutes
                "test_time_seconds": 300,   # 5 minutes
                "container_size_mb": 1000   # 1GB
            }
        }
        
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
            
    def check_code_quality_health(self) -> Dict[str, Any]:
        """Check code quality health metrics."""
        health_status = {
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }
        
        # Load current metrics
        try:
            with open(self.metrics_file, 'r') as f:
                metrics = json.load(f)
                
            code_quality = metrics.get('metrics', {}).get('development', {}).get('code_quality', {})
            
            # Check test coverage
            coverage = code_quality.get('test_coverage', {}).get('current', 0)
            health_status['metrics']['test_coverage'] = coverage
            
            if coverage < self.health_thresholds['critical']['test_coverage']:
                health_status['issues'].append({
                    'severity': 'critical',
                    'type': 'test_coverage',
                    'message': f"Test coverage critically low: {coverage}%",
                    'recommendation': 'Add tests to increase coverage above 60%'
                })
                health_status['status'] = 'critical'
            elif coverage < self.health_thresholds['warning']['test_coverage']:
                health_status['issues'].append({
                    'severity': 'warning',
                    'type': 'test_coverage',
                    'message': f"Test coverage below target: {coverage}%",
                    'recommendation': 'Add tests to reach 85% coverage target'
                })
                if health_status['status'] == 'healthy':
                    health_status['status'] = 'warning'
                    
            # Check cyclomatic complexity
            complexity = code_quality.get('cyclomatic_complexity', {}).get('current', 0)
            health_status['metrics']['cyclomatic_complexity'] = complexity
            
            if complexity > 15:
                health_status['issues'].append({
                    'severity': 'warning',
                    'type': 'complexity',
                    'message': f"High cyclomatic complexity: {complexity}",
                    'recommendation': 'Refactor complex functions to reduce complexity'
                })
                if health_status['status'] == 'healthy':
                    health_status['status'] = 'warning'
                    
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            health_status['issues'].append({
                'severity': 'warning',
                'type': 'metrics_unavailable',
                'message': f"Unable to load metrics: {e}",
                'recommendation': 'Run metrics collection script'
            })
            
        return health_status
        
    def check_security_health(self) -> Dict[str, Any]:
        """Check security health metrics."""
        health_status = {
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }
        
        # Run security scans
        # Bandit scan
        bandit_result = self._run_command(['python', '-m', 'bandit', '-r', 'src', '-f', 'json'])
        if bandit_result and bandit_result.returncode == 0:
            try:
                bandit_data = json.loads(bandit_result.stdout)
                vuln_count = len(bandit_data.get('results', []))
                health_status['metrics']['vulnerability_count'] = vuln_count
                
                if vuln_count >= self.health_thresholds['critical']['vulnerability_count']:
                    health_status['issues'].append({
                        'severity': 'critical',
                        'type': 'security_vulnerabilities',
                        'message': f"Critical security vulnerabilities found: {vuln_count}",
                        'recommendation': 'Address security vulnerabilities immediately'
                    })
                    health_status['status'] = 'critical'
                elif vuln_count >= self.health_thresholds['warning']['vulnerability_count']:
                    health_status['issues'].append({
                        'severity': 'warning',
                        'type': 'security_vulnerabilities',
                        'message': f"Security vulnerabilities found: {vuln_count}",
                        'recommendation': 'Review and address security vulnerabilities'
                    })
                    if health_status['status'] == 'healthy':
                        health_status['status'] = 'warning'
                        
            except json.JSONDecodeError:
                health_status['issues'].append({
                    'severity': 'warning',
                    'type': 'security_scan_failed',
                    'message': 'Security scan failed to parse results',
                    'recommendation': 'Check security scanning tools'
                })
                
        # Check for secrets in repository
        secrets_result = self._run_command(['git', 'log', '--all', '--full-history', '--', '*.key', '*.pem', '*.p12'])
        if secrets_result and secrets_result.stdout.strip():
            health_status['issues'].append({
                'severity': 'critical',
                'type': 'potential_secrets',
                'message': 'Potential secrets detected in git history',
                'recommendation': 'Review and remove any committed secrets'
            })
            health_status['status'] = 'critical'
            
        return health_status
        
    def check_git_health(self) -> Dict[str, Any]:
        """Check git repository health."""
        health_status = {
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }
        
        # Check last commit date
        last_commit_result = self._run_command(['git', 'log', '-1', '--format=%ct'])
        if last_commit_result and last_commit_result.returncode == 0:
            try:
                last_commit_timestamp = int(last_commit_result.stdout.strip())
                last_commit_date = datetime.datetime.fromtimestamp(last_commit_timestamp)
                days_since_commit = (datetime.datetime.now() - last_commit_date).days
                
                health_status['metrics']['days_since_last_commit'] = days_since_commit
                
                if days_since_commit >= self.health_thresholds['critical']['days_since_last_commit']:
                    health_status['issues'].append({
                        'severity': 'warning',
                        'type': 'stale_repository',
                        'message': f"No commits for {days_since_commit} days",
                        'recommendation': 'Check if repository is actively maintained'
                    })
                    if health_status['status'] == 'healthy':
                        health_status['status'] = 'warning'
                        
            except ValueError:
                health_status['issues'].append({
                    'severity': 'warning',
                    'type': 'git_history_error',
                    'message': 'Unable to parse git history',
                    'recommendation': 'Check git repository integrity'
                })
                
        # Check for large files
        large_files_result = self._run_command(['find', '.', '-type', 'f', '-size', '+10M'])
        if large_files_result and large_files_result.stdout.strip():
            large_files = large_files_result.stdout.strip().split('\n')
            if len(large_files) > 0:
                health_status['issues'].append({
                    'severity': 'warning',
                    'type': 'large_files',
                    'message': f"Large files detected: {len(large_files)} files >10MB",
                    'recommendation': 'Consider using Git LFS for large files'
                })
                if health_status['status'] == 'healthy':
                    health_status['status'] = 'warning'
                    
        return health_status
        
    def check_dependency_health(self) -> Dict[str, Any]:
        """Check dependency health."""
        health_status = {
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }
        
        # Check for outdated dependencies
        pip_list_result = self._run_command(['pip', 'list', '--outdated', '--format=json'])
        if pip_list_result and pip_list_result.returncode == 0:
            try:
                outdated_packages = json.loads(pip_list_result.stdout)
                health_status['metrics']['outdated_packages'] = len(outdated_packages)
                
                if len(outdated_packages) > 10:
                    health_status['issues'].append({
                        'severity': 'warning',
                        'type': 'outdated_dependencies',
                        'message': f"Many outdated dependencies: {len(outdated_packages)}",
                        'recommendation': 'Run dependency update automation'
                    })
                    if health_status['status'] == 'healthy':
                        health_status['status'] = 'warning'
                        
            except json.JSONDecodeError:
                pass
                
        # Check for vulnerable dependencies
        safety_result = self._run_command(['python', '-m', 'safety', 'check', '--json'])
        if safety_result and safety_result.returncode != 0:
            # safety returns non-zero if vulnerabilities found
            try:
                safety_data = json.loads(safety_result.stdout)
                vuln_count = len(safety_data)
                health_status['metrics']['vulnerable_dependencies'] = vuln_count
                
                if vuln_count > 0:
                    health_status['issues'].append({
                        'severity': 'critical',
                        'type': 'vulnerable_dependencies',
                        'message': f"Vulnerable dependencies found: {vuln_count}",
                        'recommendation': 'Update vulnerable dependencies immediately'
                    })
                    health_status['status'] = 'critical'
                    
            except json.JSONDecodeError:
                pass
                
        return health_status
        
    def check_build_health(self) -> Dict[str, Any]:
        """Check build and CI health."""
        health_status = {
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }
        
        # Check if tests can run
        test_result = self._run_command(['python', '-m', 'pytest', '--collect-only'])
        if test_result and test_result.returncode != 0:
            health_status['issues'].append({
                'severity': 'critical',
                'type': 'test_collection_failed',
                'message': 'Test collection failed',
                'recommendation': 'Fix test configuration and dependencies'
            })
            health_status['status'] = 'critical'
            
        # Check if requirements can be installed
        if (self.repo_root / 'requirements.txt').exists():
            install_check = self._run_command(['pip', 'check'])
            if install_check and install_check.returncode != 0:
                health_status['issues'].append({
                    'severity': 'warning',
                    'type': 'dependency_conflicts',
                    'message': 'Dependency conflicts detected',
                    'recommendation': 'Resolve dependency version conflicts'
                })
                if health_status['status'] == 'healthy':
                    health_status['status'] = 'warning'
                    
        return health_status
        
    def generate_health_report(self, health_results: Dict[str, Dict]) -> str:
        """Generate comprehensive health report."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate overall status
        statuses = [result['status'] for result in health_results.values()]
        if 'critical' in statuses:
            overall_status = 'critical'
        elif 'warning' in statuses:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
            
        # Count issues by severity
        all_issues = []
        for result in health_results.values():
            all_issues.extend(result['issues'])
            
        critical_count = len([i for i in all_issues if i['severity'] == 'critical'])
        warning_count = len([i for i in all_issues if i['severity'] == 'warning'])
        
        report = f"""# Repository Health Report
Generated: {timestamp}
Overall Status: **{overall_status.upper()}** {"🔴" if overall_status == "critical" else "🟡" if overall_status == "warning" else "🟢"}

## Summary
- Critical Issues: {critical_count}
- Warning Issues: {warning_count}
- Total Checks: {len(health_results)}

"""

        # Add details for each health check
        for check_name, result in health_results.items():
            status_emoji = "🔴" if result['status'] == "critical" else "🟡" if result['status'] == "warning" else "🟢"
            report += f"## {check_name.replace('_', ' ').title()} {status_emoji}\n\n"
            
            if result['issues']:
                for issue in result['issues']:
                    severity_emoji = "🔴" if issue['severity'] == "critical" else "🟡"
                    report += f"### {severity_emoji} {issue['type'].replace('_', ' ').title()}\n"
                    report += f"**Issue**: {issue['message']}\n\n"
                    report += f"**Recommendation**: {issue['recommendation']}\n\n"
            else:
                report += "No issues detected ✅\n\n"
                
            # Add metrics if available
            if result['metrics']:
                report += "**Metrics**:\n"
                for metric, value in result['metrics'].items():
                    report += f"- {metric.replace('_', ' ').title()}: {value}\n"
                report += "\n"
                
        # Add action items
        if critical_count > 0 or warning_count > 0:
            report += "## Immediate Actions Required\n\n"
            
            priority_issues = [i for i in all_issues if i['severity'] == 'critical']
            if priority_issues:
                report += "### Critical (Fix Immediately)\n"
                for i, issue in enumerate(priority_issues, 1):
                    report += f"{i}. {issue['recommendation']}\n"
                report += "\n"
                
            warning_issues = [i for i in all_issues if i['severity'] == 'warning']
            if warning_issues:
                report += "### Warnings (Address Soon)\n"
                for i, issue in enumerate(warning_issues, 1):
                    report += f"{i}. {issue['recommendation']}\n"
                report += "\n"
        else:
            report += "## All Checks Passed ✅\n\nRepository is in good health!\n\n"
            
        return report
        
    def send_alert(self, report: str, severity: str) -> None:
        """Send health alert through configured channels."""
        if not self.config['monitoring']['enabled']:
            return
            
        alert_channels = self.config['monitoring']['alert_channels']
        
        if 'console' in alert_channels:
            print(report)
            
        if 'email' in alert_channels and severity in ['critical', 'warning']:
            self._send_email_alert(report, severity)
            
        if 'slack' in alert_channels and self.config['slack']['webhook_url']:
            self._send_slack_alert(report, severity)
            
        if 'webhook' in alert_channels and self.config['webhook']['url']:
            self._send_webhook_alert(report, severity)
            
    def _send_email_alert(self, report: str, severity: str) -> None:
        """Send email alert."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['sender']
            msg['To'] = ', '.join(self.config['email']['recipients'])
            msg['Subject'] = f"Repository Health Alert - {severity.upper()}"
            
            msg.attach(MIMEText(report, 'plain'))
            
            server = smtplib.SMTP(self.config['email']['smtp_server'], self.config['email']['smtp_port'])
            server.send_message(msg)
            server.quit()
            
            logger.info("Email alert sent successfully")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            
    def _send_slack_alert(self, report: str, severity: str) -> None:
        """Send Slack alert."""
        try:
            color = "#FF0000" if severity == "critical" else "#FFFF00" if severity == "warning" else "#00FF00"
            
            payload = {
                "channel": self.config['slack']['channel'],
                "attachments": [{
                    "color": color,
                    "title": f"Repository Health Alert - {severity.upper()}",
                    "text": report[:3000],  # Slack has character limits
                    "ts": datetime.datetime.now().timestamp()
                }]
            }
            
            response = requests.post(self.config['slack']['webhook_url'], json=payload)
            if response.status_code == 200:
                logger.info("Slack alert sent successfully")
            else:
                logger.error(f"Failed to send Slack alert: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            
    def _send_webhook_alert(self, report: str, severity: str) -> None:
        """Send webhook alert."""
        try:
            payload = {
                "timestamp": datetime.datetime.now().isoformat(),
                "severity": severity,
                "repository": "gan-cyber-range-sim",
                "report": report
            }
            
            headers = self.config['webhook'].get('headers', {})
            response = requests.post(self.config['webhook']['url'], json=payload, headers=headers)
            
            if response.status_code == 200:
                logger.info("Webhook alert sent successfully")
            else:
                logger.error(f"Failed to send webhook alert: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            
    def run_health_check(self) -> None:
        """Run complete repository health check."""
        logger.info("Starting repository health check...")
        
        health_checks = {
            'code_quality': self.check_code_quality_health,
            'security': self.check_security_health,
            'git_repository': self.check_git_health,
            'dependencies': self.check_dependency_health,
            'build_system': self.check_build_health
        }
        
        health_results = {}
        
        for check_name, check_function in health_checks.items():
            try:
                logger.info(f"Running {check_name} health check...")
                health_results[check_name] = check_function()
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                health_results[check_name] = {
                    'status': 'warning',
                    'issues': [{
                        'severity': 'warning',
                        'type': 'check_failed',
                        'message': f"Health check failed: {e}",
                        'recommendation': f"Fix {check_name} health check implementation"
                    }],
                    'metrics': {}
                }
                
        # Generate report
        report = self.generate_health_report(health_results)
        
        # Save report
        report_file = self.repo_root / 'repository-health-report.md'
        with open(report_file, 'w') as f:
            f.write(report)
            
        # Determine overall severity
        statuses = [result['status'] for result in health_results.values()]
        overall_severity = 'critical' if 'critical' in statuses else 'warning' if 'warning' in statuses else 'healthy'
        
        # Send alerts
        self.send_alert(report, overall_severity)
        
        logger.info(f"Health check completed. Report saved to {report_file}")
        logger.info(f"Overall health status: {overall_severity}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Repository health monitoring')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    monitor = RepositoryHealthMonitor(args.repo_root, args.config)
    monitor.run_health_check()

if __name__ == '__main__':
    main()