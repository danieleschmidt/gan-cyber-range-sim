#!/usr/bin/env python3
"""
Terragon SDLC Metrics Collection Script
Automatically collects and updates project metrics for SDLC monitoring.
"""

import json
import os
import subprocess
import sys
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and aggregates SDLC metrics from various sources."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.metrics_file = self.repo_root / ".github" / "project-metrics.json"
        self.metrics_data = self._load_metrics()
        
    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing metrics configuration."""
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Metrics file not found: {self.metrics_file}")
            sys.exit(1)
            
    def _save_metrics(self) -> None:
        """Save updated metrics back to file."""
        self.metrics_data["metadata"]["last_updated"] = datetime.datetime.now().isoformat()
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_data, f, indent=2)
            
    def _run_command(self, command: list, capture_output: bool = True) -> Optional[subprocess.CompletedProcess]:
        """Run shell command safely."""
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                cwd=self.repo_root
            )
            return result
        except subprocess.SubprocessError as e:
            logger.warning(f"Command failed: {' '.join(command)} - {e}")
            return None
            
    def collect_code_quality_metrics(self) -> None:
        """Collect code quality metrics."""
        logger.info("Collecting code quality metrics...")
        
        # Test coverage
        coverage_result = self._run_command(['python', '-m', 'pytest', '--cov=src', '--cov-report=json'])
        if coverage_result and coverage_result.returncode == 0:
            try:
                with open(self.repo_root / 'coverage.json', 'r') as f:
                    coverage_data = json.load(f)
                    coverage_percent = coverage_data.get('totals', {}).get('percent_covered', 0)
                    self.metrics_data['metrics']['development']['code_quality']['test_coverage']['current'] = round(coverage_percent, 1)
            except FileNotFoundError:
                logger.warning("Coverage report not found")
                
        # Cyclomatic complexity
        complexity_result = self._run_command(['python', '-m', 'radon', 'cc', 'src', '--json'])
        if complexity_result and complexity_result.returncode == 0:
            try:
                complexity_data = json.loads(complexity_result.stdout)
                total_complexity = 0
                total_functions = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item.get('type') == 'function':
                            total_complexity += item.get('complexity', 0)
                            total_functions += 1
                            
                avg_complexity = total_complexity / total_functions if total_functions > 0 else 0
                self.metrics_data['metrics']['development']['code_quality']['cyclomatic_complexity']['current'] = round(avg_complexity, 1)
            except (json.JSONDecodeError, ZeroDivisionError):
                logger.warning("Failed to parse complexity data")
                
        # Lines of code
        loc_result = self._run_command(['find', 'src', '-name', '*.py', '-exec', 'wc', '-l', '{}', '+'])
        if loc_result and loc_result.returncode == 0:
            lines = loc_result.stdout.strip().split('\n')
            if lines:
                total_line = lines[-1]  # Last line contains total
                try:
                    total_loc = int(total_line.split()[0])
                    self.metrics_data['metrics']['development']['code_quality']['lines_of_code'] = total_loc
                except (ValueError, IndexError):
                    logger.warning("Failed to parse lines of code")
                    
    def collect_security_metrics(self) -> None:
        """Collect security metrics."""
        logger.info("Collecting security metrics...")
        
        # Bandit security scan
        bandit_result = self._run_command(['python', '-m', 'bandit', '-r', 'src', '-f', 'json'])
        if bandit_result:
            try:
                bandit_data = json.loads(bandit_result.stdout)
                results = bandit_data.get('results', [])
                
                vulnerability_count = len(results)
                severity_breakdown = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
                
                for result in results:
                    severity = result.get('issue_severity', '').lower()
                    if severity in severity_breakdown:
                        severity_breakdown[severity] += 1
                        
                self.metrics_data['metrics']['development']['security']['vulnerability_count']['current'] = vulnerability_count
                self.metrics_data['metrics']['development']['security']['vulnerability_count']['severity_breakdown'] = severity_breakdown
            except json.JSONDecodeError:
                logger.warning("Failed to parse bandit output")
                
        # Safety dependency scan
        safety_result = self._run_command(['python', '-m', 'safety', 'check', '--json'])
        if safety_result:
            try:
                safety_data = json.loads(safety_result.stdout)
                dependency_vulnerabilities = len(safety_data)
                
                current_vuln_count = self.metrics_data['metrics']['development']['security']['vulnerability_count']['current']
                self.metrics_data['metrics']['development']['security']['vulnerability_count']['current'] = current_vuln_count + dependency_vulnerabilities
            except json.JSONDecodeError:
                logger.warning("Failed to parse safety output")
                
    def collect_performance_metrics(self) -> None:
        """Collect performance metrics."""
        logger.info("Collecting performance metrics...")
        
        # Container image size (if Dockerfile exists)
        if (self.repo_root / 'Dockerfile').exists():
            # Build image to get size
            build_result = self._run_command(['docker', 'build', '-t', 'gan-cyber-range:metrics', '.'])
            if build_result and build_result.returncode == 0:
                inspect_result = self._run_command(['docker', 'images', 'gan-cyber-range:metrics', '--format', '{{.Size}}'])
                if inspect_result and inspect_result.returncode == 0:
                    size_str = inspect_result.stdout.strip()
                    try:
                        # Parse size (e.g., "1.2GB" -> 1200)
                        if 'GB' in size_str:
                            size_mb = float(size_str.replace('GB', '')) * 1024
                        elif 'MB' in size_str:
                            size_mb = float(size_str.replace('MB', ''))
                        else:
                            size_mb = 0
                            
                        self.metrics_data['metrics']['development']['performance']['container_image_size']['current'] = round(size_mb)
                    except ValueError:
                        logger.warning("Failed to parse container size")
                        
    def collect_git_metrics(self) -> None:
        """Collect git-based metrics."""
        logger.info("Collecting git metrics...")
        
        # Commit frequency (last 30 days)
        thirty_days_ago = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        commit_result = self._run_command(['git', 'log', '--since', thirty_days_ago, '--oneline'])
        if commit_result and commit_result.returncode == 0:
            commit_count = len(commit_result.stdout.strip().split('\n')) if commit_result.stdout.strip() else 0
            self.metrics_data['metrics']['development']['git_activity'] = {
                'commits_last_30_days': commit_count,
                'commit_frequency': 'daily' if commit_count > 20 else 'weekly' if commit_count > 4 else 'monthly'
            }
            
        # Contributors
        contributor_result = self._run_command(['git', 'shortlog', '-sn'])
        if contributor_result and contributor_result.returncode == 0:
            contributors = len(contributor_result.stdout.strip().split('\n')) if contributor_result.stdout.strip() else 0
            self.metrics_data['metrics']['team']['collaboration']['active_contributors'] = contributors
            
    def collect_documentation_metrics(self) -> None:
        """Collect documentation metrics."""
        logger.info("Collecting documentation metrics...")
        
        # Count documentation files
        doc_files = list(self.repo_root.glob('**/*.md')) + list(self.repo_root.glob('docs/**/*'))
        doc_count = len([f for f in doc_files if f.is_file()])
        
        # Count Python docstrings
        py_files = list(self.repo_root.glob('src/**/*.py'))
        documented_functions = 0
        total_functions = 0
        
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                # Simple heuristic: count function definitions and docstrings
                function_lines = [line for line in content.split('\n') if line.strip().startswith('def ')]
                total_functions += len(function_lines)
                
                # Count docstrings (simplified)
                docstring_count = content.count('"""') + content.count("'''")
                documented_functions += min(len(function_lines), docstring_count // 2)
                
            except Exception as e:
                logger.warning(f"Failed to analyze {py_file}: {e}")
                
        doc_coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 0
        
        self.metrics_data['metrics']['compliance']['governance']['documentation_coverage']['current'] = round(doc_coverage, 1)
        self.metrics_data['metrics']['development']['documentation'] = {
            'total_doc_files': doc_count,
            'function_documentation_coverage': round(doc_coverage, 1)
        }
        
    def generate_report(self) -> str:
        """Generate a summary report of collected metrics."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""
# SDLC Metrics Report
Generated: {timestamp}

## Code Quality
- Test Coverage: {self.metrics_data['metrics']['development']['code_quality']['test_coverage']['current']}%
- Cyclomatic Complexity: {self.metrics_data['metrics']['development']['code_quality']['cyclomatic_complexity']['current']}
- Technical Debt Ratio: {self.metrics_data['metrics']['development']['code_quality']['technical_debt_ratio']['current']}%

## Security
- Vulnerabilities: {self.metrics_data['metrics']['development']['security']['vulnerability_count']['current']}
- Severity Breakdown: {self.metrics_data['metrics']['development']['security']['vulnerability_count']['severity_breakdown']}

## Performance
- Container Image Size: {self.metrics_data['metrics']['development']['performance']['container_image_size']['current']}MB

## Documentation
- Documentation Coverage: {self.metrics_data['metrics']['compliance']['governance']['documentation_coverage']['current']}%

## Recommendations
"""
        
        # Add recommendations based on metrics
        recommendations = []
        
        coverage = self.metrics_data['metrics']['development']['code_quality']['test_coverage']['current']
        if coverage < 80:
            recommendations.append(f"- Increase test coverage from {coverage}% to 85%+")
            
        vulns = self.metrics_data['metrics']['development']['security']['vulnerability_count']['current']
        if vulns > 0:
            recommendations.append(f"- Address {vulns} security vulnerabilities")
            
        complexity = self.metrics_data['metrics']['development']['code_quality']['cyclomatic_complexity']['current']
        if complexity > 10:
            recommendations.append(f"- Reduce cyclomatic complexity from {complexity} to <10")
            
        if not recommendations:
            recommendations.append("- All metrics are within target ranges ✅")
            
        report += '\n'.join(recommendations)
        return report
        
    def run_collection(self) -> None:
        """Run full metrics collection."""
        logger.info("Starting SDLC metrics collection...")
        
        try:
            self.collect_code_quality_metrics()
            self.collect_security_metrics()
            self.collect_performance_metrics()
            self.collect_git_metrics()
            self.collect_documentation_metrics()
            
            self._save_metrics()
            
            # Generate and save report
            report = self.generate_report()
            report_file = self.repo_root / 'metrics-report.md'
            with open(report_file, 'w') as f:
                f.write(report)
                
            logger.info(f"Metrics collection completed. Report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            sys.exit(1)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect SDLC metrics')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--output-format', choices=['json', 'markdown'], default='markdown', help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    collector = MetricsCollector(args.repo_root)
    collector.run_collection()
    
    if args.output_format == 'json':
        print(json.dumps(collector.metrics_data, indent=2))

if __name__ == '__main__':
    main()