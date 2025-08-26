#!/usr/bin/env python3
"""Security and quality gate validations for GAN Cyber Range Simulator."""

import sys
import os
import asyncio
import logging
import json
import re
import time
import hashlib
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from robust_cyber_range import RobustLogger


class SecurityLevel(Enum):
    """Security assessment levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    PASS = "pass"


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAIL = "fail"


@dataclass
class SecurityIssue:
    """Security issue representation."""
    severity: SecurityLevel
    category: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: Optional[str] = None
    cve_references: List[str] = field(default_factory=list)


@dataclass
class QualityIssue:
    """Code quality issue representation."""
    severity: QualityLevel
    category: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    fix_suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    timestamp: datetime = field(default_factory=datetime.now)
    security_score: float = 0.0
    quality_score: float = 0.0
    security_issues: List[SecurityIssue] = field(default_factory=list)
    quality_issues: List[QualityIssue] = field(default_factory=list)
    passed_checks: int = 0
    failed_checks: int = 0
    total_files_scanned: int = 0
    scan_duration_seconds: float = 0.0
    
    @property
    def overall_score(self) -> float:
        return (self.security_score + self.quality_score) / 2
    
    @property
    def passed(self) -> bool:
        return (self.security_score >= 70 and 
                self.quality_score >= 70 and 
                not any(issue.severity == SecurityLevel.CRITICAL for issue in self.security_issues))


class SecurityValidator:
    """Comprehensive security validation and scanning."""
    
    def __init__(self):
        self.logger = RobustLogger("SecurityValidator")
        self.dangerous_patterns = [
            # Code injection patterns
            (r'eval\s*\(', SecurityLevel.CRITICAL, "Code injection via eval()"),
            (r'exec\s*\(', SecurityLevel.CRITICAL, "Code injection via exec()"),
            (r'__import__\s*\(', SecurityLevel.HIGH, "Dynamic import usage"),
            (r'compile\s*\(', SecurityLevel.HIGH, "Dynamic code compilation"),
            
            # SQL injection patterns
            (r'SELECT.*\+.*', SecurityLevel.HIGH, "Potential SQL injection"),
            (r'INSERT.*\+.*', SecurityLevel.HIGH, "Potential SQL injection"),
            (r'UPDATE.*\+.*', SecurityLevel.HIGH, "Potential SQL injection"),
            (r'DELETE.*\+.*', SecurityLevel.HIGH, "Potential SQL injection"),
            
            # Command injection patterns
            (r'os\.system\s*\(', SecurityLevel.HIGH, "OS command execution"),
            (r'subprocess\.call\s*\(', SecurityLevel.MEDIUM, "Subprocess call"),
            (r'shell\s*=\s*True', SecurityLevel.HIGH, "Shell injection risk"),
            
            # File system access patterns
            (r'\.\./', SecurityLevel.HIGH, "Directory traversal"),
            (r'open\s*\(.*\+', SecurityLevel.MEDIUM, "Dynamic file path"),
            
            # Network security patterns
            (r'ssl_context\s*=\s*None', SecurityLevel.HIGH, "Disabled SSL verification"),
            (r'verify\s*=\s*False', SecurityLevel.HIGH, "Disabled certificate verification"),
            (r'check_hostname\s*=\s*False', SecurityLevel.MEDIUM, "Disabled hostname check"),
            
            # Authentication patterns
            (r'password\s*=\s*["\'].*["\']', SecurityLevel.HIGH, "Hardcoded password"),
            (r'secret\s*=\s*["\'].*["\']', SecurityLevel.HIGH, "Hardcoded secret"),
            (r'api_key\s*=\s*["\'].*["\']', SecurityLevel.HIGH, "Hardcoded API key"),
            (r'token\s*=\s*["\'].*["\']', SecurityLevel.HIGH, "Hardcoded token"),
            
            # Cryptographic patterns
            (r'md5\s*\(', SecurityLevel.MEDIUM, "Weak hash algorithm MD5"),
            (r'sha1\s*\(', SecurityLevel.MEDIUM, "Weak hash algorithm SHA1"),
            (r'DES|RC4', SecurityLevel.HIGH, "Weak encryption algorithm"),
            
            # Input validation patterns
            (r'input\s*\(\)', SecurityLevel.LOW, "User input without validation"),
            (r'raw_input\s*\(\)', SecurityLevel.LOW, "Raw user input"),
            
            # Deserialization patterns
            (r'pickle\.loads?\s*\(', SecurityLevel.HIGH, "Unsafe deserialization"),
            (r'marshal\.loads?\s*\(', SecurityLevel.HIGH, "Unsafe deserialization"),
            (r'yaml\.load\s*\(', SecurityLevel.MEDIUM, "Unsafe YAML loading"),
            
            # Web security patterns
            (r'<script.*>', SecurityLevel.HIGH, "Potential XSS vector"),
            (r'innerHTML\s*=', SecurityLevel.MEDIUM, "DOM manipulation risk"),
            (r'document\.write', SecurityLevel.MEDIUM, "DOM injection risk")
        ]
        
        self.secure_coding_patterns = [
            # Good patterns that increase security score
            (r'hashlib\.sha256', 5, "Strong hash algorithm"),
            (r'hashlib\.sha512', 5, "Strong hash algorithm"),
            (r'secrets\.token_hex', 5, "Cryptographically secure random"),
            (r'ssl\._create_default_context', 5, "Secure SSL context"),
            (r'verify\s*=\s*True', 3, "Certificate verification enabled"),
            (r'@login_required', 3, "Authentication required"),
            (r'csrf_token', 3, "CSRF protection"),
            (r'escape\s*\(', 2, "Output escaping"),
            (r'sanitize\s*\(', 3, "Input sanitization")
        ]
    
    async def scan_directory(self, directory: Path) -> List[SecurityIssue]:
        """Scan directory for security issues."""
        self.logger.info(f"Scanning directory for security issues: {directory}")
        
        issues = []
        python_files = list(directory.rglob("*.py"))
        
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
                
            try:
                file_issues = await self._scan_file(file_path)
                issues.extend(file_issues)
                
            except Exception as e:
                self.logger.warning(f"Failed to scan file {file_path}", exception=e)
        
        self.logger.info(f"Security scan complete: {len(issues)} issues found")
        return issues
    
    async def _scan_file(self, file_path: Path) -> List[SecurityIssue]:
        """Scan individual file for security issues."""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                line_issues = self._scan_line(line, file_path, line_num)
                issues.extend(line_issues)
                
        except Exception as e:
            self.logger.warning(f"Failed to read file {file_path}", exception=e)
        
        return issues
    
    def _scan_line(self, line: str, file_path: Path, line_num: int) -> List[SecurityIssue]:
        """Scan individual line for security patterns."""
        issues = []
        
        # Check for dangerous patterns
        for pattern, severity, description in self.dangerous_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                # Skip if in comments
                if line.strip().startswith('#'):
                    continue
                    
                issue = SecurityIssue(
                    severity=severity,
                    category="code_analysis",
                    description=description,
                    file_path=str(file_path),
                    line_number=line_num,
                    recommendation=self._get_recommendation(pattern)
                )
                issues.append(issue)
        
        return issues
    
    def _get_recommendation(self, pattern: str) -> str:
        """Get security recommendation for pattern."""
        recommendations = {
            r'eval\s*\(': "Use ast.literal_eval() for safe evaluation or avoid dynamic code execution",
            r'exec\s*\(': "Avoid dynamic code execution; use safer alternatives",
            r'os\.system\s*\(': "Use subprocess.run() with shell=False instead",
            r'shell\s*=\s*True': "Set shell=False and pass command as list",
            r'ssl_context\s*=\s*None': "Create proper SSL context for secure connections",
            r'verify\s*=\s*False': "Enable certificate verification for production",
            r'password\s*=\s*["\'].*["\']': "Use environment variables or secure configuration",
            r'pickle\.loads?\s*\(': "Use safer serialization formats like JSON"
        }
        
        for key, recommendation in recommendations.items():
            if key in pattern:
                return recommendation
        
        return "Review code for security implications"
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during scanning."""
        skip_patterns = [
            "test_",
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            ".pytest_cache"
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def calculate_security_score(self, issues: List[SecurityIssue], total_files: int) -> float:
        """Calculate security score based on issues found."""
        if total_files == 0:
            return 100.0
        
        # Weight by severity
        severity_weights = {
            SecurityLevel.CRITICAL: 50,
            SecurityLevel.HIGH: 25,
            SecurityLevel.MEDIUM: 10,
            SecurityLevel.LOW: 2
        }
        
        total_penalty = sum(
            severity_weights.get(issue.severity, 0) 
            for issue in issues
        )
        
        # Calculate score (100 - penalties, minimum 0)
        base_score = 100
        file_penalty_ratio = min(total_penalty / total_files, base_score)
        
        return max(0, base_score - file_penalty_ratio)


class QualityValidator:
    """Code quality analysis and validation."""
    
    def __init__(self):
        self.logger = RobustLogger("QualityValidator")
        self.quality_patterns = [
            # Code quality issues
            (r'^\s*print\s*\(', QualityLevel.POOR, "Print statement in production code"),
            (r'TODO|FIXME|HACK', QualityLevel.FAIR, "TODO/FIXME comments"),
            (r'^\s*pass\s*$', QualityLevel.FAIR, "Empty pass statement"),
            (r'except:\s*$', QualityLevel.POOR, "Bare except clause"),
            (r'except\s+Exception\s*:\s*pass', QualityLevel.POOR, "Silent exception handling"),
            
            # Complexity issues
            (r'if.*and.*and.*and', QualityLevel.FAIR, "Complex conditional"),
            (r'lambda.*:.*lambda', QualityLevel.POOR, "Nested lambda"),
            
            # Naming issues
            (r'\b[a-z]\b\s*=', QualityLevel.FAIR, "Single letter variable"),
            (r'def\s+[a-z]\s*\(', QualityLevel.FAIR, "Single letter function name"),
            
            # Documentation issues
            (r'def\s+\w+\([^)]*\):\s*$', QualityLevel.FAIR, "Missing function docstring"),
            (r'class\s+\w+[^:]*:\s*$', QualityLevel.FAIR, "Missing class docstring"),
        ]
        
        self.good_practices = [
            # Good patterns that increase quality score
            (r'""".*"""', 5, "Docstring present"),
            (r'@\w+', 3, "Decorator usage"),
            (r'typing\.|Type\[|List\[|Dict\[', 5, "Type hints"),
            (r'logging\.\w+', 3, "Proper logging"),
            (r'raise\s+\w+Error', 3, "Specific exception raising"),
            (r'with\s+open', 3, "Context manager usage"),
            (r'f".*{.*}"', 2, "F-string usage"),
            (r'@dataclass', 3, "Dataclass usage"),
            (r'async\s+def', 3, "Async function"),
            (r'await\s+', 3, "Proper async usage")
        ]
    
    async def analyze_directory(self, directory: Path) -> List[QualityIssue]:
        """Analyze directory for code quality issues."""
        self.logger.info(f"Analyzing directory for quality issues: {directory}")
        
        issues = []
        python_files = list(directory.rglob("*.py"))
        
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
                
            try:
                file_issues = await self._analyze_file(file_path)
                issues.extend(file_issues)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze file {file_path}", exception=e)
        
        self.logger.info(f"Quality analysis complete: {len(issues)} issues found")
        return issues
    
    async def _analyze_file(self, file_path: Path) -> List[QualityIssue]:
        """Analyze individual file for quality issues."""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Check file-level metrics
            issues.extend(self._check_file_metrics(file_path, lines))
            
            # Check line-level issues
            for line_num, line in enumerate(lines, 1):
                line_issues = self._analyze_line(line, file_path, line_num)
                issues.extend(line_issues)
                
        except Exception as e:
            self.logger.warning(f"Failed to read file {file_path}", exception=e)
        
        return issues
    
    def _check_file_metrics(self, file_path: Path, lines: List[str]) -> List[QualityIssue]:
        """Check file-level quality metrics."""
        issues = []
        
        # File length check
        if len(lines) > 500:
            issues.append(QualityIssue(
                severity=QualityLevel.FAIR,
                category="file_metrics",
                description=f"File too long: {len(lines)} lines",
                file_path=str(file_path),
                fix_suggestion="Consider splitting into smaller modules"
            ))
        
        # Function count check
        function_count = sum(1 for line in lines if re.match(r'^\s*def\s+', line))
        if function_count > 20:
            issues.append(QualityIssue(
                severity=QualityLevel.FAIR,
                category="file_metrics", 
                description=f"Too many functions: {function_count}",
                file_path=str(file_path),
                fix_suggestion="Consider splitting into multiple files"
            ))
        
        return issues
    
    def _analyze_line(self, line: str, file_path: Path, line_num: int) -> List[QualityIssue]:
        """Analyze individual line for quality issues."""
        issues = []
        
        # Skip comments and empty lines
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            return issues
        
        # Check for quality patterns
        for pattern, severity, description in self.quality_patterns:
            if re.search(pattern, line):
                issue = QualityIssue(
                    severity=severity,
                    category="code_quality",
                    description=description,
                    file_path=str(file_path),
                    line_number=line_num,
                    fix_suggestion=self._get_quality_fix(pattern)
                )
                issues.append(issue)
        
        # Check line length
        if len(line) > 120:
            issues.append(QualityIssue(
                severity=QualityLevel.FAIR,
                category="formatting",
                description=f"Line too long: {len(line)} characters",
                file_path=str(file_path),
                line_number=line_num,
                fix_suggestion="Break line into multiple lines"
            ))
        
        return issues
    
    def _get_quality_fix(self, pattern: str) -> str:
        """Get quality improvement suggestion for pattern."""
        fixes = {
            r'^\s*print\s*\(': "Use logging instead of print statements",
            r'TODO|FIXME|HACK': "Complete or remove TODO comments",
            r'except:\s*$': "Catch specific exceptions",
            r'^\s*pass\s*$': "Implement functionality or add docstring",
            r'def\s+\w+\([^)]*\):\s*$': "Add docstring describing function purpose"
        }
        
        for key, fix in fixes.items():
            if key in pattern:
                return fix
        
        return "Review and improve code quality"
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis."""
        return QualityValidator._should_skip_file_static(file_path)
    
    @staticmethod
    def _should_skip_file_static(file_path: Path) -> bool:
        """Static method to check if file should be skipped."""
        skip_patterns = [
            "test_",
            "__pycache__",
            ".git",
            ".venv", 
            "venv",
            ".pytest_cache",
            "minimal_test.py"  # Skip our test file
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def calculate_quality_score(self, issues: List[QualityIssue], total_files: int) -> float:
        """Calculate quality score based on issues found."""
        if total_files == 0:
            return 100.0
        
        # Weight by severity
        severity_weights = {
            QualityLevel.FAIL: 40,
            QualityLevel.POOR: 20,
            QualityLevel.FAIR: 10,
            QualityLevel.GOOD: 2
        }
        
        total_penalty = sum(
            severity_weights.get(issue.severity, 0)
            for issue in issues
        )
        
        # Calculate score
        base_score = 100
        file_penalty_ratio = min(total_penalty / total_files, base_score)
        
        return max(0, base_score - file_penalty_ratio)


class ComplianceValidator:
    """Validate compliance with security and coding standards."""
    
    def __init__(self):
        self.logger = RobustLogger("ComplianceValidator")
        self.required_files = [
            "README.md",
            "LICENSE",
            "requirements.txt",
            ".gitignore"
        ]
        
        self.recommended_files = [
            "SECURITY.md",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            "CHANGELOG.md"
        ]
    
    async def validate_project_structure(self, project_root: Path) -> List[QualityIssue]:
        """Validate project structure and required files."""
        issues = []
        
        # Check required files
        for required_file in self.required_files:
            if not (project_root / required_file).exists():
                issues.append(QualityIssue(
                    severity=QualityLevel.POOR,
                    category="project_structure",
                    description=f"Missing required file: {required_file}",
                    fix_suggestion=f"Create {required_file} file"
                ))
        
        # Check recommended files
        for recommended_file in self.recommended_files:
            if not (project_root / recommended_file).exists():
                issues.append(QualityIssue(
                    severity=QualityLevel.FAIR,
                    category="project_structure", 
                    description=f"Missing recommended file: {recommended_file}",
                    fix_suggestion=f"Consider adding {recommended_file}"
                ))
        
        # Check for test directory
        test_dirs = ["tests", "test"]
        if not any((project_root / test_dir).exists() for test_dir in test_dirs):
            issues.append(QualityIssue(
                severity=QualityLevel.POOR,
                category="testing",
                description="No test directory found",
                fix_suggestion="Create tests/ directory with test files"
            ))
        
        return issues


class QualityGateRunner:
    """Main quality gate runner that orchestrates all validations."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.logger = RobustLogger("QualityGateRunner")
        self.security_validator = SecurityValidator()
        self.quality_validator = QualityValidator()
        self.compliance_validator = ComplianceValidator()
    
    async def run_all_validations(self) -> ValidationReport:
        """Run all quality gates and generate comprehensive report."""
        start_time = time.time()
        self.logger.info(f"Starting quality gate validations for {self.project_root}")
        
        report = ValidationReport()
        
        try:
            # Security validation
            self.logger.info("Running security validation...")
            security_issues = await self.security_validator.scan_directory(self.project_root)
            report.security_issues = security_issues
            
            # Quality validation
            self.logger.info("Running quality validation...")
            quality_issues = await self.quality_validator.analyze_directory(self.project_root)
            
            # Compliance validation
            self.logger.info("Running compliance validation...")
            compliance_issues = await self.compliance_validator.validate_project_structure(self.project_root)
            
            # Combine quality and compliance issues
            report.quality_issues = quality_issues + compliance_issues
            
            # Count files scanned
            python_files = list(self.project_root.rglob("*.py"))
            report.total_files_scanned = len([f for f in python_files 
                                            if not self._should_skip_file(f)])
            
            # Calculate scores
            report.security_score = self.security_validator.calculate_security_score(
                security_issues, report.total_files_scanned)
            report.quality_score = self.quality_validator.calculate_quality_score(
                report.quality_issues, report.total_files_scanned)
            
            # Count checks
            report.passed_checks = report.total_files_scanned
            report.failed_checks = len(security_issues) + len(report.quality_issues)
            
            report.scan_duration_seconds = time.time() - start_time
            
            self.logger.info(f"Quality gate validation complete in {report.scan_duration_seconds:.2f}s")
            
            return report
            
        except Exception as e:
            self.logger.error("Quality gate validation failed", exception=e)
            report.scan_duration_seconds = time.time() - start_time
            return report
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        return QualityValidator._should_skip_file_static(file_path)
    
    def generate_report(self, report: ValidationReport) -> str:
        """Generate human-readable report."""
        lines = []
        lines.append("🛡️  SECURITY AND QUALITY GATE VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"📅 Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"⏱️  Scan Duration: {report.scan_duration_seconds:.2f} seconds")
        lines.append(f"📁 Files Scanned: {report.total_files_scanned}")
        lines.append("")
        
        # Overall summary
        status = "✅ PASSED" if report.passed else "❌ FAILED"
        lines.append(f"🎯 OVERALL STATUS: {status}")
        lines.append(f"📊 Overall Score: {report.overall_score:.1f}/100")
        lines.append("")
        
        # Security results
        lines.append(f"🔒 SECURITY SCORE: {report.security_score:.1f}/100")
        if report.security_issues:
            lines.append(f"⚠️  Security Issues Found: {len(report.security_issues)}")
            
            # Group by severity
            critical = [i for i in report.security_issues if i.severity == SecurityLevel.CRITICAL]
            high = [i for i in report.security_issues if i.severity == SecurityLevel.HIGH]
            medium = [i for i in report.security_issues if i.severity == SecurityLevel.MEDIUM]
            low = [i for i in report.security_issues if i.severity == SecurityLevel.LOW]
            
            if critical:
                lines.append(f"   🚨 Critical: {len(critical)}")
            if high:
                lines.append(f"   🔴 High: {len(high)}")
            if medium:
                lines.append(f"   🟡 Medium: {len(medium)}")
            if low:
                lines.append(f"   🟢 Low: {len(low)}")
        else:
            lines.append("✅ No security issues found!")
        lines.append("")
        
        # Quality results
        lines.append(f"⚡ QUALITY SCORE: {report.quality_score:.1f}/100")
        if report.quality_issues:
            lines.append(f"⚠️  Quality Issues Found: {len(report.quality_issues)}")
        else:
            lines.append("✅ No quality issues found!")
        lines.append("")
        
        # Detailed issues (top 10 most critical)
        all_issues = []
        for issue in report.security_issues:
            all_issues.append(("SECURITY", issue.severity.value, issue.description, 
                             issue.file_path, issue.line_number))
        
        for issue in report.quality_issues:
            all_issues.append(("QUALITY", issue.severity.value, issue.description,
                             issue.file_path, issue.line_number))
        
        if all_issues:
            lines.append("🔍 TOP ISSUES TO ADDRESS:")
            lines.append("-" * 40)
            
            # Sort by severity and show top 10
            severity_order = {"critical": 0, "high": 1, "fail": 2, "poor": 3, "medium": 4, "fair": 5}
            sorted_issues = sorted(all_issues, key=lambda x: severity_order.get(x[1], 10))
            
            for i, (category, severity, description, file_path, line_num) in enumerate(sorted_issues[:10], 1):
                file_info = f"{Path(file_path).name}:{line_num}" if file_path and line_num else "N/A"
                lines.append(f"{i:2d}. [{category}] {description}")
                lines.append(f"     📍 {file_info}")
                lines.append("")
        
        # Recommendations
        lines.append("💡 RECOMMENDATIONS:")
        lines.append("-" * 20)
        
        if report.security_score < 80:
            lines.append("• Review and fix security issues before production deployment")
        if report.quality_score < 70:
            lines.append("• Improve code quality to meet standards")
        if report.passed:
            lines.append("• ✅ All quality gates passed! Ready for production")
        else:
            lines.append("• ❌ Address critical issues before proceeding")
        
        lines.append("")
        lines.append("🏁 Quality gate validation complete")
        
        return "\n".join(lines)
    
    async def save_report(self, report: ValidationReport, output_file: Path = None) -> Path:
        """Save validation report to file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.project_root / f"quality_gate_report_{timestamp}.json"
        
        # Prepare data for JSON serialization
        report_data = {
            'timestamp': report.timestamp.isoformat(),
            'security_score': report.security_score,
            'quality_score': report.quality_score,
            'overall_score': report.overall_score,
            'passed': report.passed,
            'total_files_scanned': report.total_files_scanned,
            'scan_duration_seconds': report.scan_duration_seconds,
            'security_issues': [
                {
                    'severity': issue.severity.value,
                    'category': issue.category,
                    'description': issue.description,
                    'file_path': issue.file_path,
                    'line_number': issue.line_number,
                    'recommendation': issue.recommendation
                }
                for issue in report.security_issues
            ],
            'quality_issues': [
                {
                    'severity': issue.severity.value,
                    'category': issue.category,
                    'description': issue.description,
                    'file_path': issue.file_path,
                    'line_number': issue.line_number,
                    'fix_suggestion': issue.fix_suggestion
                }
                for issue in report.quality_issues
            ]
        }
        
        output_file.write_text(json.dumps(report_data, indent=2))
        self.logger.info(f"Report saved to: {output_file}")
        
        return output_file


async def main():
    """Main function to run quality gate validations."""
    print("🎯 GAN CYBER RANGE - SECURITY & QUALITY GATE VALIDATION")
    print("=" * 65)
    
    # Initialize quality gate runner
    project_root = Path.cwd()
    runner = QualityGateRunner(project_root)
    
    # Run all validations
    report = await runner.run_all_validations()
    
    # Generate and display report
    report_text = runner.generate_report(report)
    print(report_text)
    
    # Save report
    report_file = await runner.save_report(report)
    print(f"\n💾 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    exit_code = 0 if report.passed else 1
    print(f"\n🚪 Exiting with code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)