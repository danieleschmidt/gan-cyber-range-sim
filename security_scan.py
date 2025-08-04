#!/usr/bin/env python3
"""
Security scanning for GAN Cyber Range Simulator.
Checks for common security issues and vulnerabilities.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple


class SecurityScanner:
    """Security scanner for Python code."""
    
    def __init__(self):
        # Dangerous patterns to look for
        self.dangerous_patterns = {
            'hardcoded_secrets': [
                re.compile(r'password\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
                re.compile(r'api_key\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
                re.compile(r'secret\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
                re.compile(r'token\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
            ],
            'command_injection': [
                re.compile(r'os\.system\s*\(', re.IGNORECASE),
                re.compile(r'subprocess\.call\s*\(.*shell\s*=\s*True', re.IGNORECASE),
                re.compile(r'subprocess\.run\s*\(.*shell\s*=\s*True', re.IGNORECASE),
                re.compile(r'eval\s*\(', re.IGNORECASE),
                re.compile(r'exec\s*\(', re.IGNORECASE),
            ],
            'path_traversal': [
                re.compile(r'\.\./', re.IGNORECASE),
                re.compile(r'\.\.\\\\', re.IGNORECASE),
            ],
            'sql_injection_risk': [
                re.compile(r'f["\'].*SELECT.*{', re.IGNORECASE),
                re.compile(r'%.*SELECT', re.IGNORECASE),
                re.compile(r'\+.*SELECT', re.IGNORECASE),
            ],
            'weak_crypto': [
                re.compile(r'md5\s*\(', re.IGNORECASE),
                re.compile(r'sha1\s*\(', re.IGNORECASE),
                re.compile(r'random\.random\s*\(', re.IGNORECASE),
            ],
            'insecure_defaults': [
                re.compile(r'debug\s*=\s*True', re.IGNORECASE),
                re.compile(r'verify\s*=\s*False', re.IGNORECASE),
                re.compile(r'ssl_verify\s*=\s*False', re.IGNORECASE),
            ]
        }
        
        # Known secure patterns (whitelist)
        self.secure_patterns = [
            re.compile(r'hashlib\.pbkdf2_hmac'),  # Good password hashing
            re.compile(r'secrets\.token_'),       # Secure random generation
            re.compile(r'bcrypt\.'),              # Good password hashing
            re.compile(r'cryptography\.'),        # Good crypto library
        ]
    
    def scan_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan a single file for security issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                line_issues = self._scan_line(line, line_num, file_path)
                issues.extend(line_issues)
        
        except Exception as e:
            issues.append({
                'type': 'scan_error',
                'severity': 'low',
                'file': str(file_path),
                'line': 0,
                'message': f'Could not scan file: {e}'
            })
        
        return issues
    
    def _scan_line(self, line: str, line_num: int, file_path: Path) -> List[Dict[str, Any]]:
        """Scan a single line for security issues."""
        issues = []
        
        # Skip comments and docstrings for some checks
        stripped_line = line.strip()
        if stripped_line.startswith('#') or stripped_line.startswith('"""') or stripped_line.startswith("'''"):
            return issues
        
        # Check for dangerous patterns
        for category, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                if pattern.search(line):
                    # Check if it's a whitelisted pattern
                    is_whitelisted = any(secure.search(line) for secure in self.secure_patterns)
                    if not is_whitelisted:
                        severity = self._get_severity(category)
                        issues.append({
                            'type': category,
                            'severity': severity,
                            'file': str(file_path),
                            'line': line_num,
                            'message': f'{category.replace("_", " ").title()} detected',
                            'code': line.strip()
                        })
        
        return issues
    
    def _get_severity(self, category: str) -> str:
        """Get severity level for a security issue category."""
        severity_map = {
            'hardcoded_secrets': 'critical',
            'command_injection': 'critical',
            'path_traversal': 'high',
            'sql_injection_risk': 'high',
            'weak_crypto': 'medium',
            'insecure_defaults': 'medium'
        }
        return severity_map.get(category, 'low')
    
    def scan_directory(self, directory: Path) -> Dict[str, Any]:
        """Scan directory for security issues."""
        all_issues = []
        files_scanned = 0
        
        # Get all Python files
        python_files = list(directory.rglob('*.py'))
        
        for file_path in python_files:
            # Skip test files and virtual environments
            if any(skip in str(file_path) for skip in ['test_', '__pycache__', '.pyc', 'venv/', '.venv/']):
                continue
            
            file_issues = self.scan_file(file_path)
            all_issues.extend(file_issues)
            files_scanned += 1
        
        # Categorize issues by severity
        issues_by_severity = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
        
        for issue in all_issues:
            severity = issue.get('severity', 'low')
            issues_by_severity[severity].append(issue)
        
        return {
            'files_scanned': files_scanned,
            'total_issues': len(all_issues),
            'issues_by_severity': issues_by_severity,
            'all_issues': all_issues
        }


def check_file_permissions(directory: Path) -> List[Dict[str, Any]]:
    """Check file permissions for security issues."""
    issues = []
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            # Check for world-writable files
            stat = file_path.stat()
            mode = stat.st_mode
            
            if mode & 0o002:  # World writable
                issues.append({
                    'type': 'world_writable_file',
                    'severity': 'high',
                    'file': str(file_path),
                    'message': 'File is world-writable',
                    'permissions': oct(mode)[-3:]
                })
            
            # Check for executable files that shouldn't be
            if file_path.suffix in ['.py', '.txt', '.md', '.json', '.yaml', '.yml']:
                if mode & 0o111:  # Any execute bit
                    issues.append({
                        'type': 'unnecessary_executable',
                        'severity': 'low',
                        'file': str(file_path),
                        'message': 'File has unnecessary execute permissions',
                        'permissions': oct(mode)[-3:]
                    })
    
    return issues


def check_dependencies() -> List[Dict[str, Any]]:
    """Check for known vulnerable dependencies."""
    issues = []
    
    # Check if requirements.txt exists
    req_files = ['requirements.txt', 'pyproject.toml']
    
    for req_file in req_files:
        req_path = Path(req_file)
        if req_path.exists():
            try:
                content = req_path.read_text()
                
                # Known vulnerable patterns (simplified)
                vulnerable_patterns = [
                    (re.compile(r'django\s*<\s*3\.0', re.IGNORECASE), 'Django < 3.0 has known vulnerabilities'),
                    (re.compile(r'flask\s*<\s*1\.0', re.IGNORECASE), 'Flask < 1.0 has known vulnerabilities'),
                    (re.compile(r'requests\s*<\s*2\.20', re.IGNORECASE), 'Requests < 2.20 has known vulnerabilities'),
                ]
                
                for pattern, message in vulnerable_patterns:
                    if pattern.search(content):
                        issues.append({
                            'type': 'vulnerable_dependency',
                            'severity': 'high',
                            'file': req_file,
                            'message': message
                        })
            
            except Exception as e:
                issues.append({
                    'type': 'dependency_check_error',
                    'severity': 'low',
                    'file': req_file,
                    'message': f'Could not check dependencies: {e}'
                })
    
    return issues


def main():
    """Run security scan."""
    print("🔒 Starting Security Scan for GAN Cyber Range Simulator")
    print("=" * 60)
    
    # Get project directory
    project_dir = Path(__file__).parent
    src_dir = project_dir / 'src'
    
    scanner = SecurityScanner()
    
    # Scan source code
    print("🔍 Scanning source code for security issues...")
    code_results = scanner.scan_directory(src_dir)
    
    print(f"   Scanned {code_results['files_scanned']} files")
    print(f"   Found {code_results['total_issues']} potential issues")
    
    # Check file permissions
    print("\n🔍 Checking file permissions...")
    permission_issues = check_file_permissions(project_dir)
    print(f"   Found {len(permission_issues)} permission issues")
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    dependency_issues = check_dependencies()
    print(f"   Found {len(dependency_issues)} dependency issues")
    
    # Combine all issues
    all_issues = (code_results['all_issues'] + 
                  permission_issues + 
                  dependency_issues)
    
    # Report results
    print("\n" + "=" * 60)
    print("📊 Security Scan Results")
    print("=" * 60)
    
    if not all_issues:
        print("🎉 No security issues found!")
        return 0
    
    # Group by severity
    by_severity = {'critical': [], 'high': [], 'medium': [], 'low': []}
    for issue in all_issues:
        severity = issue.get('severity', 'low')
        by_severity[severity].append(issue)
    
    # Report by severity
    for severity in ['critical', 'high', 'medium', 'low']:
        issues = by_severity[severity]
        if issues:
            emoji = {'critical': '🚨', 'high': '⚠️', 'medium': '⚡', 'low': 'ℹ️'}[severity]
            print(f"\n{emoji} {severity.upper()} Issues ({len(issues)}):")
            
            for issue in issues[:5]:  # Show first 5 issues
                print(f"   📍 {issue['file']}:{issue.get('line', '?')}")
                print(f"      {issue['message']}")
                if 'code' in issue:
                    print(f"      Code: {issue['code']}")
            
            if len(issues) > 5:
                print(f"   ... and {len(issues) - 5} more {severity} issues")
    
    # Summary
    critical_count = len(by_severity['critical'])
    high_count = len(by_severity['high'])
    
    print(f"\n📈 Summary:")
    print(f"   Critical: {critical_count}")
    print(f"   High: {high_count}")
    print(f"   Medium: {len(by_severity['medium'])}")
    print(f"   Low: {len(by_severity['low'])}")
    
    if critical_count > 0:
        print("\n🚨 CRITICAL issues found - immediate attention required!")
        return 2
    elif high_count > 0:
        print("\n⚠️  HIGH priority issues found - should be addressed soon")
        return 1
    else:
        print("\n✅ No critical or high priority security issues found")
        return 0


if __name__ == "__main__":
    sys.exit(main())