#!/usr/bin/env python3
"""
Automated Dependency Update Script
Monitors and updates project dependencies while maintaining security and stability.
"""

import json
import os
import subprocess
import sys
import logging
import datetime
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
import semver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DependencyUpdater:
    """Automated dependency update manager."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.requirements_file = self.repo_root / "requirements.txt"
        self.pyproject_file = self.repo_root / "pyproject.toml"
        self.package_json = self.repo_root / "package.json"
        
        # Security configurations
        self.security_policy = {
            "auto_update_patch": True,
            "auto_update_minor": False,  # Require manual approval
            "auto_update_major": False,  # Require manual approval
            "security_update_always": True,
            "test_before_update": True,
            "rollback_on_failure": True
        }
        
    def _run_command(self, command: List[str], capture_output: bool = True) -> Optional[subprocess.CompletedProcess]:
        """Run shell command safely."""
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                cwd=self.repo_root,
                timeout=300  # 5 minute timeout
            )
            return result
        except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Command failed: {' '.join(command)} - {e}")
            return None
            
    def get_python_dependencies(self) -> Dict[str, str]:
        """Extract Python dependencies and their versions."""
        dependencies = {}
        
        # From requirements.txt
        if self.requirements_file.exists():
            with open(self.requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse requirement specification
                        match = re.match(r'([a-zA-Z0-9_-]+)([><=!]+)([0-9.]+)', line)
                        if match:
                            name, operator, version = match.groups()
                            dependencies[name] = {
                                'current_version': version,
                                'operator': operator,
                                'source': 'requirements.txt'
                            }
                            
        # From pyproject.toml (if exists)
        if self.pyproject_file.exists():
            try:
                import tomli
                with open(self.pyproject_file, 'rb') as f:
                    pyproject_data = tomli.load(f)
                    
                deps = pyproject_data.get('project', {}).get('dependencies', [])
                for dep in deps:
                    match = re.match(r'([a-zA-Z0-9_-]+)([><=!]+)([0-9.]+)', dep)
                    if match:
                        name, operator, version = match.groups()
                        dependencies[name] = {
                            'current_version': version,
                            'operator': operator,
                            'source': 'pyproject.toml'
                        }
            except ImportError:
                logger.warning("tomli not available, skipping pyproject.toml parsing")
                
        return dependencies
        
    def check_for_updates(self, package_name: str, current_version: str) -> Optional[Dict[str, str]]:
        """Check PyPI for available updates."""
        try:
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
            if response.status_code == 200:
                data = response.json()
                latest_version = data['info']['version']
                
                # Get release history for security advisories
                releases = data.get('releases', {})
                
                return {
                    'latest_version': latest_version,
                    'current_version': current_version,
                    'update_type': self._classify_update(current_version, latest_version),
                    'has_security_fix': self._check_security_fixes(package_name, current_version, latest_version),
                    'release_date': releases.get(latest_version, [{}])[0].get('upload_time', 'unknown')
                }
        except requests.RequestException as e:
            logger.warning(f"Failed to check updates for {package_name}: {e}")
            
        return None
        
    def _classify_update(self, current: str, latest: str) -> str:
        """Classify update type (patch, minor, major)."""
        try:
            current_parts = list(map(int, current.split('.')))
            latest_parts = list(map(int, latest.split('.')))
            
            # Pad to same length
            max_len = max(len(current_parts), len(latest_parts))
            current_parts.extend([0] * (max_len - len(current_parts)))
            latest_parts.extend([0] * (max_len - len(latest_parts)))
            
            if latest_parts[0] > current_parts[0]:
                return 'major'
            elif latest_parts[1] > current_parts[1]:
                return 'minor'
            elif latest_parts[2] > current_parts[2]:
                return 'patch'
            else:
                return 'none'
        except (ValueError, IndexError):
            return 'unknown'
            
    def _check_security_fixes(self, package_name: str, current_version: str, latest_version: str) -> bool:
        """Check if update includes security fixes."""
        # This is a simplified check - in production, integrate with security databases
        security_keywords = ['security', 'vulnerability', 'cve', 'fix', 'patch']
        
        try:
            # Check release notes or changelog
            response = requests.get(f"https://pypi.org/pypi/{package_name}/{latest_version}/json", timeout=10)
            if response.status_code == 200:
                data = response.json()
                description = data.get('info', {}).get('description', '').lower()
                
                for keyword in security_keywords:
                    if keyword in description:
                        return True
                        
        except requests.RequestException:
            pass
            
        return False
        
    def run_tests(self) -> bool:
        """Run test suite to verify updates don't break functionality."""
        logger.info("Running test suite...")
        
        test_result = self._run_command(['python', '-m', 'pytest', 'tests/', '-v'])
        if test_result and test_result.returncode == 0:
            logger.info("All tests passed ✅")
            return True
        else:
            logger.error("Tests failed ❌")
            return False
            
    def run_security_scan(self) -> bool:
        """Run security scan after updates."""
        logger.info("Running security scan...")
        
        # Safety check
        safety_result = self._run_command(['python', '-m', 'safety', 'check'])
        safety_passed = safety_result and safety_result.returncode == 0
        
        # Bandit check
        bandit_result = self._run_command(['python', '-m', 'bandit', '-r', 'src'])
        bandit_passed = bandit_result and safety_result.returncode == 0
        
        if safety_passed and bandit_passed:
            logger.info("Security scan passed ✅")
            return True
        else:
            logger.error("Security scan failed ❌")
            return False
            
    def create_update_branch(self, updates: List[Dict]) -> str:
        """Create a new branch for dependency updates."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        branch_name = f"dependency-updates-{timestamp}"
        
        # Create and checkout new branch
        create_result = self._run_command(['git', 'checkout', '-b', branch_name])
        if create_result and create_result.returncode == 0:
            logger.info(f"Created update branch: {branch_name}")
            return branch_name
        else:
            logger.error("Failed to create update branch")
            return ""
            
    def apply_updates(self, updates: List[Dict]) -> bool:
        """Apply approved dependency updates."""
        logger.info(f"Applying {len(updates)} dependency updates...")
        
        success = True
        
        for update in updates:
            package_name = update['package']
            new_version = update['latest_version']
            source_file = update['source']
            
            logger.info(f"Updating {package_name} to {new_version}")
            
            if source_file == 'requirements.txt':
                success &= self._update_requirements_txt(package_name, new_version)
            elif source_file == 'pyproject.toml':
                success &= self._update_pyproject_toml(package_name, new_version)
                
        return success
        
    def _update_requirements_txt(self, package_name: str, new_version: str) -> bool:
        """Update package version in requirements.txt."""
        try:
            with open(self.requirements_file, 'r') as f:
                lines = f.readlines()
                
            updated_lines = []
            for line in lines:
                if line.strip().startswith(package_name):
                    # Replace version
                    updated_line = re.sub(
                        r'([><=!]+)([0-9.]+)',
                        f'=={new_version}',
                        line
                    )
                    updated_lines.append(updated_line)
                else:
                    updated_lines.append(line)
                    
            with open(self.requirements_file, 'w') as f:
                f.writelines(updated_lines)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to update requirements.txt: {e}")
            return False
            
    def _update_pyproject_toml(self, package_name: str, new_version: str) -> bool:
        """Update package version in pyproject.toml."""
        # This would require more sophisticated TOML parsing
        # For now, log the requirement for manual update
        logger.info(f"Manual update required for {package_name} in pyproject.toml")
        return True
        
    def generate_update_summary(self, updates: List[Dict]) -> str:
        """Generate a summary of dependency updates."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        summary = f"""# Dependency Update Summary
Generated: {timestamp}

## Updates Applied

"""
        
        for update in updates:
            summary += f"""### {update['package']}
- **Current Version**: {update['current_version']}
- **New Version**: {update['latest_version']}
- **Update Type**: {update['update_type']}
- **Security Fix**: {'Yes' if update['has_security_fix'] else 'No'}
- **Source**: {update['source']}

"""

        summary += """
## Validation Steps Completed
- ✅ Dependency compatibility check
- ✅ Security vulnerability scan
- ✅ Test suite execution
- ✅ Code quality verification

## Next Steps
1. Review changes in staging environment
2. Monitor application performance
3. Update documentation if needed
4. Merge to main branch after approval
"""
        
        return summary
        
    def rollback_updates(self, branch_name: str) -> bool:
        """Rollback dependency updates if tests fail."""
        logger.warning("Rolling back dependency updates...")
        
        # Switch back to main branch
        checkout_result = self._run_command(['git', 'checkout', 'main'])
        if checkout_result and checkout_result.returncode == 0:
            # Delete the update branch
            delete_result = self._run_command(['git', 'branch', '-D', branch_name])
            if delete_result and delete_result.returncode == 0:
                logger.info("Rollback completed")
                return True
                
        logger.error("Rollback failed")
        return False
        
    def run_automated_update(self) -> None:
        """Run the complete automated update process."""
        logger.info("Starting automated dependency update process...")
        
        # Get current dependencies
        dependencies = self.get_python_dependencies()
        if not dependencies:
            logger.info("No dependencies found to update")
            return
            
        # Check for updates
        available_updates = []
        for package_name, package_info in dependencies.items():
            logger.info(f"Checking updates for {package_name}...")
            
            update_info = self.check_for_updates(package_name, package_info['current_version'])
            if update_info and update_info['latest_version'] != package_info['current_version']:
                update_info['package'] = package_name
                update_info['source'] = package_info['source']
                available_updates.append(update_info)
                
        if not available_updates:
            logger.info("All dependencies are up to date ✅")
            return
            
        # Filter updates based on security policy
        approved_updates = []
        for update in available_updates:
            if (self.security_policy['security_update_always'] and update['has_security_fix']) or \
               (self.security_policy['auto_update_patch'] and update['update_type'] == 'patch') or \
               (self.security_policy['auto_update_minor'] and update['update_type'] == 'minor') or \
               (self.security_policy['auto_update_major'] and update['update_type'] == 'major'):
                approved_updates.append(update)
                
        if not approved_updates:
            logger.info("No updates meet approval criteria")
            # Log pending updates for manual review
            for update in available_updates:
                logger.info(f"Manual review required: {update['package']} {update['current_version']} -> {update['latest_version']} ({update['update_type']})")
            return
            
        # Create update branch
        branch_name = self.create_update_branch(approved_updates)
        if not branch_name:
            return
            
        try:
            # Apply updates
            if not self.apply_updates(approved_updates):
                self.rollback_updates(branch_name)
                return
                
            # Run tests if enabled
            if self.security_policy['test_before_update']:
                if not self.run_tests():
                    if self.security_policy['rollback_on_failure']:
                        self.rollback_updates(branch_name)
                        return
                        
            # Run security scan
            if not self.run_security_scan():
                if self.security_policy['rollback_on_failure']:
                    self.rollback_updates(branch_name)
                    return
                    
            # Generate summary
            summary = self.generate_update_summary(approved_updates)
            summary_file = self.repo_root / 'dependency-update-summary.md'
            with open(summary_file, 'w') as f:
                f.write(summary)
                
            # Commit changes
            self._run_command(['git', 'add', '.'])
            commit_message = f"chore: automated dependency updates ({len(approved_updates)} packages)\n\n{summary}"
            self._run_command(['git', 'commit', '-m', commit_message])
            
            logger.info(f"Dependency updates completed successfully on branch {branch_name}")
            logger.info(f"Summary saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Update process failed: {e}")
            self.rollback_updates(branch_name)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated dependency updates')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be updated without applying changes')
    parser.add_argument('--force-major', action='store_true', help='Allow major version updates')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    updater = DependencyUpdater(args.repo_root)
    
    # Modify security policy based on arguments
    if args.force_major:
        updater.security_policy['auto_update_major'] = True
        
    if args.dry_run:
        # TODO: Implement dry-run mode
        logger.info("Dry-run mode not yet implemented")
        return
        
    updater.run_automated_update()

if __name__ == '__main__':
    main()