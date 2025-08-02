#!/usr/bin/env python3
"""
Security Policy Validation Script for GAN Cyber Range Simulator

This script validates security policies and configurations to ensure
defensive security practices are maintained throughout the codebase.
"""

import os
import sys
import yaml
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple


class SecurityPolicyValidator:
    """Validates security policies and configurations."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate_all(self) -> bool:
        """Run all security validations."""
        print("🔒 Starting security policy validation...")
        
        # Core validation functions
        self.validate_container_security()
        self.validate_network_policies()
        self.validate_secret_patterns()
        self.validate_deployment_configs()
        self.validate_isolation_policies()
        
        # Report results
        self.report_results()
        
        return len(self.errors) == 0
    
    def validate_container_security(self):
        """Validate Docker container security configurations."""
        dockerfile_path = self.repo_root / "Dockerfile"
        if not dockerfile_path.exists():
            self.warnings.append("No Dockerfile found for container security validation")
            return
            
        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()
                
            # Check for security best practices
            if "USER root" in content and "USER " not in content.split("USER root")[1]:
                self.errors.append("Dockerfile runs as root without switching to non-root user")
                
            if "--no-cache-dir" not in content:
                self.warnings.append("Consider using --no-cache-dir in pip installs")
                
            if "ADD" in content and "COPY" not in content:
                self.warnings.append("Consider using COPY instead of ADD for security")
                
        except Exception as e:
            self.errors.append(f"Failed to validate Dockerfile: {e}")
    
    def validate_network_policies(self):
        """Validate network isolation policies."""
        config_dirs = ["config", "deployments", "k8s"]
        
        for config_dir in config_dirs:
            config_path = self.repo_root / config_dir
            if not config_path.exists():
                continue
                
            for yaml_file in config_path.glob("**/*.yaml"):
                try:
                    with open(yaml_file, 'r') as f:
                        doc = yaml.safe_load(f)
                        
                    if isinstance(doc, dict) and doc.get("kind") == "NetworkPolicy":
                        self.validate_network_policy_spec(doc, yaml_file)
                        
                except Exception as e:
                    self.warnings.append(f"Could not parse {yaml_file}: {e}")
    
    def validate_network_policy_spec(self, policy: Dict[str, Any], filepath: Path):
        """Validate specific network policy configuration."""
        spec = policy.get("spec", {})
        
        # Check for proper isolation
        if not spec.get("policyTypes"):
            self.warnings.append(f"{filepath}: NetworkPolicy missing policyTypes specification")
            
        # Ensure egress is restricted
        egress = spec.get("egress", [])
        if not egress:
            self.warnings.append(f"{filepath}: No egress rules defined - may be too restrictive")
        else:
            for rule in egress:
                if not rule.get("to"):
                    self.errors.append(f"{filepath}: Egress rule allows all destinations")
    
    def validate_secret_patterns(self):
        """Validate that no secrets are hardcoded in configuration files."""
        secret_patterns = [
            r'password\s*[=:]\s*["\']?[^"\'\s]+["\']?',
            r'api[_-]key\s*[=:]\s*["\']?[^"\'\s]+["\']?',
            r'secret[_-]key\s*[=:]\s*["\']?[^"\'\s]+["\']?',
            r'token\s*[=:]\s*["\']?[^"\'\s]+["\']?',
            r'private[_-]key\s*[=:]\s*["\']?[^"\'\s]+["\']?',
        ]
        
        for pattern in secret_patterns:
            for config_file in self.repo_root.glob("**/*.yaml"):
                if self.is_excluded_path(config_file):
                    continue
                    
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                        
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        self.warnings.append(
                            f"{config_file}: Potential hardcoded secret pattern found"
                        )
                except Exception:
                    continue
    
    def validate_deployment_configs(self):
        """Validate deployment configurations for security."""
        docker_compose = self.repo_root / "docker-compose.yml"
        if docker_compose.exists():
            self.validate_docker_compose_security(docker_compose)
    
    def validate_docker_compose_security(self, compose_file: Path):
        """Validate Docker Compose security settings."""
        try:
            with open(compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
                
            services = compose_config.get("services", {})
            
            for service_name, service_config in services.items():
                # Check for privileged containers
                if service_config.get("privileged"):
                    self.errors.append(
                        f"Service {service_name} runs in privileged mode - security risk"
                    )
                
                # Check for host network mode
                if service_config.get("network_mode") == "host":
                    self.warnings.append(
                        f"Service {service_name} uses host networking - potential security risk"
                    )
                
                # Check for volume mounts
                volumes = service_config.get("volumes", [])
                for volume in volumes:
                    if isinstance(volume, str) and volume.startswith("/"):
                        if ":" in volume and volume.split(":")[0] in ["/", "/etc", "/usr"]:
                            self.errors.append(
                                f"Service {service_name} mounts sensitive host directory"
                            )
                            
        except Exception as e:
            self.errors.append(f"Failed to validate Docker Compose: {e}")
    
    def validate_isolation_policies(self):
        """Validate cyber range isolation policies."""
        # Check for environment variable validation
        env_files = list(self.repo_root.glob("**/.env*"))
        
        for env_file in env_files:
            if env_file.name in [".env", ".env.local", ".env.production"]:
                self.warnings.append(
                    f"Environment file {env_file} should not be committed to version control"
                )
    
    def is_excluded_path(self, path: Path) -> bool:
        """Check if path should be excluded from validation."""
        excluded_patterns = [
            "tests/",
            "docs/examples/",
            ".git/",
            "__pycache__/",
            "node_modules/",
        ]
        
        path_str = str(path)
        return any(pattern in path_str for pattern in excluded_patterns)
    
    def report_results(self):
        """Report validation results."""
        if self.errors:
            print("\n❌ Security Policy Violations:")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print("\n⚠️  Security Warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            print("✅ All security policy validations passed!")
        
        print(f"\nValidation Summary:")
        print(f"  Errors: {len(self.errors)}")
        print(f"  Warnings: {len(self.warnings)}")


def main():
    """Main entry point for security policy validation."""
    validator = SecurityPolicyValidator()
    
    if not validator.validate_all():
        print("\n🚨 Security policy validation failed!")
        sys.exit(1)
    else:
        print("\n🔒 Security policy validation completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()