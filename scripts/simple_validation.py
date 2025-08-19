#!/usr/bin/env python3
"""Simple validation script for quality gates structure."""

import sys
from pathlib import Path

def validate_file_structure():
    """Validate that all required files exist."""
    print("🔍 Validating file structure...")
    
    required_files = [
        "src/gan_cyber_range/quality/__init__.py",
        "src/gan_cyber_range/quality/quality_gates.py",
        "src/gan_cyber_range/quality/progressive_validator.py",
        "src/gan_cyber_range/quality/automated_pipeline.py",
        "src/gan_cyber_range/quality/monitoring.py",
        "src/gan_cyber_range/quality/validation_framework.py",
        "src/gan_cyber_range/quality/intelligent_optimizer.py",
        "src/gan_cyber_range/quality/auto_scaler.py",
        "scripts/run_quality_pipeline.py",
        "docs/workflows/quality-gates-setup.md",
        "quality_config.yaml",
        "tests/quality/test_quality_gates.py",
        "tests/quality/test_progressive_validator.py",
        "tests/quality/test_monitoring.py",
        "tests/quality/test_integration.py"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            existing_files.append(file_path)
            print(f"  ✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  ❌ {file_path}")
    
    print(f"\nSummary: {len(existing_files)}/{len(required_files)} files exist")
    
    if missing_files:
        print("Missing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    return True


def validate_code_structure():
    """Validate basic code structure."""
    print("\n🔍 Validating code structure...")
    
    try:
        # Check quality gates module
        quality_gates_file = Path("src/gan_cyber_range/quality/quality_gates.py")
        if quality_gates_file.exists():
            content = quality_gates_file.read_text()
            
            required_classes = [
                "class QualityGate",
                "class QualityGateResult", 
                "class QualityGateRunner",
                "class TestCoverageGate",
                "class SecurityGate",
                "class CodeQualityGate",
                "class PerformanceGate",
                "class ComplianceGate"
            ]
            
            for class_name in required_classes:
                if class_name in content:
                    print(f"  ✅ {class_name}")
                else:
                    print(f"  ❌ {class_name}")
                    return False
        
        # Check progressive validator
        validator_file = Path("src/gan_cyber_range/quality/progressive_validator.py")
        if validator_file.exists():
            content = validator_file.read_text()
            
            required_components = [
                "class ProgressiveValidator",
                "class ValidationStage",
                "class ValidationResult",
                "class ValidationMetrics"
            ]
            
            for component in required_components:
                if component in content:
                    print(f"  ✅ {component}")
                else:
                    print(f"  ❌ {component}")
                    return False
        
        # Check monitoring components
        monitoring_file = Path("src/gan_cyber_range/quality/monitoring.py")
        if monitoring_file.exists():
            content = monitoring_file.read_text()
            
            required_components = [
                "class QualityMetric",
                "class MetricsCollector", 
                "class QualityMonitor",
                "class QualityDashboard",
                "class QualityTrendAnalyzer"
            ]
            
            for component in required_components:
                if component in content:
                    print(f"  ✅ {component}")
                else:
                    print(f"  ❌ {component}")
                    return False
        
        print("✅ Code structure validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Code structure validation failed: {e}")
        return False


def validate_configuration():
    """Validate configuration files."""
    print("\n🔍 Validating configuration...")
    
    try:
        # Check quality config
        config_file = Path("quality_config.yaml")
        if config_file.exists():
            content = config_file.read_text()
            
            required_sections = [
                "pipeline:",
                "quality_thresholds:",
                "gates:",
                "deployment_criteria:",
                "auto_fix:",
                "notifications:",
                "reporting:"
            ]
            
            for section in required_sections:
                if section in content:
                    print(f"  ✅ {section}")
                else:
                    print(f"  ❌ {section}")
                    return False
        
        # Check GitHub workflow setup documentation
        workflow_doc = Path("docs/workflows/quality-gates-setup.md")
        if workflow_doc.exists():
            content = workflow_doc.read_text()
            
            required_sections = [
                "progressive-validation:",
                "security-scan:",
                "deployment-readiness:",
                "notify-results:"
            ]
            
            for section in required_sections:
                if section in content:
                    print(f"  ✅ {section}")
                else:
                    print(f"  ❌ {section}")
                    return False
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def validate_tests():
    """Validate test structure."""
    print("\n🔍 Validating tests...")
    
    try:
        test_files = [
            "tests/quality/test_quality_gates.py",
            "tests/quality/test_progressive_validator.py", 
            "tests/quality/test_monitoring.py",
            "tests/quality/test_integration.py"
        ]
        
        for test_file in test_files:
            file_path = Path(test_file)
            if file_path.exists():
                content = file_path.read_text()
                
                # Check for pytest structure
                if "import pytest" in content and "def test_" in content:
                    print(f"  ✅ {test_file}")
                else:
                    print(f"  ⚠️ {test_file} (missing pytest structure)")
            else:
                print(f"  ❌ {test_file}")
                return False
        
        print("✅ Test validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Test validation failed: {e}")
        return False


def validate_documentation():
    """Validate documentation completeness."""
    print("\n🔍 Validating documentation...")
    
    try:
        # Check for docstrings in main modules
        modules_to_check = [
            "src/gan_cyber_range/quality/quality_gates.py",
            "src/gan_cyber_range/quality/progressive_validator.py",
            "src/gan_cyber_range/quality/automated_pipeline.py"
        ]
        
        for module_path in modules_to_check:
            file_path = Path(module_path)
            if file_path.exists():
                content = file_path.read_text()
                
                # Check for module docstring
                if '"""' in content:
                    print(f"  ✅ {module_path}")
                else:
                    print(f"  ⚠️ {module_path} (missing docstrings)")
            else:
                print(f"  ❌ {module_path}")
        
        print("✅ Documentation validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Documentation validation failed: {e}")
        return False


def main():
    """Run all validations."""
    print("🚀 Quality Gates Implementation Validation")
    print("=" * 50)
    
    validations = [
        ("File Structure", validate_file_structure),
        ("Code Structure", validate_code_structure),
        ("Configuration", validate_configuration),
        ("Tests", validate_tests),
        ("Documentation", validate_documentation)
    ]
    
    results = []
    
    for name, validation_func in validations:
        try:
            result = validation_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} validation crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:<20} {status}")
        if result:
            passed += 1
    
    total = len(results)
    print("-" * 50)
    print(f"Total: {passed}/{total} validations passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("Quality Gates implementation structure is complete and ready for testing.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} VALIDATIONS FAILED")
        print("Review and fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())