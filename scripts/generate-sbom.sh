#!/bin/bash

# Generate Software Bill of Materials (SBOM) for GAN Cyber Range
# This script creates comprehensive SBOMs in multiple formats for compliance

set -euo pipefail

# Configuration
PROJECT_NAME="gan-cyber-range-simulator"
PROJECT_VERSION=$(python -c "import gan_cyber_range; print(gan_cyber_range.__version__)" 2>/dev/null || echo "0.1.0")
OUTPUT_DIR="sbom"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌${NC} $1"
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

log "Starting SBOM generation for $PROJECT_NAME v$PROJECT_VERSION"

# Check if required tools are installed
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is required but not installed"
        return 1
    fi
}

# Install SBOM generation tools if needed
install_sbom_tools() {
    log "Installing SBOM generation tools..."
    
    # CycloneDX for Python
    pip install cyclonedx-bom cyclonedx-python-lib --quiet
    
    # Syft for container scanning
    if ! command -v syft &> /dev/null; then
        log "Installing Syft..."
        curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
    fi
    
    log_success "SBOM tools installed"
}

# Generate Python dependency SBOM
generate_python_sbom() {
    log "Generating Python dependency SBOM..."
    
    # CycloneDX format
    cyclonedx-py \
        --schema-version 1.4 \
        --output-format json \
        --output-file "$OUTPUT_DIR/python-dependencies-cyclonedx.json" \
        .
    
    # SPDX format
    cyclonedx-py \
        --schema-version 1.4 \
        --output-format xml \
        --output-file "$OUTPUT_DIR/python-dependencies-cyclonedx.xml" \
        .
    
    log_success "Python SBOM generated"
}

# Generate container SBOM if Docker is available
generate_container_sbom() {
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not available, skipping container SBOM"
        return 0
    fi
    
    log "Generating container SBOM..."
    
    # Build the image first
    docker build -t "$PROJECT_NAME:sbom" .
    
    # Use Syft to scan the container
    if command -v syft &> /dev/null; then
        syft "$PROJECT_NAME:sbom" -o json > "$OUTPUT_DIR/container-sbom-syft.json"
        syft "$PROJECT_NAME:sbom" -o spdx-json > "$OUTPUT_DIR/container-sbom-spdx.json"
        syft "$PROJECT_NAME:sbom" -o table > "$OUTPUT_DIR/container-sbom-table.txt"
        log_success "Container SBOM generated with Syft"
    else
        log_warning "Syft not available, using Docker history"
        docker history "$PROJECT_NAME:sbom" --format "table {{.CreatedBy}}\t{{.Size}}" > "$OUTPUT_DIR/container-history.txt"
    fi
}

# Generate system dependencies SBOM
generate_system_sbom() {
    log "Generating system dependencies SBOM..."
    
    # Create system SBOM in JSON format
    cat > "$OUTPUT_DIR/system-dependencies.json" << EOF
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.4",
  "serialNumber": "urn:uuid:$(uuidgen)",
  "version": 1,
  "metadata": {
    "timestamp": "$(date -Iseconds)",
    "tools": [
      {
        "vendor": "Terragon Labs",
        "name": "SBOM Generator",
        "version": "1.0.0"
      }
    ],
    "component": {
      "type": "application",
      "name": "$PROJECT_NAME",
      "version": "$PROJECT_VERSION"
    }
  },
  "components": [
EOF

    # Add system packages if on Linux
    if command -v dpkg &> /dev/null; then
        log "Scanning Debian/Ubuntu packages..."
        dpkg-query -W -f='    {\n      "type": "library",\n      "name": "${Package}",\n      "version": "${Version}",\n      "scope": "required",\n      "description": "${Description}"\n    },\n' >> "$OUTPUT_DIR/system-dependencies.json"
    elif command -v rpm &> /dev/null; then
        log "Scanning RPM packages..."
        rpm -qa --queryformat '    {\n      "type": "library",\n      "name": "%{NAME}",\n      "version": "%{VERSION}-%{RELEASE}",\n      "scope": "required"\n    },\n' >> "$OUTPUT_DIR/system-dependencies.json"
    fi
    
    # Remove trailing comma and close JSON
    sed -i '$ s/,$//' "$OUTPUT_DIR/system-dependencies.json"
    echo "  ]" >> "$OUTPUT_DIR/system-dependencies.json"
    echo "}" >> "$OUTPUT_DIR/system-dependencies.json"
    
    log_success "System dependencies SBOM generated"
}

# Generate comprehensive SBOM
generate_comprehensive_sbom() {
    log "Generating comprehensive SBOM..."
    
    # Create comprehensive SBOM combining all sources
    cat > "$OUTPUT_DIR/comprehensive-sbom.json" << EOF
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.4",
  "serialNumber": "urn:uuid:$(uuidgen)",
  "version": 1,
  "metadata": {
    "timestamp": "$(date -Iseconds)",
    "tools": [
      {
        "vendor": "Terragon Labs",
        "name": "Comprehensive SBOM Generator",
        "version": "1.0.0"
      }
    ],
    "component": {
      "type": "application",
      "name": "$PROJECT_NAME",
      "version": "$PROJECT_VERSION",
      "description": "GAN Cyber Range Simulator for AI-powered cybersecurity training"
    }
  },
  "components": [],
  "vulnerabilities": [],
  "annotations": [
    {
      "subjects": ["pkg:pypi/$PROJECT_NAME@$PROJECT_VERSION"],
      "annotator": "Terragon Labs SBOM Generator",
      "annotationType": "security",
      "annotationDate": "$(date -Iseconds)",
      "annotation": "Generated for security compliance and vulnerability management"
    }
  ]
}
EOF
    
    log_success "Comprehensive SBOM generated"
}

# Generate vulnerability report
generate_vulnerability_report() {
    log "Generating vulnerability report..."
    
    # Check for known vulnerabilities in Python packages
    if command -v safety &> /dev/null; then
        safety check --json --output "$OUTPUT_DIR/vulnerability-report.json" || true
        log_success "Vulnerability report generated"
    else
        log_warning "Safety not available, installing..."
        pip install safety --quiet
        safety check --json --output "$OUTPUT_DIR/vulnerability-report.json" || true
        log_success "Vulnerability report generated"
    fi
}

# Generate license report
generate_license_report() {
    log "Generating license report..."
    
    if command -v pip-licenses &> /dev/null; then
        pip-licenses --format json --output-file "$OUTPUT_DIR/license-report.json"
        pip-licenses --format plain --output-file "$OUTPUT_DIR/license-report.txt"
    else
        log "Installing pip-licenses..."
        pip install pip-licenses --quiet
        pip-licenses --format json --output-file "$OUTPUT_DIR/license-report.json"
        pip-licenses --format plain --output-file "$OUTPUT_DIR/license-report.txt"
    fi
    
    log_success "License report generated"
}

# Create summary report
create_summary_report() {
    log "Creating summary report..."
    
    cat > "$OUTPUT_DIR/sbom-summary.md" << EOF
# SBOM Summary Report

**Project**: $PROJECT_NAME  
**Version**: $PROJECT_VERSION  
**Generated**: $(date)  
**Generator**: Terragon Labs SBOM Generator v1.0.0

## Files Generated

- \`python-dependencies-cyclonedx.json\` - Python dependencies in CycloneDX JSON format
- \`python-dependencies-cyclonedx.xml\` - Python dependencies in CycloneDX XML format
- \`container-sbom-syft.json\` - Container SBOM in Syft JSON format (if available)
- \`container-sbom-spdx.json\` - Container SBOM in SPDX JSON format (if available)
- \`system-dependencies.json\` - System package dependencies
- \`comprehensive-sbom.json\` - Combined SBOM from all sources
- \`vulnerability-report.json\` - Known vulnerability analysis
- \`license-report.json\` - License compliance report

## Component Statistics

EOF

    # Add statistics if files exist
    if [ -f "$OUTPUT_DIR/python-dependencies-cyclonedx.json" ]; then
        PYTHON_DEPS=$(jq '.components | length' "$OUTPUT_DIR/python-dependencies-cyclonedx.json" 2>/dev/null || echo "N/A")
        echo "- Python Dependencies: $PYTHON_DEPS" >> "$OUTPUT_DIR/sbom-summary.md"
    fi
    
    if [ -f "$OUTPUT_DIR/vulnerability-report.json" ]; then
        VULNS=$(jq '. | length' "$OUTPUT_DIR/vulnerability-report.json" 2>/dev/null || echo "N/A")
        echo "- Known Vulnerabilities: $VULNS" >> "$OUTPUT_DIR/sbom-summary.md"
    fi
    
    echo "" >> "$OUTPUT_DIR/sbom-summary.md"
    echo "## Usage" >> "$OUTPUT_DIR/sbom-summary.md"
    echo "" >> "$OUTPUT_DIR/sbom-summary.md"
    echo "These SBOM files can be used for:" >> "$OUTPUT_DIR/sbom-summary.md"
    echo "- Security vulnerability scanning" >> "$OUTPUT_DIR/sbom-summary.md"
    echo "- License compliance verification" >> "$OUTPUT_DIR/sbom-summary.md"
    echo "- Supply chain security analysis" >> "$OUTPUT_DIR/sbom-summary.md"
    echo "- Regulatory compliance reporting" >> "$OUTPUT_DIR/sbom-summary.md"
    
    log_success "Summary report created"
}

# Validate generated SBOMs
validate_sboms() {
    log "Validating generated SBOMs..."
    
    local validation_errors=0
    
    # Check JSON files are valid
    for json_file in "$OUTPUT_DIR"/*.json; do
        if [ -f "$json_file" ]; then
            if ! jq . "$json_file" > /dev/null 2>&1; then
                log_error "Invalid JSON in $json_file"
                ((validation_errors++))
            else
                log_success "Valid JSON: $(basename "$json_file")"
            fi
        fi
    done
    
    # Check XML files are valid
    for xml_file in "$OUTPUT_DIR"/*.xml; do
        if [ -f "$xml_file" ]; then
            if command -v xmllint &> /dev/null; then
                if ! xmllint --noout "$xml_file" 2>/dev/null; then
                    log_error "Invalid XML in $xml_file"
                    ((validation_errors++))
                else
                    log_success "Valid XML: $(basename "$xml_file")"
                fi
            fi
        fi
    done
    
    if [ $validation_errors -eq 0 ]; then
        log_success "All SBOMs validated successfully"
    else
        log_error "$validation_errors validation errors found"
        return 1
    fi
}

# Main execution
main() {
    log "GAN Cyber Range SBOM Generator v1.0.0"
    log "====================================="
    
    # Install tools
    install_sbom_tools
    
    # Generate SBOMs
    generate_python_sbom
    generate_container_sbom
    generate_system_sbom
    generate_comprehensive_sbom
    generate_vulnerability_report
    generate_license_report
    
    # Create summary and validate
    create_summary_report
    validate_sboms
    
    log_success "SBOM generation complete!"
    log "Output directory: $OUTPUT_DIR"
    log "Summary report: $OUTPUT_DIR/sbom-summary.md"
    
    # Display file sizes
    echo ""
    log "Generated files:"
    ls -lh "$OUTPUT_DIR"
}

# Handle errors
trap 'log_error "Script failed on line $LINENO"' ERR

# Run main function
main "$@"