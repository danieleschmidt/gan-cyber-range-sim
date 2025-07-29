#!/bin/bash
set -euo pipefail

# Development environment setup script for GAN Cyber Range Simulator

echo "🚀 Setting up GAN Cyber Range development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
required_version="3.10"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "🔧 Installing dependencies..."
pip install -e ".[dev,security,ml]"

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "🔧 Creating directory structure..."
mkdir -p {logs,data,models,config/local}

# Set up local configuration
if [ ! -f ".env.local" ]; then
    echo "🔧 Creating local environment file..."
    cat > .env.local << 'EOF'
# Local development environment variables
ENVIRONMENT=development
LOG_LEVEL=DEBUG
KUBERNETES_NAMESPACE=cyber-range-dev

# API Configuration
API_HOST=localhost
API_PORT=8000

# Security Configuration
ENABLE_AUTHENTICATION=false
RATE_LIMIT_ENABLED=false

# ML Configuration
MODEL_CACHE_DIR=./models
TRAINING_DATA_DIR=./data
EOF
fi

# Validate installation
echo "🧪 Validating installation..."
python -c "import gan_cyber_range; print('✅ Package import successful')"

# Run basic tests
echo "🧪 Running basic tests..."
pytest tests/unit/ -v --no-cov

# Security check
echo "🔒 Running security checks..."
bandit -r src/ -x tests/ -q

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Review configuration in .env.local"
echo "  3. Run tests: pytest"
echo "  4. Start development server: python -m gan_cyber_range.api"
echo ""
echo "For more information, see docs/DEVELOPMENT.md"