#!/bin/bash
set -e

echo "🔄 Running post-start setup..."

# Ensure correct permissions
chmod +x /workspace/.devcontainer/post-create.sh
chmod +x /workspace/scripts/*.sh

# Start background services if needed
echo "🚀 Starting background services..."

# Check if Kubernetes cluster is accessible
if kubectl cluster-info &> /dev/null; then
    echo "✅ Kubernetes cluster is accessible"
    
    # Create development namespace if it doesn't exist
    kubectl create namespace cyber-range-dev --dry-run=client -o yaml | kubectl apply -f -
    
    # Set default namespace for convenience
    kubectl config set-context --current --namespace=cyber-range-dev
else
    echo "⚠️  Kubernetes cluster not accessible. Some features may be limited."
fi

# Activate Python virtual environment if it exists
if [ -d "/workspace/.venv" ]; then
    source /workspace/.venv/bin/activate
    echo "✅ Activated Python virtual environment"
fi

# Update pre-commit hooks in background
echo "🔧 Updating pre-commit hooks in background..."
pre-commit autoupdate &

# Display development status
echo ""
echo "🎯 GAN Cyber Range Development Environment Ready!"
echo ""
echo "📊 Environment Status:"
echo "  - Python: $(python --version)"
echo "  - Kubectl: $(kubectl version --client --short 2>/dev/null || echo 'Not available')"
echo "  - Helm: $(helm version --short 2>/dev/null || echo 'Not available')"
echo "  - Docker: $(docker --version 2>/dev/null || echo 'Not available')"
echo ""
echo "🚀 Next steps:"
echo "  1. Copy .env.example to .env and configure your API keys"
echo "  2. Run 'range-up' to start the development environment"
echo "  3. Run 'cr-test' to verify everything is working"
echo ""

# Check for required environment variables
echo "🔍 Checking environment configuration..."
missing_vars=()

if [ -z "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    missing_vars+=("OPENAI_API_KEY")
fi

if [ -z "$ANTHROPIC_API_KEY" ] && [ "$ANTHROPIC_API_KEY" = "your_anthropic_api_key_here" ]; then
    missing_vars+=("ANTHROPIC_API_KEY")
fi

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo "⚠️  Missing required environment variables:"
    for var in "${missing_vars[@]}"; do
        echo "    - $var"
    done
    echo "   Please update your .env file with your API keys"
fi

echo "✅ Post-start setup complete!"