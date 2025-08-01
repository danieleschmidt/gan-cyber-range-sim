#!/bin/bash
set -e

echo "🚀 Setting up GAN Cyber Range development environment..."

# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install additional security tools for development
sudo apt-get install -y \
    nmap \
    nikto \
    sqlmap \
    gobuster \
    john \
    hashcat \
    metasploit-framework \
    wireshark-common \
    tcpdump \
    netcat \
    socat \
    curl \
    wget \
    jq \
    yq \
    tree \
    htop \
    vim \
    git-extras

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Install development dependencies
pip install -e ".[dev,security,docs,ml]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Setup Kubernetes tools
echo "☸️ Setting up Kubernetes tools..."
# Install k9s for cluster management
curl -sS https://webinstall.dev/k9s | bash
sudo mv ~/.local/bin/k9s /usr/local/bin/

# Install kubectx and kubens
sudo git clone https://github.com/ahmetb/kubectx /opt/kubectx
sudo ln -s /opt/kubectx/kubectx /usr/local/bin/kubectx
sudo ln -s /opt/kubectx/kubens /usr/local/bin/kubens

# Install stern for log streaming
curl -L https://github.com/stern/stern/releases/latest/download/stern_linux_amd64.tar.gz | sudo tar xz -C /usr/local/bin stern

# Setup development aliases and functions
echo "🔗 Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# GAN Cyber Range Development Aliases
alias k='kubectl'
alias kgp='kubectl get pods'
alias kgs='kubectl get services'
alias kgd='kubectl get deployments'
alias kns='kubens'
alias kctx='kubectx'
alias logs='stern'

# Python development aliases
alias pytest-cov='pytest --cov=gan_cyber_range --cov-report=html --cov-report=term'
alias lint='pre-commit run --all-files'
alias format='black src/ tests/ && isort src/ tests/'
alias typecheck='mypy src/'
alias security-scan='bandit -r src/'

# Docker aliases
alias dps='docker ps'
alias dimg='docker images'
alias dlog='docker logs'
alias dexec='docker exec -it'

# Cyber range specific aliases
alias range-up='docker-compose -f docker-compose.dev.yml up -d'
alias range-down='docker-compose -f docker-compose.dev.yml down'
alias range-logs='docker-compose -f docker-compose.dev.yml logs -f'
alias range-status='kubectl get all -n cyber-range'

# Quick development functions
cr-test() {
    echo "🧪 Running comprehensive test suite..."
    pytest-cov && lint && typecheck && security-scan
}

cr-deploy-local() {
    echo "🚀 Deploying cyber range locally..."
    ./scripts/setup-dev.sh
    kubectl apply -f deployments/development/
}

cr-reset() {
    echo "🔄 Resetting development environment..."
    docker-compose -f docker-compose.dev.yml down -v
    kubectl delete namespace cyber-range --ignore-not-found
    docker system prune -f
}
EOF

# Create .kube directory if it doesn't exist
mkdir -p ~/.kube

# Create development configuration
echo "📝 Creating development configuration..."
mkdir -p /workspace/config/development
cat > /workspace/config/development/local.env << 'EOF'
# Development environment configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# AI/ML Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=gan-cyber-range-dev

# Database Configuration
DATABASE_URL=postgresql://cyber_range:password@postgres:5432/cyber_range_dev
REDIS_URL=redis://redis:6379/0

# Kubernetes Configuration
KUBERNETES_NAMESPACE=cyber-range-dev
KUBECONFIG=/workspace/.kube/config

# Security Configuration
SECRET_KEY=development-secret-key-change-in-production
ENCRYPTION_KEY=development-encryption-key-32-chars
JWT_SECRET=development-jwt-secret

# Monitoring Configuration
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
JAEGER_ENABLED=true

# Development Flags
MOCK_SECURITY_TOOLS=true
FAST_MODE=true
SKIP_AUTH=false
EOF

# Setup git hooks
echo "🔧 Setting up git hooks..."
mkdir -p .git/hooks
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
echo "🔍 Running pre-push checks..."
pre-commit run --all-files
if [ $? -ne 0 ]; then
    echo "❌ Pre-commit checks failed. Push aborted."
    exit 1
fi

echo "🧪 Running tests..."
python -m pytest tests/ -x
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Push aborted."
    exit 1
fi

echo "✅ All checks passed. Proceeding with push."
EOF
chmod +x .git/hooks/pre-push

# Generate development certificates
echo "🔐 Generating development certificates..."
mkdir -p /workspace/certs/development
openssl req -x509 -newkey rsa:4096 -keyout /workspace/certs/development/key.pem -out /workspace/certs/development/cert.pem -days 365 -nodes -subj "/C=US/ST=Dev/L=Local/O=GAN-Cyber-Range/CN=localhost"

# Setup completion scripts
echo "⚡ Setting up shell completions..."
kubectl completion bash | sudo tee /etc/bash_completion.d/kubectl > /dev/null
helm completion bash | sudo tee /etc/bash_completion.d/helm > /dev/null

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  - range-up          # Start development environment"
echo "  - cr-test           # Run all tests and checks"
echo "  - cr-deploy-local   # Deploy to local cluster"
echo "  - k9s               # Kubernetes cluster management"
echo ""
echo "📚 Documentation: https://gan-cyber-range.readthedocs.io"
echo "💬 Support: https://discord.gg/gan-cyber-range"