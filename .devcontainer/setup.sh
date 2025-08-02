#!/bin/bash

# DevContainer post-create setup script
# This script runs after the container is created to set up the development environment

set -e

echo "🚀 Starting GAN Cyber Range Simulator development environment setup..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install additional security tools for cyber range development
echo "🔧 Installing security development tools..."
sudo apt-get install -y \
    nmap \
    netcat-openbsd \
    tcpdump \
    wireshark-common \
    curl \
    wget \
    jq \
    yq \
    tree \
    htop \
    tmux \
    vim \
    git-lfs \
    postgresql-client \
    redis-tools

# Install Python development dependencies
echo "🐍 Setting up Python environment..."
python -m pip install --upgrade pip setuptools wheel

# Install the project in development mode
if [ -f "requirements.txt" ]; then
    echo "📝 Installing Python requirements..."
    pip install -r requirements.txt
fi

if [ -f "pyproject.toml" ]; then
    echo "📦 Installing project in development mode..."
    pip install -e ".[dev,test]"
fi

# Install pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    pre-commit install --hook-type commit-msg
    echo "✅ Pre-commit hooks installed"
else
    echo "⚠️  pre-commit not found, skipping hook installation"
fi

# Set up Git configuration
echo "🔧 Configuring Git..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input
git config --global core.safecrlf true

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p \
    logs \
    data \
    .vscode \
    .pytest_cache \
    .mypy_cache \
    .ruff_cache

# Set up environment variables
echo "🌍 Setting up environment variables..."
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "✅ Created .env from .env.example"
fi

# Initialize local development database (if docker-compose includes it)
echo "🗄️  Checking database setup..."
if docker-compose ps db &> /dev/null; then
    echo "Database service found in docker-compose"
    # Wait for database to be ready
    until docker-compose exec db pg_isready; do
        echo "Waiting for database..."
        sleep 2
    done
    echo "✅ Database is ready"
fi

# Set up Kubernetes tools
echo "☸️  Setting up Kubernetes development tools..."
# Create local kubeconfig for development
mkdir -p ~/.kube
if [ ! -f ~/.kube/config ]; then
    # Create a basic kubeconfig for local development
    cat > ~/.kube/config << 'EOF'
apiVersion: v1
clusters: []
contexts: []
current-context: ""
kind: Config
preferences: {}
users: []
EOF
    chmod 600 ~/.kube/config
fi

# Install additional Python tools for security research
echo "🔬 Installing security research tools..."
pip install \
    jupyter \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    scikit-learn \
    networkx \
    plotly \
    dash

# Set up Jupyter notebook environment
echo "📊 Setting up Jupyter environment..."
jupyter --generate-config
mkdir -p notebooks

# Create useful development aliases
echo "⚡ Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# GAN Cyber Range Development Aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Project-specific aliases
alias pytest-cov='pytest --cov=gan_cyber_range --cov-report=html --cov-report=term'
alias ruff-check='ruff check .'
alias ruff-fix='ruff check --fix .'
alias mypy-check='mypy src/'
alias format-code='black . && isort .'
alias run-tests='pytest tests/ -v'
alias run-security-tests='pytest tests/security/ -v'
alias docker-build='docker build -t gan-cyber-range:dev .'
alias k='kubectl'
alias kns='kubectl config set-context --current --namespace'
alias kctx='kubectl config use-context'

# Docker compose helpers
alias dc='docker-compose'
alias dcup='docker-compose up -d'
alias dcdown='docker-compose down'
alias dclogs='docker-compose logs -f'
alias dcps='docker-compose ps'

# Git helpers
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -10'
alias gd='git diff'
alias gb='git branch'
alias gco='git checkout'

EOF

# Make scripts executable
echo "🔐 Setting script permissions..."
find scripts/ -name "*.sh" -type f -exec chmod +x {} \; 2>/dev/null || true

# Install Node.js dependencies if package.json exists
if [ -f "package.json" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Set up VSCode workspace settings
echo "⚙️  Setting up VSCode workspace..."
if [ ! -f ".vscode/settings.json" ]; then
    mkdir -p .vscode
    cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "/usr/local/bin/python",
    "python.terminal.activateEnvironment": false,
    "terminal.integrated.defaultProfile.linux": "bash",
    "files.watcherExclude": {
        "**/.git/objects/**": true,
        "**/.git/subtree-cache/**": true,
        "**/node_modules/*/**": true,
        "**/.pytest_cache/**": true,
        "**/.mypy_cache/**": true,
        "**/.ruff_cache/**": true,
        "**/logs/**": true,
        "**/data/**": true
    }
}
EOF
fi

# Final setup verification
echo "🔍 Verifying setup..."
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo "Git version: $(git --version)"
echo "Docker version: $(docker --version)"
echo "Kubectl version: $(kubectl version --client)"
echo "Helm version: $(helm version --short)"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Copy .env.example to .env and configure your environment variables"
echo "   2. Run 'docker-compose up -d' to start the development stack"
echo "   3. Run 'pytest' to verify tests are working"
echo "   4. Visit http://localhost:8080 to access the development dashboard"
echo ""
echo "🔗 Useful commands:"
echo "   - 'make help' - Show available make targets"
echo "   - 'pytest-cov' - Run tests with coverage"
echo "   - 'format-code' - Format code with black and isort"
echo "   - 'ruff-check' - Run linting with ruff"
echo ""