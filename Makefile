# Makefile for GAN Cyber Range Simulator
.PHONY: help install test lint format security docs clean docker k8s-deploy

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development setup
install: ## Install development dependencies
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

install-dev: ## Install development dependencies with extras
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev,security,docs,ml]"
	pre-commit install

# Testing
test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-security: ## Run security-focused tests
	pytest tests/security/ -v -m security

test-coverage: ## Run tests with coverage report
	pytest tests/ --cov=gan_cyber_range --cov-report=html --cov-report=term

# Code quality
lint: ## Run all linting tools
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

# Security
security: ## Run security checks
	bandit -r src/
	safety check
	semgrep --config=auto src/

security-baseline: ## Create security baseline for secrets detection
	detect-secrets scan --baseline .secrets.baseline

# Documentation
docs: ## Build documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve

docs-deploy: ## Deploy documentation
	mkdocs gh-deploy

# Docker operations
docker-build: ## Build Docker images
	docker build -t gan-cyber-range:latest .
	docker build -f deployments/Dockerfile.agent -t gan-cyber-range-agent:latest .

docker-run: ## Run application with Docker Compose
	docker-compose up -d

docker-stop: ## Stop Docker Compose services
	docker-compose down

docker-logs: ## View Docker Compose logs
	docker-compose logs -f

# Kubernetes operations
k8s-namespace: ## Create Kubernetes namespace
	kubectl create namespace cyber-range --dry-run=client -o yaml | kubectl apply -f -

k8s-deploy: k8s-namespace ## Deploy to Kubernetes
	kubectl apply -f deployments/k8s/ -n cyber-range

k8s-delete: ## Delete Kubernetes deployment
	kubectl delete -f deployments/k8s/ -n cyber-range

k8s-status: ## Check Kubernetes deployment status
	kubectl get all -n cyber-range

k8s-logs: ## View application logs in Kubernetes
	kubectl logs -f deployment/gan-cyber-range -n cyber-range

# Local development
dev-setup: ## Setup local development environment
	./scripts/dev-setup.sh

dev-start: ## Start development services
	docker-compose -f docker-compose.dev.yml up -d
	uvicorn gan_cyber_range.api:app --reload --host 0.0.0.0 --port 8000

dev-stop: ## Stop development services
	docker-compose -f docker-compose.dev.yml down

# Database operations
db-upgrade: ## Run database migrations
	alembic upgrade head

db-downgrade: ## Rollback database migration
	alembic downgrade -1

db-revision: ## Create new database migration
	alembic revision --autogenerate -m "$(msg)"

# Cleanup
clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

clean-docker: ## Clean up Docker images and containers
	docker-compose down -v --remove-orphans
	docker system prune -f

# Release operations
version: ## Show current version
	python -c "from src.gan_cyber_range import __version__; print(__version__)"

build: ## Build distribution packages
	python -m build

release-test: ## Upload to test PyPI
	python -m twine upload --repository testpypi dist/*

release: ## Upload to PyPI
	python -m twine upload dist/*

# Validation and verification
verify: lint test security ## Run all verification checks

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Environment-specific targets
prod-deploy: ## Deploy to production environment
	@echo "Deploying to production..."
	kubectl apply -f deployments/k8s/production/ -n cyber-range-prod

staging-deploy: ## Deploy to staging environment
	@echo "Deploying to staging..."
	kubectl apply -f deployments/k8s/staging/ -n cyber-range-staging

# Monitoring and metrics
metrics: ## Start monitoring stack
	docker-compose -f deployments/monitoring/docker-compose.yml up -d

metrics-stop: ## Stop monitoring stack
	docker-compose -f deployments/monitoring/docker-compose.yml down

# Performance testing
perf-test: ## Run performance tests
	pytest tests/performance/ -v

load-test: ## Run load tests
	locust -f tests/load/locustfile.py --host=http://localhost:8000

# Security scanning
scan-deps: ## Scan dependencies for vulnerabilities
	safety check
	pip-audit

scan-code: ## Scan code for security issues
	bandit -r src/
	semgrep --config=auto src/

scan-containers: ## Scan container images for vulnerabilities
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy image gan-cyber-range:latest

scan-secrets: ## Scan for committed secrets and credentials
	detect-secrets scan --baseline .secrets.baseline

security-policy-check: ## Validate security policies and configurations
	python scripts/validate_security_policies.py

security-full: scan-deps scan-code scan-secrets security-policy-check ## Run comprehensive security scan
	@echo "✅ Full security scan completed"

compliance-check: ## Check compliance with security frameworks
	@echo "Running compliance validation..."
	python scripts/validate_security_policies.py
	detect-secrets scan --baseline .secrets.baseline --force-use-all-plugins
	bandit -r src/ -f json -o security_reports/bandit-report.json || true
	@echo "✅ Compliance check completed"

isolation-test: ## Test cyber range isolation mechanisms
	pytest tests/security/test_isolation.py -v -m isolation
	@echo "✅ Isolation tests completed"

# Backup and restore
backup: ## Create backup of important data
	./scripts/backup.sh

restore: ## Restore from backup
	./scripts/restore.sh $(backup_file)

# Development utilities
shell: ## Start Python shell with project context
	python -c "from gan_cyber_range import *; import IPython; IPython.embed()"

jupyter: ## Start Jupyter notebook server
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Quick development workflows
quick-test: format lint test-unit ## Quick development test cycle

full-verify: format lint security test docs ## Full verification before commit

ci-test: ## Run tests as in CI environment
	pytest tests/ --cov=gan_cyber_range --cov-report=xml --cov-report=term

# Variables for dynamic targets
PYTEST_ARGS ?= -v
DOCKER_TAG ?= latest
NAMESPACE ?= cyber-range