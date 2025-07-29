# Contributing to GAN Cyber Range Simulator

Thank you for your interest in contributing to this cybersecurity research project! This guide outlines how to contribute safely and effectively.

## Quick Start

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create** a feature branch
4. **Make** your changes
5. **Test** thoroughly in isolated environment
6. **Submit** a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/gan-cyber-range-sim.git
cd gan-cyber-range-sim

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Setup pre-commit hooks
pre-commit install
```

## Types of Contributions

### Priority Areas
- 🛡️ New defensive techniques and detection methods
- 🔍 Additional vulnerability scenarios for training
- 📊 Enhanced metrics and visualization capabilities
- 🐛 Bug fixes and performance improvements
- 📚 Documentation improvements

### Security Research Contributions
- Novel attack simulation techniques
- Advanced defense mechanisms
- Realistic network topologies
- Integration with security tools
- Research reproducibility features

## Security Guidelines

### Responsible Research
- **Test Only in Isolation**: Use containerized environments
- **No Real Vulnerabilities**: Don't introduce actual security flaws
- **Ethical Use Only**: Ensure defensive focus of contributions
- **Legal Compliance**: Follow all applicable laws and regulations

### Code Review Requirements
- All security-related code requires two approvals
- Vulnerability simulations must be clearly documented
- No hardcoded credentials or sensitive data
- Proper input validation and sanitization

## Pull Request Process

### Before Submitting
- [ ] Code follows project style guidelines
- [ ] Tests pass in isolated environment
- [ ] Security review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

### PR Template
```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Security enhancement

## Security Considerations
- Testing environment: [isolated/containerized]
- Potential impact: [none/low/medium/high]
- Review requirements: [standard/security]

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Security tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No sensitive data exposed
```

## Coding Standards

### Python Style
- Follow PEP 8
- Use type hints
- Maximum line length: 88 characters
- Use Black for formatting
- Use isort for imports

### Security Code Patterns
```python
# Good: Parameterized queries
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))

# Bad: String concatenation
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
```

## Testing Requirements

### Test Categories
- **Unit Tests**: Individual component functionality
- **Integration Tests**: Component interaction testing
- **Security Tests**: Vulnerability simulation validation
- **End-to-End Tests**: Complete scenario testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gan_cyber_range

# Run security tests only
pytest tests/security/

# Run performance tests
pytest tests/performance/
```

## Documentation

### Required Documentation
- Code comments for complex logic
- API documentation using docstrings
- Security considerations for new features
- Setup and configuration guides

### Documentation Style
- Use Markdown for all documentation
- Include code examples
- Provide security context
- Link to relevant resources

## Community

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Security Issues**: security@gan-cyber-range.org

### Getting Help
- Check existing issues and documentation
- Use GitHub Discussions for questions
- Follow Code of Conduct guidelines
- Be patient and respectful

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Security Disclosure

If you discover a security vulnerability, please report it privately to security@gan-cyber-range.org before creating public issues or pull requests.