# Testing Framework Documentation

This document describes the comprehensive testing infrastructure for the GAN Cyber Range Simulator project.

## Overview

Our testing strategy follows industry best practices with multiple layers of testing:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions 
- **Security Tests**: Validate security controls and isolation
- **Performance Tests**: Ensure scalability and responsiveness
- **End-to-End Tests**: Validate complete user scenarios

## Test Categories

### 1. Unit Tests (`tests/unit/`)

Test individual functions, classes, and modules in isolation.

**Characteristics:**
- Fast execution (< 1 second per test)
- No external dependencies
- High code coverage target (90%+)
- Mock all external services

**Example:**
```bash
# Run all unit tests
pytest tests/unit/ -m unit

# Run with coverage
pytest tests/unit/ --cov=gan_cyber_range --cov-report=html
```

### 2. Integration Tests (`tests/integration/`)

Test interactions between components and external services.

**Characteristics:**
- Moderate execution time (1-30 seconds per test)
- May use test databases, containers
- Test API endpoints, database operations
- Validate component integration

**Example:**
```bash
# Run integration tests
pytest tests/integration/ -m integration

# Run with external services
docker-compose -f docker-compose.test.yml up -d
pytest tests/integration/
docker-compose -f docker-compose.test.yml down
```

### 3. Security Tests (`tests/security/`)

Validate security controls, isolation mechanisms, and threat protection.

**Characteristics:**
- Comprehensive security validation
- Container escape prevention tests
- Network isolation validation
- Input sanitization verification

**Example:**
```bash
# Run security test suite
pytest tests/security/ -m security

# Run isolation tests specifically
pytest tests/security/test_isolation.py -v
```

### 4. Performance Tests (`tests/performance/`)

Ensure the system meets performance and scalability requirements.

**Characteristics:**
- Load testing for concurrent users
- Stress testing for resource limits
- Agent response time validation
- Memory usage monitoring

**Example:**
```bash
# Run performance benchmarks
pytest tests/performance/ -m performance

# Run with resource monitoring
pytest tests/performance/ --benchmark-only
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)

Key configuration settings:

- **Coverage Target**: 80% minimum
- **Timeout**: 300 seconds maximum per test
- **Markers**: Organized by test category
- **Output**: HTML and XML reports for CI/CD

### Environment Configuration

Test environments are configured through:

- `.env.test`: Test-specific environment variables
- `conftest.py`: Shared fixtures and setup
- Docker Compose: External service dependencies

### Fixtures

Common fixtures available in all tests:

| Fixture | Purpose |
|---------|---------|
| `mock_kubernetes_client` | Mock Kubernetes operations |
| `mock_llm_client` | Mock AI/LLM interactions |
| `isolated_environment` | Security-isolated test environment |
| `sample_vulnerabilities` | Test vulnerability data |
| `sample_attack_scenarios` | Test attack scenarios |
| `test_config` | Test configuration settings |

## Running Tests

### Local Development

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m security
pytest -m performance

# Run with coverage
pytest --cov=gan_cyber_range --cov-report=html

# Run specific test file
pytest tests/unit/test_agents.py -v

# Run tests matching pattern
pytest -k "test_agent" -v

# Run tests with debugging
pytest tests/unit/test_agents.py::test_agent_creation -v -s
```

### CI/CD Pipeline

```bash
# Install test dependencies
pip install -r requirements.txt
pip install -e ".[test]"

# Run test suite with strict settings
pytest --strict-markers --strict-config \
       --cov=gan_cyber_range \
       --cov-report=xml \
       --cov-fail-under=80 \
       --junit-xml=junit.xml

# Security-specific tests
pytest tests/security/ --strict-markers -v
```

### Docker-based Testing

```bash
# Build test image
docker build -f Dockerfile.test -t gan-cyber-range:test .

# Run tests in container
docker run --rm gan-cyber-range:test pytest

# Run with volume mounts for coverage reports
docker run --rm \
  -v $(pwd)/htmlcov:/app/htmlcov \
  gan-cyber-range:test \
  pytest --cov-report=html
```

## Test Data and Fixtures

### Test Data Location

- `tests/fixtures/`: Factory classes for test objects
- `tests/data/`: Static test data files
- `tests/integration/fixtures/`: Integration test scenarios

### Fixture Factories

Common patterns for creating test data:

```python
# Agent factory
def create_test_agent(agent_type="red", model="mock"):
    return AgentFactory.create(
        type=agent_type,
        model=model,
        config=get_test_config()
    )

# Scenario factory
def create_test_scenario(name="basic", difficulty="beginner"):
    return ScenarioFactory.create(
        name=name,
        difficulty=difficulty,
        vulnerabilities=get_test_vulnerabilities()
    )
```

## Security Testing

### Isolation Testing

Critical security tests include:

1. **Container Escape Prevention**: Verify containers cannot escape isolation
2. **Network Segmentation**: Validate network policies work correctly
3. **Resource Limits**: Ensure resource quotas are enforced
4. **Privilege Escalation**: Test against unauthorized privilege gains
5. **Data Leakage**: Verify no sensitive data exposure

### Example Security Test

```python
@pytest.mark.security
async def test_container_isolation(isolated_environment):
    """Test that containers cannot escape isolation."""
    # Create isolated container
    container = await create_isolated_container(isolated_environment)
    
    # Attempt various escape techniques
    escape_attempts = [
        "access_host_filesystem",
        "privilege_escalation", 
        "network_bypass",
        "resource_exhaustion"
    ]
    
    for attempt in escape_attempts:
        result = await attempt_container_escape(container, attempt)
        assert result.blocked, f"Container escape via {attempt} was not blocked"
        assert result.logged, f"Container escape attempt {attempt} was not logged"
```

## Performance Testing

### Benchmarking

Performance tests validate:

1. **Agent Response Time**: < 5 seconds per decision
2. **Concurrent Simulations**: Support 10+ parallel scenarios  
3. **Resource Usage**: Stay within defined limits
4. **Memory Leaks**: No memory growth over time
5. **Database Performance**: Query optimization validation

### Load Testing

```python
@pytest.mark.performance
async def test_concurrent_agent_performance():
    """Test performance with multiple concurrent agents."""
    # Create multiple agents
    agents = [create_test_agent() for _ in range(10)]
    
    # Run concurrent operations
    start_time = time.time()
    results = await asyncio.gather(*[
        agent.make_decision(scenario) for agent in agents
    ])
    execution_time = time.time() - start_time
    
    # Validate performance requirements
    assert execution_time < 30, "Concurrent operations too slow"
    assert all(r.success for r in results), "Some operations failed"
```

## Continuous Integration

### GitHub Actions Integration

```yaml
# .github/workflows/test.yml
- name: Run Test Suite
  run: |
    pytest --strict-markers \
           --cov=gan_cyber_range \
           --cov-report=xml \
           --junit-xml=junit.xml \
           --timeout=300

- name: Security Tests
  run: |
    pytest tests/security/ -m security --strict-markers

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Quality Gates

Tests must pass these quality gates:

- **Coverage**: Minimum 80% code coverage
- **Security**: All security tests pass
- **Performance**: No performance regressions
- **Linting**: Code style compliance (ruff, black)
- **Type Checking**: MyPy type validation

## Debugging Tests

### Common Debug Commands

```bash
# Run single test with debug output
pytest tests/unit/test_agents.py::test_agent_creation -v -s

# Run with pdb debugger
pytest tests/unit/test_agents.py::test_agent_creation --pdb

# Run with increased logging
pytest tests/unit/ --log-cli-level=DEBUG

# Run failed tests only
pytest --lf

# Run with coverage and debug
pytest --cov=gan_cyber_range --cov-report=term-missing -v
```

### Test Debugging Tips

1. **Use fixtures**: Leverage shared fixtures for consistent test setup
2. **Mock external services**: Avoid dependencies on external APIs
3. **Isolate failures**: Run single tests to isolate issues
4. **Check logs**: Review test logs for detailed error information
5. **Validate assumptions**: Ensure test data matches expected formats

## Best Practices

### Writing Tests

1. **Descriptive Names**: Use clear, descriptive test names
2. **Single Responsibility**: Each test should validate one behavior
3. **Arrange-Act-Assert**: Structure tests with clear phases
4. **Independent Tests**: Tests should not depend on each other
5. **Meaningful Assertions**: Validate specific expected outcomes

### Test Organization

1. **Consistent Structure**: Follow established directory organization
2. **Appropriate Markers**: Tag tests with relevant markers
3. **Shared Fixtures**: Use fixtures for common setup patterns
4. **Documentation**: Document complex test scenarios
5. **Regular Maintenance**: Keep tests updated with code changes

### Security Considerations

1. **No Real Credentials**: Never use production credentials in tests
2. **Isolated Environments**: Use completely isolated test environments
3. **Cleanup**: Always clean up resources after tests
4. **Logging Validation**: Verify security events are logged
5. **Audit Trail**: Maintain audit trail for security test results

---

For questions about testing or to contribute to test coverage, see our [Contributing Guidelines](../../CONTRIBUTING.md) or join discussions in our [Discord Community](https://discord.gg/gan-cyber-range).