# Testing Framework

This document describes the comprehensive testing framework for the GAN Cyber Range Simulator.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Global test configuration and fixtures
├── fixtures/                # Shared test fixtures
│   ├── __init__.py
│   └── cyber_range_fixtures.py
├── unit/                    # Unit tests - fast, isolated
│   ├── __init__.py
│   └── test_agents.py
├── integration/             # Integration tests - component interactions
│   ├── __init__.py
│   └── test_cyber_range.py
├── security/                # Security-focused tests
│   ├── __init__.py
│   └── test_isolation.py
├── performance/             # Performance and load tests
│   ├── __init__.py
│   └── test_agent_performance.py
└── e2e/                     # End-to-end tests - full workflows
    ├── __init__.py
    └── test_full_simulation.py
```

## Test Categories

### Unit Tests
- **Purpose**: Test individual components in isolation
- **Speed**: Fast (< 1 second per test)
- **Scope**: Single functions, classes, or modules
- **Mocking**: Heavy use of mocks to isolate dependencies
- **Example**: Testing agent decision-making logic

```bash
# Run unit tests only
pytest tests/unit/ -m unit
```

### Integration Tests
- **Purpose**: Test component interactions and data flow
- **Speed**: Medium (1-10 seconds per test)
- **Scope**: Multiple components working together
- **Dependencies**: May require external services (database, Redis)
- **Example**: Testing agent-environment interactions

```bash
# Run integration tests
pytest tests/integration/ -m integration
```

### Security Tests
- **Purpose**: Validate security controls and isolation
- **Speed**: Variable (depending on security scans)
- **Scope**: Security boundaries, access controls, vulnerability detection
- **Focus**: Ensure cyber range isolation and security
- **Example**: Testing container escape prevention

```bash
# Run security tests
pytest tests/security/ -m security
```

### Performance Tests
- **Purpose**: Validate system performance under load
- **Speed**: Slow (10+ seconds per test)
- **Scope**: Response times, throughput, resource usage
- **Tools**: pytest-benchmark for detailed performance metrics
- **Example**: Testing agent response times under concurrent load

```bash
# Run performance tests
pytest tests/performance/ -m performance
```

### End-to-End Tests
- **Purpose**: Test complete user workflows and scenarios
- **Speed**: Very slow (minutes per test)
- **Scope**: Full system integration from user perspective
- **Environment**: Requires full test environment
- **Example**: Complete attack-defense simulation

```bash
# Run end-to-end tests
pytest tests/e2e/ -m e2e
```

## Test Markers

Use pytest markers to categorize and filter tests:

```python
@pytest.mark.unit
def test_agent_initialization():
    """Unit test for agent initialization."""
    pass

@pytest.mark.integration
@pytest.mark.security
def test_agent_isolation():
    """Integration test for agent security isolation."""
    pass

@pytest.mark.performance
@pytest.mark.slow
def test_concurrent_agents():
    """Performance test for concurrent agent operations."""
    pass

@pytest.mark.e2e
@pytest.mark.slow
def test_full_simulation():
    """End-to-end test for complete simulation."""
    pass
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/gan_cyber_range

# Run specific test file
pytest tests/unit/test_agents.py

# Run specific test function
pytest tests/unit/test_agents.py::test_red_team_agent_initialization

# Run tests by marker
pytest -m unit
pytest -m "unit or integration"
pytest -m "not slow"
```

### Parallel Test Execution

```bash
# Install pytest-xdist for parallel execution
pip install pytest-xdist

# Run tests in parallel
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

### Performance Testing

```bash
# Install pytest-benchmark
pip install pytest-benchmark

# Run benchmark tests
pytest --benchmark-only

# Generate benchmark report
pytest --benchmark-only --benchmark-html=benchmark_report.html
```

### Coverage Reporting

```bash
# Generate HTML coverage report
pytest --cov=src/gan_cyber_range --cov-report=html

# Generate XML coverage report (for CI)
pytest --cov=src/gan_cyber_range --cov-report=xml

# Fail if coverage below threshold
pytest --cov=src/gan_cyber_range --cov-fail-under=80
```

## Test Configuration

### Environment Variables

Set these environment variables for testing:

```bash
# Test configuration
export TESTING=true
export TEST_DATABASE_URL=postgresql://test_user:password@localhost:5432/test_db
export TEST_REDIS_URL=redis://localhost:6379/1

# Mock external services
export MOCK_LLM_APIS=true
export MOCK_SECURITY_TOOLS=true
export MOCK_KUBERNETES=true

# Test timeouts
export TEST_TIMEOUT=300
export ASYNC_TEST_TIMEOUT=60
```

### Test Data

Test data is stored in `tests/data/` directory:

```
tests/data/
├── attack_scenarios/       # Sample attack scenarios
├── defense_configs/        # Defense configuration samples
├── vulnerability_data/     # Vulnerability test data
├── log_samples/           # Sample log files
└── network_topologies/    # Test network configurations
```

## Fixtures and Mocking

### Available Fixtures

The testing framework provides comprehensive fixtures:

```python
# Cyber range fixtures
mock_cyber_range_environment
mock_vulnerable_service
sample_attack_scenario
sample_defense_scenario

# Agent fixtures
mock_red_team_agent
mock_blue_team_agent
mock_llm_responses

# Infrastructure fixtures
mock_kubernetes_client
mock_security_tools
temp_workspace

# Performance fixtures
performance_benchmarks
load_test_config
```

### Using Fixtures

```python
def test_agent_attack_planning(mock_red_team_agent, sample_attack_scenario):
    """Test agent attack planning with fixtures."""
    plan = mock_red_team_agent.plan_attack(sample_attack_scenario)
    assert "reconnaissance" in plan
    assert "exploitation" in plan
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e ".[dev,security]"
      
      - name: Run unit tests
        run: pytest tests/unit/ -m unit
      
      - name: Run integration tests
        run: pytest tests/integration/ -m integration
      
      - name: Run security tests
        run: pytest tests/security/ -m security
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Test Quality Gates

- **Unit Test Coverage**: ≥ 90%
- **Integration Test Coverage**: ≥ 80%
- **Security Test Coverage**: ≥ 95%
- **Performance Regression**: < 10% degradation
- **Test Execution Time**: < 30 minutes for full suite

## Debugging Tests

### Debugging Failed Tests

```bash
# Run with verbose output
pytest -v

# Show local variables on failure
pytest --tb=long

# Drop into debugger on failure
pytest --pdb

# Run single test in debug mode
pytest tests/unit/test_agents.py::test_specific_function -s --pdb
```

### Logging During Tests

```python
import logging

def test_with_logging():
    logger = logging.getLogger(__name__)
    logger.info("Test started")
    
    # Test code here
    
    logger.info("Test completed")
```

## Best Practices

### Writing Effective Tests

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Arrange-Act-Assert**: Structure tests clearly
3. **Single Responsibility**: Each test should verify one specific behavior
4. **Independence**: Tests should not depend on other tests
5. **Deterministic**: Tests should produce consistent results

### Example Test Structure

```python
def test_red_team_agent_executes_sql_injection_attack():
    """Test that red team agent can execute SQL injection attack."""
    # Arrange
    agent = create_test_red_team_agent()
    vulnerable_webapp = create_mock_webapp_with_sql_vulnerability()
    
    # Act
    attack_result = agent.execute_attack(
        target=vulnerable_webapp,
        attack_type="sql_injection"
    )
    
    # Assert
    assert attack_result.success is True
    assert attack_result.attack_type == "sql_injection"
    assert "database_access_gained" in attack_result.achievements
```

### Mock Guidelines

1. **Mock External Dependencies**: Always mock external APIs, databases, etc.
2. **Verify Interactions**: Use mocks to verify method calls and parameters
3. **Keep Mocks Simple**: Don't over-complicate mock setup
4. **Reset Mocks**: Reset mocks between tests to avoid interference

### Performance Test Guidelines

1. **Baseline Metrics**: Establish performance baselines
2. **Consistent Environment**: Run performance tests in consistent environments
3. **Statistical Significance**: Run multiple iterations for reliable results
4. **Resource Monitoring**: Monitor CPU, memory, and network usage
5. **Regression Detection**: Fail tests if performance degrades significantly

## Troubleshooting

### Common Issues

**Tests hanging or timing out:**
- Check for infinite loops or blocking operations
- Verify async/await usage is correct
- Ensure external dependencies are properly mocked

**Flaky tests:**
- Add proper wait conditions for async operations
- Use deterministic test data
- Avoid timing-dependent assertions

**Import errors:**
- Ensure PYTHONPATH includes `src` directory
- Check that all dependencies are installed
- Verify test file naming conventions

**Coverage issues:**
- Run tests with coverage to identify uncovered code
- Add tests for edge cases and error conditions
- Review coverage reports regularly

For additional help, see the [troubleshooting guide](../guides/developer/testing.md) or open an issue on GitHub.