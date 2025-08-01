# ADR-0002: Kubernetes Isolation Strategy

## Status
Accepted

## Context
The cyber range must provide strong isolation to prevent simulated attacks from affecting the host system or escaping to the broader network. We need a comprehensive isolation strategy that balances security with realistic attack scenarios.

## Decision
We implement multi-layered isolation using:

1. **Network Policies**: Strict ingress/egress rules limiting communication to cyber-range namespace
2. **Pod Security Standards**: Enforced security contexts preventing privilege escalation
3. **Resource Quotas**: CPU, memory, and storage limits to prevent resource exhaustion
4. **RBAC**: Minimal service account permissions for range components
5. **Admission Controllers**: Validate and potentially modify resource creation
6. **Runtime Security**: Falco rules for detecting suspicious behavior

Isolation layers:
- Namespace-level isolation with network policies
- Pod-level security contexts and capabilities restrictions
- Container-level resource limits and security scanning
- Host-level monitoring and alerting

## Consequences

### Positive
- Strong security boundaries prevent breakout scenarios
- Resource limits ensure system stability during intensive simulations
- Network isolation contains all attack traffic within the range
- Comprehensive monitoring provides visibility into all activities

### Negative
- Complex configuration increases setup complexity
- Performance overhead from security controls
- Some realistic attack scenarios may be limited by security constraints
- Requires Kubernetes expertise for proper configuration and maintenance