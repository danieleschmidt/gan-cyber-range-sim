# ADR-002: Use Kubernetes for Container Orchestration

## Status

Accepted

## Context

Given our decision to adopt microservices architecture (ADR-001), we need a container orchestration platform that can:
- Manage multiple services with complex networking requirements
- Provide strong security isolation between cyber range components
- Support dynamic scaling based on simulation load
- Offer built-in service discovery and load balancing
- Enable network policy enforcement for security boundaries
- Support both development and production deployments

Alternative options considered:
- **Docker Compose**: Too simple for complex networking and scaling requirements
- **Docker Swarm**: Limited ecosystem and advanced networking capabilities
- **Nomad**: Less mature ecosystem, limited security policy features
- **Custom Solution**: Too much development overhead, reinventing solved problems

Kubernetes emerged as the clear choice due to its:
- Mature ecosystem and extensive community support
- Native support for network policies and security contexts
- Built-in secrets management and RBAC
- Extensive monitoring and observability integrations
- Support for both on-premises and cloud deployments

## Decision

We will use Kubernetes as our container orchestration platform with the following configuration:

### Core Components
- **Namespaces**: Separate namespaces for different simulation environments
- **Network Policies**: Strict ingress/egress rules between services
- **Security Contexts**: Non-root containers with restricted capabilities
- **Resource Limits**: CPU/memory limits on all pods to prevent resource exhaustion
- **Secrets Management**: Kubernetes secrets for API keys and sensitive configuration

### Security Model
- **Pod Security Standards**: Restricted security context for all workloads
- **Network Segmentation**: NetworkPolicies isolate cyber range from cluster services
- **RBAC**: Minimal permissions for service accounts
- **Image Security**: Only trusted, scanned container images
- **Admission Controllers**: Policy enforcement at pod creation

### Deployment Strategy
- **Helm Charts**: Standardized deployment templates
- **GitOps**: Infrastructure as code with automatic synchronization
- **Environment Promotion**: Dev → Staging → Production pipeline
- **Rolling Updates**: Zero-downtime deployments

## Consequences

### Positive
- **Security Isolation**: Strong network and compute isolation through namespaces and policies
- **Scalability**: Horizontal pod autoscaling based on metrics
- **High Availability**: Built-in pod restart and rescheduling
- **Service Discovery**: DNS-based service discovery and load balancing
- **Ecosystem**: Rich ecosystem of security, monitoring, and networking tools
- **Portability**: Runs consistently across different cloud providers and on-premises
- **DevOps Integration**: Native CI/CD integration with GitOps patterns

### Negative
- **Complexity**: Significant learning curve for Kubernetes concepts and operations
- **Resource Overhead**: Kubernetes control plane consumes cluster resources
- **Debugging Complexity**: More complex troubleshooting with multiple abstraction layers
- **Configuration Verbosity**: YAML manifests can become large and complex
- **Cluster Management**: Need for cluster maintenance, upgrades, and monitoring

### Neutral
- **Infrastructure Requirements**: Requires multi-node cluster for high availability
- **Networking Requirements**: Need for CNI plugin selection and configuration
- **Storage Requirements**: Need for persistent volume provisioning strategy
- **Monitoring Stack**: Requires Prometheus/Grafana or similar observability platform