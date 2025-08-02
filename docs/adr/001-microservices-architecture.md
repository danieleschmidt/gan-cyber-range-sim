# ADR-001: Adopt Microservices Architecture

## Status

Accepted

## Context

The GAN Cyber Range Simulator requires a flexible, scalable architecture that can support:
- Independent deployment and scaling of different components
- Isolation between vulnerable services and control systems
- Easy addition of new vulnerability types and attack scenarios
- Clear separation of concerns between red team, blue team, and orchestration systems
- Security boundaries that prevent simulation from affecting host systems

Traditional monolithic architectures would create tight coupling between components, making it difficult to:
- Scale individual services based on demand
- Maintain security isolation
- Allow independent development of agent capabilities
- Support different programming languages and frameworks for specialized components

## Decision

We will adopt a microservices architecture with the following key services:

1. **Orchestrator Service**: Central coordination and scenario management
2. **Red Team Agent Service**: Attack planning and execution
3. **Blue Team Agent Service**: Defense monitoring and response
4. **Vulnerable Service Pool**: Dynamically deployed target applications
5. **Metrics Collection Service**: Centralized monitoring and analytics
6. **Network Topology Service**: Virtual network management
7. **Scenario Management Service**: Training curriculum and configuration

Each service will:
- Run in its own container with defined resource limits
- Communicate through well-defined APIs (REST/gRPC)
- Maintain its own data store when needed
- Be independently deployable and scalable
- Follow security-first design principles

## Consequences

### Positive
- **Scalability**: Individual services can be scaled based on demand
- **Security Isolation**: Strong boundaries between components reduce attack surface
- **Development Velocity**: Teams can work independently on different services
- **Technology Flexibility**: Services can use different languages/frameworks as appropriate
- **Fault Tolerance**: Failure of one service doesn't bring down entire system
- **Testability**: Individual services can be tested in isolation
- **Deployment Flexibility**: Rolling updates and A/B testing become possible

### Negative
- **Complexity**: Increased operational complexity with service discovery, monitoring
- **Network Overhead**: Inter-service communication adds latency and failure points
- **Data Consistency**: Distributed data management requires careful transaction design
- **Debugging Difficulty**: Tracing issues across multiple services is more complex
- **Infrastructure Requirements**: Need for container orchestration (Kubernetes)

### Neutral
- **Learning Curve**: Team needs to learn microservices patterns and Kubernetes
- **Tooling Requirements**: Need for service mesh, monitoring, and observability tools
- **Configuration Management**: More complex configuration with multiple services