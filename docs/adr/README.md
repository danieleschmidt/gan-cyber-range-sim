# Architecture Decision Records (ADRs)

## Overview

This directory contains Architecture Decision Records (ADRs) for the GAN Cyber Range Simulator project. ADRs document the architectural decisions made during the development of this system.

## Format

Each ADR follows the standard format:
- **Title**: Short noun phrase describing the decision
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: What is the issue motivating this decision
- **Decision**: What is the change we're proposing or have agreed to
- **Consequences**: What becomes easier or more difficult to do

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [001](001-microservices-architecture.md) | Adopt Microservices Architecture | Accepted |
| [002](002-kubernetes-orchestration.md) | Use Kubernetes for Container Orchestration | Accepted |
| [003](003-llm-agent-framework.md) | LLM-Based Agent Framework | Accepted |
| [004](004-security-isolation-model.md) | Security Isolation Model | Accepted |
| [005](005-metrics-and-observability.md) | Metrics and Observability Strategy | Accepted |

## Creating New ADRs

1. Copy the template from `template.md`
2. Number sequentially (e.g., `006-new-decision.md`)
3. Fill in all sections
4. Create a pull request for review
5. Update this index when merged

## Template

See [template.md](template.md) for the standard ADR template.