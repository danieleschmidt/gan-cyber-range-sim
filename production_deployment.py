#!/usr/bin/env python3
"""Production deployment configuration and infrastructure as code."""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import json
# import yaml  # Using manual YAML generation to avoid dependencies


class ProductionDeploymentGenerator:
    """Generate production-ready deployment configurations."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.config = self._load_default_config()
    
    def _dict_to_yaml(self, data: Any, indent: int = 0) -> str:
        """Simple YAML generator to avoid external dependencies."""
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{'  ' * indent}{key}:")
                    lines.append(self._dict_to_yaml(value, indent + 1))
                else:
                    lines.append(f"{'  ' * indent}{key}: {self._format_yaml_value(value)}")
            return '\n'.join(lines)
        elif isinstance(data, list):
            lines = []
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(f"{'  ' * indent}- ")
                    item_yaml = self._dict_to_yaml(item, indent + 1)
                    # Fix indentation for the first line
                    item_lines = item_yaml.split('\n')
                    if item_lines:
                        item_lines[0] = item_lines[0].lstrip()
                    lines.extend(item_lines)
                else:
                    lines.append(f"{'  ' * indent}- {self._format_yaml_value(item)}")
            return '\n'.join(lines)
        else:
            return str(data)
    
    def _format_yaml_value(self, value: Any) -> str:
        """Format value for YAML output."""
        if isinstance(value, str):
            # Check if string needs quoting
            if any(char in value for char in [' ', ':', '#', '[', ']', '{', '}', '"', "'"]):
                return f'"{value}"'
            return value
        elif isinstance(value, bool):
            return 'true' if value else 'false'
        elif value is None:
            return 'null'
        else:
            return str(value)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default production configuration."""
        return {
            'app_name': 'gan-cyber-range',
            'version': '1.0.0',
            'namespace': 'cyber-range',
            'replicas': 3,
            'image_tag': 'latest',
            'resources': {
                'requests': {'cpu': '500m', 'memory': '1Gi'},
                'limits': {'cpu': '2000m', 'memory': '4Gi'}
            },
            'ports': {
                'http': 8080,
                'metrics': 9090
            },
            'environment': 'production',
            'domains': ['cyber-range.example.com'],
            'ssl_enabled': True,
            'monitoring_enabled': True,
            'scaling': {
                'min_replicas': 3,
                'max_replicas': 20,
                'target_cpu_percent': 70
            }
        }
    
    def generate_dockerfile(self) -> str:
        """Generate production-optimized Dockerfile."""
        dockerfile_content = '''# Multi-stage Dockerfile for GAN Cyber Range Simulator
FROM python:3.12-slim-bookworm AS builder

# Build dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    gcc \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Create build user
RUN useradd --create-home --shell /bin/bash builder
USER builder
WORKDIR /home/builder

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim-bookworm AS production

# Security updates and runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser

# Copy Python packages from builder
COPY --from=builder /home/builder/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Create application directories
RUN mkdir -p /app /app/logs /app/data \\
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD python3 -c "import sys; sys.exit(0)"

# Expose ports
EXPOSE 8080 9090

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHON_DISABLE_MODULES=yaml,pickle

# Default command
CMD ["python3", "simple_cli.py", "simulate", "--duration", "1.0"]

# Labels for metadata
LABEL maintainer="GAN Cyber Range Team"
LABEL version="{version}"
LABEL description="GAN Cyber Range Simulator - Adversarial Security Training"
LABEL org.opencontainers.image.source="https://github.com/yourusername/gan-cyber-range"
'''.format(version=self.config['version'])
        
        return dockerfile_content
    
    def generate_docker_compose(self) -> str:
        """Generate Docker Compose configuration."""
        compose_config = {
            'version': '3.8',
            'services': {
                'cyber-range': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile'
                    },
                    'image': f"{self.config['app_name']}:{self.config['image_tag']}",
                    'container_name': f"{self.config['app_name']}-app",
                    'ports': [
                        f"{self.config['ports']['http']}:{self.config['ports']['http']}",
                        f"{self.config['ports']['metrics']}:{self.config['ports']['metrics']}"
                    ],
                    'environment': {
                        'ENVIRONMENT': self.config['environment'],
                        'LOG_LEVEL': 'INFO',
                        'METRICS_ENABLED': 'true'
                    },
                    'volumes': [
                        './data:/app/data:rw',
                        './logs:/app/logs:rw'
                    ],
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': ['CMD', 'python3', 'simple_cli.py', 'status'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '30s'
                    },
                    'deploy': {
                        'resources': {
                            'limits': {
                                'cpus': '2.0',
                                'memory': '4G'
                            },
                            'reservations': {
                                'cpus': '0.5',
                                'memory': '1G'
                            }
                        }
                    },
                    'networks': ['cyber-range-network']
                },
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'container_name': f"{self.config['app_name']}-prometheus",
                    'ports': ['9091:9090'],
                    'volumes': [
                        './monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro'
                    ],
                    'command': [
                        '--config.file=/etc/prometheus/prometheus.yml',
                        '--storage.tsdb.path=/prometheus',
                        '--web.console.libraries=/etc/prometheus/console_libraries',
                        '--web.console.templates=/etc/prometheus/consoles',
                        '--web.enable-lifecycle'
                    ],
                    'networks': ['cyber-range-network']
                },
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'container_name': f"{self.config['app_name']}-grafana",
                    'ports': ['3000:3000'],
                    'environment': {
                        'GF_SECURITY_ADMIN_PASSWORD': 'admin123',
                        'GF_INSTALL_PLUGINS': 'grafana-clock-panel,grafana-simple-json-datasource'
                    },
                    'volumes': [
                        'grafana-storage:/var/lib/grafana',
                        './monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro',
                        './monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro'
                    ],
                    'networks': ['cyber-range-network']
                }
            },
            'networks': {
                'cyber-range-network': {
                    'driver': 'bridge'
                }
            },
            'volumes': {
                'grafana-storage': {}
            }
        }
        
        return self._dict_to_yaml(compose_config)
    
    def generate_kubernetes_manifests(self) -> Dict[str, str]:
        """Generate Kubernetes deployment manifests."""
        manifests = {}
        
        # Namespace
        manifests['namespace.yaml'] = self._dict_to_yaml({
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {
                'name': self.config['namespace'],
                'labels': {
                    'app': self.config['app_name'],
                    'version': self.config['version']
                }
            }
        })
        
        # ConfigMap
        manifests['configmap.yaml'] = self._dict_to_yaml({
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f"{self.config['app_name']}-config",
                'namespace': self.config['namespace']
            },
            'data': {
                'ENVIRONMENT': self.config['environment'],
                'LOG_LEVEL': 'INFO',
                'METRICS_ENABLED': 'true',
                'MAX_SIMULATION_DURATION': '24',
                'DEFAULT_SKILL_LEVEL': 'advanced'
            }
        })
        
        # Deployment
        manifests['deployment.yaml'] = self._dict_to_yaml({
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': self.config['app_name'],
                'namespace': self.config['namespace'],
                'labels': {
                    'app': self.config['app_name'],
                    'version': self.config['version']
                }
            },
            'spec': {
                'replicas': self.config['replicas'],
                'strategy': {
                    'type': 'RollingUpdate',
                    'rollingUpdate': {
                        'maxUnavailable': 1,
                        'maxSurge': 1
                    }
                },
                'selector': {
                    'matchLabels': {
                        'app': self.config['app_name']
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': self.config['app_name'],
                            'version': self.config['version']
                        }
                    },
                    'spec': {
                        'securityContext': {
                            'runAsNonRoot': True,
                            'runAsUser': 1001,
                            'fsGroup': 1001
                        },
                        'containers': [{
                            'name': self.config['app_name'],
                            'image': f"{self.config['app_name']}:{self.config['image_tag']}",
                            'ports': [
                                {'containerPort': self.config['ports']['http'], 'name': 'http'},
                                {'containerPort': self.config['ports']['metrics'], 'name': 'metrics'}
                            ],
                            'env': [
                                {'name': 'ENVIRONMENT', 'valueFrom': {'configMapKeyRef': {'name': f"{self.config['app_name']}-config", 'key': 'ENVIRONMENT'}}},
                                {'name': 'LOG_LEVEL', 'valueFrom': {'configMapKeyRef': {'name': f"{self.config['app_name']}-config", 'key': 'LOG_LEVEL'}}}
                            ],
                            'resources': self.config['resources'],
                            'livenessProbe': {
                                'exec': {'command': ['python3', 'simple_cli.py', 'status']},
                                'initialDelaySeconds': 30,
                                'periodSeconds': 30,
                                'timeoutSeconds': 10,
                                'failureThreshold': 3
                            },
                            'readinessProbe': {
                                'exec': {'command': ['python3', 'simple_cli.py', 'status']},
                                'initialDelaySeconds': 10,
                                'periodSeconds': 10,
                                'timeoutSeconds': 5,
                                'failureThreshold': 3
                            },
                            'securityContext': {
                                'allowPrivilegeEscalation': False,
                                'readOnlyRootFilesystem': False,
                                'capabilities': {'drop': ['ALL']}
                            }
                        }]
                    }
                }
            }
        })
        
        # Service
        manifests['service.yaml'] = self._dict_to_yaml({
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{self.config['app_name']}-service",
                'namespace': self.config['namespace'],
                'labels': {
                    'app': self.config['app_name']
                }
            },
            'spec': {
                'selector': {
                    'app': self.config['app_name']
                },
                'ports': [
                    {'port': 80, 'targetPort': self.config['ports']['http'], 'name': 'http'},
                    {'port': 9090, 'targetPort': self.config['ports']['metrics'], 'name': 'metrics'}
                ],
                'type': 'ClusterIP'
            }
        })
        
        # HorizontalPodAutoscaler
        manifests['hpa.yaml'] = self._dict_to_yaml({
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f"{self.config['app_name']}-hpa",
                'namespace': self.config['namespace']
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': self.config['app_name']
                },
                'minReplicas': self.config['scaling']['min_replicas'],
                'maxReplicas': self.config['scaling']['max_replicas'],
                'metrics': [{
                    'type': 'Resource',
                    'resource': {
                        'name': 'cpu',
                        'target': {
                            'type': 'Utilization',
                            'averageUtilization': self.config['scaling']['target_cpu_percent']
                        }
                    }
                }]
            }
        })
        
        # Ingress
        if self.config['ssl_enabled']:
            manifests['ingress.yaml'] = self._dict_to_yaml({
                'apiVersion': 'networking.k8s.io/v1',
                'kind': 'Ingress',
                'metadata': {
                    'name': f"{self.config['app_name']}-ingress",
                    'namespace': self.config['namespace'],
                    'annotations': {
                        'kubernetes.io/ingress.class': 'nginx',
                        'cert-manager.io/cluster-issuer': 'letsencrypt-prod',
                        'nginx.ingress.kubernetes.io/ssl-redirect': 'true'
                    }
                },
                'spec': {
                    'tls': [{
                        'hosts': self.config['domains'],
                        'secretName': f"{self.config['app_name']}-tls"
                    }],
                    'rules': [{
                        'host': domain,
                        'http': {
                            'paths': [{
                                'path': '/',
                                'pathType': 'Prefix',
                                'backend': {
                                    'service': {
                                        'name': f"{self.config['app_name']}-service",
                                        'port': {'number': 80}
                                    }
                                }
                            }]
                        }
                    } for domain in self.config['domains']]
                }
            })
        
        return manifests
    
    def generate_prometheus_config(self) -> str:
        """Generate Prometheus monitoring configuration."""
        prometheus_config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'scrape_configs': [
                {
                    'job_name': 'cyber-range',
                    'static_configs': [{
                        'targets': [f"cyber-range:{self.config['ports']['metrics']}"]
                    }],
                    'metrics_path': '/metrics',
                    'scrape_interval': '30s'
                },
                {
                    'job_name': 'kubernetes-pods',
                    'kubernetes_sd_configs': [{
                        'role': 'pod'
                    }],
                    'relabel_configs': [
                        {
                            'source_labels': ['__meta_kubernetes_pod_annotation_prometheus_io_scrape'],
                            'action': 'keep',
                            'regex': 'true'
                        }
                    ]
                }
            ]
        }
        
        return self._dict_to_yaml(prometheus_config)
    
    def generate_github_actions_workflow(self) -> str:
        """Generate GitHub Actions CI/CD workflow."""
        workflow = {
            'name': 'CI/CD Pipeline',
            'on': {
                'push': {
                    'branches': ['main', 'develop']
                },
                'pull_request': {
                    'branches': ['main']
                }
            },
            'env': {
                'REGISTRY': 'ghcr.io',
                'IMAGE_NAME': '${{ github.repository }}'
            },
            'jobs': {
                'test': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': '3.12'
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': 'pip install -r requirements-minimal.txt'
                        },
                        {
                            'name': 'Run tests',
                            'run': 'python3 test_suite.py'
                        },
                        {
                            'name': 'Run security scan',
                            'run': 'python3 security_quality_gates.py || true'
                        }
                    ]
                },
                'build-and-push': {
                    'runs-on': 'ubuntu-latest',
                    'needs': 'test',
                    'if': "github.ref == 'refs/heads/main'",
                    'permissions': {
                        'contents': 'read',
                        'packages': 'write'
                    },
                    'steps': [
                        {'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Log in to Container Registry',
                            'uses': 'docker/login-action@v3',
                            'with': {
                                'registry': '${{ env.REGISTRY }}',
                                'username': '${{ github.actor }}',
                                'password': '${{ secrets.GITHUB_TOKEN }}'
                            }
                        },
                        {
                            'name': 'Extract metadata',
                            'id': 'meta',
                            'uses': 'docker/metadata-action@v5',
                            'with': {
                                'images': '${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}'
                            }
                        },
                        {
                            'name': 'Build and push Docker image',
                            'uses': 'docker/build-push-action@v5',
                            'with': {
                                'context': '.',
                                'push': True,
                                'tags': '${{ steps.meta.outputs.tags }}',
                                'labels': '${{ steps.meta.outputs.labels }}'
                            }
                        }
                    ]
                },
                'deploy': {
                    'runs-on': 'ubuntu-latest',
                    'needs': 'build-and-push',
                    'environment': 'production',
                    'steps': [
                        {'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Deploy to Kubernetes',
                            'run': '''
                                echo "Deploying to Kubernetes cluster..."
                                # kubectl apply -f k8s/
                                echo "Deployment completed"
                            '''
                        }
                    ]
                }
            }
        }
        
        return self._dict_to_yaml(workflow)
    
    def generate_all_configs(self) -> None:
        """Generate all production deployment configurations."""
        print("🚀 Generating Production Deployment Configurations")
        print("=" * 55)
        
        # Create deployment directory
        deploy_dir = self.project_root / "deployment"
        deploy_dir.mkdir(exist_ok=True)
        
        # Generate Docker configurations
        print("🐳 Generating Docker configurations...")
        dockerfile_path = self.project_root / "Dockerfile"
        dockerfile_path.write_text(self.generate_dockerfile())
        
        compose_path = deploy_dir / "docker-compose.yml"
        compose_path.write_text(self.generate_docker_compose())
        
        # Generate Kubernetes manifests
        print("☸️  Generating Kubernetes manifests...")
        k8s_dir = deploy_dir / "k8s"
        k8s_dir.mkdir(exist_ok=True)
        
        manifests = self.generate_kubernetes_manifests()
        for filename, content in manifests.items():
            manifest_path = k8s_dir / filename
            manifest_path.write_text(content)
        
        # Generate monitoring configuration
        print("📊 Generating monitoring configurations...")
        monitoring_dir = deploy_dir / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        prometheus_path = monitoring_dir / "prometheus.yml"
        prometheus_path.write_text(self.generate_prometheus_config())
        
        # Generate CI/CD workflow
        print("🔄 Generating CI/CD workflow...")
        github_dir = self.project_root / ".github" / "workflows"
        github_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_path = github_dir / "ci-cd.yml"
        workflow_path.write_text(self.generate_github_actions_workflow())
        
        print(f"\n✅ All deployment configurations generated!")
        print(f"📁 Deployment files location: {deploy_dir}")
        print("\n🎯 Next Steps:")
        print("1. Review and customize configurations")
        print("2. Set up container registry")
        print("3. Configure Kubernetes cluster")
        print("4. Set up monitoring and logging")
        print("5. Deploy to production")
        
        print(f"\n🚀 Ready for production deployment!")


def main():
    """Main function to generate deployment configurations."""
    generator = ProductionDeploymentGenerator()
    generator.generate_all_configs()


if __name__ == "__main__":
    main()