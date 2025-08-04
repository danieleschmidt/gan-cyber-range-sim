"""Kubernetes management for cyber range deployment."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from kubernetes import client, config
from kubernetes.client.rest import ApiException


class KubernetesManager:
    """Manages Kubernetes resources for cyber range deployment."""
    
    def __init__(self, namespace: str = "cyber-range", config_path: Optional[str] = None):
        self.namespace = namespace
        self.logger = logging.getLogger(f"KubernetesManager.{namespace}")
        
        # Initialize Kubernetes client
        try:
            if config_path:
                config.load_kube_config(config_file=config_path)
            else:
                # Try in-cluster config first, then local config
                try:
                    config.load_incluster_config()
                except config.ConfigException:
                    config.load_kube_config()
            
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.networking_v1 = client.NetworkingV1Api()
            
            self.logger.info("Kubernetes client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
    
    async def create_namespace(self) -> bool:
        """Create the cyber range namespace."""
        try:
            namespace_body = client.V1Namespace(
                metadata=client.V1ObjectMeta(
                    name=self.namespace,
                    labels={
                        "app": "gan-cyber-range",
                        "environment": "simulation",
                        "security-isolation": "enabled"
                    }
                )
            )
            
            self.v1.create_namespace(body=namespace_body)
            self.logger.info(f"Created namespace: {self.namespace}")
            return True
            
        except ApiException as e:
            if e.status == 409:  # Already exists
                self.logger.info(f"Namespace {self.namespace} already exists")
                return True
            else:
                self.logger.error(f"Failed to create namespace: {e}")
                return False
    
    async def deploy_vulnerable_service(self, service_config: Dict[str, Any]) -> bool:
        """Deploy a vulnerable service to the cluster."""
        service_name = service_config["name"]
        
        try:
            # Create deployment
            deployment = self._create_deployment_spec(service_config)
            self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )
            
            # Create service
            service = self._create_service_spec(service_config)
            self.v1.create_namespaced_service(
                namespace=self.namespace,
                body=service
            )
            
            self.logger.info(f"Deployed vulnerable service: {service_name}")
            return True
            
        except ApiException as e:
            self.logger.error(f"Failed to deploy service {service_name}: {e}")
            return False
    
    async def deploy_monitoring_stack(self) -> bool:
        """Deploy monitoring stack (Prometheus, Grafana)."""
        try:
            # Deploy Prometheus
            prometheus_config = {
                "name": "prometheus",
                "image": "prom/prometheus:latest",
                "ports": [9090],
                "config_map": "prometheus-config"
            }
            await self.deploy_vulnerable_service(prometheus_config)
            
            # Deploy Grafana
            grafana_config = {
                "name": "grafana",
                "image": "grafana/grafana:latest",
                "ports": [3000],
                "env": {
                    "GF_SECURITY_ADMIN_PASSWORD": "admin123"
                }
            }
            await self.deploy_vulnerable_service(grafana_config)
            
            self.logger.info("Monitoring stack deployed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deploy monitoring stack: {e}")
            return False
    
    async def create_network_policies(self) -> bool:
        """Create network policies for isolation."""
        try:
            # Default deny all policy
            deny_all_policy = client.V1NetworkPolicy(
                metadata=client.V1ObjectMeta(
                    name="default-deny-all",
                    namespace=self.namespace
                ),
                spec=client.V1NetworkPolicySpec(
                    pod_selector=client.V1LabelSelector(),
                    policy_types=["Ingress", "Egress"]
                )
            )
            
            self.networking_v1.create_namespaced_network_policy(
                namespace=self.namespace,
                body=deny_all_policy
            )
            
            # Allow internal communication
            allow_internal_policy = client.V1NetworkPolicy(
                metadata=client.V1ObjectMeta(
                    name="allow-internal",
                    namespace=self.namespace
                ),
                spec=client.V1NetworkPolicySpec(
                    pod_selector=client.V1LabelSelector(
                        match_labels={"app": "gan-cyber-range"}
                    ),
                    ingress=[
                        client.V1NetworkPolicyIngressRule(
                            _from=[
                                client.V1NetworkPolicyPeer(
                                    namespace_selector=client.V1LabelSelector(
                                        match_labels={"name": self.namespace}
                                    )
                                )
                            ]
                        )
                    ],
                    egress=[
                        client.V1NetworkPolicyEgressRule(
                            to=[
                                client.V1NetworkPolicyPeer(
                                    namespace_selector=client.V1LabelSelector(
                                        match_labels={"name": self.namespace}
                                    )
                                )
                            ]
                        )
                    ],
                    policy_types=["Ingress", "Egress"]
                )
            )
            
            self.networking_v1.create_namespaced_network_policy(
                namespace=self.namespace,
                body=allow_internal_policy
            )
            
            self.logger.info("Network policies created")
            return True
            
        except ApiException as e:
            self.logger.error(f"Failed to create network policies: {e}")
            return False
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get status of a deployed service."""
        try:
            # Get deployment status
            deployment = self.apps_v1.read_namespaced_deployment(
                name=service_name,
                namespace=self.namespace
            )
            
            # Get service status
            service = self.v1.read_namespaced_service(
                name=service_name,
                namespace=self.namespace
            )
            
            # Get pods
            pods = self.v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app={service_name}"
            )
            
            return {
                "name": service_name,
                "deployment_ready": deployment.status.ready_replicas or 0,
                "deployment_replicas": deployment.status.replicas or 0,
                "service_ip": service.spec.cluster_ip,
                "service_ports": [port.port for port in service.spec.ports],
                "pods": [
                    {
                        "name": pod.metadata.name,
                        "status": pod.status.phase,
                        "ip": pod.status.pod_ip
                    }
                    for pod in pods.items
                ]
            }
            
        except ApiException as e:
            self.logger.error(f"Failed to get service status for {service_name}: {e}")
            return {"name": service_name, "error": str(e)}
    
    async def scale_service(self, service_name: str, replicas: int) -> bool:
        """Scale a service to specified number of replicas."""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=service_name,
                namespace=self.namespace
            )
            
            deployment.spec.replicas = replicas
            
            self.apps_v1.patch_namespaced_deployment(
                name=service_name,
                namespace=self.namespace,
                body=deployment
            )
            
            self.logger.info(f"Scaled {service_name} to {replicas} replicas")
            return True
            
        except ApiException as e:
            self.logger.error(f"Failed to scale {service_name}: {e}")
            return False
    
    async def delete_service(self, service_name: str) -> bool:
        """Delete a service and its deployment."""
        try:
            # Delete deployment
            self.apps_v1.delete_namespaced_deployment(
                name=service_name,
                namespace=self.namespace
            )
            
            # Delete service
            self.v1.delete_namespaced_service(
                name=service_name,
                namespace=self.namespace
            )
            
            self.logger.info(f"Deleted service: {service_name}")
            return True
            
        except ApiException as e:
            self.logger.error(f"Failed to delete {service_name}: {e}")
            return False
    
    async def cleanup_namespace(self) -> bool:
        """Clean up the entire cyber range namespace."""
        try:
            self.v1.delete_namespace(name=self.namespace)
            self.logger.info(f"Deleted namespace: {self.namespace}")
            return True
            
        except ApiException as e:
            self.logger.error(f"Failed to delete namespace: {e}")
            return False
    
    def _create_deployment_spec(self, service_config: Dict[str, Any]) -> client.V1Deployment:
        """Create deployment specification."""
        name = service_config["name"]
        image = service_config.get("image", f"gan-cyber-range/{name}:latest")
        ports = service_config.get("ports", [80])
        env_vars = service_config.get("env", {})
        
        # Create environment variables
        env_list = [
            client.V1EnvVar(name=key, value=str(value))
            for key, value in env_vars.items()
        ]
        
        # Create container ports
        container_ports = [
            client.V1ContainerPort(container_port=port)
            for port in ports
        ]
        
        # Resource limits for security
        resources = client.V1ResourceRequirements(
            limits={
                "cpu": "500m",
                "memory": "512Mi"
            },
            requests={
                "cpu": "100m",
                "memory": "128Mi"
            }
        )
        
        # Security context
        security_context = client.V1SecurityContext(
            run_as_non_root=True,
            run_as_user=1000,
            read_only_root_filesystem=False,  # Some apps need writable filesystem
            allow_privilege_escalation=False
        )
        
        container = client.V1Container(
            name=name,
            image=image,
            ports=container_ports,
            env=env_list,
            resources=resources,
            security_context=security_context
        )
        
        pod_spec = client.V1PodSpec(
            containers=[container],
            security_context=client.V1PodSecurityContext(
                run_as_non_root=True,
                run_as_user=1000,
                fs_group=2000
            )
        )
        
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    "app": name,
                    "component": "gan-cyber-range",
                    "security": "vulnerable"
                }
            ),
            spec=pod_spec
        )
        
        spec = client.V1DeploymentSpec(
            replicas=service_config.get("replicas", 1),
            selector=client.V1LabelSelector(
                match_labels={"app": name}
            ),
            template=template
        )
        
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name=name,
                namespace=self.namespace,
                labels={
                    "app": name,
                    "component": "gan-cyber-range"
                }
            ),
            spec=spec
        )
        
        return deployment
    
    def _create_service_spec(self, service_config: Dict[str, Any]) -> client.V1Service:
        """Create service specification."""
        name = service_config["name"]
        ports = service_config.get("ports", [80])
        
        service_ports = [
            client.V1ServicePort(
                port=port,
                target_port=port,
                protocol="TCP"
            )
            for port in ports
        ]
        
        spec = client.V1ServiceSpec(
            selector={"app": name},
            ports=service_ports,
            type="ClusterIP"
        )
        
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=name,
                namespace=self.namespace,
                labels={
                    "app": name,
                    "component": "gan-cyber-range"
                }
            ),
            spec=spec
        )
        
        return service
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information."""
        try:
            # Get nodes
            nodes = self.v1.list_node()
            
            # Get namespace status
            namespace_info = self.v1.read_namespace(name=self.namespace)
            
            return {
                "nodes": len(nodes.items),
                "namespace": self.namespace,
                "namespace_status": namespace_info.status.phase,
                "cluster_healthy": True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cluster info: {e}")
            return {
                "cluster_healthy": False,
                "error": str(e)
            }