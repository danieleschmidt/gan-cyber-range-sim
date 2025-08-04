"""Network isolation and security controls."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class IsolationLevel(Enum):
    """Network isolation levels."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class NetworkRule:
    """Network access rule."""
    source: str
    destination: str
    port: int
    protocol: str = "TCP"
    action: str = "ALLOW"  # ALLOW, DENY
    priority: int = 100
    description: str = ""


@dataclass
class IsolationPolicy:
    """Network isolation policy."""
    name: str
    level: IsolationLevel
    rules: List[NetworkRule] = field(default_factory=list)
    allowed_external_ips: Set[str] = field(default_factory=set)
    allowed_internal_subnets: Set[str] = field(default_factory=set)
    blocked_ports: Set[int] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NetworkIsolation:
    """Network isolation manager for cyber range."""
    
    def __init__(self, isolation_level: IsolationLevel = IsolationLevel.STRICT):
        self.isolation_level = isolation_level
        self.policies: Dict[str, IsolationPolicy] = {}
        self.active_rules: List[NetworkRule] = []
        self.logger = logging.getLogger("NetworkIsolation")
        
        # Default isolation settings
        self._setup_default_policies()
    
    def _setup_default_policies(self) -> None:
        """Setup default isolation policies."""
        # Basic isolation policy
        basic_policy = IsolationPolicy(
            name="basic",
            level=IsolationLevel.BASIC,
            allowed_internal_subnets={"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"},
            blocked_ports={22, 23, 3389}  # SSH, Telnet, RDP
        )
        
        basic_policy.rules.extend([
            NetworkRule(
                source="any",
                destination="internal",
                port=80,
                action="ALLOW",
                description="Allow HTTP traffic"
            ),
            NetworkRule(
                source="any",
                destination="internal",
                port=443,
                action="ALLOW",
                description="Allow HTTPS traffic"
            ),
            NetworkRule(
                source="external",
                destination="internal",
                port=22,
                action="DENY",
                priority=10,
                description="Block external SSH"
            )
        ])
        
        # Strict isolation policy
        strict_policy = IsolationPolicy(
            name="strict",
            level=IsolationLevel.STRICT,
            allowed_internal_subnets={"10.0.1.0/24"},  # Very limited subnet
            blocked_ports={22, 23, 21, 3389, 5985, 5986, 135, 139, 445}
        )
        
        strict_policy.rules.extend([
            NetworkRule(
                source="internal",
                destination="internal",
                port=80,
                action="ALLOW",
                description="Internal HTTP only"
            ),
            NetworkRule(
                source="internal",
                destination="internal",
                port=443,
                action="ALLOW",
                description="Internal HTTPS only"
            ),
            NetworkRule(
                source="external",
                destination="internal",
                port=0,  # All ports
                action="DENY",
                priority=1,
                description="Block all external traffic"
            ),
            NetworkRule(
                source="internal",
                destination="external",
                port=0,  # All ports
                action="DENY",
                priority=1,
                description="Block all outbound traffic"
            )
        ])
        
        # Paranoid isolation policy
        paranoid_policy = IsolationPolicy(
            name="paranoid",
            level=IsolationLevel.PARANOID,
            allowed_internal_subnets={"10.0.1.0/26"},  # Very small subnet
            blocked_ports=set(range(1, 1024))  # Block all privileged ports
        )
        
        paranoid_policy.rules.extend([
            NetworkRule(
                source="any",
                destination="any",
                port=0,
                action="DENY",
                priority=1,
                description="Default deny all"
            ),
            NetworkRule(
                source="simulation",
                destination="simulation",
                port=8080,
                action="ALLOW",
                priority=50,
                description="Allow simulation internal communication"
            )
        ])
        
        # Register policies
        self.policies["basic"] = basic_policy
        self.policies["strict"] = strict_policy
        self.policies["paranoid"] = paranoid_policy
    
    async def apply_isolation_policy(self, policy_name: str, namespace: str = "cyber-range") -> bool:
        """Apply network isolation policy to namespace."""
        if policy_name not in self.policies:
            self.logger.error(f"Unknown policy: {policy_name}")
            return False
        
        policy = self.policies[policy_name]
        self.logger.info(f"Applying isolation policy '{policy_name}' to namespace '{namespace}'")
        
        try:
            # Generate Kubernetes network policies
            k8s_policies = self._generate_kubernetes_policies(policy, namespace)
            
            # Apply policies (would integrate with Kubernetes API)
            for k8s_policy in k8s_policies:
                success = await self._apply_kubernetes_policy(k8s_policy, namespace)
                if not success:
                    self.logger.error(f"Failed to apply policy: {k8s_policy['metadata']['name']}")
                    return False
            
            # Update active rules
            self.active_rules = policy.rules.copy()
            
            self.logger.info(f"Successfully applied isolation policy '{policy_name}'")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to apply isolation policy: {e}")
            return False
    
    def _generate_kubernetes_policies(self, policy: IsolationPolicy, namespace: str) -> List[Dict[str, Any]]:
        """Generate Kubernetes NetworkPolicy objects."""
        policies = []
        
        # Default deny all policy
        deny_all = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"{policy.name}-deny-all",
                "namespace": namespace
            },
            "spec": {
                "podSelector": {},
                "policyTypes": ["Ingress", "Egress"]
            }
        }
        policies.append(deny_all)
        
        # Allow internal communication policy
        if policy.allowed_internal_subnets:
            allow_internal = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {
                    "name": f"{policy.name}-allow-internal",
                    "namespace": namespace
                },
                "spec": {
                    "podSelector": {
                        "matchLabels": {
                            "app": "gan-cyber-range"
                        }
                    },
                    "policyTypes": ["Ingress", "Egress"],
                    "ingress": [
                        {
                            "from": [
                                {
                                    "namespaceSelector": {
                                        "matchLabels": {
                                            "name": namespace
                                        }
                                    }
                                }
                            ]
                        }
                    ],
                    "egress": [
                        {
                            "to": [
                                {
                                    "namespaceSelector": {
                                        "matchLabels": {
                                            "name": namespace
                                        }
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
            policies.append(allow_internal)
        
        # Generate policies for specific rules
        for i, rule in enumerate(policy.rules):
            if rule.action == "ALLOW":
                rule_policy = self._generate_rule_policy(rule, policy.name, namespace, i)
                if rule_policy:
                    policies.append(rule_policy)
        
        return policies
    
    def _generate_rule_policy(
        self,
        rule: NetworkRule,
        policy_name: str,
        namespace: str,
        rule_index: int
    ) -> Optional[Dict[str, Any]]:
        """Generate Kubernetes policy for a specific rule."""
        if rule.action != "ALLOW":
            return None  # Only generate policies for ALLOW rules
        
        policy = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"{policy_name}-rule-{rule_index}",
                "namespace": namespace
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app": "gan-cyber-range"
                    }
                },
                "policyTypes": []
            }
        }
        
        # Configure ingress rules
        if rule.source != "internal":
            policy["spec"]["policyTypes"].append("Ingress")
            ingress_rule = {}
            
            if rule.port > 0:
                ingress_rule["ports"] = [
                    {
                        "protocol": rule.protocol,
                        "port": rule.port
                    }
                ]
            
            if rule.source == "any":
                # Allow from anywhere (use with caution)
                pass
            elif rule.source == "internal":
                ingress_rule["from"] = [
                    {
                        "namespaceSelector": {
                            "matchLabels": {
                                "name": namespace
                            }
                        }
                    }
                ]
            
            policy["spec"]["ingress"] = [ingress_rule]
        
        # Configure egress rules
        if rule.destination != "internal":
            policy["spec"]["policyTypes"].append("Egress")
            egress_rule = {}
            
            if rule.port > 0:
                egress_rule["ports"] = [
                    {
                        "protocol": rule.protocol,
                        "port": rule.port
                    }
                ]
            
            if rule.destination == "external":
                # Allow to external (specific external IPs could be configured)
                pass
            elif rule.destination == "internal":
                egress_rule["to"] = [
                    {
                        "namespaceSelector": {
                            "matchLabels": {
                                "name": namespace
                            }
                        }
                    }
                ]
            
            policy["spec"]["egress"] = [egress_rule]
        
        return policy
    
    async def _apply_kubernetes_policy(self, policy: Dict[str, Any], namespace: str) -> bool:
        """Apply Kubernetes network policy."""
        try:
            # This would integrate with the KubernetesManager
            # For now, we'll simulate the application
            policy_name = policy["metadata"]["name"]
            self.logger.info(f"Applied Kubernetes policy: {policy_name}")
            
            # Simulate async operation
            await asyncio.sleep(0.1)
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to apply Kubernetes policy: {e}")
            return False
    
    def check_network_access(
        self,
        source_ip: str,
        destination_ip: str,
        port: int,
        protocol: str = "TCP"
    ) -> bool:
        """Check if network access is allowed by current policies."""
        # Check against active rules
        for rule in sorted(self.active_rules, key=lambda r: r.priority):
            if self._rule_matches(rule, source_ip, destination_ip, port, protocol):
                allowed = rule.action == "ALLOW"
                self.logger.debug(
                    f"Network access {source_ip}:{port} -> {destination_ip}:{port} : "
                    f"{'ALLOWED' if allowed else 'DENIED'} by rule: {rule.description}"
                )
                return allowed
        
        # Default deny if no rules match
        self.logger.debug(
            f"Network access {source_ip}:{port} -> {destination_ip}:{port} : "
            f"DENIED (no matching rules)"
        )
        return False
    
    def _rule_matches(
        self,
        rule: NetworkRule,
        source_ip: str,
        destination_ip: str,
        port: int,
        protocol: str
    ) -> bool:
        """Check if a rule matches the given network access."""
        # Check protocol
        if rule.protocol.upper() != protocol.upper():
            return False
        
        # Check port (0 means all ports)
        if rule.port != 0 and rule.port != port:
            return False
        
        # Check source
        if rule.source == "any":
            source_match = True
        elif rule.source == "internal":
            source_match = self._is_internal_ip(source_ip)
        elif rule.source == "external":
            source_match = not self._is_internal_ip(source_ip)
        else:
            source_match = source_ip == rule.source
        
        # Check destination
        if rule.destination == "any":
            dest_match = True
        elif rule.destination == "internal":
            dest_match = self._is_internal_ip(destination_ip)
        elif rule.destination == "external":
            dest_match = not self._is_internal_ip(destination_ip)
        else:
            dest_match = destination_ip == rule.destination
        
        return source_match and dest_match
    
    def _is_internal_ip(self, ip: str) -> bool:
        """Check if IP is in internal subnet ranges."""
        import ipaddress
        
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            # Check against configured internal subnets
            for policy in self.policies.values():
                for subnet_str in policy.allowed_internal_subnets:
                    subnet = ipaddress.ip_network(subnet_str)
                    if ip_obj in subnet:
                        return True
            
            # Default internal ranges
            default_internal = [
                ipaddress.ip_network("10.0.0.0/8"),
                ipaddress.ip_network("172.16.0.0/12"),
                ipaddress.ip_network("192.168.0.0/16"),
                ipaddress.ip_network("127.0.0.0/8")
            ]
            
            for subnet in default_internal:
                if ip_obj in subnet:
                    return True
            
            return False
        
        except ValueError:
            return False
    
    def add_custom_rule(
        self,
        source: str,
        destination: str,
        port: int,
        action: str = "ALLOW",
        protocol: str = "TCP",
        description: str = ""
    ) -> bool:
        """Add a custom network rule."""
        try:
            rule = NetworkRule(
                source=source,
                destination=destination,
                port=port,
                protocol=protocol,
                action=action.upper(),
                description=description or f"Custom rule: {source} -> {destination}:{port}"
            )
            
            self.active_rules.append(rule)
            self.logger.info(f"Added custom rule: {rule.description}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to add custom rule: {e}")
            return False
    
    def remove_custom_rule(self, description: str) -> bool:
        """Remove a custom network rule by description."""
        initial_count = len(self.active_rules)
        self.active_rules = [
            rule for rule in self.active_rules
            if rule.description != description
        ]
        
        removed_count = initial_count - len(self.active_rules)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} rule(s) with description: {description}")
            return True
        else:
            self.logger.warning(f"No rules found with description: {description}")
            return False
    
    def get_isolation_status(self) -> Dict[str, Any]:
        """Get current isolation status."""
        return {
            "isolation_level": self.isolation_level.value,
            "active_policies": list(self.policies.keys()),
            "active_rules_count": len(self.active_rules),
            "rules": [
                {
                    "source": rule.source,
                    "destination": rule.destination,
                    "port": rule.port,
                    "protocol": rule.protocol,
                    "action": rule.action,
                    "priority": rule.priority,
                    "description": rule.description
                }
                for rule in self.active_rules
            ]
        }
    
    async def cleanup_policies(self, namespace: str) -> bool:
        """Clean up all network policies in namespace."""
        try:
            self.logger.info(f"Cleaning up network policies in namespace: {namespace}")
            
            # This would integrate with Kubernetes API to delete policies
            # For now, simulate cleanup
            await asyncio.sleep(0.1)
            
            self.active_rules.clear()
            self.logger.info("Network policies cleaned up successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup network policies: {e}")
            return False