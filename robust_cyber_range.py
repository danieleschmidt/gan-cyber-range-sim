#!/usr/bin/env python3
"""Robust, production-ready GAN Cyber Range Simulator with comprehensive error handling."""

import sys
import os
import asyncio
import logging
import json
import signal
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import uuid
import hashlib
import time
from contextlib import asynccontextmanager
import tempfile

# Add src to path for imports  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from minimal_test import MockCyberRange, MockRedTeamAgent, MockBlueTeamAgent, MockAgentAction


@dataclass
class SystemHealth:
    """System health status tracking."""
    status: str = "unknown"  # healthy, degraded, unhealthy
    last_check: datetime = field(default_factory=datetime.now)
    components: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    uptime_seconds: float = 0.0
    
    def is_healthy(self) -> bool:
        return self.status == "healthy"


class SecurityError(Exception):
    """Security-related errors."""
    pass


class ValidationError(Exception):
    """Input validation errors."""
    pass


class SimulationError(Exception):
    """Simulation execution errors."""
    pass


class ResourceExhaustedError(Exception):
    """Resource limit exceeded errors."""
    pass


class RobustLogger:
    """Enhanced logging with structured output and error tracking."""
    
    def __init__(self, name: str, log_level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Structured logging handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s'
        )
        handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        
        self.error_count = 0
        self.warning_count = 0
        self.start_time = time.time()
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning with tracking."""
        self.warning_count += 1
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.warning(message)
    
    def error(self, message: str, exception: Exception = None, **kwargs):
        """Log error with exception details and tracking."""
        self.error_count += 1
        if exception:
            message += f" | Exception: {type(exception).__name__}: {str(exception)}"
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.error(message)
        
        if exception and hasattr(exception, '__traceback__'):
            self.logger.error(f"Traceback: {traceback.format_exception(type(exception), exception, exception.__traceback__)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'uptime_seconds': time.time() - self.start_time
        }


class InputValidator:
    """Input validation and sanitization."""
    
    @staticmethod
    def validate_services(services: List[str]) -> List[str]:
        """Validate and sanitize service list."""
        if not services or not isinstance(services, list):
            raise ValidationError("Services must be a non-empty list")
        
        valid_services = ['webapp', 'database', 'api-gateway', 'fileserver', 'mailserver']
        sanitized = []
        
        for service in services:
            if not isinstance(service, str):
                raise ValidationError(f"Service name must be string, got {type(service)}")
            
            service = service.strip().lower()
            if not service:
                continue
            
            # Basic sanitization
            if not service.replace('-', '').replace('_', '').isalnum():
                raise ValidationError(f"Invalid service name: {service}")
            
            if service in valid_services:
                sanitized.append(service)
            else:
                raise ValidationError(f"Unknown service: {service}. Valid services: {valid_services}")
        
        if not sanitized:
            raise ValidationError("No valid services specified")
        
        return sanitized
    
    @staticmethod
    def validate_duration(duration: Union[str, float]) -> float:
        """Validate simulation duration."""
        try:
            duration = float(duration)
        except (ValueError, TypeError):
            raise ValidationError(f"Duration must be a number, got {type(duration)}")
        
        if duration <= 0:
            raise ValidationError("Duration must be positive")
        
        if duration > 24:  # Max 24 hours
            raise ValidationError("Duration cannot exceed 24 hours")
        
        return duration
    
    @staticmethod
    def validate_skill_level(skill: str) -> str:
        """Validate agent skill level."""
        if not isinstance(skill, str):
            raise ValidationError(f"Skill level must be string, got {type(skill)}")
        
        skill = skill.strip().lower()
        valid_skills = ['beginner', 'intermediate', 'advanced']
        
        if skill not in valid_skills:
            raise ValidationError(f"Invalid skill level: {skill}. Valid levels: {valid_skills}")
        
        return skill
    
    @staticmethod
    def validate_strategy(strategy: str) -> str:
        """Validate defense strategy."""
        if not isinstance(strategy, str):
            raise ValidationError(f"Strategy must be string, got {type(strategy)}")
        
        strategy = strategy.strip().lower()
        valid_strategies = ['reactive', 'proactive']
        
        if strategy not in valid_strategies:
            raise ValidationError(f"Invalid strategy: {strategy}. Valid strategies: {valid_strategies}")
        
        return strategy


class ResourceMonitor:
    """Monitor system resources and enforce limits."""
    
    def __init__(self):
        self.start_time = time.time()
        self.max_memory_mb = 1024  # 1GB limit
        self.max_duration_seconds = 3600  # 1 hour limit
        self.max_actions_per_agent = 1000
        self.logger = RobustLogger("ResourceMonitor")
    
    def check_resource_limits(self, context: Dict[str, Any] = None) -> None:
        """Check if resource limits are exceeded."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Check duration limit
        if uptime > self.max_duration_seconds:
            raise ResourceExhaustedError(f"Maximum runtime exceeded: {uptime:.1f}s")
        
        # Check action count limits
        if context:
            red_actions = context.get('red_actions_count', 0)
            blue_actions = context.get('blue_actions_count', 0)
            
            if red_actions > self.max_actions_per_agent:
                raise ResourceExhaustedError(f"Red team action limit exceeded: {red_actions}")
            
            if blue_actions > self.max_actions_per_agent:
                raise ResourceExhaustedError(f"Blue team action limit exceeded: {blue_actions}")
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        return {
            'uptime_seconds': time.time() - self.start_time,
            'max_memory_mb': self.max_memory_mb,
            'max_duration_seconds': self.max_duration_seconds
        }


class SecurityManager:
    """Security measures and threat detection."""
    
    def __init__(self):
        self.logger = RobustLogger("SecurityManager")
        self.suspicious_patterns = [
            r'\.\./', r'<script', r'eval\(', r'exec\(', r'__import__',
            r'file://', r'javascript:', r'data:'
        ]
        self.failed_attempts = {}
        self.max_failed_attempts = 5
    
    def sanitize_input(self, data: Any) -> Any:
        """Sanitize input data to prevent injection attacks."""
        if isinstance(data, str):
            # Basic sanitization
            import re
            for pattern in self.suspicious_patterns:
                if re.search(pattern, data, re.IGNORECASE):
                    raise SecurityError(f"Suspicious pattern detected: {pattern}")
            return data.strip()
        
        elif isinstance(data, dict):
            return {k: self.sanitize_input(v) for k, v in data.items()}
        
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        
        return data
    
    def check_rate_limits(self, identifier: str) -> None:
        """Check rate limits for operations."""
        current_time = time.time()
        
        # Clean old entries
        cutoff_time = current_time - 300  # 5 minutes
        self.failed_attempts = {
            k: v for k, v in self.failed_attempts.items() 
            if v['last_attempt'] > cutoff_time
        }
        
        if identifier in self.failed_attempts:
            attempts = self.failed_attempts[identifier]
            if attempts['count'] >= self.max_failed_attempts:
                raise SecurityError(f"Rate limit exceeded for {identifier}")
    
    def record_failed_attempt(self, identifier: str) -> None:
        """Record a failed attempt."""
        current_time = time.time()
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = {'count': 0, 'last_attempt': current_time}
        
        self.failed_attempts[identifier]['count'] += 1
        self.failed_attempts[identifier]['last_attempt'] = current_time
        
        self.logger.warning(f"Failed attempt recorded", identifier=identifier, 
                          count=self.failed_attempts[identifier]['count'])


class HealthChecker:
    """System health monitoring and diagnostics."""
    
    def __init__(self):
        self.logger = RobustLogger("HealthChecker")
        self.start_time = datetime.now()
        self.last_health_check = None
        self.health_history = []
    
    async def perform_health_check(self) -> SystemHealth:
        """Perform comprehensive health check."""
        health = SystemHealth(
            last_check=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds()
        )
        
        try:
            # Check core components
            await self._check_core_components(health)
            
            # Check system resources
            await self._check_system_resources(health)
            
            # Check recent errors
            await self._check_error_rates(health)
            
            # Determine overall status
            health.status = self._calculate_overall_status(health)
            
            self.last_health_check = health
            self.health_history.append(health)
            
            # Keep only recent history
            if len(self.health_history) > 100:
                self.health_history = self.health_history[-100:]
            
            return health
            
        except Exception as e:
            health.status = "unhealthy"
            health.errors.append(f"Health check failed: {e}")
            self.logger.error("Health check failed", exception=e)
            return health
    
    async def _check_core_components(self, health: SystemHealth) -> None:
        """Check core system components."""
        # Test basic functionality
        try:
            # Test cyber range creation
            test_range = MockCyberRange(['webapp'])
            health.components['cyber_range'] = "healthy"
            
            # Test agent creation  
            test_red = MockRedTeamAgent()
            test_blue = MockBlueTeamAgent()
            health.components['agents'] = "healthy"
            
        except Exception as e:
            health.components['core'] = "unhealthy"
            health.errors.append(f"Core component check failed: {e}")
    
    async def _check_system_resources(self, health: SystemHealth) -> None:
        """Check system resource usage."""
        try:
            # Check disk space
            import shutil
            disk_usage = shutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb < 1.0:  # Less than 1GB free
                health.components['disk'] = "degraded"
                health.errors.append(f"Low disk space: {free_gb:.1f}GB free")
            else:
                health.components['disk'] = "healthy"
            
            health.metrics['disk_free_gb'] = free_gb
            
        except Exception as e:
            health.components['disk'] = "unknown"
            health.errors.append(f"Disk check failed: {e}")
    
    async def _check_error_rates(self, health: SystemHealth) -> None:
        """Check recent error rates."""
        if len(self.health_history) >= 5:
            recent_checks = self.health_history[-5:]
            total_errors = sum(len(check.errors) for check in recent_checks)
            error_rate = total_errors / len(recent_checks)
            
            health.metrics['error_rate'] = error_rate
            
            if error_rate > 2.0:  # More than 2 errors per check on average
                health.components['error_rate'] = "degraded"
            else:
                health.components['error_rate'] = "healthy"
        else:
            health.components['error_rate'] = "healthy"
    
    def _calculate_overall_status(self, health: SystemHealth) -> str:
        """Calculate overall system health status."""
        component_statuses = list(health.components.values())
        
        if not component_statuses:
            return "unknown"
        
        if "unhealthy" in component_statuses:
            return "unhealthy"
        elif "degraded" in component_statuses:
            return "degraded"
        elif all(status == "healthy" for status in component_statuses):
            return "healthy"
        else:
            return "degraded"


class RobustCyberRange:
    """Production-ready cyber range with comprehensive error handling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = RobustLogger("RobustCyberRange")
        self.resource_monitor = ResourceMonitor()
        self.security_manager = SecurityManager()
        self.health_checker = HealthChecker()
        self.config = self._validate_config(config)
        
        self.simulation_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        self.shutdown_requested = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.logger.info("Robust Cyber Range initialized", 
                        simulation_id=self.simulation_id,
                        config=self.config)
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize configuration."""
        validated = {
            'services': InputValidator.validate_services(config.get('services', ['webapp'])),
            'duration': InputValidator.validate_duration(config.get('duration', 0.1)),
            'red_skill': InputValidator.validate_skill_level(config.get('red_skill', 'advanced')),
            'blue_skill': InputValidator.validate_skill_level(config.get('blue_skill', 'advanced')),
            'strategy': InputValidator.validate_strategy(config.get('strategy', 'proactive')),
            'debug': bool(config.get('debug', False)),
            'max_rounds': min(int(config.get('max_rounds', 100)), 1000),  # Cap at 1000 rounds
            'output_dir': str(config.get('output_dir', tempfile.gettempdir()))
        }
        
        # Sanitize all string values
        for key, value in validated.items():
            if isinstance(value, str):
                validated[key] = self.security_manager.sanitize_input(value)
        
        return validated
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown")
        self.shutdown_requested = True
    
    async def run_simulation(self) -> Dict[str, Any]:
        """Run simulation with comprehensive error handling."""
        self.logger.info("Starting robust simulation", config=self.config)
        
        try:
            # Pre-flight checks
            await self._perform_preflight_checks()
            
            # Initialize components with error handling
            cyber_range, red_team, blue_team = await self._initialize_components()
            
            # Run simulation with monitoring
            results = await self._execute_simulation(cyber_range, red_team, blue_team)
            
            # Post-simulation cleanup
            await self._cleanup_components(cyber_range, red_team, blue_team)
            
            return results
            
        except ValidationError as e:
            self.logger.error("Validation error", exception=e)
            raise
        except SecurityError as e:
            self.logger.error("Security error", exception=e) 
            raise
        except ResourceExhaustedError as e:
            self.logger.error("Resource exhausted", exception=e)
            raise
        except SimulationError as e:
            self.logger.error("Simulation error", exception=e)
            raise
        except Exception as e:
            self.logger.error("Unexpected error", exception=e)
            raise SimulationError(f"Simulation failed: {e}") from e
    
    async def _perform_preflight_checks(self) -> None:
        """Perform pre-simulation checks."""
        self.logger.info("Performing preflight checks")
        
        # Health check
        health = await self.health_checker.perform_health_check()
        if not health.is_healthy():
            raise SimulationError(f"System unhealthy: {health.errors}")
        
        # Resource check
        self.resource_monitor.check_resource_limits()
        
        # Security check
        self.security_manager.check_rate_limits(self.simulation_id)
        
        # Output directory check
        output_dir = Path(self.config['output_dir'])
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        elif not output_dir.is_dir():
            raise ValidationError(f"Output path is not a directory: {output_dir}")
    
    async def _initialize_components(self) -> tuple:
        """Initialize simulation components with error handling."""
        self.logger.info("Initializing simulation components")
        
        try:
            # Create cyber range
            cyber_range = MockCyberRange(
                vulnerable_services=self.config['services']
            )
            
            # Create agents with timeout protection
            red_team = MockRedTeamAgent(
                name=f"RobustRedTeam_{self.simulation_id[:8]}",
                skill_level=self.config['red_skill']
            )
            
            blue_team = MockBlueTeamAgent(
                name=f"RobustBlueTeam_{self.simulation_id[:8]}",
                skill_level=self.config['blue_skill'],
                defense_strategy=self.config['strategy']
            )
            
            return cyber_range, red_team, blue_team
            
        except Exception as e:
            raise SimulationError(f"Component initialization failed: {e}") from e
    
    async def _execute_simulation(self, cyber_range, red_team, blue_team) -> Dict[str, Any]:
        """Execute simulation with monitoring and error recovery."""
        self.logger.info("Executing simulation")
        
        results = {
            'simulation_id': self.simulation_id,
            'start_time': self.start_time.isoformat(),
            'config': self.config,
            'status': 'running',
            'rounds_completed': 0,
            'errors': [],
            'metrics': {}
        }
        
        try:
            # Run simulation with timeout protection
            simulation_timeout = self.config['duration'] * 3600 + 60  # Duration + 1 minute buffer
            
            simulation_results = await asyncio.wait_for(
                cyber_range.simulate(
                    red_team=red_team,
                    blue_team=blue_team, 
                    duration_hours=self.config['duration'],
                    realtime_factor=60
                ),
                timeout=simulation_timeout
            )
            
            # Process results
            results.update({
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'duration': str(simulation_results.duration),
                'total_attacks': simulation_results.total_attacks,
                'services_compromised': simulation_results.services_compromised,
                'attacks_blocked': simulation_results.attacks_blocked,
                'patches_deployed': simulation_results.patches_deployed,
                'compromise_rate': simulation_results.compromise_rate,
                'defense_effectiveness': simulation_results.defense_effectiveness,
                'red_team_actions': simulation_results.red_team_actions,
                'blue_team_actions': simulation_results.blue_team_actions,
                'agent_stats': {
                    'red_team': red_team.get_stats(),
                    'blue_team': blue_team.get_stats()
                }
            })
            
            self.logger.info("Simulation completed successfully", 
                           duration=results['duration'],
                           total_attacks=results['total_attacks'])
            
            return results
            
        except asyncio.TimeoutError:
            results['status'] = 'timeout'
            results['errors'].append('Simulation timed out')
            self.logger.error("Simulation timed out")
            raise SimulationError("Simulation timed out")
            
        except Exception as e:
            results['status'] = 'failed'
            results['errors'].append(str(e))
            self.logger.error("Simulation execution failed", exception=e)
            raise
    
    async def _cleanup_components(self, cyber_range, red_team, blue_team) -> None:
        """Clean up simulation components."""
        self.logger.info("Cleaning up simulation components")
        
        try:
            # Deactivate agents
            if red_team.active:
                red_team.deactivate()
            if blue_team.active:
                blue_team.deactivate()
                
        except Exception as e:
            self.logger.warning("Cleanup warning", exception=e)
    
    async def get_health_status(self) -> SystemHealth:
        """Get current system health."""
        return await self.health_checker.perform_health_check()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        return {
            'simulation_id': self.simulation_id,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'resource_usage': self.resource_monitor.get_resource_usage(),
            'logging_stats': self.logger.get_stats(),
            'config': self.config
        }


class RobustCLI:
    """Production-ready CLI with comprehensive error handling."""
    
    def __init__(self):
        self.logger = RobustLogger("RobustCLI")
    
    async def run_simulation_command(self, args: Dict[str, Any]) -> int:
        """Run simulation command with robust error handling."""
        try:
            self.logger.info("Starting robust simulation", args=args)
            
            # Create robust cyber range
            robust_range = RobustCyberRange(args)
            
            # Run simulation
            results = await robust_range.run_simulation()
            
            # Display results
            self._display_results(results)
            
            # Save results
            self._save_results(results, args.get('output_dir', tempfile.gettempdir()))
            
            return 0
            
        except ValidationError as e:
            print(f"❌ Validation Error: {e}")
            return 1
        except SecurityError as e:
            print(f"🔒 Security Error: {e}")
            return 1
        except ResourceExhaustedError as e:
            print(f"🔥 Resource Error: {e}")
            return 1
        except SimulationError as e:
            print(f"⚠️  Simulation Error: {e}")
            return 1
        except Exception as e:
            print(f"💥 Unexpected Error: {e}")
            if args.get('debug'):
                traceback.print_exc()
            return 1
    
    def _display_results(self, results: Dict[str, Any]) -> None:
        """Display simulation results."""
        print("\n" + "="*60)
        print("🎉 ROBUST SIMULATION COMPLETE!")
        print("="*60)
        
        print(f"📊 SIMULATION RESULTS:")
        print(f"   Status: {results['status']}")
        print(f"   Duration: {results.get('duration', 'N/A')}")
        print(f"   Total Attacks: {results.get('total_attacks', 0)}")
        print(f"   Services Compromised: {results.get('services_compromised', 0)}")
        print(f"   Attacks Blocked: {results.get('attacks_blocked', 0)}")
        print(f"   Defense Effectiveness: {results.get('defense_effectiveness', 0):.2%}")
        
        if results.get('errors'):
            print(f"\n⚠️  ERRORS ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"   • {error}")
    
    def _save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """Save results to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"robust_simulation_{timestamp}.json"
            filepath = Path(output_dir) / filename
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"💾 Results saved to: {filepath}")
            
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")


# Demo function to test robust features
async def demo_robust_system():
    """Demonstrate robust system capabilities."""
    print("🎯 ROBUST GAN CYBER RANGE DEMONSTRATION")
    print("="*60)
    
    # Test various scenarios
    test_configs = [
        # Normal case
        {
            'name': 'Normal Simulation',
            'config': {
                'services': ['webapp', 'database'],
                'duration': 0.05,
                'red_skill': 'advanced',
                'blue_skill': 'advanced',
                'strategy': 'proactive'
            }
        },
        
        # Edge cases
        {
            'name': 'Single Service',
            'config': {
                'services': ['webapp'],
                'duration': 0.01,
                'red_skill': 'beginner',
                'blue_skill': 'beginner'
            }
        }
    ]
    
    cli = RobustCLI()
    
    for test in test_configs:
        print(f"\n🧪 Testing: {test['name']}")
        print("-" * 40)
        
        try:
            exit_code = await cli.run_simulation_command(test['config'])
            status = "✅ PASSED" if exit_code == 0 else "❌ FAILED"
            print(f"{status} - Exit Code: {exit_code}")
        except Exception as e:
            print(f"❌ FAILED - Exception: {e}")
    
    print("\n🎉 Robust system demonstration complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(demo_robust_system())
    else:
        print("Usage: python3 robust_cyber_range.py demo")
        print("This demonstrates the robust, production-ready cyber range system.")