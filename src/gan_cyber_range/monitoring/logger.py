"""Structured logging for cyber range operations."""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class StructuredLogger:
    """Structured logger for cyber range operations."""
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        log_file: Optional[Path] = None,
        enable_console: bool = True,
        enable_structured: bool = True
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        if enable_console:
            self._setup_console_handler(enable_structured)
        
        if log_file:
            self._setup_file_handler(log_file, enable_structured)
    
    def _setup_console_handler(self, structured: bool = True) -> None:
        """Setup console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, log_file: Path, structured: bool = True) -> None:
        """Setup file logging handler."""
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        
        if structured:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with structured data."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with structured data."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with structured data."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with structured data."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with structured data."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs) -> None:
        """Internal logging method with structured data."""
        extra = {"extra_fields": kwargs} if kwargs else {}
        self.logger.log(level, message, extra=extra)
    
    def log_agent_action(
        self,
        agent_name: str,
        agent_type: str,
        action_type: str,
        target: str,
        success: bool,
        execution_time: float,
        **metadata
    ) -> None:
        """Log agent action with structured data."""
        self.info(
            f"Agent action: {action_type}",
            agent_name=agent_name,
            agent_type=agent_type,
            action_type=action_type,
            target=target,
            success=success,
            execution_time_ms=execution_time * 1000,
            **metadata
        )
    
    def log_simulation_event(
        self,
        event_type: str,
        simulation_id: str,
        round_number: int,
        **metadata
    ) -> None:
        """Log simulation event with structured data."""
        self.info(
            f"Simulation event: {event_type}",
            event_type=event_type,
            simulation_id=simulation_id,
            round_number=round_number,
            **metadata
        )
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        source: str,
        target: str,
        **metadata
    ) -> None:
        """Log security event with structured data."""
        self.warning(
            f"Security event: {event_type}",
            event_type=event_type,
            severity=severity,
            source=source,
            target=target,
            **metadata
        )
    
    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        **metadata
    ) -> None:
        """Log performance metric with structured data."""
        self.info(
            f"Performance metric: {metric_name}",
            metric_name=metric_name,
            value=value,
            unit=unit,
            **metadata
        )
    
    def log_error_with_context(
        self,
        error: Exception,
        context: str,
        **metadata
    ) -> None:
        """Log error with context and structured data."""
        self.error(
            f"Error in {context}: {str(error)}",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            **metadata,
            exc_info=True
        )
    
    def log_kubernetes_operation(
        self,
        operation: str,
        resource_type: str,
        resource_name: str,
        namespace: str,
        success: bool,
        **metadata
    ) -> None:
        """Log Kubernetes operation with structured data."""
        level = self.info if success else self.error
        level(
            f"K8s {operation}: {resource_type}/{resource_name}",
            operation=operation,
            resource_type=resource_type,
            resource_name=resource_name,
            namespace=namespace,
            success=success,
            **metadata
        )
    
    def log_health_check(
        self,
        check_name: str,
        status: str,
        response_time_ms: float,
        **metadata
    ) -> None:
        """Log health check result with structured data."""
        level = self.info if status == "healthy" else self.warning
        level(
            f"Health check {check_name}: {status}",
            check_name=check_name,
            status=status,
            response_time_ms=response_time_ms,
            **metadata
        )


def setup_global_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_structured: bool = True
) -> StructuredLogger:
    """Setup global logging configuration."""
    # Setup main logger
    main_logger = StructuredLogger(
        name="gan_cyber_range",
        level=level,
        log_file=log_dir / "cyber_range.log" if log_dir else None,
        enable_structured=enable_structured
    )
    
    # Setup component loggers
    components = [
        "gan_cyber_range.agents",
        "gan_cyber_range.environment",
        "gan_cyber_range.monitoring",
        "gan_cyber_range.kubernetes"
    ]
    
    for component in components:
        component_logger = logging.getLogger(component)
        component_logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplicate logs by not adding handlers to child loggers
        component_logger.propagate = True
    
    return main_logger


class AuditLogger:
    """Specialized logger for audit events."""
    
    def __init__(self, log_file: Path):
        self.logger = StructuredLogger(
            name="audit",
            level="INFO",
            log_file=log_file,
            enable_console=False,
            enable_structured=True
        )
    
    def log_simulation_start(
        self,
        simulation_id: str,
        user: str,
        config: Dict[str, Any]
    ) -> None:
        """Log simulation start."""
        self.logger.info(
            "Simulation started",
            event_type="simulation_start",
            simulation_id=simulation_id,
            user=user,
            config=config
        )
    
    def log_simulation_end(
        self,
        simulation_id: str,
        user: str,
        duration_seconds: float,
        results: Dict[str, Any]
    ) -> None:
        """Log simulation end."""
        self.logger.info(
            "Simulation completed",
            event_type="simulation_end",
            simulation_id=simulation_id,
            user=user,
            duration_seconds=duration_seconds,
            results=results
        )
    
    def log_deployment_operation(
        self,
        operation: str,
        user: str,
        resources: List[str],
        success: bool
    ) -> None:
        """Log deployment operation."""
        self.logger.info(
            f"Deployment {operation}",
            event_type="deployment_operation",
            operation=operation,
            user=user,
            resources=resources,
            success=success
        )
    
    def log_security_violation(
        self,
        violation_type: str,
        source: str,
        details: Dict[str, Any]
    ) -> None:
        """Log security violation."""
        self.logger.warning(
            f"Security violation: {violation_type}",
            event_type="security_violation",
            violation_type=violation_type,
            source=source,
            details=details
        )