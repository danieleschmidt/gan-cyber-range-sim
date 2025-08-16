"""Comprehensive error handling and recovery system."""

import logging
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union
from functools import wraps
import json


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class CyberRangeError(Exception):
    """Base exception for all cyber range errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "CR_GENERIC",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "traceback": traceback.format_exc() if self.cause else None
        }


class AgentError(CyberRangeError):
    """Errors related to agent operations."""
    
    def __init__(self, message: str, agent_id: Optional[str] = None, **kwargs):
        self.agent_id = agent_id
        super().__init__(
            message, 
            error_code=f"AGENT_{kwargs.get('error_code', 'ERROR')}", 
            **kwargs
        )


class SecurityError(CyberRangeError):
    """Security-related errors."""
    
    def __init__(self, message: str, security_context: Optional[Dict] = None, **kwargs):
        self.security_context = security_context or {}
        super().__init__(
            message,
            error_code=f"SEC_{kwargs.get('error_code', 'VIOLATION')}",
            severity=kwargs.get('severity', ErrorSeverity.HIGH),
            **kwargs
        )


class InfrastructureError(CyberRangeError):
    """Infrastructure and deployment errors."""
    
    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        self.component = component
        super().__init__(
            message,
            error_code=f"INFRA_{kwargs.get('error_code', 'FAILURE')}",
            **kwargs
        )


class ErrorHandler:
    """Centralized error handling and recovery."""
    
    def __init__(self, logger_name: str = "cyber_range"):
        self.logger = logging.getLogger(logger_name)
        self.error_counts = {}
        self.recovery_strategies = {}
        
    def register_recovery_strategy(
        self, 
        error_code: str, 
        strategy_func: callable
    ):
        """Register a recovery strategy for specific error codes."""
        self.recovery_strategies[error_code] = strategy_func
    
    def handle_error(
        self, 
        error: Union[Exception, CyberRangeError],
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True
    ) -> Optional[Any]:
        """Handle error with logging and optional recovery."""
        
        # Convert to CyberRangeError if needed
        if not isinstance(error, CyberRangeError):
            error = CyberRangeError(
                message=str(error),
                cause=error,
                context=context
            )
        
        # Log error
        error_dict = error.to_dict()
        if context:
            error_dict["context"].update(context)
            
        self.logger.error(
            f"Error occurred: {error.error_code}",
            extra={"error_details": error_dict}
        )
        
        # Track error frequency
        self.error_counts[error.error_code] = (
            self.error_counts.get(error.error_code, 0) + 1
        )
        
        # Attempt recovery if enabled and strategy exists
        if attempt_recovery and error.error_code in self.recovery_strategies:
            try:
                recovery_result = self.recovery_strategies[error.error_code](error)
                self.logger.info(f"Recovery successful for {error.error_code}")
                return recovery_result
            except Exception as recovery_error:
                self.logger.error(
                    f"Recovery failed for {error.error_code}: {recovery_error}"
                )
        
        # Re-raise if critical
        if error.severity == ErrorSeverity.CRITICAL:
            raise error
        
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "error_counts": self.error_counts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "registered_strategies": list(self.recovery_strategies.keys())
        }


def error_recovery(
    error_codes: Optional[list] = None,
    max_retries: int = 3,
    backoff_seconds: float = 1.0
):
    """Decorator for automatic error recovery and retries."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_error = e
                    
                    # Check if this error code should trigger recovery
                    if error_codes and hasattr(e, 'error_code'):
                        if e.error_code not in error_codes:
                            raise e
                    
                    # Don't retry on last attempt
                    if attempt == max_retries:
                        break
                    
                    # Exponential backoff
                    import time
                    time.sleep(backoff_seconds * (2 ** attempt))
                    
                    logging.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}"
                    )
            
            # If we get here, all retries failed
            raise last_error
        
        return wrapper
    return decorator


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_error(error: Exception, **kwargs) -> Optional[Any]:
    """Convenience function for global error handling."""
    return global_error_handler.handle_error(error, **kwargs)


def register_recovery_strategy(error_code: str, strategy_func: callable):
    """Convenience function to register recovery strategies globally."""
    global_error_handler.register_recovery_strategy(error_code, strategy_func)


# Pre-defined recovery strategies
def restart_component_strategy(error: CyberRangeError) -> Any:
    """Generic strategy to restart a failed component."""
    if hasattr(error, 'component') and error.component:
        logging.info(f"Attempting to restart component: {error.component}")
        # Implementation would depend on component type
        return {"action": "restart", "component": error.component}
    return None


def reset_agent_strategy(error: AgentError) -> Any:
    """Strategy to reset a failed agent."""
    if error.agent_id:
        logging.info(f"Attempting to reset agent: {error.agent_id}")
        # Implementation would reset agent state
        return {"action": "reset", "agent_id": error.agent_id}
    return None


# Register default recovery strategies
register_recovery_strategy("INFRA_SERVICE_DOWN", restart_component_strategy)
register_recovery_strategy("AGENT_UNRESPONSIVE", reset_agent_strategy)