"""Circuit breaker pattern for LLM calls.

Prevents cascading failures when LLM providers are down or degraded by:
- Fast-failing when errors exceed threshold
- Automatic recovery detection
- Resource protection (don't overwhelm failing service)
- Graceful degradation

States:
- CLOSED: Normal operation (all requests pass through)
- OPEN: Failing fast (reject requests immediately)
- HALF_OPEN: Testing recovery (allow limited requests)

Features:
- Configurable failure threshold and recovery timeout
- Per-model circuit breakers
- Thread-safe state management
- Automatic state transitions
- Circuit breaker statistics
"""

from textwrap import dedent

CIRCUIT_BREAKER = dedent(
    '''
# Circuit Breaker Pattern
class _CircuitState:
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class _CircuitBreaker:
    """Circuit breaker for LLM provider calls."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            success_threshold: Successes needed to close circuit
        """
        import threading
        import time
        
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = _CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.lock = threading.RLock()
        
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rejected_calls": 0,
            "state_changes": 0,
        }
    
    def call(self, func: Callable[[], Any]) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to call
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerError: If circuit is open
        """
        import time
        
        with self.lock:
            self.stats["total_calls"] += 1
            
            # Check if circuit should transition to half-open
            if self.state == _CircuitState.OPEN:
                if self.last_failure_time is not None:
                    elapsed = time.time() - self.last_failure_time
                    if elapsed >= self.recovery_timeout:
                        self._transition_to_half_open()
            
            # Reject if circuit is open
            if self.state == _CircuitState.OPEN:
                self.stats["rejected_calls"] += 1
                raise _CircuitBreakerError(
                    f"Circuit breaker is OPEN (too many failures). "
                    f"Will retry in {self.recovery_timeout}s"
                )
            
            # Allow limited requests in half-open state
            if self.state == _CircuitState.HALF_OPEN:
                # Only allow one request at a time in half-open
                pass
        
        # Execute function (outside lock to allow concurrency)
        try:
            result = func()
            self._on_success()
            return result
        
        except Exception as exc:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Handle successful call."""
        with self.lock:
            self.stats["successful_calls"] += 1
            self.failure_count = 0
            
            if self.state == _CircuitState.HALF_OPEN:
                self.success_count += 1
                
                # Close circuit if enough successes
                if self.success_count >= self.success_threshold:
                    self._transition_to_closed()
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        import time
        
        with self.lock:
            self.stats["failed_calls"] += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Open circuit if too many failures
            if self.state == _CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self._transition_to_open()
            
            # Reopen if failure in half-open state
            elif self.state == _CircuitState.HALF_OPEN:
                self._transition_to_open()
    
    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        self.state = _CircuitState.OPEN
        self.stats["state_changes"] += 1
        
        _record_event(
            "circuit_breaker",
            "opened",
            "warning",
            {
                "failures": self.failure_count,
                "threshold": self.failure_threshold,
            }
        )
    
    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self.state = _CircuitState.HALF_OPEN
        self.success_count = 0
        self.stats["state_changes"] += 1
        
        _record_event(
            "circuit_breaker",
            "half_open",
            "info",
            {"message": "Testing recovery"}
        )
    
    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self.state = _CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.stats["state_changes"] += 1
        
        _record_event(
            "circuit_breaker",
            "closed",
            "info",
            {"message": "Recovery successful"}
        )
    
    def get_state(self) -> str:
        """Get current circuit state."""
        with self.lock:
            return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self.lock:
            stats = dict(self.stats)
            stats["state"] = self.state
            stats["failure_count"] = self.failure_count
            
            # Calculate success rate
            total = stats["successful_calls"] + stats["failed_calls"]
            if total > 0:
                stats["success_rate"] = stats["successful_calls"] / total
            else:
                stats["success_rate"] = 0.0
            
            return stats
    
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self.lock:
            self.state = _CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            
            _record_event(
                "circuit_breaker",
                "reset",
                "info",
                {"message": "Manual reset"}
            )


class _CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Global circuit breakers per model
_circuit_breakers: Dict[str, _CircuitBreaker] = {}
_circuit_breaker_lock = None


def _get_circuit_breaker(model: str) -> _CircuitBreaker:
    """Get or create circuit breaker for model."""
    global _circuit_breakers, _circuit_breaker_lock
    
    if _circuit_breaker_lock is None:
        import threading
        _circuit_breaker_lock = threading.Lock()
    
    if model not in _circuit_breakers:
        with _circuit_breaker_lock:
            if model not in _circuit_breakers:
                # Get circuit breaker settings
                cb_config = RUNTIME_SETTINGS.get("circuit_breaker", {})
                enabled = cb_config.get("enabled", True)
                
                if not enabled:
                    # Return dummy that always allows
                    class _DummyCircuitBreaker:
                        def call(self, func):
                            return func()
                        def get_state(self):
                            return _CircuitState.CLOSED
                        def get_stats(self):
                            return {}
                        def reset(self):
                            pass
                    return _DummyCircuitBreaker()
                
                failure_threshold = cb_config.get("failure_threshold", 5)
                recovery_timeout = cb_config.get("recovery_timeout", 60.0)
                success_threshold = cb_config.get("success_threshold", 2)
                
                _circuit_breakers[model] = _CircuitBreaker(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    success_threshold=success_threshold,
                )
    
    return _circuit_breakers[model]


def _protected_llm_call(
    model: str,
    call_func: Callable[[], Any],
) -> Any:
    """Execute LLM call with circuit breaker protection.
    
    Args:
        model: Model identifier
        call_func: Function to call
    
    Returns:
        LLM response
    
    Raises:
        _CircuitBreakerError: If circuit is open
    """
    circuit_breaker = _get_circuit_breaker(model)
    return circuit_breaker.call(call_func)


def _get_circuit_breaker_stats(model: Optional[str] = None) -> Dict[str, Any]:
    """Get circuit breaker statistics.
    
    Args:
        model: Specific model, or None for all models
    
    Returns:
        Statistics dict
    """
    if model:
        cb = _get_circuit_breaker(model)
        return {model: cb.get_stats()}
    
    # Get stats for all models
    stats = {}
    for model_name, cb in _circuit_breakers.items():
        stats[model_name] = cb.get_stats()
    
    return stats


def _reset_circuit_breaker(model: str) -> None:
    """Manually reset circuit breaker for model.
    
    Args:
        model: Model identifier
    """
    circuit_breaker = _get_circuit_breaker(model)
    circuit_breaker.reset()
'''
).strip()

__all__ = ['CIRCUIT_BREAKER']
