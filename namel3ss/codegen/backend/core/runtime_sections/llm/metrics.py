"""LLM observability metrics for N3 runtime.

Tracks key metrics for LLM operations:
- Call latency (P50, P95, P99)
- Cache hit/miss rates
- Error rates by provider
- Token usage and costs
- Request volumes
"""

from textwrap import dedent

LLM_METRICS = dedent(
    '''
# LLM Observability Metrics
class _LLMMetrics:
    """Track LLM performance and usage metrics."""
    
    def __init__(self):
        """Initialize metrics tracking."""
        import threading
        from collections import defaultdict
        
        self.lock = threading.RLock()
        self.metrics = {
            "calls": defaultdict(int),  # By model
            "errors": defaultdict(int),  # By error type
            "latencies": defaultdict(list),  # By model
            "tokens": defaultdict(int),  # By model
            "cache_hits": 0,
            "cache_misses": 0,
        }
    
    def record_call(
        self,
        model: str,
        latency_ms: float,
        tokens: Optional[int] = None,
        cached: bool = False,
    ) -> None:
        """Record successful LLM call."""
        with self.lock:
            self.metrics["calls"][model] += 1
            self.metrics["latencies"][model].append(latency_ms)
            
            if tokens:
                self.metrics["tokens"][model] += tokens
            
            if cached:
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["cache_misses"] += 1
    
    def record_error(self, model: str, error_type: str) -> None:
        """Record LLM call error."""
        with self.lock:
            error_key = f"{model}:{error_type}"
            self.metrics["errors"][error_key] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self.lock:
            stats: Dict[str, Any] = {
                "total_calls": sum(self.metrics["calls"].values()),
                "total_errors": sum(self.metrics["errors"].values()),
                "cache_hit_rate": 0.0,
                "by_model": {},
            }
            
            # Calculate cache hit rate
            total_cache_ops = self.metrics["cache_hits"] + self.metrics["cache_misses"]
            if total_cache_ops > 0:
                stats["cache_hit_rate"] = self.metrics["cache_hits"] / total_cache_ops
            
            # Per-model statistics
            for model in self.metrics["calls"].keys():
                latencies = self.metrics["latencies"].get(model, [])
                
                if latencies:
                    latencies_sorted = sorted(latencies)
                    n = len(latencies_sorted)
                    
                    stats["by_model"][model] = {
                        "calls": self.metrics["calls"][model],
                        "tokens": self.metrics["tokens"].get(model, 0),
                        "latency_p50": latencies_sorted[int(n * 0.5)] if n > 0 else 0,
                        "latency_p95": latencies_sorted[int(n * 0.95)] if n > 0 else 0,
                        "latency_p99": latencies_sorted[int(n * 0.99)] if n > 0 else 0,
                        "latency_avg": sum(latencies) / len(latencies) if latencies else 0,
                    }
            
            # Error breakdown
            stats["errors"] = dict(self.metrics["errors"])
            
            return stats
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self.lock:
            from collections import defaultdict
            
            self.metrics = {
                "calls": defaultdict(int),
                "errors": defaultdict(int),
                "latencies": defaultdict(list),
                "tokens": defaultdict(int),
                "cache_hits": 0,
                "cache_misses": 0,
            }


# Global metrics instance
_llm_metrics: Optional[_LLMMetrics] = None


def _get_llm_metrics() -> _LLMMetrics:
    """Get or create global metrics tracker."""
    global _llm_metrics
    
    if _llm_metrics is None:
        _llm_metrics = _LLMMetrics()
    
    return _llm_metrics


def _record_llm_metrics(
    model: str,
    latency_ms: float,
    tokens: Optional[int] = None,
    cached: bool = False,
    error: Optional[str] = None,
) -> None:
    """Record LLM operation metrics.
    
    Args:
        model: Model identifier
        latency_ms: Request latency in milliseconds
        tokens: Number of tokens used (if available)
        cached: Whether response was from cache
        error: Error type if call failed
    """
    metrics = _get_llm_metrics()
    
    if error:
        metrics.record_error(model, error)
    else:
        metrics.record_call(model, latency_ms, tokens, cached)


def _get_llm_stats() -> Dict[str, Any]:
    """Get current LLM metrics snapshot."""
    metrics = _get_llm_metrics()
    return metrics.get_stats()


def _reset_llm_metrics() -> None:
    """Reset LLM metrics."""
    metrics = _get_llm_metrics()
    metrics.reset()
'''
).strip()

__all__ = ['LLM_METRICS']
