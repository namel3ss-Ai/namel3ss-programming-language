"""LLM response caching system for N3 runtime.

Provides TTL-based caching for LLM responses to reduce:
- API costs (avoid duplicate calls)
- Latency (cache hits <1ms vs API calls ~1-5s)
- Rate limit issues (fewer API requests)

Features:
- TTL-based expiration (default 1 hour)
- Configurable cache size (default 1000 entries)
- Content-based hashing (prompt + model + params)
- Thread-safe operations
- Optional persistence to disk
- Cache statistics (hits, misses, evictions)
"""

from textwrap import dedent

LLM_CACHE = dedent(
    '''
# LLM Response Caching
class _LLMCache:
    """Thread-safe TTL cache for LLM responses."""
    
    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        """Initialize cache.
        
        Args:
            maxsize: Maximum number of cached responses
            ttl: Time-to-live in seconds (default 1 hour)
        """
        import threading
        from collections import OrderedDict
        import time
        
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "errors": 0,
        }
    
    def _hash_key(self, prompt: str, model: str, params: Dict[str, Any]) -> str:
        """Generate cache key from prompt, model, and parameters."""
        import hashlib
        import json
        
        # Normalize parameters for consistent hashing
        normalized_params = {
            k: v for k, v in sorted(params.items())
            if k not in ("stream", "timeout")  # Exclude non-deterministic params
        }
        
        key_data = {
            "prompt": prompt,
            "model": model,
            "params": normalized_params,
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached response if available and not expired."""
        import time
        
        key = self._hash_key(prompt, model, params)
        
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None
            
            response, timestamp = self.cache[key]
            
            # Check if expired
            if time.time() - timestamp > self.ttl:
                del self.cache[key]
                self.stats["evictions"] += 1
                self.stats["misses"] += 1
                return None
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            self.stats["hits"] += 1
            return response
    
    def put(self, prompt: str, model: str, params: Dict[str, Any], response: Any) -> None:
        """Store response in cache."""
        import time
        
        key = self._hash_key(prompt, model, params)
        
        with self.lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.maxsize and key not in self.cache:
                self.cache.popitem(last=False)
                self.stats["evictions"] += 1
            
            self.cache[key] = (response, time.time())
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self.lock:
            total = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total if total > 0 else 0.0
            
            return {
                **self.stats,
                "size": len(self.cache),
                "hit_rate": hit_rate,
            }


# Global cache instance
_llm_cache: Optional[_LLMCache] = None


def _get_llm_cache() -> _LLMCache:
    """Get or create global LLM cache."""
    global _llm_cache
    
    if _llm_cache is None:
        # Get cache settings from runtime config
        cache_config = RUNTIME_SETTINGS.get("llm_cache", {})
        enabled = cache_config.get("enabled", True)
        
        if not enabled:
            # Return dummy cache that always misses
            class _DummyCache:
                def get(self, *args, **kwargs): return None
                def put(self, *args, **kwargs): pass
                def clear(self): pass
                def get_stats(self): return {}
            return _DummyCache()
        
        maxsize = cache_config.get("maxsize", 1000)
        ttl = cache_config.get("ttl", 3600)  # 1 hour default
        
        _llm_cache = _LLMCache(maxsize=maxsize, ttl=ttl)
    
    return _llm_cache


def _cached_llm_call(
    prompt: str,
    model: str,
    params: Dict[str, Any],
    call_func: Callable[[], Any],
) -> Any:
    """Execute LLM call with caching.
    
    Args:
        prompt: The prompt text
        model: Model identifier
        params: Call parameters
        call_func: Function to call if cache miss
    
    Returns:
        LLM response (from cache or fresh call)
    """
    cache = _get_llm_cache()
    
    # Try cache first
    cached_response = cache.get(prompt, model, params)
    if cached_response is not None:
        _record_event(
            "llm_cache",
            "hit",
            "info",
            {"model": model, "prompt_length": len(prompt)}
        )
        return cached_response
    
    # Cache miss - make actual call
    try:
        response = call_func()
        
        # Store in cache
        cache.put(prompt, model, params, response)
        
        _record_event(
            "llm_cache",
            "miss",
            "info",
            {"model": model, "prompt_length": len(prompt)}
        )
        
        return response
    
    except Exception as exc:
        _record_event(
            "llm_cache",
            "error",
            "error",
            {"model": model, "error": str(exc)}
        )
        raise


def _clear_llm_cache() -> Dict[str, Any]:
    """Clear LLM cache and return final stats."""
    cache = _get_llm_cache()
    stats = cache.get_stats()
    cache.clear()
    return stats
'''
).strip()

__all__ = ['LLM_CACHE']
