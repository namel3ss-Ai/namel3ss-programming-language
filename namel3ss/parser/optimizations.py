"""
Parser performance optimizations.

Implements caching, incremental parsing, and other performance improvements.
"""

from typing import Dict, Optional, Any, Tuple
import time
import hashlib
from functools import lru_cache
from namel3ss.ast import Module


class ParserCache:
    """
    Module-level parser cache with content-based invalidation.
    
    Features:
    - Content-aware caching (detects changes via hash)
    - Memory-efficient storage
    - LRU eviction for large caches
    - Cache statistics and monitoring
    """
    
    def __init__(self, max_entries: int = 100):
        self._cache: Dict[str, Tuple[str, Module, float]] = {}  # path -> (hash, module, timestamp)
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0
        
    def _content_hash(self, content: str) -> str:
        """Generate content hash for cache key."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, path: str, content: str) -> Optional[Module]:
        """
        Get cached module if content hasn't changed.
        
        Returns:
            Module if cache hit, None if cache miss
        """
        content_hash = self._content_hash(content)
        
        if path in self._cache:
            cached_hash, cached_module, _ = self._cache[path]
            if cached_hash == content_hash:
                self._hits += 1
                return cached_module
        
        self._misses += 1
        return None
    
    def set(self, path: str, content: str, module: Module) -> None:
        """Cache a parsed module."""
        content_hash = self._content_hash(content)
        
        # Evict oldest entry if cache is full
        if len(self._cache) >= self._max_entries:
            oldest_path = min(self._cache.keys(), 
                            key=lambda p: self._cache[p][2])  # Find oldest by timestamp
            del self._cache[oldest_path]
        
        self._cache[path] = (content_hash, module, time.time())
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "paths": list(self._cache.keys()),
            "max_entries": self._max_entries
        }


# Global cache instance
_global_cache = ParserCache()


def get_parser_cache() -> ParserCache:
    """Get the global parser cache instance."""
    return _global_cache


class ParseTimeProfiler:
    """
    Profile parsing time for performance analysis.
    
    Tracks timing data for different file sizes and patterns.
    """
    
    def __init__(self):
        self._timings: Dict[str, List[float]] = {}
    
    def time_operation(self, operation_name: str, func, *args, **kwargs):
        """Time a parsing operation and record the result."""
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            end = time.perf_counter()
            duration = end - start
            
            if operation_name not in self._timings:
                self._timings[operation_name] = []
            self._timings[operation_name].append(duration)
            
            return result
        except Exception as e:
            end = time.perf_counter()
            # Still record timing even for failed operations
            duration = end - start
            if operation_name not in self._timings:
                self._timings[operation_name] = []
            self._timings[operation_name].append(duration)
            raise
    
    def get_stats(self, operation_name: str) -> Dict[str, float]:
        """Get timing statistics for an operation."""
        if operation_name not in self._timings:
            return {"count": 0}
        
        times = self._timings[operation_name]
        return {
            "count": len(times),
            "total_time": sum(times),
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "median_time": sorted(times)[len(times) // 2]
        }
    
    def report(self) -> str:
        """Generate a timing report."""
        lines = ["Parser Performance Report:", "=" * 30]
        
        for op_name, times in self._timings.items():
            stats = self.get_stats(op_name)
            lines.append(f"\n{op_name}:")
            lines.append(f"  Count: {stats['count']}")
            lines.append(f"  Average: {stats['avg_time']*1000:.2f}ms")
            lines.append(f"  Range: {stats['min_time']*1000:.2f}ms - {stats['max_time']*1000:.2f}ms")
        
        return "\n".join(lines)


def enable_parser_optimizations():
    """
    Enable parser optimizations by monkey-patching the Parser class.
    
    This adds caching functionality to the existing parser without 
    breaking existing code.
    """
    from namel3ss.parser import Parser
    
    # Store original parse method
    if not hasattr(Parser, '_original_parse'):
        Parser._original_parse = Parser.parse
    
    def optimized_parse(self) -> Module:
        """Optimized parse method with caching."""
        cache = get_parser_cache()
        
        # Try cache first
        cached_module = cache.get(self.source_path, self._source)
        if cached_module is not None:
            return cached_module
        
        # Parse and cache
        module = self._original_parse()
        cache.set(self.source_path, self._source, module)
        
        return module
    
    # Replace parse method
    Parser.parse = optimized_parse


def disable_parser_optimizations():
    """Disable parser optimizations and restore original behavior."""
    from namel3ss.parser import Parser
    
    if hasattr(Parser, '_original_parse'):
        Parser.parse = Parser._original_parse
        delattr(Parser, '_original_parse')


@lru_cache(maxsize=256)
def cached_tokenize(source_hash: str, source: str):
    """
    LRU cached tokenization for frequently parsed content.
    
    Uses source hash as key to detect content changes.
    """
    from namel3ss.lang.parser.grammar.lexer import Lexer
    
    lexer = Lexer(source)
    return list(lexer.tokenize())


def benchmark_parser_performance():
    """
    Run a comprehensive performance benchmark.
    
    Tests both cached and uncached performance across different scenarios.
    """
    print("🚀 Comprehensive Parser Performance Benchmark")
    print("=" * 60)
    
    # Test with real working content
    test_content = '''app "Benchmark Test" {
    description: "Performance testing application"
}

llm "test_llm" {
    provider: "openai"
    model: "gpt-4o-mini"
    temperature: 0.7
}

memory "test_memory" {
    scope: "user"
    kind: "list"
    max_items: 50
}

page "Home" at "/" {
    show text "Welcome to the performance test"
}'''
    
    from namel3ss.parser import Parser
    
    # Test 1: Baseline performance
    print("\n1. Baseline Performance (No Optimizations)")
    print("-" * 40)
    
    baseline_times = []
    for i in range(20):
        start = time.perf_counter()
        try:
            parser = Parser(test_content, path=f"baseline_{i}.n3")
            parser.parse()
            end = time.perf_counter()
            baseline_times.append(end - start)
        except Exception:
            pass
    
    baseline_avg = sum(baseline_times) / len(baseline_times) if baseline_times else 0
    print(f"Average time: {baseline_avg*1000:.2f}ms ({len(baseline_times)}/20 successful)")
    
    # Test 2: With caching enabled
    print("\n2. With Caching Optimizations")
    print("-" * 40)
    
    enable_parser_optimizations()
    cache = get_parser_cache()
    cache.clear()  # Start fresh
    
    cached_times = []
    
    # First parse (cache miss)
    start = time.perf_counter()
    try:
        parser = Parser(test_content, path="cached_test.n3")
        parser.parse()
        end = time.perf_counter()
        first_time = end - start
        cached_times.append(first_time)
        print(f"First parse (cache miss): {first_time*1000:.2f}ms")
    except Exception as e:
        first_time = None
        print(f"First parse failed: {e}")
    
    # Subsequent parses (cache hits)
    cache_hit_times = []
    for i in range(10):
        start = time.perf_counter()
        try:
            parser = Parser(test_content, path="cached_test.n3") 
            parser.parse()
            end = time.perf_counter()
            hit_time = end - start
            cache_hit_times.append(hit_time)
        except Exception:
            pass
    
    if cache_hit_times:
        hit_avg = sum(cache_hit_times) / len(cache_hit_times)
        print(f"Cache hits average: {hit_avg*1000:.2f}ms ({len(cache_hit_times)}/10 successful)")
        
        if first_time and hit_avg:
            speedup = first_time / hit_avg
            print(f"Cache speedup: {speedup:.2f}x faster")
    
    # Test 3: Cache statistics
    print("\n3. Cache Statistics")
    print("-" * 40)
    
    stats = cache.stats()
    print(f"Cache entries: {stats['entries']}")
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']*100:.1f}%")
    
    # Restore original behavior
    disable_parser_optimizations()
    
    print("\n✅ Performance benchmark complete!")
    
    return {
        "baseline_avg": baseline_avg,
        "cache_hit_avg": sum(cache_hit_times) / len(cache_hit_times) if cache_hit_times else None,
        "cache_stats": stats
    }