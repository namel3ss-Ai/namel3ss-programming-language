"""
Demo of parser performance optimizations.

Shows how to enable caching and measure performance improvements.
"""

from namel3ss.parser import Parser, enable_parser_cache, disable_parser_cache, clear_parser_cache
import time


def demo_performance_optimization():
    """Demonstrate parser performance optimization benefits."""
    
    print("🚀 N3 Parser Performance Optimization Demo")
    print("=" * 50)
    
    # Sample N3 content  
    sample_content = '''app "Performance Demo" {
    description: "Testing parser performance optimizations"
}

llm "demo_llm" {
    provider: "openai"
    model: "gpt-4o-mini"
    temperature: 0.7
    max_tokens: 500
}

memory "demo_memory" {
    scope: "user"
    kind: "list"
    max_items: 50
}

prompt "demo_prompt" {
    model: "demo_llm"
    args: {
        user_input: string
    }
    template: "User: {user_input}\\nAssistant:"
    output_schema: {
        response: string
    }
}

page "Demo" at "/demo" {
    show text: "Performance Demo Page"
    show form: {
        field: {
            name: "input"
            type: "text"
            required: true
        }
        submit: "Submit"
    }
}'''
    
    # Test 1: Without optimizations
    print("\\n1. Without Optimizations")
    print("-" * 30)
    
    disable_parser_cache()  # Ensure optimizations are off
    
    parse_times = []
    for i in range(10):
        start = time.perf_counter()
        parser = Parser(sample_content, path=f"demo_{i}.n3")
        module = parser.parse()
        end = time.perf_counter()
        parse_times.append(end - start)
    
    avg_time_no_opt = sum(parse_times) / len(parse_times)
    print(f"Average parse time: {avg_time_no_opt*1000:.3f}ms")
    print(f"Total time for 10 parses: {sum(parse_times)*1000:.3f}ms")
    
    # Test 2: With caching enabled
    print("\\n2. With Caching Enabled")
    print("-" * 30)
    
    cache_stats = enable_parser_cache(max_entries=50)
    clear_parser_cache()  # Start fresh
    
    # First parse (cache miss)
    start = time.perf_counter()
    parser = Parser(sample_content, path="demo_cached.n3")
    module = parser.parse()
    end = time.perf_counter()
    first_parse_time = end - start
    
    print(f"First parse (cache miss): {first_parse_time*1000:.3f}ms")
    
    # Multiple cache hits
    cache_hit_times = []
    for i in range(10):
        start = time.perf_counter()
        parser = Parser(sample_content, path="demo_cached.n3")
        module = parser.parse()
        end = time.perf_counter()
        cache_hit_times.append(end - start)
    
    avg_cache_hit_time = sum(cache_hit_times) / len(cache_hit_times)
    print(f"Average cache hit time: {avg_cache_hit_time*1000:.3f}ms")
    print(f"Total time for 10 cache hits: {sum(cache_hit_times)*1000:.3f}ms")
    
    # Performance comparison
    print("\\n3. Performance Comparison")
    print("-" * 30)
    
    speedup = avg_time_no_opt / avg_cache_hit_time
    time_savings = (avg_time_no_opt - avg_cache_hit_time) * 1000
    
    print(f"Speedup: {speedup:.1f}x faster with caching")
    print(f"Time savings: {time_savings:.3f}ms per parse")
    
    # Cache statistics
    stats = cache_stats()
    print(f"\\nCache Statistics:")
    print(f"- Cache entries: {stats['entries']}")
    print(f"- Cache hits: {stats['hits']}")
    print(f"- Cache misses: {stats['misses']}")
    print(f"- Hit rate: {stats['hit_rate']*100:.1f}%")
    
    # Test 3: Memory usage estimation
    print("\\n4. Memory Usage Optimization")
    print("-" * 30)
    
    # Test with multiple different files
    different_files = []
    for i in range(5):
        content = sample_content.replace("Performance Demo", f"Demo {i}")
        parser = Parser(content, path=f"unique_demo_{i}.n3")
        module = parser.parse()
        different_files.append((content, module))
    
    final_stats = cache_stats()
    print(f"Cache entries after 5 different files: {final_stats['entries']}")
    print(f"Total operations: {final_stats['hits'] + final_stats['misses']}")
    print(f"Overall hit rate: {final_stats['hit_rate']*100:.1f}%")
    
    # Cleanup
    disable_parser_cache()
    
    print("\\n✅ Performance optimization demo complete!")
    print(f"\\n💡 Key Takeaway: Caching provides {speedup:.1f}x speedup for repeated parsing")
    
    return {
        "baseline_avg": avg_time_no_opt,
        "cache_hit_avg": avg_cache_hit_time,
        "speedup": speedup,
        "final_stats": final_stats
    }


if __name__ == "__main__":
    demo_performance_optimization()