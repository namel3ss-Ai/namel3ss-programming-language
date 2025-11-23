"""
Performance profiling for N3 parser optimization.

Measures parsing time for files of different sizes and complexity.
"""

import time
import cProfile
import pstats
import io
from typing import List, Dict, Any
from namel3ss.parser import Parser
from namel3ss.lang.parser import parse_module
from pathlib import Path


class ParserPerformanceProfiler:
    """Profile parser performance and identify optimization opportunities."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    def time_parse_file(self, file_path: str, iterations: int = 10) -> Dict[str, Any]:
        """Time parsing a file multiple times and return statistics."""
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Test unified parser
        unified_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                parse_module(source, path=file_path)
                end = time.perf_counter()
                unified_times.append(end - start)
            except Exception as e:
                unified_times.append(None)  # Failed parse
        
        # Test hybrid parser
        hybrid_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                parser = Parser(source, path=file_path)
                parser.parse()
                end = time.perf_counter()
                hybrid_times.append(end - start)
            except Exception as e:
                hybrid_times.append(None)  # Failed parse
        
        # Calculate statistics
        valid_unified = [t for t in unified_times if t is not None]
        valid_hybrid = [t for t in hybrid_times if t is not None]
        
        result = {
            "file_path": file_path,
            "file_size_lines": len(source.split('\n')),
            "file_size_chars": len(source),
            "unified_parser": {
                "success_rate": len(valid_unified) / iterations,
                "avg_time": sum(valid_unified) / len(valid_unified) if valid_unified else None,
                "min_time": min(valid_unified) if valid_unified else None,
                "max_time": max(valid_unified) if valid_unified else None,
            },
            "hybrid_parser": {
                "success_rate": len(valid_hybrid) / iterations,
                "avg_time": sum(valid_hybrid) / len(valid_hybrid) if valid_hybrid else None,
                "min_time": min(valid_hybrid) if valid_hybrid else None,
                "max_time": max(valid_hybrid) if valid_hybrid else None,
            },
        }
        
        self.results.append(result)
        return result
    
    def profile_parser_details(self, file_path: str) -> Dict[str, Any]:
        """Profile parser with detailed function-level statistics."""
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Profile unified parser
        unified_profiler = cProfile.Profile()
        unified_profiler.enable()
        try:
            parse_module(source, path=file_path)
            unified_success = True
        except Exception as e:
            unified_success = False
            unified_error = str(e)
        unified_profiler.disable()
        
        # Profile hybrid parser 
        hybrid_profiler = cProfile.Profile()
        hybrid_profiler.enable()
        try:
            parser = Parser(source, path=file_path)
            parser.parse()
            hybrid_success = True
        except Exception as e:
            hybrid_success = False
            hybrid_error = str(e)
        hybrid_profiler.disable()
        
        # Extract profile statistics
        def extract_stats(profiler):
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            return s.getvalue()
        
        return {
            "file_path": file_path,
            "unified_parser": {
                "success": unified_success,
                "error": unified_error if not unified_success else None,
                "profile": extract_stats(unified_profiler) if unified_success else None
            },
            "hybrid_parser": {
                "success": hybrid_success,
                "error": hybrid_error if not hybrid_success else None,
                "profile": extract_stats(hybrid_profiler) if hybrid_success else None
            }
        }
    
    def benchmark_file_sizes(self) -> None:
        """Benchmark parsing performance across different file sizes."""
        # Generate test files of different sizes
        base_content = '''app "Performance Test" {
    description: "Testing parser performance"
}

llm "test_model" {
    provider: "openai"
    model: "gpt-4o-mini"
}

page "Home" at "/" {
    show text "Welcome"
    show form "TestForm" {
        field "name" type="text" required=true
        field "email" type="email" required=true
    }
}
'''
        
        test_sizes = [1, 5, 10, 20, 50]  # Multiples of base content
        
        print("\\n=== Parser Performance Benchmark ===")
        print(f"{'Size (lines)':<12} {'Size (chars)':<12} {'Unified (ms)':<15} {'Hybrid (ms)':<15} {'Ratio':<10}")
        print("-" * 70)
        
        for multiplier in test_sizes:
            # Create test content
            content = base_content * multiplier
            temp_file = f"/tmp/perf_test_{multiplier}.n3"
            
            with open(temp_file, 'w') as f:
                f.write(content)
            
            # Benchmark
            result = self.time_parse_file(temp_file, iterations=20)
            
            lines = result["file_size_lines"]
            chars = result["file_size_chars"]
            unified_ms = result["unified_parser"]["avg_time"] * 1000 if result["unified_parser"]["avg_time"] else "FAIL"
            hybrid_ms = result["hybrid_parser"]["avg_time"] * 1000 if result["hybrid_parser"]["avg_time"] else "FAIL"
            
            if isinstance(unified_ms, (int, float)) and isinstance(hybrid_ms, (int, float)):
                ratio = f"{hybrid_ms/unified_ms:.2f}x"
            else:
                ratio = "N/A"
            
            print(f"{lines:<12} {chars:<12} {unified_ms:<15} {hybrid_ms:<15} {ratio:<10}")


def main():
    """Run parser performance analysis."""
    profiler = ParserPerformanceProfiler()
    
    print("🚀 N3 Parser Performance Analysis")
    print("=" * 50)
    
    # Test existing files
    test_files = [
        "./examples/memory_chat_demo.n3",
        "./namel3ss/project_templates/crud_service/files/app.n3",
        "./tests/lsp/data/dashboard.n3"
    ]
    
    print("\\n1. Testing Real Files:")
    for file_path in test_files:
        if Path(file_path).exists():
            print(f"\\n📄 Testing: {file_path}")
            result = profiler.time_parse_file(file_path, iterations=10)
            
            lines = result["file_size_lines"]
            chars = result["file_size_chars"]
            
            unified = result["unified_parser"]
            hybrid = result["hybrid_parser"]
            
            print(f"   Size: {lines} lines, {chars} characters")
            
            if unified['avg_time'] is not None:
                print(f"   Unified Parser: {unified['success_rate']*100:.0f}% success, {unified['avg_time']*1000:.2f}ms avg")
            else:
                print(f"   Unified Parser: {unified['success_rate']*100:.0f}% success, FAILED")
            
            if hybrid['avg_time'] is not None:
                print(f"   Hybrid Parser:  {hybrid['success_rate']*100:.0f}% success, {hybrid['avg_time']*1000:.2f}ms avg")
            else:
                print(f"   Hybrid Parser:  {hybrid['success_rate']*100:.0f}% success, FAILED")
            
            if unified['avg_time'] and hybrid['avg_time']:
                ratio = hybrid['avg_time'] / unified['avg_time']
                print(f"   Performance: Hybrid is {ratio:.2f}x unified parser time")
            elif unified['avg_time'] and not hybrid['avg_time']:
                print(f"   Performance: Unified parser works, hybrid fails")
            elif not unified['avg_time'] and hybrid['avg_time']:
                print(f"   Performance: Hybrid parser works, unified fails")
            else:
                print(f"   Performance: Both parsers failed")
    
    print("\\n2. Scalability Benchmark:")
    profiler.benchmark_file_sizes()
    
    print("\\n✅ Performance analysis complete!")


if __name__ == "__main__":
    main()