"""
Locust load testing suite for generated backends with async chains.

This benchmark suite validates:
- Throughput improvements from async execution
- Latency percentiles (P50, P95, P99) under load
- Streaming endpoint performance
- Concurrent chain execution
- Rate limiting behavior

Usage:
    # Start your generated backend first
    cd generated_backend && uvicorn app.main:app --host 0.0.0.0 --port 8000
    
    # Run benchmark (from this directory)
    locust -f locustfile.py --host http://localhost:8000
    
    # Access Web UI at http://localhost:8089
    
    # Or run headless
    locust -f locustfile.py --host http://localhost:8000 \
           --users 100 --spawn-rate 10 --run-time 60s --headless
"""

from locust import HttpUser, task, between, events
import json
import time
import random


class ChainExecutionUser(HttpUser):
    """User that executes AI chains via the generated backend."""
    
    wait_time = between(0.5, 2.0)  # Wait 0.5-2s between requests
    
    def on_start(self):
        """Called when a simulated user starts."""
        self.chain_names = [
            "summarize_text",
            "analyze_sentiment",
            "generate_questions",
            "extract_entities",
            "translate_text",
        ]
        
        self.sample_inputs = [
            {"text": "The quick brown fox jumps over the lazy dog. " * 10},
            {"text": "AI is transforming industries worldwide. " * 10},
            {"text": "Climate change requires immediate action. " * 10},
            {"text": "Quantum computing promises exponential speedups. " * 10},
            {"text": "Space exploration pushes human boundaries. " * 10},
        ]
    
    @task(5)
    def execute_chain(self):
        """Execute a chain via POST /api/chains/{chain_name}."""
        chain_name = random.choice(self.chain_names)
        payload = random.choice(self.sample_inputs)
        
        with self.client.post(
            f"/api/chains/{chain_name}",
            json=payload,
            catch_response=True,
            name="/api/chains/[chain]"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") == "ok":
                        response.success()
                    else:
                        response.failure(f"Chain failed: {data.get('error')}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 404:
                # Chain not registered, skip
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(3)
    def execute_llm_connector(self):
        """Execute an LLM connector via POST /api/llm/{connector}."""
        connector_name = random.choice(["gpt4", "claude", "gemini", "llama"])
        payload = {
            "prompt": f"Explain {random.choice(['AI', 'ML', 'NLP', 'CV'])} in simple terms.",
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        with self.client.post(
            f"/api/llm/{connector_name}",
            json=payload,
            catch_response=True,
            name="/api/llm/[connector]"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # Connector not registered, skip
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def stream_chain(self):
        """Stream chain execution via GET /api/chains/{chain_name}/stream."""
        chain_name = random.choice(self.chain_names)
        params = {
            "text": "Generate a story about space exploration."
        }
        
        start_time = time.time()
        chunk_count = 0
        first_chunk_time = None
        
        with self.client.get(
            f"/api/chains/{chain_name}/stream",
            params=params,
            stream=True,
            catch_response=True,
            name="/api/chains/[chain]/stream"
        ) as response:
            if response.status_code == 200:
                try:
                    for line in response.iter_lines():
                        if line:
                            chunk_count += 1
                            if first_chunk_time is None:
                                first_chunk_time = time.time()
                            
                            if line.decode() == "data: [DONE]":
                                break
                    
                    elapsed = time.time() - start_time
                    ttft = first_chunk_time - start_time if first_chunk_time else 0
                    
                    # Record custom metrics
                    events.request.fire(
                        request_type="GET",
                        name="/api/chains/[chain]/stream (TTFT)",
                        response_time=ttft * 1000,
                        response_length=0,
                        exception=None,
                        context={}
                    )
                    
                    response.success()
                except Exception as e:
                    response.failure(f"Streaming error: {e}")
            elif response.status_code == 404:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Check backend health via GET /health."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


class ParallelChainUser(HttpUser):
    """User that tests parallel chain execution."""
    
    wait_time = between(1.0, 3.0)
    
    @task
    def execute_parallel_chain(self):
        """Execute a chain with parallel steps enabled."""
        payload = {
            "inputs": [
                {"text": "Analyze this text."},
                {"text": "Summarize this content."},
                {"text": "Extract key points."},
            ]
        }
        
        headers = {
            "X-Enable-Parallel": "true"  # Custom header to enable parallel execution
        }
        
        with self.client.post(
            "/api/chains/parallel_processing",
            json=payload,
            headers=headers,
            catch_response=True,
            name="/api/chains/parallel_processing"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Check if parallel execution was faster
                    if "metadata" in data and "elapsed_ms" in data["metadata"]:
                        elapsed_ms = data["metadata"]["elapsed_ms"]
                        if elapsed_ms < 5000:  # Under 5s is good for parallel
                            response.success()
                        else:
                            response.failure(f"Slow parallel execution: {elapsed_ms}ms")
                    else:
                        response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON")
            elif response.status_code == 404:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class ConcurrentLoadUser(HttpUser):
    """User for testing high concurrent load."""
    
    wait_time = between(0.1, 0.5)  # Very short wait for high concurrency
    
    @task
    def rapid_fire_requests(self):
        """Send rapid requests to test rate limiting and concurrency."""
        payload = {"text": "Quick test."}
        
        with self.client.post(
            "/api/chains/test_chain",
            json=payload,
            catch_response=True,
            name="/api/chains/test_chain (rapid)"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                # Rate limited - expected behavior
                response.success()
            elif response.status_code == 404:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


# Custom event listeners for detailed metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts."""
    print("\n" + "="*80)
    print("Starting Backend Performance Benchmark")
    print("="*80)
    print(f"Target: {environment.host}")
    print(f"Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}")
    print("="*80 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops - print summary."""
    print("\n" + "="*80)
    print("Benchmark Complete - Summary")
    print("="*80)
    
    stats = environment.stats
    
    if stats.total.num_requests > 0:
        print(f"Total Requests: {stats.total.num_requests}")
        print(f"Total Failures: {stats.total.num_failures}")
        print(f"Requests/sec: {stats.total.total_rps:.2f}")
        print(f"Failure Rate: {stats.total.fail_ratio * 100:.2f}%")
        print()
        print("Latency Percentiles:")
        print(f"  P50 (median): {stats.total.get_response_time_percentile(0.5):.0f}ms")
        print(f"  P75: {stats.total.get_response_time_percentile(0.75):.0f}ms")
        print(f"  P90: {stats.total.get_response_time_percentile(0.90):.0f}ms")
        print(f"  P95: {stats.total.get_response_time_percentile(0.95):.0f}ms")
        print(f"  P99: {stats.total.get_response_time_percentile(0.99):.0f}ms")
        print(f"  Min: {stats.total.min_response_time:.0f}ms")
        print(f"  Max: {stats.total.max_response_time:.0f}ms")
        print(f"  Avg: {stats.total.avg_response_time:.0f}ms")
    
    print("="*80 + "\n")


# Performance threshold checking
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Monitor individual requests for performance issues."""
    # Warn if response time exceeds threshold
    if response_time > 10000:  # 10 seconds
        print(f"⚠️  SLOW REQUEST: {name} took {response_time:.0f}ms")
    
    # Warn on errors
    if exception:
        print(f"❌ ERROR: {name} - {exception}")
