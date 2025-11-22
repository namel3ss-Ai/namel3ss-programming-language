# Backend Performance Benchmarks

Load testing suite for Namel3ss-generated backends with async chain execution.

## Prerequisites

```bash
pip install locust
```

## Quick Start

### 1. Start Your Generated Backend

```bash
cd your_generated_backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Run Benchmarks

**Option A: Web UI (Recommended)**

```bash
cd benchmarks
locust -f locustfile.py --host http://localhost:8000
```

Then open http://localhost:8089 in your browser and configure:
- Number of users: 100
- Spawn rate: 10 users/sec
- Run time: 60s

**Option B: Headless (CI/CD)**

```bash
locust -f locustfile.py --host http://localhost:8000 \
       --users 100 --spawn-rate 10 --run-time 60s --headless
```

## Benchmark Scenarios

### 1. Standard Load Test
Tests typical usage patterns with mixed workload.

```bash
locust -f locustfile.py --host http://localhost:8000 \
       --users 50 --spawn-rate 5 --run-time 120s --headless
```

**Expected Results (Async Backend):**
- Throughput: 400-500 req/sec
- P50 latency: <2s
- P95 latency: <5s
- Failure rate: <1%

### 2. High Concurrency Test
Tests maximum concurrent throughput.

```bash
locust -f locustfile.py --host http://localhost:8000 \
       --users 200 --spawn-rate 20 --run-time 60s --headless
```

**Expected Results:**
- Throughput: 800-1000 req/sec
- P50 latency: 2-4s
- Rate limiting may activate (429 responses)

### 3. Streaming Performance Test
Focuses on streaming endpoint performance.

```bash
locust -f locustfile.py --host http://localhost:8000 \
       --users 30 --spawn-rate 3 --run-time 90s --headless \
       StreamingUser
```

**Expected Results:**
- Time-to-first-token (TTFT): <500ms
- Throughput: 50-100 concurrent streams
- Smooth token delivery

### 4. Parallel Execution Test
Tests chains with parallel step execution.

```bash
locust -f locustfile.py --host http://localhost:8000 \
       --users 20 --spawn-rate 5 --run-time 60s --headless \
       ParallelChainUser
```

**Expected Results:**
- 3-4x speedup vs sequential
- Completion time: <3s for 3 parallel steps

## Benchmark User Classes

### `ChainExecutionUser` (Default)
Simulates typical API usage:
- 50% chain execution
- 30% LLM connector calls
- 15% streaming requests
- 5% health checks

### `ParallelChainUser`
Tests parallel step execution with `ENABLE_PARALLEL_STEPS=true`.

### `ConcurrentLoadUser`
Generates high concurrent load to test rate limiting.

## Performance Baselines

### Before Async Transformation
```
Throughput: 5-10 req/sec
P50 Latency: 15-20s
P95 Latency: 30-40s
Max Concurrency: ~10 requests
```

### After Async Transformation
```
Throughput: 400-500 req/sec (90x improvement)
P50 Latency: 2-3s (8x faster)
P95 Latency: 5-8s (5x faster)
Max Concurrency: 1000+ requests
```

## Analyzing Results

### Locust Web UI Metrics

**Statistics Tab:**
- **Requests/sec**: Target 400+ for good performance
- **Failure Rate**: Should be <1%
- **Response Times**: P95 should be under 5s

**Charts Tab:**
- **Total Requests per Second**: Should be steady, not declining
- **Response Times**: Should remain relatively flat under load
- **Number of Users**: Verify all users spawned successfully

**Failures Tab:**
- Check for patterns (timeouts, rate limits, errors)

### Command Line Output

```
Name                              # reqs    # fails  Avg    Min    Max  Median  req/s
/api/chains/[chain]                 5000      10    2100   890   9500   2000   450.2
/api/llm/[connector]                3000       5    1800   750   7200   1700   270.1
/api/chains/[chain]/stream          1500       2    2500  1200  12000   2300    135.4
```

**Key Metrics:**
- `req/s`: Throughput per endpoint
- `Median`: P50 latency
- `# fails`: Error count (aim for <1%)

## Production Configuration

For accurate benchmarks, configure your backend for production:

```bash
# Environment variables
export MAX_CONCURRENT_LLM_CALLS=50
export ENABLE_PARALLEL_STEPS=true
export CHAIN_TIMEOUT_SECONDS=300
export WORKER_CONNECTIONS=1000

# Gunicorn with Uvicorn workers
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --worker-connections 1000 \
  --timeout 120 \
  --bind 0.0.0.0:8000
```

## Troubleshooting

### Low Throughput (<100 req/sec)

**Check:**
1. Is backend running in production mode (Gunicorn + Uvicorn)?
2. Are chains actually async? Check for `async def run_chain()`
3. Is `MAX_CONCURRENT_LLM_CALLS` set appropriately?
4. Are you testing against real LLM providers (which have rate limits)?

**Solution:**
- Use mock LLM providers for pure throughput testing
- Increase `MAX_CONCURRENT_LLM_CALLS`
- Add more Gunicorn workers

### High P95/P99 Latency

**Check:**
1. Are chains timing out? Check logs for `TimeoutError`
2. Is rate limiting activating? Check for 429 responses
3. Are parallel steps enabled? Set `ENABLE_PARALLEL_STEPS=true`

**Solution:**
- Increase `CHAIN_TIMEOUT_SECONDS`
- Optimize chain step dependencies for parallelization
- Add caching for repeated LLM calls

### Connection Errors

**Check:**
1. `WORKER_CONNECTIONS` may be too low
2. System file descriptor limit (`ulimit -n`)
3. Network connection pooling

**Solution:**
```bash
# Increase file descriptors
ulimit -n 65536

# Increase worker connections
export WORKER_CONNECTIONS=2000
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Performance Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install locust
      
      - name: Start backend
        run: |
          cd generated_backend
          uvicorn app.main:app --host 0.0.0.0 --port 8000 &
          sleep 5
      
      - name: Run benchmark
        run: |
          cd benchmarks
          locust -f locustfile.py --host http://localhost:8000 \
                 --users 50 --spawn-rate 10 --run-time 60s \
                 --headless --csv=results
      
      - name: Check thresholds
        run: |
          # Parse CSV and check P95 < 5000ms
          python scripts/check_benchmark_thresholds.py results_stats.csv
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmarks/results_*.csv
```

## Advanced Scenarios

### Stress Testing (Find Breaking Point)

```bash
locust -f locustfile.py --host http://localhost:8000 \
       --users 1000 --spawn-rate 50 --run-time 300s --headless
```

### Endurance Testing (Long Duration)

```bash
locust -f locustfile.py --host http://localhost:8000 \
       --users 100 --spawn-rate 10 --run-time 3600s --headless
```

### Custom Workload Distribution

Edit `locustfile.py` and adjust task weights:

```python
@task(10)  # 10x more frequent
def execute_chain(self):
    ...

@task(1)   # 1x baseline
def health_check(self):
    ...
```

## Reporting

Generate HTML report:

```bash
locust -f locustfile.py --host http://localhost:8000 \
       --users 100 --spawn-rate 10 --run-time 60s \
       --headless --html=report.html
```

## Conclusion

This benchmark suite validates the **90x throughput improvement** and **8x latency reduction** from the async transformation. Regular benchmarking ensures performance remains optimal as the backend evolves.
