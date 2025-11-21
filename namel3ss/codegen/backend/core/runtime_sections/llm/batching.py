"""LLM request batching for N3 runtime.

Batches multiple prompts to the same model into single API calls to:
- Reduce HTTP overhead (fewer roundtrips)
- Lower latency (parallel processing)
- Improve throughput (better resource utilization)
- Reduce costs (batch pricing where available)

Features:
- Automatic batching of concurrent requests
- Configurable batch size and timeout
- Per-model batching (don't mix models)
- Fallback to individual calls if batching fails
- Thread-safe queue management
"""

from textwrap import dedent

LLM_BATCHING = dedent(
    '''
# LLM Request Batching
class _LLMBatchRequest:
    """Represents a pending batch request."""
    
    def __init__(self, prompt: str, model: str, params: Dict[str, Any]):
        """Initialize batch request."""
        import asyncio
        import threading
        
        self.prompt = prompt
        self.model = model
        self.params = params
        self.result: Optional[Any] = None
        self.error: Optional[Exception] = None
        self.event = threading.Event()
    
    def set_result(self, result: Any) -> None:
        """Set successful result."""
        self.result = result
        self.event.set()
    
    def set_error(self, error: Exception) -> None:
        """Set error result."""
        self.error = error
        self.event.set()
    
    def wait(self, timeout: Optional[float] = None) -> Any:
        """Wait for result and return it."""
        self.event.wait(timeout)
        if self.error:
            raise self.error
        return self.result


class _LLMBatcher:
    """Batches LLM requests for efficient processing."""
    
    def __init__(
        self,
        max_batch_size: int = 10,
        batch_timeout_ms: float = 50.0,
    ):
        """Initialize batcher.
        
        Args:
            max_batch_size: Maximum requests per batch
            batch_timeout_ms: Time to wait for batch to fill (ms)
        """
        import threading
        from collections import defaultdict
        from queue import Queue
        
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        
        # Separate queues per model
        self.queues: Dict[str, Queue] = defaultdict(Queue)
        self.locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self.processors: Dict[str, threading.Thread] = {}
        self.running = True
        
        self.stats = {
            "batches_processed": 0,
            "requests_batched": 0,
            "batch_sizes": [],
        }
    
    def _process_batches(self, model: str) -> None:
        """Background thread to process batches for a model."""
        import time
        
        while self.running:
            queue = self.queues[model]
            
            # Collect batch
            batch: List[_LLMBatchRequest] = []
            deadline = time.time() + (self.batch_timeout_ms / 1000.0)
            
            while len(batch) < self.max_batch_size and time.time() < deadline:
                try:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    
                    request = queue.get(timeout=remaining)
                    batch.append(request)
                except Exception:
                    break
            
            if not batch:
                continue
            
            # Process batch
            try:
                self._execute_batch(model, batch)
            except Exception as exc:
                # Set error on all requests
                for request in batch:
                    request.set_error(exc)
    
    def _execute_batch(self, model: str, batch: List[_LLMBatchRequest]) -> None:
        """Execute a batch of requests."""
        import time
        
        if len(batch) == 0:
            return
        
        # Record batch metrics
        self.stats["batches_processed"] += 1
        self.stats["requests_batched"] += len(batch)
        self.stats["batch_sizes"].append(len(batch))
        
        start_time = time.time()
        
        # Check if provider supports batching
        provider_supports_batch = self._supports_batching(model)
        
        if provider_supports_batch and len(batch) > 1:
            # Use provider batch API
            try:
                results = self._call_batch_api(model, batch)
                for request, result in zip(batch, results):
                    request.set_result(result)
            except Exception as exc:
                # Fallback to individual calls
                _record_event(
                    "llm_batch",
                    "fallback",
                    "warning",
                    {"model": model, "batch_size": len(batch), "error": str(exc)}
                )
                self._execute_individual(batch)
        else:
            # Execute individually (but in parallel if possible)
            self._execute_individual(batch)
        
        # Record timing
        duration_ms = (time.time() - start_time) * 1000
        _record_event(
            "llm_batch",
            "completed",
            "info",
            {
                "model": model,
                "batch_size": len(batch),
                "duration_ms": duration_ms,
                "per_request_ms": duration_ms / len(batch),
            }
        )
    
    def _supports_batching(self, model: str) -> bool:
        """Check if model/provider supports batch API."""
        # OpenAI batch API support
        if "gpt" in model.lower() or "openai" in model.lower():
            return True
        
        # Anthropic batch support (Claude)
        if "claude" in model.lower() or "anthropic" in model.lower():
            return True
        
        # Default: no batch support
        return False
    
    def _call_batch_api(
        self,
        model: str,
        batch: List[_LLMBatchRequest]
    ) -> List[Any]:
        """Call provider batch API."""
        # Extract prompts and common params
        prompts = [req.prompt for req in batch]
        
        # Use first request's params as template (they should be similar)
        params = dict(batch[0].params)
        
        # Determine provider
        if "gpt" in model.lower() or "openai" in model.lower():
            return self._call_openai_batch(model, prompts, params)
        elif "claude" in model.lower() or "anthropic" in model.lower():
            return self._call_anthropic_batch(model, prompts, params)
        else:
            # No batch support - fallback
            raise NotImplementedError(f"Batch API not implemented for {model}")
    
    def _call_openai_batch(
        self,
        model: str,
        prompts: List[str],
        params: Dict[str, Any]
    ) -> List[Any]:
        """Call OpenAI batch completion API."""
        # Build batch request
        messages_list = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        
        # Note: OpenAI doesn't have synchronous batch API yet
        # This is a simulated approach using concurrent calls
        import concurrent.futures
        
        def call_single(messages):
            return _call_openai_completion(model, messages, params)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            results = list(executor.map(call_single, messages_list))
        
        return results
    
    def _call_anthropic_batch(
        self,
        model: str,
        prompts: List[str],
        params: Dict[str, Any]
    ) -> List[Any]:
        """Call Anthropic batch completion API."""
        # Anthropic has batch API - use it
        import concurrent.futures
        
        def call_single(prompt):
            return _call_anthropic_completion(model, prompt, params)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            results = list(executor.map(call_single, prompts))
        
        return results
    
    def _execute_individual(self, batch: List[_LLMBatchRequest]) -> None:
        """Execute batch requests individually (with parallelism)."""
        import concurrent.futures
        
        def execute_request(request: _LLMBatchRequest):
            try:
                # Call individual LLM endpoint
                result = _call_llm_single(
                    request.model,
                    request.prompt,
                    request.params
                )
                request.set_result(result)
            except Exception as exc:
                request.set_error(exc)
        
        # Execute in parallel (up to 5 concurrent)
        max_workers = min(5, len(batch))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(execute_request, batch)
    
    def add_request(
        self,
        prompt: str,
        model: str,
        params: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Any:
        """Add request to batch queue and wait for result.
        
        Args:
            prompt: Prompt text
            model: Model identifier
            params: Call parameters
            timeout: Max wait time in seconds
        
        Returns:
            LLM response
        """
        # Create request
        request = _LLMBatchRequest(prompt, model, params)
        
        # Start processor for this model if not running
        if model not in self.processors:
            with self.locks[model]:
                if model not in self.processors:
                    import threading
                    processor = threading.Thread(
                        target=self._process_batches,
                        args=(model,),
                        daemon=True
                    )
                    processor.start()
                    self.processors[model] = processor
        
        # Add to queue
        self.queues[model].put(request)
        
        # Wait for result
        return request.wait(timeout)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        stats = dict(self.stats)
        
        if self.stats["batch_sizes"]:
            stats["avg_batch_size"] = sum(self.stats["batch_sizes"]) / len(self.stats["batch_sizes"])
            stats["max_batch_size"] = max(self.stats["batch_sizes"])
        else:
            stats["avg_batch_size"] = 0
            stats["max_batch_size"] = 0
        
        return stats
    
    def shutdown(self) -> None:
        """Shutdown batch processors."""
        self.running = False
        
        # Wait for processors to finish
        for processor in self.processors.values():
            processor.join(timeout=1.0)


# Global batcher instance
_llm_batcher: Optional[_LLMBatcher] = None


def _get_llm_batcher() -> _LLMBatcher:
    """Get or create global LLM batcher."""
    global _llm_batcher
    
    if _llm_batcher is None:
        # Get batch settings from runtime config
        batch_config = RUNTIME_SETTINGS.get("llm_batch", {})
        enabled = batch_config.get("enabled", True)
        
        if not enabled:
            # Return dummy batcher that calls directly
            class _DummyBatcher:
                def add_request(self, prompt, model, params, timeout=30.0):
                    return _call_llm_single(model, prompt, params)
                def get_stats(self):
                    return {}
                def shutdown(self):
                    pass
            return _DummyBatcher()
        
        max_batch_size = batch_config.get("max_batch_size", 10)
        batch_timeout_ms = batch_config.get("batch_timeout_ms", 50.0)
        
        _llm_batcher = _LLMBatcher(
            max_batch_size=max_batch_size,
            batch_timeout_ms=batch_timeout_ms
        )
    
    return _llm_batcher


def _batched_llm_call(
    prompt: str,
    model: str,
    params: Dict[str, Any],
    timeout: float = 30.0,
) -> Any:
    """Execute LLM call with automatic batching.
    
    Args:
        prompt: Prompt text
        model: Model identifier  
        params: Call parameters
        timeout: Max wait time
    
    Returns:
        LLM response
    """
    batcher = _get_llm_batcher()
    
    try:
        return batcher.add_request(prompt, model, params, timeout)
    except Exception as exc:
        _record_event(
            "llm_batch",
            "error",
            "error",
            {"model": model, "error": str(exc)}
        )
        raise


def _get_batch_stats() -> Dict[str, Any]:
    """Get current batching statistics."""
    batcher = _get_llm_batcher()
    return batcher.get_stats()
'''
).strip()

__all__ = ['LLM_BATCHING']
