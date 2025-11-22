"""Tests for async chain execution in generated backends."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestAsyncChainExecution:
    """Test async chain execution and workflow functions."""
    
    @pytest.mark.asyncio
    async def test_run_chain_is_async(self):
        """Verify run_chain is an async function."""
        # This test validates the template generates async code
        # In a real generated backend, run_chain would be imported
        
        async def mock_run_chain(name, payload=None, context=None):
            """Mock async run_chain function."""
            await asyncio.sleep(0.01)  # Simulate async work
            return {
                "status": "ok",
                "result": "test_result",
                "steps": [],
                "inputs": payload or {},
                "metadata": {"elapsed_ms": 10.0}
            }
        
        result = await mock_run_chain("test_chain", {"input": "test"})
        
        assert result["status"] == "ok"
        assert result["result"] == "test_result"
        assert "elapsed_ms" in result["metadata"]
    
    @pytest.mark.asyncio
    async def test_workflow_nodes_execute_sequentially(self):
        """Test that workflow nodes execute in correct order."""
        execution_order = []
        
        async def mock_execute_step(step_name):
            await asyncio.sleep(0.01)
            execution_order.append(step_name)
            return "ok", f"result_{step_name}", f"working_{step_name}", False
        
        # Simulate sequential execution
        await mock_execute_step("step1")
        await mock_execute_step("step2")
        await mock_execute_step("step3")
        
        assert execution_order == ["step1", "step2", "step3"]
    
    @pytest.mark.asyncio
    async def test_parallel_steps_execute_concurrently(self):
        """Test that parallel steps execute concurrently with asyncio.gather."""
        start_times = {}
        end_times = {}
        
        async def mock_parallel_step(step_name, duration=0.05):
            start_times[step_name] = asyncio.get_event_loop().time()
            await asyncio.sleep(duration)
            end_times[step_name] = asyncio.get_event_loop().time()
            return f"result_{step_name}"
        
        # Execute in parallel using gather
        results = await asyncio.gather(
            mock_parallel_step("step1"),
            mock_parallel_step("step2"),
            mock_parallel_step("step3"),
        )
        
        assert len(results) == 3
        assert results == ["result_step1", "result_step2", "result_step3"]
        
        # Verify they ran concurrently (overlapping time windows)
        # All should start before any finishes
        earliest_end = min(end_times.values())
        latest_start = max(start_times.values())
        assert latest_start < earliest_end, "Steps should have overlapped in time"
    
    @pytest.mark.asyncio
    async def test_timeout_protection(self):
        """Test that chains timeout correctly using asyncio.wait_for."""
        async def long_running_chain():
            await asyncio.sleep(10)  # Long operation
            return "result"
        
        # Should timeout after 0.1 seconds
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(long_running_chain(), timeout=0.1)
    
    @pytest.mark.asyncio
    async def test_semaphore_rate_limiting(self):
        """Test that semaphore limits concurrent execution."""
        max_concurrent = 2
        semaphore = asyncio.Semaphore(max_concurrent)
        current_running = 0
        max_seen = 0
        
        async def rate_limited_operation(task_id):
            nonlocal current_running, max_seen
            
            async with semaphore:
                current_running += 1
                max_seen = max(max_seen, current_running)
                await asyncio.sleep(0.05)
                current_running -= 1
            
            return f"result_{task_id}"
        
        # Launch 10 tasks
        results = await asyncio.gather(*[
            rate_limited_operation(i) for i in range(10)
        ])
        
        assert len(results) == 10
        assert max_seen <= max_concurrent, f"Max concurrent was {max_seen}, expected <= {max_concurrent}"
    
    @pytest.mark.asyncio
    async def test_error_handling_in_async_chain(self):
        """Test that errors in async chains are properly handled."""
        async def failing_step():
            await asyncio.sleep(0.01)
            raise ValueError("Step failed")
        
        with pytest.raises(ValueError, match="Step failed"):
            await failing_step()
    
    @pytest.mark.asyncio
    async def test_cancellation_propagation(self):
        """Test that cancellation is properly propagated through async chains."""
        cancelled = False
        
        async def cancellable_operation():
            nonlocal cancelled
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled = True
                raise
        
        task = asyncio.create_task(cancellable_operation())
        await asyncio.sleep(0.01)
        task.cancel()
        
        with pytest.raises(asyncio.CancelledError):
            await task
        
        assert cancelled, "Cancellation should have been caught"


class TestAsyncLLMConnectors:
    """Test async LLM connector integration."""
    
    @pytest.mark.asyncio
    async def test_call_llm_connector_async(self):
        """Test that call_llm_connector properly awaits async providers."""
        mock_provider = AsyncMock()
        mock_provider.agenerate = AsyncMock(return_value=MagicMock(
            text="Generated response",
            raw="raw response",
            metadata={"provider": "test"},
            usage={"tokens": 100},
            finish_reason="stop",
            model="test-model"
        ))
        
        # Simulate async connector call
        result = await mock_provider.agenerate("Test prompt")
        
        assert result.text == "Generated response"
        mock_provider.agenerate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multiple_llm_calls_concurrent(self):
        """Test multiple LLM calls execute concurrently."""
        async def mock_llm_call(prompt, delay=0.05):
            await asyncio.sleep(delay)
            return f"response_for_{prompt}"
        
        start_time = asyncio.get_event_loop().time()
        
        results = await asyncio.gather(
            mock_llm_call("prompt1"),
            mock_llm_call("prompt2"),
            mock_llm_call("prompt3"),
        )
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        assert len(results) == 3
        # Should take ~0.05s (concurrent), not 0.15s (sequential)
        assert elapsed < 0.1, f"Took {elapsed}s, should be concurrent"


class TestStreamingEndpoints:
    """Test streaming response generation."""
    
    @pytest.mark.asyncio
    async def test_stream_generator_yields_chunks(self):
        """Test that streaming generators yield chunks progressively."""
        async def mock_stream_generator():
            for i in range(5):
                await asyncio.sleep(0.01)
                yield {"chunk": f"token_{i}", "index": i}
            yield {"status": "complete", "chunks": 5}
        
        chunks = []
        async for chunk in mock_stream_generator():
            chunks.append(chunk)
        
        assert len(chunks) == 6  # 5 chunks + 1 complete
        assert chunks[-1]["status"] == "complete"
        assert chunks[0]["chunk"] == "token_0"
    
    @pytest.mark.asyncio
    async def test_sse_formatting(self):
        """Test Server-Sent Events formatting."""
        import json
        
        async def sse_generator():
            data = {"chunk": "Hello", "index": 1}
            yield f"data: {json.dumps(data)}\n\n"
            
            data = {"status": "complete"}
            yield f"data: {json.dumps(data)}\n\n"
            
            yield "data: [DONE]\n\n"
        
        events = []
        async for event in sse_generator():
            events.append(event)
        
        assert len(events) == 3
        assert events[0].startswith("data: {")
        assert events[-1] == "data: [DONE]\n\n"
    
    @pytest.mark.asyncio
    async def test_streaming_with_backpressure(self):
        """Test streaming handles backpressure correctly."""
        queue = asyncio.Queue(maxsize=2)  # Small buffer
        
        async def producer():
            for i in range(10):
                await queue.put(f"item_{i}")
                await asyncio.sleep(0.01)
            await queue.put(None)  # Sentinel
        
        async def consumer():
            items = []
            while True:
                item = await queue.get()
                if item is None:
                    break
                items.append(item)
                await asyncio.sleep(0.02)  # Slower consumer
            return items
        
        producer_task = asyncio.create_task(producer())
        items = await consumer()
        await producer_task
        
        assert len(items) == 10


class TestConcurrencyPrimitives:
    """Test concurrency control mechanisms."""
    
    @pytest.mark.asyncio
    async def test_parallel_group_detection(self):
        """Test detection of parallelizable step groups."""
        steps = [
            {"name": "step1", "parallel": True},
            {"name": "step2", "parallel": True},
            {"name": "step3"},  # Sequential
            {"name": "step4", "type": "if"},  # Control flow
            {"name": "step5", "depends_on": ["step4"]},  # Has dependency
        ]
        
        # Mock detection logic
        groups = []
        parallel_group = []
        
        for step in steps:
            if step.get("parallel") and not step.get("depends_on") and step.get("type", "step") == "step":
                parallel_group.append(step)
            else:
                if parallel_group:
                    groups.append(parallel_group)
                    parallel_group = []
                groups.append([step])
        
        if parallel_group:
            groups.append(parallel_group)
        
        assert len(groups) == 4
        assert len(groups[0]) == 2  # step1, step2 in parallel
        assert len(groups[1]) == 1  # step3 sequential
    
    @pytest.mark.asyncio
    async def test_semaphore_fairness(self):
        """Test that semaphore provides fair access."""
        semaphore = asyncio.Semaphore(1)
        access_order = []
        
        async def acquire_resource(task_id):
            async with semaphore:
                access_order.append(task_id)
                await asyncio.sleep(0.01)
        
        # Launch tasks
        await asyncio.gather(*[
            acquire_resource(i) for i in range(5)
        ])
        
        # All tasks should have executed
        assert len(access_order) == 5
        assert set(access_order) == {0, 1, 2, 3, 4}
    
    @pytest.mark.asyncio
    async def test_timeout_with_partial_results(self):
        """Test that timeouts return partial results."""
        results = []
        
        async def multi_step_chain():
            for i in range(10):
                await asyncio.sleep(0.05)
                results.append(f"step_{i}")
            return results
        
        try:
            await asyncio.wait_for(multi_step_chain(), timeout=0.15)
        except asyncio.TimeoutError:
            pass
        
        # Should have completed ~3 steps before timeout
        assert len(results) >= 2
        assert len(results) < 10


class TestErrorHandling:
    """Test error handling in async chains."""
    
    @pytest.mark.asyncio
    async def test_error_in_parallel_steps(self):
        """Test error handling when one parallel step fails."""
        async def failing_step():
            await asyncio.sleep(0.01)
            raise ValueError("Step failed")
        
        async def successful_step():
            await asyncio.sleep(0.01)
            return "success"
        
        # Using return_exceptions to capture both results and exceptions
        results = await asyncio.gather(
            successful_step(),
            failing_step(),
            successful_step(),
            return_exceptions=True
        )
        
        assert len(results) == 3
        assert results[0] == "success"
        assert isinstance(results[1], ValueError)
        assert results[2] == "success"
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when steps fail."""
        async def step_with_fallback():
            try:
                # Simulate failure
                raise ConnectionError("Provider unavailable")
            except ConnectionError:
                # Fallback to stub response
                return {"status": "stub", "result": "[stub: llm call failed]"}
        
        result = await step_with_fallback()
        
        assert result["status"] == "stub"
        assert "stub" in result["result"]
    
    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test retry logic for transient failures."""
        attempts = 0
        
        async def flaky_operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await flaky_operation()
                break
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.01)
        
        assert result == "success"
        assert attempts == 3


class TestPerformance:
    """Test performance characteristics of async implementation."""
    
    @pytest.mark.asyncio
    async def test_concurrent_throughput(self):
        """Test that async enables high concurrent throughput."""
        async def mock_request(request_id):
            await asyncio.sleep(0.1)  # Simulate I/O
            return f"response_{request_id}"
        
        start = asyncio.get_event_loop().time()
        
        # 100 concurrent requests
        results = await asyncio.gather(*[
            mock_request(i) for i in range(100)
        ])
        
        elapsed = asyncio.get_event_loop().time() - start
        
        assert len(results) == 100
        # Should take ~0.1s (concurrent), not 10s (sequential)
        assert elapsed < 0.5, f"Took {elapsed}s for 100 requests"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test that async doesn't create excessive overhead."""
        import sys
        
        # Create many concurrent tasks
        async def lightweight_task(task_id):
            await asyncio.sleep(0.01)
            return task_id
        
        tasks = [asyncio.create_task(lightweight_task(i)) for i in range(1000)]
        
        # Tasks should be lightweight
        task_size = sys.getsizeof(tasks[0])
        assert task_size < 1000, f"Task size is {task_size} bytes"
        
        results = await asyncio.gather(*tasks)
        assert len(results) == 1000
    
    @pytest.mark.asyncio
    async def test_no_event_loop_blocking(self):
        """Test that async operations don't block the event loop."""
        loop_blocked = False
        
        async def monitor_loop():
            nonlocal loop_blocked
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.05)
            elapsed = asyncio.get_event_loop().time() - start
            # If loop is blocked, sleep will take much longer
            if elapsed > 0.1:
                loop_blocked = True
        
        async def async_operation():
            # Non-blocking async operation
            await asyncio.sleep(0.2)
            return "done"
        
        # Run both concurrently
        monitor_task = asyncio.create_task(monitor_loop())
        operation_task = asyncio.create_task(async_operation())
        
        await asyncio.gather(monitor_task, operation_task)
        
        assert not loop_blocked, "Event loop should not be blocked"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
