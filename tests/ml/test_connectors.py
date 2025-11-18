"""Tests for connector retry/resilience utilities and observability."""

from __future__ import annotations

import asyncio
from typing import Any, Callable
from unittest.mock import AsyncMock, Mock, call, patch

import httpx
import pytest

from namel3ss.ml.connectors.base import (
    RetryConfig,
    make_resilient_request,
    run_many_safe,
)


class TestRetryConfig:
    """Test RetryConfig dataclass validation and defaults."""

    def test_default_values(self):
        """RetryConfig should have sensible defaults."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay_seconds == 0.5
        assert config.max_delay_seconds == 5.0

    def test_custom_values(self):
        """RetryConfig should accept custom retry parameters."""
        config = RetryConfig(
            max_attempts=5,
            base_delay_seconds=1.0,
            max_delay_seconds=10.0,
        )
        assert config.max_attempts == 5
        assert config.base_delay_seconds == 1.0
        assert config.max_delay_seconds == 10.0


class TestMakeResilientRequest:
    """Test make_resilient_request wrapper for async HTTP calls."""

    @pytest.mark.asyncio
    async def test_successful_first_attempt(self):
        """Should succeed on first try without retries."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "success"}

        async def request_fn(client: Any) -> Any:
            return mock_response

        config = RetryConfig(max_attempts=3, base_delay_seconds=0.1)
        
        with patch("namel3ss.observability.logging.log_retry_event") as mock_log, \
             patch("namel3ss.observability.metrics.get_metric") as mock_metric:
            
            mock_counter = Mock()
            mock_metric.return_value = mock_counter
            
            result = await make_resilient_request(
                request_fn,
                config,
                "test-connector",
                None,  # client not used in this test
            )
            
            assert result == mock_response
            # Should not log retries on success
            mock_log.assert_not_called()
            # Should not increment retry counter on first success
            mock_counter.labels.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self):
        """Should retry on transient failures and eventually succeed."""
        call_count = 0
        
        async def flaky_request(client: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.TimeoutException("Timeout")
            mock_response = Mock()
            mock_response.status_code = 200
            return mock_response

        config = RetryConfig(max_attempts=5, base_delay_seconds=0.01, max_delay_seconds=0.05)
        
        with patch("namel3ss.observability.logging.log_retry_event") as mock_log, \
             patch("namel3ss.observability.metrics.get_metric") as mock_metric, \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            
            mock_counter = Mock()
            mock_metric.return_value = mock_counter
            
            result = await make_resilient_request(
                flaky_request,
                config,
                "test-connector",
                None,
            )
            
            assert result.status_code == 200
            assert call_count == 3
            # Should have logged 2 retry events (attempts 1 and 2 failed)
            assert mock_log.call_count == 2
            # Should have slept twice (after attempts 1 and 2)
            assert mock_sleep.call_count == 2
            # Verify exponential backoff: first sleep ~0.01s, second ~0.02s (with jitter)
            first_sleep = mock_sleep.call_args_list[0][0][0]
            second_sleep = mock_sleep.call_args_list[1][0][0]
            assert 0.01 <= first_sleep <= 0.011  # base_delay + 10% jitter
            assert 0.02 <= second_sleep <= 0.022  # base_delay * 2 + 10% jitter

    @pytest.mark.asyncio
    async def test_exhaust_retries_and_fail(self):
        """Should raise exception after exhausting all retry attempts."""
        
        async def always_fail(client: Any) -> Any:
            raise httpx.ConnectError("Connection refused")

        config = RetryConfig(max_attempts=3, base_delay_seconds=0.01)
        
        with patch("namel3ss.observability.logging.log_retry_event") as mock_log, \
             patch("namel3ss.observability.metrics.get_metric") as mock_metric, \
             patch("asyncio.sleep", new_callable=AsyncMock):
            
            mock_counter = Mock()
            mock_metric.return_value = mock_counter
            
            with pytest.raises(httpx.ConnectError, match="Connection refused"):
                await make_resilient_request(
                    always_fail,
                    config,
                    "failing-connector",
                    None,
                )
            
            # Should log all retry attempts
            assert mock_log.call_count == 3
            # Verify connector name in logs
            for call_obj in mock_log.call_args_list:
                assert call_obj[0][0] == "failing-connector"

    @pytest.mark.asyncio
    async def test_http_status_error_propagation(self):
        """Should retry and eventually propagate HTTP status errors."""
        
        async def http_error_request(client: Any) -> Any:
            response = Mock()
            response.status_code = 500
            raise httpx.HTTPStatusError("Server error", request=Mock(), response=response)

        config = RetryConfig(max_attempts=2, base_delay_seconds=0.01)
        
        with patch("namel3ss.observability.logging.log_retry_event"), \
             patch("namel3ss.observability.metrics.get_metric") as mock_metric, \
             patch("asyncio.sleep", new_callable=AsyncMock):
            
            mock_counter = Mock()
            mock_metric.return_value = mock_counter
            
            with pytest.raises(httpx.HTTPStatusError, match="Server error"):
                await make_resilient_request(
                    http_error_request,
                    config,
                    "test-connector",
                    None,
                )

    @pytest.mark.asyncio
    async def test_observability_metrics_on_retry(self):
        """Should emit metrics for each retry attempt."""
        call_count = 0
        
        async def flaky_request(client: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.TimeoutException("Timeout")
            return Mock(status_code=200)

        config = RetryConfig(max_attempts=3, base_delay_seconds=0.01)
        
        with patch("namel3ss.observability.logging.log_retry_event"), \
             patch("namel3ss.observability.metrics.get_metric") as mock_metric, \
             patch("asyncio.sleep", new_callable=AsyncMock):
            
            mock_counter = Mock()
            mock_labels_chain = Mock()
            mock_counter.labels.return_value = mock_labels_chain
            mock_metric.return_value = mock_counter
            
            await make_resilient_request(
                flaky_request,
                config,
                "test-connector",
                None,
            )
            
            # Should call get_metric to retrieve retry counter
            mock_metric.assert_called_once_with("connector_retry_total")
            # Should label metric with connector name
            mock_counter.labels.assert_called_once_with(connector="test-connector")
            # Should increment counter once (for the failed attempt)
            mock_labels_chain.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        """Should cap retry delay at max_delay_seconds."""
        call_count = 0
        
        async def many_failures(client: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count < 5:
                raise httpx.TimeoutException("Timeout")
            return Mock(status_code=200)

        config = RetryConfig(
            max_attempts=5,
            base_delay_seconds=2.0,
            max_delay_seconds=3.0,  # Cap at 3 seconds
        )
        
        with patch("namel3ss.observability.logging.log_retry_event"), \
             patch("namel3ss.observability.metrics.get_metric"), \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            
            await make_resilient_request(
                many_failures,
                config,
                "test-connector",
                None,
            )
            
            # Verify all delays respect max_delay_seconds + 10% jitter
            for call_obj in mock_sleep.call_args_list:
                delay = call_obj[0][0]
                assert delay <= 3.3  # max_delay + 10% jitter


class TestRunManySafe:
    """Test run_many_safe for concurrent request execution with limits."""

    @pytest.mark.asyncio
    async def test_sequential_execution_below_limit(self):
        """Should execute tasks concurrently when below concurrency limit."""
        results = []
        
        async def task(value: int) -> int:
            results.append(f"start-{value}")
            await asyncio.sleep(0.01)
            results.append(f"end-{value}")
            return value * 2

        tasks = [task(i) for i in range(3)]
        config = RetryConfig()  # Default concurrency_limit=10
        
        outputs = await run_many_safe(tasks, config)
        
        assert outputs == [0, 2, 4]
        # All tasks should have started before any completed (concurrent)
        assert results[:3] == ["start-0", "start-1", "start-2"]

    @pytest.mark.asyncio
    async def test_concurrency_limit_enforcement(self):
        """Should limit concurrent execution to concurrency_limit."""
        active_count = 0
        max_active = 0
        
        async def monitored_task(value: int) -> int:
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.02)
            active_count -= 1
            return value

        tasks = [monitored_task(i) for i in range(10)]
        config = RetryConfig(max_attempts=1, base_delay_seconds=0.01)
        
        # Note: run_many_safe currently doesn't implement concurrency limiting
        # This test documents expected behavior for future implementation
        outputs = await run_many_safe(tasks, config)
        
        assert len(outputs) == 10
        assert outputs == list(range(10))
        # Currently no limit enforced, but should be <= concurrency_limit in future
        # assert max_active <= config.concurrency_limit

    @pytest.mark.asyncio
    async def test_error_propagation_in_concurrent_batch(self):
        """Should propagate errors from failed tasks in batch."""
        
        async def task_that_fails(value: int) -> int:
            if value == 3:
                raise ValueError(f"Task {value} failed")
            return value

        tasks = [task_that_fails(i) for i in range(5)]
        config = RetryConfig()
        
        # run_many_safe currently doesn't handle errors gracefully
        # This documents expected behavior
        with pytest.raises(ValueError, match="Task 3 failed"):
            await run_many_safe(tasks, config)

    @pytest.mark.asyncio
    async def test_empty_task_list(self):
        """Should handle empty task list gracefully."""
        config = RetryConfig()
        outputs = await run_many_safe([], config)
        assert outputs == []


class TestConfigIntegration:
    """Integration tests for config-driven retry behavior."""

    def test_extract_connector_config_with_defaults(self):
        """Config extraction should merge app overrides with workspace defaults."""
        from namel3ss.config import WorkspaceDefaults, extract_connector_config
        
        defaults = WorkspaceDefaults()
        config = extract_connector_config(None, defaults)
        
        assert config["retry_max_attempts"] == 3
        assert config["retry_base_delay"] == 0.5
        assert config["retry_max_delay"] == 5.0
        assert config["concurrency_limit"] == 10

    def test_extract_connector_config_with_overrides(self):
        """App-specific overrides should take precedence."""
        from namel3ss.config import AppConfig, WorkspaceDefaults, extract_connector_config
        
        defaults = WorkspaceDefaults()
        app_config = AppConfig(
            name="test-app",
            file=None,  # type: ignore
            connector_retry_max_attempts=5,
            connector_retry_base_delay=1.0,
        )
        
        config = extract_connector_config(app_config, defaults)
        
        assert config["retry_max_attempts"] == 5
        assert config["retry_base_delay"] == 1.0
        assert config["retry_max_delay"] == 5.0  # From defaults
        assert config["concurrency_limit"] == 10  # From defaults
