"""Resilient async connector utilities."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Optional, Sequence, TypeVar

import httpx

from namel3ss.observability import logging as obs_logging
from namel3ss.observability import metrics as obs_metrics


class ConnectorError(Exception):
    """Base error for connector failures."""


class TransientNetworkError(ConnectorError):
    """Raised when a transient network issue occurs."""


class RateLimitError(ConnectorError):
    """Raised when the upstream service rate limits."""


@dataclass(slots=True)
class RetryConfig:
    """Configuration for resilient requests."""

    max_attempts: int = 3
    base_delay: float = 0.5  # seconds
    max_delay: float = 5.0
    jitter: float = 0.2
    timeout: Optional[float] = 30.0

    def compute_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        return delay + random.uniform(0, self.jitter)


T = TypeVar("T")

_logger = obs_logging.get_logger(__name__)
_emit_metric = obs_metrics.get_metric("connector_retry_total")


async def make_resilient_request(
    request_fn: Callable[[], Awaitable[T]],
    *,
    retry_config: RetryConfig | None = None,
    name: str = "connector_request",
    retry_on_status: Sequence[int] | None = None,
) -> T:
    cfg = retry_config or RetryConfig()
    retry_on = set(retry_on_status or {408, 409, 425, 429, 500, 502, 503, 504})
    attempt = 1
    start = time.perf_counter()

    while True:
        try:
            if cfg.timeout:
                return await asyncio.wait_for(request_fn(), timeout=cfg.timeout)
            return await request_fn()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in retry_on and attempt < cfg.max_attempts:
                await _handle_retry(name, attempt, exc, cfg, start)
                attempt += 1
                continue
            raise RateLimitError(str(exc)) if status == 429 else ConnectorError(str(exc)) from exc
        except (httpx.TransportError, asyncio.TimeoutError) as exc:
            if attempt < cfg.max_attempts:
                await _handle_retry(name, attempt, exc, cfg, start)
                attempt += 1
                continue
            raise TransientNetworkError(str(exc)) from exc


async def _handle_retry(name: str, attempt: int, exc: Exception, cfg: RetryConfig, start: float) -> None:
    delay = cfg.compute_delay(attempt)
    obs_logging.log_retry_event(
        provider=name,
        model=None,
        attempt=attempt,
        delay=delay,
        reason=str(exc),
        extras={"elapsed": round(time.perf_counter() - start, 3)},
    )
    _emit_metric(1, labels={"name": name, "attempt": str(attempt)})
    await asyncio.sleep(delay)


async def run_many_safe(
    coroutines: Iterable[Awaitable[T]],
    *,
    concurrency: int = 5,
    suppress_exceptions: bool = True,
) -> list[Optional[T]]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = list(coroutines)
    results: list[Optional[T]] = [None] * len(tasks)

    async def _runner(idx: int, coro: Awaitable[T]) -> None:
        async with semaphore:
            try:
                results[idx] = await coro
            except Exception as exc:  # noqa: BLE001
                _logger.exception("Concurrent task failed", exc_info=exc)
                if not suppress_exceptions:
                    raise
                results[idx] = None

    await asyncio.gather(*(_runner(i, coro) for i, coro in enumerate(tasks)))
    return results
