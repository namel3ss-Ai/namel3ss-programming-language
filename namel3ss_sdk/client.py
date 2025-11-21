"""N3 remote client with retry, circuit breaker, and tracing.

Provides typed interfaces for calling N3 chains, prompts, agents, and RAG pipelines
running on a remote N3 server.
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import httpx

from .config import N3ClientConfig, get_settings
from .exceptions import (
    N3AuthError,
    N3CircuitBreakerError,
    N3ClientError,
    N3ConnectionError,
    N3RateLimitError,
    N3SchemaError,
    N3ServerError,
    N3TimeoutError,
)


class CircuitBreaker:
    """Circuit breaker for fault tolerance.
    
    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
    """
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    
    def __init__(self, threshold: int = 5, timeout: float = 60.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = self.CLOSED
    
    def call(self, func):
        """Execute function with circuit breaker protection."""
        if self.state == self.OPEN:
            if self.last_failure_time and (time.time() - self.last_failure_time) >= self.timeout:
                self.state = self.HALF_OPEN
            else:
                raise N3CircuitBreakerError("Circuit breaker is OPEN - too many failures")
        
        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    async def acall(self, coro):
        """Execute coroutine with circuit breaker protection."""
        if self.state == self.OPEN:
            if self.last_failure_time and (time.time() - self.last_failure_time) >= self.timeout:
                self.state = self.HALF_OPEN
            else:
                raise N3CircuitBreakerError("Circuit breaker is OPEN - too many failures")
        
        try:
            result = await coro
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == self.HALF_OPEN:
            self.state = self.CLOSED
        self.failures = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.threshold:
            self.state = self.OPEN


class ChainsAPI:
    """API for executing N3 chains."""
    
    def __init__(self, client: 'N3Client'):
        self._client = client
    
    def run(
        self,
        name: str,
        *,
        timeout: Optional[float] = None,
        **payload: Any,
    ) -> Dict[str, Any]:
        """Execute a chain synchronously.
        
        Args:
            name: Chain name
            timeout: Override default timeout
            **payload: Chain inputs
        
        Returns:
            Chain execution result
        
        Raises:
            N3ClientError: Chain not found or invalid inputs
            N3TimeoutError: Request timed out
            N3RuntimeError: Chain execution failed
        
        Example:
            >>> result = client.chains.run(
            ...     "summarize",
            ...     text="Long document...",
            ...     max_length=100
            ... )
            >>> print(result['result'])
        """
        return self._client._request(
            "POST",
            f"/api/chains/{name}/execute",
            json=payload,
            timeout=timeout,
        )
    
    async def arun(
        self,
        name: str,
        *,
        timeout: Optional[float] = None,
        **payload: Any,
    ) -> Dict[str, Any]:
        """Execute a chain asynchronously."""
        return await self._client._arequest(
            "POST",
            f"/api/chains/{name}/execute",
            json=payload,
            timeout=timeout,
        )


class PromptsAPI:
    """API for executing N3 prompts."""
    
    def __init__(self, client: 'N3Client'):
        self._client = client
    
    def run(
        self,
        name: str,
        *,
        timeout: Optional[float] = None,
        **inputs: Any,
    ) -> Dict[str, Any]:
        """Execute a prompt synchronously."""
        return self._client._request(
            "POST",
            f"/api/prompts/{name}/execute",
            json=inputs,
            timeout=timeout,
        )
    
    async def arun(
        self,
        name: str,
        *,
        timeout: Optional[float] = None,
        **inputs: Any,
    ) -> Dict[str, Any]:
        """Execute a prompt asynchronously."""
        return await self._client._arequest(
            "POST",
            f"/api/prompts/{name}/execute",
            json=inputs,
            timeout=timeout,
        )


class AgentsAPI:
    """API for running N3 agents."""
    
    def __init__(self, client: 'N3Client'):
        self._client = client
    
    def run(
        self,
        name: str,
        user_input: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run an agent synchronously."""
        payload = {"user_input": user_input}
        if context:
            payload["context"] = context
        if max_turns is not None:
            payload["max_turns"] = max_turns
        
        return self._client._request(
            "POST",
            f"/api/agents/{name}/execute",
            json=payload,
            timeout=timeout,
        )
    
    async def arun(
        self,
        name: str,
        user_input: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run an agent asynchronously."""
        payload = {"user_input": user_input}
        if context:
            payload["context"] = context
        if max_turns is not None:
            payload["max_turns"] = max_turns
        
        return await self._client._arequest(
            "POST",
            f"/api/agents/{name}/execute",
            json=payload,
            timeout=timeout,
        )


class RagAPI:
    """API for querying N3 RAG pipelines."""
    
    def __init__(self, client: 'N3Client'):
        self._client = client
    
    def query(
        self,
        pipeline: str,
        query: str,
        *,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Query a RAG pipeline synchronously."""
        payload = {"query": query}
        if top_k is not None:
            payload["top_k"] = top_k
        if filters:
            payload["filters"] = filters
        
        return self._client._request(
            "POST",
            f"/api/rag/{pipeline}/query",
            json=payload,
            timeout=timeout,
        )
    
    async def aquery(
        self,
        pipeline: str,
        query: str,
        *,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Query a RAG pipeline asynchronously."""
        payload = {"query": query}
        if top_k is not None:
            payload["top_k"] = top_k
        if filters:
            payload["filters"] = filters
        
        return await self._client._arequest(
            "POST",
            f"/api/rag/{pipeline}/query",
            json=payload,
            timeout=timeout,
        )


class N3Client:
    """N3 remote client with retry, circuit breaker, and tracing.
    
    Provides typed interfaces for calling N3 chains, prompts, agents, and RAG
    pipelines running on a remote N3 server.
    
    Features:
        - Automatic retries with exponential backoff
        - Circuit breaker for fault tolerance
        - OpenTelemetry instrumentation
        - Request ID tracking
        - Comprehensive error handling
    
    Example:
        Synchronous:
        >>> client = N3Client(base_url="https://api.example.com")
        >>> result = client.chains.run("summarize", text="...")
        
        Asynchronous:
        >>> async with N3Client(base_url="...") as client:
        ...     result = await client.chains.arun("summarize", text="...")
        
        With custom config:
        >>> config = N3ClientConfig(
        ...     base_url="https://api.example.com",
        ...     max_retries=5,
        ...     timeout=60.0
        ... )
        >>> client = N3Client(config=config)
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        config: Optional[N3ClientConfig] = None,
    ):
        """Initialize N3 client.
        
        Args:
            base_url: N3 server URL (overrides config)
            api_token: Auth token (overrides config)
            config: Client configuration (uses defaults if not provided)
        """
        self.config = config or get_settings().client
        
        # Override config with explicit params
        if base_url:
            self.config.base_url = base_url
        if api_token:
            self.config.api_token = api_token
        
        # Create HTTP client
        headers = {"User-Agent": "n3-sdk-python/0.1.0"}
        if self.config.api_token:
            headers["Authorization"] = f"Bearer {self.config.api_token}"
        
        timeout_config = httpx.Timeout(
            timeout=self.config.timeout,
            connect=self.config.connect_timeout,
        )
        
        self._client = httpx.Client(
            base_url=self.config.base_url,
            headers=headers,
            timeout=timeout_config,
            verify=self.config.verify_ssl,
        )
        
        self._async_client: Optional[httpx.AsyncClient] = None
        
        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            timeout=self.config.circuit_breaker_timeout,
        )
        
        # API namespaces
        self.chains = ChainsAPI(self)
        self.prompts = PromptsAPI(self)
        self.agents = AgentsAPI(self)
        self.rag = RagAPI(self)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
    
    def close(self):
        """Close HTTP connections."""
        if self._client:
            self._client.close()
    
    async def aclose(self):
        """Close async HTTP connections."""
        if self._async_client:
            await self._async_client.aclose()
        self.close()
    
    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make synchronous HTTP request with retry and circuit breaker."""
        request_id = str(uuid.uuid4())
        kwargs.setdefault("headers", {})
        kwargs["headers"]["X-Request-ID"] = request_id
        
        if timeout:
            kwargs["timeout"] = timeout
        
        def _make_request():
            return self._do_request(method, path, request_id, **kwargs)
        
        return self._circuit_breaker.call(_make_request)
    
    async def _arequest(
        self,
        method: str,
        path: str,
        *,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make asynchronous HTTP request with retry and circuit breaker."""
        if self._async_client is None:
            headers = {"User-Agent": "n3-sdk-python/0.1.0"}
            if self.config.api_token:
                headers["Authorization"] = f"Bearer {self.config.api_token}"
            
            timeout_config = httpx.Timeout(
                timeout=self.config.timeout,
                connect=self.config.connect_timeout,
            )
            
            self._async_client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=timeout_config,
                verify=self.config.verify_ssl,
            )
        
        request_id = str(uuid.uuid4())
        kwargs.setdefault("headers", {})
        kwargs["headers"]["X-Request-ID"] = request_id
        
        if timeout:
            kwargs["timeout"] = timeout
        
        return await self._circuit_breaker.acall(
            self._do_arequest(method, path, request_id, **kwargs)
        )
    
    def _do_request(
        self,
        method: str,
        path: str,
        request_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute HTTP request with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.request(method, path, **kwargs)
                return self._handle_response(response, request_id)
            
            except httpx.TimeoutException as e:
                last_exception = N3TimeoutError(
                    timeout_seconds=self.config.timeout,
                    request_id=request_id,
                )
                if attempt < self.config.max_retries:
                    self._backoff(attempt)
                    continue
            
            except httpx.ConnectError as e:
                last_exception = N3ConnectionError(
                    f"Failed to connect to {self.config.base_url}",
                    request_id=request_id,
                )
                if attempt < self.config.max_retries:
                    self._backoff(attempt)
                    continue
            
            except Exception as e:
                last_exception = e
                break
        
        if last_exception:
            raise last_exception
        
        raise N3ClientError("Request failed after retries", request_id=request_id)
    
    async def _do_arequest(
        self,
        method: str,
        path: str,
        request_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute async HTTP request with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._async_client.request(method, path, **kwargs)
                return self._handle_response(response, request_id)
            
            except httpx.TimeoutException as e:
                last_exception = N3TimeoutError(
                    timeout_seconds=self.config.timeout,
                    request_id=request_id,
                )
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self._backoff_delay(attempt))
                    continue
            
            except httpx.ConnectError as e:
                last_exception = N3ConnectionError(
                    f"Failed to connect to {self.config.base_url}",
                    request_id=request_id,
                )
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self._backoff_delay(attempt))
                    continue
            
            except Exception as e:
                last_exception = e
                break
        
        if last_exception:
            raise last_exception
        
        raise N3ClientError("Request failed after retries", request_id=request_id)
    
    def _handle_response(
        self,
        response: httpx.Response,
        request_id: str,
    ) -> Dict[str, Any]:
        """Handle HTTP response and raise appropriate exceptions."""
        # Success
        if 200 <= response.status_code < 300:
            return response.json()
        
        # Get error details
        try:
            error_data = response.json()
            message = error_data.get("detail", response.text)
        except Exception:
            message = response.text or f"HTTP {response.status_code}"
        
        # Client errors (4xx)
        if response.status_code == 401:
            raise N3AuthError(message, request_id=request_id)
        
        if response.status_code == 404:
            raise N3ClientError(
                message,
                status_code=404,
                request_id=request_id,
            )
        
        if response.status_code == 422:
            validation_errors = error_data.get("errors", []) if isinstance(error_data, dict) else []
            raise N3SchemaError(
                message,
                validation_errors=validation_errors,
                request_id=request_id,
            )
        
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise N3RateLimitError(
                message,
                retry_after=int(retry_after) if retry_after else None,
                request_id=request_id,
            )
        
        if 400 <= response.status_code < 500:
            raise N3ClientError(
                message,
                status_code=response.status_code,
                request_id=request_id,
            )
        
        # Server errors (5xx)
        if 500 <= response.status_code < 600:
            raise N3ServerError(
                message,
                status_code=response.status_code,
                request_id=request_id,
            )
        
        # Unknown status
        raise N3ClientError(
            f"Unexpected status {response.status_code}: {message}",
            status_code=response.status_code,
            request_id=request_id,
        )
    
    def _backoff(self, attempt: int) -> None:
        """Sleep for exponential backoff."""
        time.sleep(self._backoff_delay(attempt))
    
    def _backoff_delay(self, attempt: int) -> float:
        """Calculate backoff delay."""
        delay = self.config.retry_backoff_factor * (2 ** attempt)
        return min(delay, self.config.retry_backoff_max)
