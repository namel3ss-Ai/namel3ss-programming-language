"""Cohere LLM provider implementation with full async + streaming support."""

import asyncio
import json
import os
from typing import AsyncIterator, Dict, Optional

import httpx

from namel3ss.ml.connectors.base import RetryConfig
from namel3ss.observability.logging import get_logger
from namel3ss.observability.metrics import record_metric

from .base import (
    LLMProvider,
    LLMResponse,
    LLMError,
    StreamChunk,
    StreamConfig,
    ProviderStreamingNotSupportedError,
)


logger = get_logger(__name__)


class CohereProvider(LLMProvider):
    """
    Cohere LLM provider with production-grade async + streaming support.
    
    Features:
    - Real async HTTP requests via httpx.AsyncClient
    - True SSE token streaming with backpressure control
    - Timeout management (stream timeout, chunk timeout)
    - Cancellation propagation and cleanup
    - Concurrency control via semaphore
    - Retry logic with jitter backoff
    
    Supports models: command, command-light, command-nightly, command-r, command-r-plus
    """
    
    DEFAULT_BASE_URL = "https://api.cohere.ai/v1"
    
    def __init__(self, *, model: str, api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, max_concurrent: int = 10, **kwargs):
        """
        Initialize Cohere provider.
        
        Args:
            model: Model name (e.g., "command", "command-r-plus")
            api_key: API key (defaults to COHERE_API_KEY env var)
            base_url: Base URL for API
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional parameters passed to LLMProvider
        """
        super().__init__(model=model, **kwargs)
        
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise LLMError(
                "Cohere API key not provided. Set COHERE_API_KEY environment variable "
                "or pass api_key parameter.",
                provider="cohere"
            )
        
        self.base_url = base_url or os.environ.get("COHERE_BASE_URL", self.DEFAULT_BASE_URL)
        self.retry_config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=60.0,
            jitter=0.2,
            timeout=60.0
        )
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, read=120.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            )
        return self._client
    
    async def _close_client(self):
        """Close async HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers for Cohere API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def _build_request_body(self, prompt: str, *, system: Optional[str] = None,
                           stream: bool = False, **kwargs) -> Dict:
        """Build request body for Cohere chat API."""
        # Cohere uses preamble for system prompt
        body = {
            "model": self.model,
            "message": prompt,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": stream,
        }
        
        if system:
            body["preamble"] = system
        
        if self.top_p is not None:
            body["p"] = self.top_p  # Cohere uses 'p' instead of 'top_p'
        
        # Cohere specific parameters
        if "k" in kwargs:  # top_k
            body["k"] = kwargs["k"]
        
        if "stop_sequences" in kwargs:
            body["stop_sequences"] = kwargs["stop_sequences"]
        
        if self.frequency_penalty is not None:
            body["frequency_penalty"] = self.frequency_penalty
        
        if self.presence_penalty is not None:
            body["presence_penalty"] = self.presence_penalty
        
        return body
    
    def _parse_response(self, response_data: Dict) -> LLMResponse:
        """Parse Cohere API response into LLMResponse."""
        try:
            content = response_data.get("text", "")
            
            # Cohere usage metadata
            meta = response_data.get("meta", {})
            billed_units = meta.get("billed_units", {})
            
            usage_dict = {
                "prompt_tokens": billed_units.get("input_tokens", 0),
                "completion_tokens": billed_units.get("output_tokens", 0),
                "total_tokens": billed_units.get("input_tokens", 0) + billed_units.get("output_tokens", 0),
            }
            
            return LLMResponse(
                content=content,
                model=response_data.get("generation_id", self.model),
                usage=usage_dict,
                finish_reason=response_data.get("finish_reason"),
                metadata={
                    "generation_id": response_data.get("generation_id"),
                    "response_id": response_data.get("response_id"),
                }
            )
        except (KeyError, ValueError) as e:
            raise LLMError(
                f"Failed to parse Cohere response: {e}",
                provider="cohere",
                original_error=e
            )
    
    def generate(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        Synchronous generate (wrapper around async).
        
        For production use, prefer agenerate() for proper async execution.
        """
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, should use agenerate instead
            raise LLMError(
                "Cannot call synchronous generate() from async context. Use agenerate() instead.",
                provider="cohere"
            )
        except RuntimeError:
            # No running loop, we can create one
            return asyncio.run(self.agenerate(prompt, system=system, **kwargs))
    
    async def agenerate(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        Generate completion using Cohere API (async).
        
        Implements:
        - Real async HTTP via httpx.AsyncClient
        - Retry with jitter backoff
        - Timeout control
        - Concurrency limiting via semaphore
        """
        url = f"{self.base_url}/chat"
        headers = self._build_headers()
        body = self._build_request_body(prompt, system=system, stream=False, **kwargs)
        
        logger.info(f"Cohere agenerate: model={self.model}, prompt_len={len(prompt)}")
        
        async with self._semaphore:
            client = await self._get_client()
            
            # Retry logic with exponential backoff
            attempt = 0
            last_error = None
            
            while attempt < self.retry_config.max_attempts:
                try:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=body,
                        timeout=self.retry_config.timeout
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        llm_response = self._parse_response(response_data)
                        
                        # Record metrics
                        record_metric("llm.generation.success", 1, 
                                    tags={"provider": "cohere", "model": self.model})
                        record_metric("llm.tokens.total", llm_response.usage.get("total_tokens", 0),
                                    tags={"provider": "cohere", "model": self.model})
                        
                        logger.info(f"Cohere agenerate complete: tokens={llm_response.usage.get('total_tokens')}")
                        return llm_response
                    
                    # Handle retryable status codes
                    if response.status_code in {429, 500, 502, 503, 504}:
                        attempt += 1
                        if attempt < self.retry_config.max_attempts:
                            delay = self.retry_config.compute_delay(attempt)
                            logger.warning(
                                f"Cohere API error {response.status_code}, retrying in {delay}s "
                                f"(attempt {attempt}/{self.retry_config.max_attempts})"
                            )
                            await asyncio.sleep(delay)
                            continue
                    
                    # Non-retryable error
                    error_msg = f"Cohere API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise LLMError(error_msg, provider="cohere", status_code=response.status_code)
                
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last_error = e
                    attempt += 1
                    if attempt < self.retry_config.max_attempts:
                        delay = self.retry_config.compute_delay(attempt)
                        logger.warning(
                            f"Cohere network error: {e}, retrying in {delay}s "
                            f"(attempt {attempt}/{self.retry_config.max_attempts})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    break
                
                except asyncio.CancelledError:
                    logger.info("Cohere agenerate cancelled")
                    raise
                
                except Exception as e:
                    record_metric("llm.generation.error", 1, 
                                tags={"provider": "cohere", "model": self.model})
                    if isinstance(e, LLMError):
                        raise
                    raise LLMError(
                        f"Cohere request failed: {e}",
                        provider="cohere",
                        original_error=e
                    )
            
            # All retries exhausted
            record_metric("llm.generation.error", 1, 
                        tags={"provider": "cohere", "model": self.model})
            raise LLMError(
                f"Cohere request failed after {self.retry_config.max_attempts} attempts: {last_error}",
                provider="cohere",
                original_error=last_error
            )
    
    async def stream_generate(self, prompt: str, *, system: Optional[str] = None,
                             stream_config: Optional[StreamConfig] = None,
                             **kwargs) -> AsyncIterator[StreamChunk]:
        """
        Stream completion using Cohere API with SSE.
        
        Implements:
        - True SSE token streaming
        - Chunk timeout (max idle time between chunks)
        - Stream timeout (max total time)
        - Backpressure control (max chunks)
        - Clean cancellation and connection closure
        """
        url = f"{self.base_url}/chat"
        headers = self._build_headers()
        body = self._build_request_body(prompt, system=system, stream=True, **kwargs)
        
        config = stream_config or StreamConfig()
        
        logger.info(f"Cohere stream_generate: model={self.model}, prompt_len={len(prompt)}")
        
        async with self._semaphore:
            client = await self._get_client()
            
            chunk_count = 0
            start_time = asyncio.get_event_loop().time()
            
            try:
                async with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=body,
                    timeout=httpx.Timeout(
                        connect=10.0,
                        read=config.chunk_timeout,
                        write=10.0,
                        pool=None
                    )
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        error_msg = f"Cohere streaming error: {response.status_code} - {error_text.decode()}"
                        logger.error(error_msg)
                        raise LLMError(error_msg, provider="cohere", status_code=response.status_code)
                    
                    # Parse SSE stream
                    async for line in response.aiter_lines():
                        # Check stream timeout
                        if config.stream_timeout:
                            elapsed = asyncio.get_event_loop().time() - start_time
                            if elapsed > config.stream_timeout:
                                logger.warning(f"Cohere stream timeout after {elapsed}s")
                                raise asyncio.TimeoutError(f"Stream exceeded timeout of {config.stream_timeout}s")
                        
                        # Check max chunks
                        if config.max_chunks and chunk_count >= config.max_chunks:
                            logger.info(f"Cohere stream reached max chunks: {config.max_chunks}")
                            return
                        
                        # Cohere uses SSE format with event types
                        if not line.strip():
                            continue
                        
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            
                            try:
                                event_data = json.loads(data)
                                event_type = event_data.get("event_type")
                                
                                # Handle text generation events
                                if event_type == "text-generation":
                                    content = event_data.get("text", "")
                                    
                                    if content:
                                        chunk_count += 1
                                        yield StreamChunk(
                                            content=content,
                                            finish_reason=None,
                                            model=self.model,
                                            metadata={"event_type": event_type}
                                        )
                                
                                elif event_type == "stream-end":
                                    # Final event with finish reason
                                    finish_reason = event_data.get("finish_reason")
                                    response_data = event_data.get("response", {})
                                    
                                    # Get usage from final event
                                    meta = response_data.get("meta", {})
                                    billed_units = meta.get("billed_units", {})
                                    usage = {
                                        "prompt_tokens": billed_units.get("input_tokens", 0),
                                        "completion_tokens": billed_units.get("output_tokens", 0),
                                        "total_tokens": billed_units.get("input_tokens", 0) + billed_units.get("output_tokens", 0),
                                    }
                                    
                                    logger.info(f"Cohere stream finished: reason={finish_reason}, chunks={chunk_count}")
                                    yield StreamChunk(
                                        content="",
                                        finish_reason=finish_reason,
                                        model=self.model,
                                        usage=usage,
                                        metadata={"event_type": event_type}
                                    )
                                    return
                                
                                elif event_type == "error":
                                    error_msg = event_data.get("message", "Unknown error")
                                    raise LLMError(
                                        f"Cohere streaming error: {error_msg}",
                                        provider="cohere"
                                    )
                            
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse Cohere SSE chunk: {e}")
                                continue
                
                # Record success
                record_metric("llm.streaming.success", 1,
                            tags={"provider": "cohere", "model": self.model})
                record_metric("llm.streaming.chunks", chunk_count,
                            tags={"provider": "cohere", "model": self.model})
            
            except asyncio.CancelledError:
                logger.info(f"Cohere stream cancelled after {chunk_count} chunks")
                record_metric("llm.streaming.cancelled", 1,
                            tags={"provider": "cohere", "model": self.model})
                raise
            
            except asyncio.TimeoutError as e:
                record_metric("llm.streaming.timeout", 1,
                            tags={"provider": "cohere", "model": self.model})
                raise LLMError(
                    f"Cohere stream timeout: {e}",
                    provider="cohere",
                    original_error=e
                )
            
            except Exception as e:
                record_metric("llm.streaming.error", 1,
                            tags={"provider": "cohere", "model": self.model})
                if isinstance(e, LLMError):
                    raise
                raise LLMError(
                    f"Cohere streaming failed: {e}",
                    provider="cohere",
                    original_error=e
                )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self._close_client()
