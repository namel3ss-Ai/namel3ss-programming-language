"""Google Gemini LLM provider implementation with full async + streaming support."""

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


class GeminiProvider(LLMProvider):
    """
    Google Gemini LLM provider with production-grade async + streaming support.
    
    Features:
    - Real async HTTP requests via httpx.AsyncClient
    - True SSE token streaming with backpressure control
    - Timeout management (stream timeout, chunk timeout)
    - Cancellation propagation and cleanup
    - Concurrency control via semaphore
    - Retry logic with jitter backoff
    
    Supports models: gemini-pro, gemini-pro-vision, gemini-ultra
    """
    
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    def __init__(self, *, model: str, api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, max_concurrent: int = 10, **kwargs):
        """
        Initialize Gemini provider.
        
        Args:
            model: Model name (e.g., "gemini-pro", "gemini-1.5-pro")
            api_key: API key (defaults to GOOGLE_API_KEY or GEMINI_API_KEY env var)
            base_url: Base URL for API
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional parameters passed to LLMProvider
        """
        super().__init__(model=model, **kwargs)
        
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise LLMError(
                "Gemini API key not provided. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable "
                "or pass api_key parameter.",
                provider="gemini"
            )
        
        self.base_url = base_url or os.environ.get("GEMINI_BASE_URL", self.DEFAULT_BASE_URL)
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
    
    def _build_request_body(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> Dict:
        """Build request body for Gemini API."""
        contents = []
        
        # Gemini uses "parts" structure
        if system:
            # System instructions can be in the first message
            contents.append({
                "role": "user",
                "parts": [{"text": system}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood."}]
            })
        
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", self.temperature),
                "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens),
            }
        }
        
        if self.top_p is not None:
            body["generationConfig"]["topP"] = self.top_p
        
        # Gemini uses topK instead of frequency/presence penalty
        if "top_k" in kwargs:
            body["generationConfig"]["topK"] = kwargs["top_k"]
        
        if "stop_sequences" in kwargs:
            body["generationConfig"]["stopSequences"] = kwargs["stop_sequences"]
        
        return body
    
    def _parse_response(self, response_data: Dict) -> LLMResponse:
        """Parse Gemini API response into LLMResponse."""
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates in response")
            
            candidate = candidates[0]
            content_parts = candidate.get("content", {}).get("parts", [])
            
            # Extract text from parts
            content = "".join(
                part.get("text", "") 
                for part in content_parts
            )
            
            # Gemini usage metadata
            usage_metadata = response_data.get("usageMetadata", {})
            usage_dict = {
                "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                "total_tokens": usage_metadata.get("totalTokenCount", 0),
            }
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage_dict,
                finish_reason=candidate.get("finishReason"),
                metadata={
                    "safety_ratings": candidate.get("safetyRatings", []),
                }
            )
        except (KeyError, ValueError) as e:
            raise LLMError(
                f"Failed to parse Gemini response: {e}",
                provider="gemini",
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
                provider="gemini"
            )
        except RuntimeError:
            # No running loop, we can create one
            return asyncio.run(self.agenerate(prompt, system=system, **kwargs))
    
    async def agenerate(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        Generate completion using Gemini API (async).
        
        Implements:
        - Real async HTTP via httpx.AsyncClient
        - Retry with jitter backoff
        - Timeout control
        - Concurrency limiting via semaphore
        """
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        body = self._build_request_body(prompt, system=system, **kwargs)
        
        logger.info(f"Gemini agenerate: model={self.model}, prompt_len={len(prompt)}")
        
        async with self._semaphore:
            client = await self._get_client()
            
            # Retry logic with exponential backoff
            attempt = 0
            last_error = None
            
            while attempt < self.retry_config.max_attempts:
                try:
                    response = await client.post(
                        url,
                        json=body,
                        timeout=self.retry_config.timeout
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        llm_response = self._parse_response(response_data)
                        
                        # Record metrics
                        record_metric("llm.generation.success", 1, 
                                    tags={"provider": "gemini", "model": self.model})
                        record_metric("llm.tokens.total", llm_response.usage.get("total_tokens", 0),
                                    tags={"provider": "gemini", "model": self.model})
                        
                        logger.info(f"Gemini agenerate complete: tokens={llm_response.usage.get('total_tokens')}")
                        return llm_response
                    
                    # Handle retryable status codes
                    if response.status_code in {429, 500, 502, 503, 504}:
                        attempt += 1
                        if attempt < self.retry_config.max_attempts:
                            delay = self.retry_config.compute_delay(attempt)
                            logger.warning(
                                f"Gemini API error {response.status_code}, retrying in {delay}s "
                                f"(attempt {attempt}/{self.retry_config.max_attempts})"
                            )
                            await asyncio.sleep(delay)
                            continue
                    
                    # Non-retryable error
                    error_msg = f"Gemini API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise LLMError(error_msg, provider="gemini", status_code=response.status_code)
                
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last_error = e
                    attempt += 1
                    if attempt < self.retry_config.max_attempts:
                        delay = self.retry_config.compute_delay(attempt)
                        logger.warning(
                            f"Gemini network error: {e}, retrying in {delay}s "
                            f"(attempt {attempt}/{self.retry_config.max_attempts})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    break
                
                except asyncio.CancelledError:
                    logger.info("Gemini agenerate cancelled")
                    raise
                
                except Exception as e:
                    record_metric("llm.generation.error", 1, 
                                tags={"provider": "gemini", "model": self.model})
                    if isinstance(e, LLMError):
                        raise
                    raise LLMError(
                        f"Gemini request failed: {e}",
                        provider="gemini",
                        original_error=e
                    )
            
            # All retries exhausted
            record_metric("llm.generation.error", 1, 
                        tags={"provider": "gemini", "model": self.model})
            raise LLMError(
                f"Gemini request failed after {self.retry_config.max_attempts} attempts: {last_error}",
                provider="gemini",
                original_error=last_error
            )
    
    async def stream_generate(self, prompt: str, *, system: Optional[str] = None,
                             stream_config: Optional[StreamConfig] = None,
                             **kwargs) -> AsyncIterator[StreamChunk]:
        """
        Stream completion using Gemini API with SSE.
        
        Implements:
        - True SSE token streaming
        - Chunk timeout (max idle time between chunks)
        - Stream timeout (max total time)
        - Backpressure control (max chunks)
        - Clean cancellation and connection closure
        """
        url = f"{self.base_url}/models/{self.model}:streamGenerateContent?key={self.api_key}&alt=sse"
        body = self._build_request_body(prompt, system=system, **kwargs)
        
        config = stream_config or StreamConfig()
        
        logger.info(f"Gemini stream_generate: model={self.model}, prompt_len={len(prompt)}")
        
        async with self._semaphore:
            client = await self._get_client()
            
            chunk_count = 0
            start_time = asyncio.get_event_loop().time()
            
            try:
                async with client.stream(
                    "POST",
                    url,
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
                        error_msg = f"Gemini streaming error: {response.status_code} - {error_text.decode()}"
                        logger.error(error_msg)
                        raise LLMError(error_msg, provider="gemini", status_code=response.status_code)
                    
                    # Parse SSE stream
                    async for line in response.aiter_lines():
                        # Check stream timeout
                        if config.stream_timeout:
                            elapsed = asyncio.get_event_loop().time() - start_time
                            if elapsed > config.stream_timeout:
                                logger.warning(f"Gemini stream timeout after {elapsed}s")
                                raise asyncio.TimeoutError(f"Stream exceeded timeout of {config.stream_timeout}s")
                        
                        # Check max chunks
                        if config.max_chunks and chunk_count >= config.max_chunks:
                            logger.info(f"Gemini stream reached max chunks: {config.max_chunks}")
                            return
                        
                        # Parse SSE format: "data: {...}"
                        if not line.strip():
                            continue
                        
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            
                            try:
                                chunk_data = json.loads(data)
                                candidates = chunk_data.get("candidates", [])
                                
                                if not candidates:
                                    continue
                                
                                candidate = candidates[0]
                                content_parts = candidate.get("content", {}).get("parts", [])
                                
                                # Extract text from parts
                                content = "".join(
                                    part.get("text", "")
                                    for part in content_parts
                                )
                                
                                if content:
                                    chunk_count += 1
                                    yield StreamChunk(
                                        content=content,
                                        finish_reason=candidate.get("finishReason"),
                                        model=self.model,
                                        metadata={
                                            "safety_ratings": candidate.get("safetyRatings", []),
                                        }
                                    )
                                
                                # Check for finish
                                if candidate.get("finishReason"):
                                    logger.info(f"Gemini stream finished: reason={candidate['finishReason']}, chunks={chunk_count}")
                                    return
                            
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse Gemini SSE chunk: {e}")
                                continue
                
                # Record success
                record_metric("llm.streaming.success", 1,
                            tags={"provider": "gemini", "model": self.model})
                record_metric("llm.streaming.chunks", chunk_count,
                            tags={"provider": "gemini", "model": self.model})
            
            except asyncio.CancelledError:
                logger.info(f"Gemini stream cancelled after {chunk_count} chunks")
                record_metric("llm.streaming.cancelled", 1,
                            tags={"provider": "gemini", "model": self.model})
                raise
            
            except asyncio.TimeoutError as e:
                record_metric("llm.streaming.timeout", 1,
                            tags={"provider": "gemini", "model": self.model})
                raise LLMError(
                    f"Gemini stream timeout: {e}",
                    provider="gemini",
                    original_error=e
                )
            
            except Exception as e:
                record_metric("llm.streaming.error", 1,
                            tags={"provider": "gemini", "model": self.model})
                if isinstance(e, LLMError):
                    raise
                raise LLMError(
                    f"Gemini streaming failed: {e}",
                    provider="gemini",
                    original_error=e
                )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self._close_client()
