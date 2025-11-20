"""Ollama LLM provider implementation with full async + streaming support."""

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


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider with production-grade async + streaming support.
    
    Features:
    - Real async HTTP requests via httpx.AsyncClient
    - True streaming with backpressure control
    - Timeout management (stream timeout, chunk timeout)
    - Cancellation propagation and cleanup
    - Concurrency control via semaphore
    - Retry logic with jitter backoff
    - Support for local LLM models (Llama, Mistral, Mixtral, etc.)
    
    Supports models: llama3, llama2, mistral, mixtral, codellama, phi, gemma, etc.
    """
    
    DEFAULT_BASE_URL = "http://localhost:11434"
    
    def __init__(self, *, model: str, base_url: Optional[str] = None, 
                 max_concurrent: int = 10, **kwargs):
        """
        Initialize Ollama provider.
        
        Args:
            model: Model name (e.g., "llama3", "mistral", "codellama")
            base_url: Base URL for Ollama API (defaults to http://localhost:11434)
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional parameters passed to LLMProvider
        """
        super().__init__(model=model, **kwargs)
        
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)
        self.retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.5,  # Faster retry for local models
            max_delay=10.0,
            jitter=0.2,
            timeout=120.0  # Longer timeout for local inference
        )
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, read=300.0),  # Longer timeout for local inference
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            )
        return self._client
    
    async def _close_client(self):
        """Close async HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    def _build_request_body(self, prompt: str, *, system: Optional[str] = None,
                           stream: bool = False, **kwargs) -> Dict:
        """Build request body for Ollama API."""
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }
        
        if system:
            body["system"] = system
        
        # Ollama-specific options
        if self.top_p is not None:
            body["options"]["top_p"] = self.top_p
        
        if "top_k" in kwargs:
            body["options"]["top_k"] = kwargs["top_k"]
        
        if "repeat_penalty" in kwargs:
            body["options"]["repeat_penalty"] = kwargs["repeat_penalty"]
        
        if "stop" in kwargs:
            body["options"]["stop"] = kwargs["stop"]
        
        # Context window size
        if "num_ctx" in kwargs:
            body["options"]["num_ctx"] = kwargs["num_ctx"]
        
        return body
    
    def _parse_response(self, response_data: Dict) -> LLMResponse:
        """Parse Ollama API response into LLMResponse."""
        try:
            content = response_data.get("response", "")
            
            # Ollama usage metadata
            usage_dict = {
                "prompt_tokens": response_data.get("prompt_eval_count", 0),
                "completion_tokens": response_data.get("eval_count", 0),
                "total_tokens": response_data.get("prompt_eval_count", 0) + response_data.get("eval_count", 0),
            }
            
            return LLMResponse(
                content=content,
                model=response_data.get("model", self.model),
                usage=usage_dict,
                finish_reason="stop" if response_data.get("done") else None,
                metadata={
                    "context": response_data.get("context", []),
                    "total_duration": response_data.get("total_duration"),
                    "load_duration": response_data.get("load_duration"),
                    "prompt_eval_duration": response_data.get("prompt_eval_duration"),
                    "eval_duration": response_data.get("eval_duration"),
                }
            )
        except (KeyError, ValueError) as e:
            raise LLMError(
                f"Failed to parse Ollama response: {e}",
                provider="ollama",
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
                provider="ollama"
            )
        except RuntimeError:
            # No running loop, we can create one
            return asyncio.run(self.agenerate(prompt, system=system, **kwargs))
    
    async def agenerate(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        Generate completion using Ollama API (async).
        
        Implements:
        - Real async HTTP via httpx.AsyncClient
        - Retry with jitter backoff
        - Timeout control
        - Concurrency limiting via semaphore
        """
        url = f"{self.base_url}/api/generate"
        body = self._build_request_body(prompt, system=system, stream=False, **kwargs)
        
        logger.info(f"Ollama agenerate: model={self.model}, prompt_len={len(prompt)}")
        
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
                                    tags={"provider": "ollama", "model": self.model})
                        record_metric("llm.tokens.total", llm_response.usage.get("total_tokens", 0),
                                    tags={"provider": "ollama", "model": self.model})
                        
                        logger.info(f"Ollama agenerate complete: tokens={llm_response.usage.get('total_tokens')}")
                        return llm_response
                    
                    # Handle retryable status codes
                    if response.status_code in {429, 500, 502, 503, 504}:
                        attempt += 1
                        if attempt < self.retry_config.max_attempts:
                            delay = self.retry_config.compute_delay(attempt)
                            logger.warning(
                                f"Ollama API error {response.status_code}, retrying in {delay}s "
                                f"(attempt {attempt}/{self.retry_config.max_attempts})"
                            )
                            await asyncio.sleep(delay)
                            continue
                    
                    # Non-retryable error
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise LLMError(error_msg, provider="ollama", status_code=response.status_code)
                
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last_error = e
                    attempt += 1
                    if attempt < self.retry_config.max_attempts:
                        delay = self.retry_config.compute_delay(attempt)
                        logger.warning(
                            f"Ollama network error: {e}, retrying in {delay}s "
                            f"(attempt {attempt}/{self.retry_config.max_attempts})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    break
                
                except asyncio.CancelledError:
                    logger.info("Ollama agenerate cancelled")
                    raise
                
                except Exception as e:
                    record_metric("llm.generation.error", 1, 
                                tags={"provider": "ollama", "model": self.model})
                    if isinstance(e, LLMError):
                        raise
                    raise LLMError(
                        f"Ollama request failed: {e}",
                        provider="ollama",
                        original_error=e
                    )
            
            # All retries exhausted
            record_metric("llm.generation.error", 1, 
                        tags={"provider": "ollama", "model": self.model})
            raise LLMError(
                f"Ollama request failed after {self.retry_config.max_attempts} attempts: {last_error}",
                provider="ollama",
                original_error=last_error
            )
    
    async def stream_generate(self, prompt: str, *, system: Optional[str] = None,
                             stream_config: Optional[StreamConfig] = None,
                             **kwargs) -> AsyncIterator[StreamChunk]:
        """
        Stream completion using Ollama API.
        
        Implements:
        - True streaming with newline-delimited JSON
        - Chunk timeout (max idle time between chunks)
        - Stream timeout (max total time)
        - Backpressure control (max chunks)
        - Clean cancellation and connection closure
        """
        url = f"{self.base_url}/api/generate"
        body = self._build_request_body(prompt, system=system, stream=True, **kwargs)
        
        config = stream_config or StreamConfig()
        
        logger.info(f"Ollama stream_generate: model={self.model}, prompt_len={len(prompt)}")
        
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
                        error_msg = f"Ollama streaming error: {response.status_code} - {error_text.decode()}"
                        logger.error(error_msg)
                        raise LLMError(error_msg, provider="ollama", status_code=response.status_code)
                    
                    # Parse newline-delimited JSON stream
                    async for line in response.aiter_lines():
                        # Check stream timeout
                        if config.stream_timeout:
                            elapsed = asyncio.get_event_loop().time() - start_time
                            if elapsed > config.stream_timeout:
                                logger.warning(f"Ollama stream timeout after {elapsed}s")
                                raise asyncio.TimeoutError(f"Stream exceeded timeout of {config.stream_timeout}s")
                        
                        # Check max chunks
                        if config.max_chunks and chunk_count >= config.max_chunks:
                            logger.info(f"Ollama stream reached max chunks: {config.max_chunks}")
                            return
                        
                        # Parse JSON line
                        if not line.strip():
                            continue
                        
                        try:
                            chunk_data = json.loads(line)
                            content = chunk_data.get("response", "")
                            done = chunk_data.get("done", False)
                            
                            if content:
                                chunk_count += 1
                                yield StreamChunk(
                                    content=content,
                                    finish_reason="stop" if done else None,
                                    model=chunk_data.get("model", self.model),
                                    metadata={
                                        "context": chunk_data.get("context", []),
                                    }
                                )
                            
                            # Check for completion
                            if done:
                                # Final chunk with usage stats
                                usage = {
                                    "prompt_tokens": chunk_data.get("prompt_eval_count", 0),
                                    "completion_tokens": chunk_data.get("eval_count", 0),
                                    "total_tokens": chunk_data.get("prompt_eval_count", 0) + chunk_data.get("eval_count", 0),
                                }
                                
                                logger.info(f"Ollama stream finished: chunks={chunk_count}, tokens={usage['total_tokens']}")
                                
                                if not content:  # If no content in final chunk, yield usage info
                                    yield StreamChunk(
                                        content="",
                                        finish_reason="stop",
                                        model=chunk_data.get("model", self.model),
                                        usage=usage,
                                        metadata={
                                            "total_duration": chunk_data.get("total_duration"),
                                            "load_duration": chunk_data.get("load_duration"),
                                            "prompt_eval_duration": chunk_data.get("prompt_eval_duration"),
                                            "eval_duration": chunk_data.get("eval_duration"),
                                        }
                                    )
                                return
                        
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse Ollama JSON chunk: {e}")
                            continue
                
                # Record success
                record_metric("llm.streaming.success", 1,
                            tags={"provider": "ollama", "model": self.model})
                record_metric("llm.streaming.chunks", chunk_count,
                            tags={"provider": "ollama", "model": self.model})
            
            except asyncio.CancelledError:
                logger.info(f"Ollama stream cancelled after {chunk_count} chunks")
                record_metric("llm.streaming.cancelled", 1,
                            tags={"provider": "ollama", "model": self.model})
                raise
            
            except asyncio.TimeoutError as e:
                record_metric("llm.streaming.timeout", 1,
                            tags={"provider": "ollama", "model": self.model})
                raise LLMError(
                    f"Ollama stream timeout: {e}",
                    provider="ollama",
                    original_error=e
                )
            
            except Exception as e:
                record_metric("llm.streaming.error", 1,
                            tags={"provider": "ollama", "model": self.model})
                if isinstance(e, LLMError):
                    raise
                raise LLMError(
                    f"Ollama streaming failed: {e}",
                    provider="ollama",
                    original_error=e
                )
    
    async def list_models(self) -> list[Dict]:
        """List available models from Ollama."""
        url = f"{self.base_url}/api/tags"
        
        client = await self._get_client()
        try:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            else:
                raise LLMError(
                    f"Failed to list Ollama models: {response.status_code} - {response.text}",
                    provider="ollama"
                )
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Failed to list Ollama models: {e}", provider="ollama", original_error=e)
    
    async def pull_model(self, model: Optional[str] = None) -> AsyncIterator[Dict]:
        """
        Pull (download) a model from Ollama registry.
        
        Args:
            model: Model name to pull (defaults to self.model)
        
        Yields:
            Progress updates as dictionaries
        """
        url = f"{self.base_url}/api/pull"
        model_name = model or self.model
        
        client = await self._get_client()
        
        try:
            async with client.stream(
                "POST",
                url,
                json={"name": model_name},
                timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=None)
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise LLMError(
                        f"Failed to pull model {model_name}: {response.status_code} - {error_text.decode()}",
                        provider="ollama"
                    )
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Failed to pull model {model_name}: {e}", provider="ollama", original_error=e)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self._close_client()
