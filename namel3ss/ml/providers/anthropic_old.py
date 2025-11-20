"""Anthropic LLM provider implementation."""

import os
from typing import AsyncIterator, Dict, Optional

from namel3ss.ml.connectors.base import make_resilient_request, RetryConfig
from namel3ss.observability.logging import get_logger
from namel3ss.observability.metrics import record_metric

from .base import LLMProvider, LLMResponse, LLMError


logger = get_logger(__name__)


class AnthropicProvider(LLMProvider):
    """
    Anthropic LLM provider.
    
    Requires ANTHROPIC_API_KEY environment variable.
    Supports models: claude-3-5-sonnet, claude-3-opus, claude-3-sonnet, etc.
    """
    
    DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
    DEFAULT_VERSION = "2023-06-01"
    
    def __init__(self, *, model: str, api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, api_version: Optional[str] = None, **kwargs):
        """
        Initialize Anthropic provider.
        
        Args:
            model: Model name (e.g., "claude-3-5-sonnet-20241022")
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: Base URL for API
            api_version: API version header
            **kwargs: Additional parameters passed to LLMProvider
        """
        super().__init__(model=model, **kwargs)
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter.",
                provider="anthropic"
            )
        
        self.base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL", self.DEFAULT_BASE_URL)
        self.api_version = api_version or self.DEFAULT_VERSION
        self.retry_config = RetryConfig(
            max_retries=3,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            retryable_status_codes={429, 500, 502, 503, 504}
        )
    
    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers for Anthropic API."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "Content-Type": "application/json",
        }
    
    def _build_request_body(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> Dict:
        """Build request body for Anthropic messages API."""
        messages = [{"role": "user", "content": prompt}]
        
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        if system:
            body["system"] = system
        
        if self.top_p is not None:
            body["top_p"] = self.top_p
        
        # Anthropic uses top_k instead of frequency/presence penalty
        if "top_k" in kwargs:
            body["top_k"] = kwargs["top_k"]
        
        # Add stop sequences if provided
        if "stop_sequences" in kwargs:
            body["stop_sequences"] = kwargs["stop_sequences"]
        
        return body
    
    def _parse_response(self, response_data: Dict) -> LLMResponse:
        """Parse Anthropic API response into LLMResponse."""
        try:
            content_blocks = response_data.get("content", [])
            if not content_blocks:
                raise ValueError("No content in response")
            
            # Concatenate all text content blocks
            content = " ".join(
                block["text"] for block in content_blocks 
                if block.get("type") == "text"
            )
            
            finish_reason = response_data.get("stop_reason")
            
            usage = response_data.get("usage", {})
            usage_dict = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            }
            
            return LLMResponse(
                content=content,
                model=response_data.get("model", self.model),
                usage=usage_dict,
                finish_reason=finish_reason,
                metadata={"response_id": response_data.get("id")}
            )
        except (KeyError, ValueError) as e:
            raise LLMError(
                f"Failed to parse Anthropic response: {e}",
                provider="anthropic",
                original_error=e
            )
    
    def generate(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate completion using Anthropic API."""
        url = f"{self.base_url}/messages"
        headers = self._build_headers()
        body = self._build_request_body(prompt, system=system, **kwargs)
        
        logger.info(f"Anthropic generate: model={self.model}, prompt_len={len(prompt)}")
        
        try:
            response = make_resilient_request(
                url=url,
                method="POST",
                headers=headers,
                json_data=body,
                retry_config=self.retry_config,
                timeout=kwargs.get("timeout", 60.0)
            )
            
            if response.status_code != 200:
                error_msg = f"Anthropic API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise LLMError(error_msg, provider="anthropic", status_code=response.status_code)
            
            response_data = response.json()
            llm_response = self._parse_response(response_data)
            
            # Record metrics
            record_metric("llm.generation.success", 1, tags={"provider": "anthropic", "model": self.model})
            record_metric("llm.tokens.total", llm_response.usage.get("total_tokens", 0),
                         tags={"provider": "anthropic", "model": self.model})
            
            logger.info(f"Anthropic generate complete: tokens={llm_response.usage.get('total_tokens')}")
            return llm_response
            
        except Exception as e:
            record_metric("llm.generation.error", 1, tags={"provider": "anthropic", "model": self.model})
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Anthropic request failed: {e}", provider="anthropic", original_error=e)
    
    async def agenerate(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> LLMResponse:
        """Async generate - falls back to sync for now."""
        # TODO: Implement true async with aiohttp
        return self.generate(prompt, system=system, **kwargs)
    
    async def stream_generate(self, prompt: str, *, system: Optional[str] = None,
                             **kwargs) -> AsyncIterator[str]:
        """Stream completion (not yet implemented)."""
        # TODO: Implement streaming with server-sent events
        raise NotImplementedError("Streaming not yet implemented for Anthropic provider")
    
    def chat(self, messages, **kwargs) -> LLMResponse:
        """Generate completion for conversation."""
        url = f"{self.base_url}/messages"
        headers = self._build_headers()
        
        # Extract system message if present
        system = None
        user_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content")
            else:
                user_messages.append(msg)
        
        body = {
            "model": self.model,
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        if system:
            body["system"] = system
        
        if self.top_p is not None:
            body["top_p"] = self.top_p
        
        try:
            response = make_resilient_request(
                url=url,
                method="POST",
                headers=headers,
                json_data=body,
                retry_config=self.retry_config,
                timeout=kwargs.get("timeout", 60.0)
            )
            
            if response.status_code != 200:
                raise LLMError(
                    f"Anthropic API error: {response.status_code} - {response.text}",
                    provider="anthropic",
                    status_code=response.status_code
                )
            
            return self._parse_response(response.json())
            
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Anthropic chat request failed: {e}", provider="anthropic", original_error=e)
