"""OpenAI LLM provider implementation."""

import os
import json
from typing import AsyncIterator, Dict, Optional

from namel3ss.ml.connectors.base import make_resilient_request, RetryConfig
from namel3ss.observability.logging import get_logger
from namel3ss.observability.metrics import record_metric

from .base import LLMProvider, LLMResponse, LLMError


logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider.
    
    Requires OPENAI_API_KEY environment variable.
    Supports models: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.
    """
    
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    
    def __init__(self, *, model: str, api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI provider.
        
        Args:
            model: Model name (e.g., "gpt-4o")
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API (for Azure OpenAI or proxies)
            **kwargs: Additional parameters passed to LLMProvider
        """
        super().__init__(model=model, **kwargs)
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter.",
                provider="openai"
            )
        
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", self.DEFAULT_BASE_URL)
        self.retry_config = RetryConfig(
            max_retries=3,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            retryable_status_codes={429, 500, 502, 503, 504}
        )
    
    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers for OpenAI API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def _build_request_body(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> Dict:
        """Build request body for OpenAI chat completions API."""
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            body["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            body["presence_penalty"] = self.presence_penalty
        
        # Add any additional parameters
        for key in ("stop", "user", "logprobs", "seed"):
            if key in kwargs:
                body[key] = kwargs[key]
        
        return body
    
    def _parse_response(self, response_data: Dict) -> LLMResponse:
        """Parse OpenAI API response into LLMResponse."""
        try:
            choice = response_data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            
            usage = response_data.get("usage", {})
            usage_dict = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            
            return LLMResponse(
                content=content,
                model=response_data.get("model", self.model),
                usage=usage_dict,
                finish_reason=finish_reason,
                metadata={"response_id": response_data.get("id")}
            )
        except (KeyError, IndexError) as e:
            raise LLMError(
                f"Failed to parse OpenAI response: {e}",
                provider="openai",
                original_error=e
            )
    
    def generate(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate completion using OpenAI API."""
        url = f"{self.base_url}/chat/completions"
        headers = self._build_headers()
        body = self._build_request_body(prompt, system=system, **kwargs)
        
        logger.info(f"OpenAI generate: model={self.model}, prompt_len={len(prompt)}")
        
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
                error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise LLMError(error_msg, provider="openai", status_code=response.status_code)
            
            response_data = response.json()
            llm_response = self._parse_response(response_data)
            
            # Record metrics
            record_metric("llm.generation.success", 1, tags={"provider": "openai", "model": self.model})
            record_metric("llm.tokens.total", llm_response.usage.get("total_tokens", 0), 
                         tags={"provider": "openai", "model": self.model})
            
            logger.info(f"OpenAI generate complete: tokens={llm_response.usage.get('total_tokens')}")
            return llm_response
            
        except Exception as e:
            record_metric("llm.generation.error", 1, tags={"provider": "openai", "model": self.model})
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"OpenAI request failed: {e}", provider="openai", original_error=e)
    
    async def agenerate(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> LLMResponse:
        """Async generate - falls back to sync for now."""
        # TODO: Implement true async with aiohttp
        return self.generate(prompt, system=system, **kwargs)
    
    async def stream_generate(self, prompt: str, *, system: Optional[str] = None, 
                             **kwargs) -> AsyncIterator[str]:
        """Stream completion (not yet implemented)."""
        # TODO: Implement streaming with server-sent events
        raise NotImplementedError("Streaming not yet implemented for OpenAI provider")
    
    def chat(self, messages, **kwargs) -> LLMResponse:
        """Generate completion for conversation."""
        url = f"{self.base_url}/chat/completions"
        headers = self._build_headers()
        
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            body["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            body["presence_penalty"] = self.presence_penalty
        
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
                    f"OpenAI API error: {response.status_code} - {response.text}",
                    provider="openai",
                    status_code=response.status_code
                )
            
            return self._parse_response(response.json())
            
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"OpenAI chat request failed: {e}", provider="openai", original_error=e)
