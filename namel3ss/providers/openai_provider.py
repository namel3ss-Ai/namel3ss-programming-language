"""OpenAI provider implementation for N3Provider.

Production-grade OpenAI API integration with async support, streaming, and batching.
"""

import os
from typing import Any, AsyncIterable, Dict, List, Optional

from .base import N3Provider, ProviderMessage, ProviderResponse, ProviderError
from .factory import register_provider_class


class OpenAIProvider(N3Provider):
    """
    OpenAI provider implementation.
    
    Supports GPT-3.5, GPT-4, and other OpenAI models with async generation,
    streaming, and batch capabilities.
    
    Configuration (via NAMEL3SS_PROVIDER_OPENAI_* or DSL config):
        - api_key: OpenAI API key (required)
        - base_url: API base URL (default: https://api.openai.com/v1)
        - organization: Optional organization ID
        - timeout: Request timeout in seconds (default: 60)
        - temperature: Sampling temperature (default: 0.7)
        - max_tokens: Maximum tokens to generate (default: 1024)
        - top_p: Nucleus sampling parameter
        - frequency_penalty: Frequency penalty
        - presence_penalty: Presence penalty
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, model, config)
        
        # Resolve API key
        self.api_key = self.config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ProviderError(
                f"OpenAI API key not found for provider '{name}'. "
                f"Set NAMEL3SS_PROVIDER_OPENAI_API_KEY or OPENAI_API_KEY environment variable, "
                f"or provide 'api_key' in config."
            )
        
        # API configuration
        self.base_url = self.config.get('base_url', 'https://api.openai.com/v1')
        self.organization = self.config.get('organization')
        self.timeout = float(self.config.get('timeout', 60))
        
        # Generation parameters
        self.temperature = float(self.config.get('temperature', 0.7))
        self.max_tokens = int(self.config.get('max_tokens', 1024))
        
        # HTTP client (initialized lazily)
        self._http_client = None
    
    def _get_http_client(self):
        """Get or create async HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout),
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                )
            except ImportError:
                raise ProviderError(
                    "httpx is required for OpenAI provider. Install it with: pip install httpx"
                )
        return self._http_client
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        if self.organization:
            headers['OpenAI-Organization'] = self.organization
        return headers
    
    def _build_request_payload(
        self,
        messages: List[ProviderMessage],
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build request payload for OpenAI API."""
        payload = {
            'model': self.model,
            'messages': [msg.to_dict() for msg in messages],
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'stream': stream,
        }
        
        # Optional parameters
        for param in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop', 'n', 'logit_bias']:
            if param in kwargs:
                payload[param] = kwargs[param]
        
        return payload
    
    async def generate(
        self,
        messages: List[ProviderMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Generate a chat completion.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            ProviderResponse with generated text and metadata
        
        Raises:
            ProviderError: If the API call fails
        """
        client = self._get_http_client()
        headers = self._build_headers()
        payload = self._build_request_payload(messages, stream=False, **kwargs)
        url = f"{self.base_url}/chat/completions"
        
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Extract response
            if 'choices' not in data or not data['choices']:
                raise ProviderError(
                    f"OpenAI API returned no choices for provider '{self.name}'"
                )
            
            choice = data['choices'][0]
            message = choice.get('message', {})
            content = message.get('content', '')
            
            return ProviderResponse(
                model=data.get('model', self.model),
                output_text=content,
                raw=data,
                usage=data.get('usage'),
                finish_reason=choice.get('finish_reason'),
            )
        
        except Exception as e:
            if hasattr(e, 'response'):
                status_code = getattr(e.response, 'status_code', None)
                error_msg = f"OpenAI API error (status {status_code}): {e}"
            else:
                error_msg = f"OpenAI API error: {e}"
            
            raise ProviderError(error_msg) from e
    
    async def generate_batch(
        self,
        batch: List[List[ProviderMessage]],
        **kwargs: Any,
    ) -> List[ProviderResponse]:
        """
        Generate completions for multiple message sequences.
        
        Uses concurrent async requests for better throughput.
        
        Args:
            batch: List of message sequences
            **kwargs: Additional parameters
        
        Returns:
            List of ProviderResponse objects
        """
        import asyncio
        
        # Execute all requests concurrently
        tasks = [self.generate(messages, **kwargs) for messages in batch]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    def supports_streaming(self) -> bool:
        """OpenAI supports streaming."""
        return True
    
    async def stream(
        self,
        messages: List[ProviderMessage],
        **kwargs: Any,
    ) -> AsyncIterable[str]:
        """
        Stream generation tokens as they arrive.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional parameters
        
        Yields:
            Incremental text chunks
        
        Raises:
            ProviderError: If streaming fails
        """
        client = self._get_http_client()
        headers = self._build_headers()
        payload = self._build_request_payload(messages, stream=True, **kwargs)
        url = f"{self.base_url}/chat/completions"
        
        try:
            async with client.stream('POST', url, json=payload, headers=headers) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line or line.strip() == '':
                        continue
                    
                    # Remove "data: " prefix
                    if line.startswith('data: '):
                        line = line[6:]
                    
                    # Check for stream termination
                    if line.strip() == '[DONE]':
                        break
                    
                    # Parse JSON chunk
                    try:
                        import json
                        chunk_data = json.loads(line)
                        
                        if 'choices' in chunk_data and chunk_data['choices']:
                            delta = chunk_data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                    
                    except json.JSONDecodeError:
                        # Skip malformed chunks
                        continue
        
        except Exception as e:
            raise ProviderError(f"OpenAI streaming error: {e}") from e
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


# Register this provider
register_provider_class('openai', OpenAIProvider)


__all__ = ['OpenAIProvider']
