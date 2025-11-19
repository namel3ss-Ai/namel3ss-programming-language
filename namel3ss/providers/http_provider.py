"""Generic HTTP provider implementation for N3Provider.

Production-grade integration with custom HTTP LLM endpoints.
"""

import os
from typing import Any, AsyncIterable, Dict, List, Optional

from .base import N3Provider, ProviderMessage, ProviderResponse, ProviderError
from .factory import register_provider_class


class HttpProvider(N3Provider):
    """
    Generic HTTP LLM provider implementation.
    
    Supports custom HTTP endpoints with configurable request/response formats.
    Ideal for enterprise-internal LLMs or custom API endpoints.
    
    Configuration (via NAMEL3SS_PROVIDER_HTTP_* or DSL config):
        - base_url: Base URL for the HTTP endpoint (required)
        - api_key: Optional API key for authentication
        - request_path: API path (default: /v1/chat/completions)
        - auth_header: Header name for API key (default: Authorization)
        - auth_prefix: Prefix for auth header value (default: Bearer )
        - timeout: Request timeout in seconds (default: 60)
        - temperature: Sampling temperature (default: 0.7)
        - max_tokens: Maximum tokens to generate (default: 1024)
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, model, config)
        
        # Resolve base URL
        self.base_url = self.config.get('base_url') or os.environ.get('HTTP_LLM_BASE_URL')
        if not self.base_url:
            raise ProviderError(
                f"HTTP base URL not found for provider '{name}'. "
                f"Set NAMEL3SS_PROVIDER_HTTP_BASE_URL or HTTP_LLM_BASE_URL, "
                f"or provide 'base_url' in config."
            )
        
        # API configuration
        self.api_key = self.config.get('api_key') or os.environ.get('HTTP_LLM_API_KEY')
        self.request_path = self.config.get('request_path', '/v1/chat/completions')
        self.auth_header = self.config.get('auth_header', 'Authorization')
        self.auth_prefix = self.config.get('auth_prefix', 'Bearer ')
        self.timeout = float(self.config.get('timeout', 60))
        
        # Generation parameters
        self.temperature = float(self.config.get('temperature', 0.7))
        self.max_tokens = int(self.config.get('max_tokens', 1024))
        
        # Request/response format customization
        self.request_format = self.config.get('request_format', 'openai')  # or 'custom'
        self.response_format = self.config.get('response_format', 'openai')  # or 'custom'
        
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
                    "httpx is required for HTTP provider. Install it with: pip install httpx"
                )
        return self._http_client
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {'Content-Type': 'application/json'}
        
        if self.api_key:
            headers[self.auth_header] = f"{self.auth_prefix}{self.api_key}"
        
        return headers
    
    def _build_url(self) -> str:
        """Build full URL."""
        base_url = self.base_url.rstrip('/')
        path = self.request_path.lstrip('/')
        return f"{base_url}/{path}"
    
    def _build_payload(
        self,
        messages: List[ProviderMessage],
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build request payload."""
        if self.request_format == 'openai':
            # OpenAI-compatible format
            return {
                'model': self.model,
                'messages': [msg.to_dict() for msg in messages],
                'temperature': kwargs.get('temperature', self.temperature),
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'stream': stream,
            }
        else:
            # Custom format - user must provide template in config
            # This is a placeholder for custom request templates
            raise ProviderError(
                f"Custom request format not yet implemented. Use 'openai' format."
            )
    
    def _parse_response(self, data: Dict[str, Any]) -> tuple:
        """
        Parse response data.
        
        Returns:
            Tuple of (output_text, usage, finish_reason)
        """
        if self.response_format == 'openai':
            # OpenAI-compatible format
            if 'choices' not in data or not data['choices']:
                raise ProviderError(
                    f"HTTP API returned no choices for provider '{self.name}'"
                )
            
            choice = data['choices'][0]
            message = choice.get('message', {})
            content = message.get('content', '')
            usage = data.get('usage')
            finish_reason = choice.get('finish_reason')
            
            return content, usage, finish_reason
        else:
            # Custom format - user must provide parsing template in config
            raise ProviderError(
                f"Custom response format not yet implemented. Use 'openai' format."
            )
    
    async def generate(
        self,
        messages: List[ProviderMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Generate a chat completion.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional parameters
        
        Returns:
            ProviderResponse with generated text and metadata
        
        Raises:
            ProviderError: If the API call fails
        """
        client = self._get_http_client()
        headers = self._build_headers()
        url = self._build_url()
        payload = self._build_payload(messages, stream=False, **kwargs)
        
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            output_text, usage, finish_reason = self._parse_response(data)
            
            return ProviderResponse(
                model=data.get('model', self.model),
                output_text=output_text,
                raw=data,
                usage=usage,
                finish_reason=finish_reason,
            )
        
        except Exception as e:
            if hasattr(e, 'response'):
                status_code = getattr(e.response, 'status_code', None)
                error_msg = f"HTTP API error (status {status_code}): {e}"
            else:
                error_msg = f"HTTP API error: {e}"
            
            raise ProviderError(error_msg) from e
    
    def supports_streaming(self) -> bool:
        """HTTP provider supports streaming if endpoint supports it."""
        return self.config.get('supports_streaming', True)
    
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
        if not self.supports_streaming():
            raise NotImplementedError(
                f"Streaming not supported by HTTP provider '{self.name}'"
            )
        
        client = self._get_http_client()
        headers = self._build_headers()
        url = self._build_url()
        payload = self._build_payload(messages, stream=True, **kwargs)
        
        try:
            async with client.stream('POST', url, json=payload, headers=headers) as response:
                response.raise_for_status()
                
                # Assume SSE format (OpenAI-compatible)
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
                        continue
        
        except Exception as e:
            raise ProviderError(f"HTTP streaming error: {e}") from e
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


# Register this provider
register_provider_class('http', HttpProvider)


__all__ = ['HttpProvider']
