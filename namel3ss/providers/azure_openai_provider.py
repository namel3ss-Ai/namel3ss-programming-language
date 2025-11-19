"""Azure OpenAI provider implementation for N3Provider.

Production-grade Azure OpenAI integration with async support and streaming.
"""

import os
from typing import Any, AsyncIterable, Dict, List, Optional

from .base import N3Provider, ProviderMessage, ProviderResponse, ProviderError
from .factory import register_provider_class


class AzureOpenAIProvider(N3Provider):
    """
    Azure OpenAI provider implementation.
    
    Supports OpenAI models deployed on Azure.
    
    Configuration (via NAMEL3SS_PROVIDER_AZURE_OPENAI_* or DSL config):
        - api_key: Azure OpenAI API key (required)
        - endpoint: Azure OpenAI endpoint URL (required)
        - deployment: Deployment name (optional, can use model name)
        - api_version: API version (default: 2023-05-15)
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
        
        # Resolve API key
        self.api_key = self.config.get('api_key') or os.environ.get('AZURE_OPENAI_API_KEY')
        if not self.api_key:
            raise ProviderError(
                f"Azure OpenAI API key not found for provider '{name}'. "
                f"Set NAMEL3SS_PROVIDER_AZURE_OPENAI_API_KEY or AZURE_OPENAI_API_KEY, "
                f"or provide 'api_key' in config."
            )
        
        # Resolve endpoint
        self.endpoint = self.config.get('endpoint') or os.environ.get('AZURE_OPENAI_ENDPOINT')
        if not self.endpoint:
            raise ProviderError(
                f"Azure OpenAI endpoint not found for provider '{name}'. "
                f"Set NAMEL3SS_PROVIDER_AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_ENDPOINT, "
                f"or provide 'endpoint' in config."
            )
        
        # Deployment name (defaults to model if not specified)
        self.deployment = self.config.get('deployment', model)
        self.api_version = self.config.get('api_version', '2023-05-15')
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
                    "httpx is required for Azure OpenAI provider. Install it with: pip install httpx"
                )
        return self._http_client
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        return {
            'api-key': self.api_key,
            'Content-Type': 'application/json',
        }
    
    def _build_url(self, stream: bool = False) -> str:
        """Build Azure OpenAI API URL."""
        # Ensure endpoint doesn't end with /
        endpoint = self.endpoint.rstrip('/')
        
        # Build URL: https://{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}
        url = (
            f"{endpoint}/openai/deployments/{self.deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )
        return url
    
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
        
        # Build payload (Azure OpenAI uses same format as OpenAI)
        payload = {
            'messages': [msg.to_dict() for msg in messages],
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'stream': False,
        }
        
        # Optional parameters
        for param in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop']:
            if param in kwargs:
                payload[param] = kwargs[param]
        
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Extract response (same format as OpenAI)
            if 'choices' not in data or not data['choices']:
                raise ProviderError(
                    f"Azure OpenAI API returned no choices for provider '{self.name}'"
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
                error_msg = f"Azure OpenAI API error (status {status_code}): {e}"
            else:
                error_msg = f"Azure OpenAI API error: {e}"
            
            raise ProviderError(error_msg) from e
    
    def supports_streaming(self) -> bool:
        """Azure OpenAI supports streaming."""
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
        url = self._build_url(stream=True)
        
        # Build payload
        payload = {
            'messages': [msg.to_dict() for msg in messages],
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'stream': True,
        }
        
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
                        continue
        
        except Exception as e:
            raise ProviderError(f"Azure OpenAI streaming error: {e}") from e
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


# Register this provider
register_provider_class('azure_openai', AzureOpenAIProvider)


__all__ = ['AzureOpenAIProvider']
