"""Anthropic provider implementation for N3Provider.

Production-grade Anthropic API integration with async support and streaming.
"""

import os
from typing import Any, AsyncIterable, Dict, List, Optional

from .base import N3Provider, ProviderMessage, ProviderResponse, ProviderError
from .factory import register_provider_class


class AnthropicProvider(N3Provider):
    """
    Anthropic provider implementation.
    
    Supports Claude 3 (Opus, Sonnet, Haiku) and Claude 2.x models with
    async generation and streaming.
    
    Configuration (via NAMEL3SS_PROVIDER_ANTHROPIC_* or DSL config):
        - api_key: Anthropic API key (required)
        - base_url: API base URL (default: https://api.anthropic.com)
        - version: API version (default: 2023-06-01)
        - timeout: Request timeout in seconds (default: 60)
        - temperature: Sampling temperature (default: 0.7)
        - max_tokens: Maximum tokens to generate (default: 1024)
        - top_p: Nucleus sampling parameter
        - top_k: Top-k sampling parameter
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, model, config)
        
        # Resolve API key
        self.api_key = self.config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ProviderError(
                f"Anthropic API key not found for provider '{name}'. "
                f"Set NAMEL3SS_PROVIDER_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY environment variable, "
                f"or provide 'api_key' in config."
            )
        
        # API configuration
        self.base_url = self.config.get('base_url', 'https://api.anthropic.com')
        self.version = self.config.get('version', '2023-06-01')
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
                    "httpx is required for Anthropic provider. Install it with: pip install httpx"
                )
        return self._http_client
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        return {
            'x-api-key': self.api_key,
            'anthropic-version': self.version,
            'Content-Type': 'application/json',
        }
    
    def _convert_messages(self, messages: List[ProviderMessage]) -> tuple:
        """
        Convert messages to Anthropic format.
        
        Anthropic requires system messages to be separate from the messages array.
        
        Returns:
            Tuple of (system_prompt, messages_array)
        """
        system_prompt = None
        converted_messages = []
        
        for msg in messages:
            if msg.role == 'system':
                # Collect system messages
                if system_prompt is None:
                    system_prompt = msg.content
                else:
                    system_prompt += '\n\n' + msg.content
            else:
                # Regular messages
                converted_messages.append({
                    'role': msg.role,
                    'content': msg.content,
                })
        
        return system_prompt, converted_messages
    
    def _build_request_payload(
        self,
        messages: List[ProviderMessage],
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build request payload for Anthropic API."""
        system_prompt, converted_messages = self._convert_messages(messages)
        
        payload = {
            'model': self.model,
            'messages': converted_messages,
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'stream': stream,
        }
        
        # Add system prompt if present
        if system_prompt:
            payload['system'] = system_prompt
        
        # Optional parameters
        for param in ['top_p', 'top_k', 'stop_sequences']:
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
        url = f"{self.base_url}/v1/messages"
        
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Extract response
            content_blocks = data.get('content', [])
            if not content_blocks:
                raise ProviderError(
                    f"Anthropic API returned no content for provider '{self.name}'"
                )
            
            # Concatenate all text blocks
            output_text = ''
            for block in content_blocks:
                if block.get('type') == 'text':
                    output_text += block.get('text', '')
            
            # Build usage info
            usage = None
            if 'usage' in data:
                usage = {
                    'prompt_tokens': data['usage'].get('input_tokens', 0),
                    'completion_tokens': data['usage'].get('output_tokens', 0),
                    'total_tokens': (
                        data['usage'].get('input_tokens', 0) + 
                        data['usage'].get('output_tokens', 0)
                    ),
                }
            
            return ProviderResponse(
                model=data.get('model', self.model),
                output_text=output_text,
                raw=data,
                usage=usage,
                finish_reason=data.get('stop_reason'),
            )
        
        except Exception as e:
            if hasattr(e, 'response'):
                status_code = getattr(e.response, 'status_code', None)
                error_msg = f"Anthropic API error (status {status_code}): {e}"
            else:
                error_msg = f"Anthropic API error: {e}"
            
            raise ProviderError(error_msg) from e
    
    def supports_streaming(self) -> bool:
        """Anthropic supports streaming."""
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
        url = f"{self.base_url}/v1/messages"
        
        try:
            async with client.stream('POST', url, json=payload, headers=headers) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line or line.strip() == '':
                        continue
                    
                    # Parse SSE format
                    if line.startswith('data: '):
                        line = line[6:]
                    elif line.startswith('event: '):
                        # Skip event lines
                        continue
                    
                    # Parse JSON chunk
                    try:
                        import json
                        chunk_data = json.loads(line)
                        
                        # Handle different event types
                        event_type = chunk_data.get('type')
                        
                        if event_type == 'content_block_delta':
                            delta = chunk_data.get('delta', {})
                            if delta.get('type') == 'text_delta':
                                text = delta.get('text', '')
                                if text:
                                    yield text
                        
                        elif event_type == 'message_stop':
                            # Stream completed
                            break
                    
                    except json.JSONDecodeError:
                        # Skip malformed chunks
                        continue
        
        except Exception as e:
            raise ProviderError(f"Anthropic streaming error: {e}") from e
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


# Register this provider
register_provider_class('anthropic', AnthropicProvider)


__all__ = ['AnthropicProvider']
