"""Anthropic LLM provider implementation."""

import os
from typing import Dict, Any, Optional, List, Iterable
from .base import BaseLLM, LLMResponse, LLMError, ChatMessage
from .factory import register_provider


class AnthropicLLM(BaseLLM):
    """
    Anthropic LLM provider.
    
    Supports Claude 3 (Opus, Sonnet, Haiku) and Claude 2.x models.
    
    Configuration:
        - api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        - api_base: Base URL for API (defaults to https://api.anthropic.com)
        - timeout: Request timeout in seconds (default: 60)
        - temperature: Sampling temperature (default: 0.7)
        - max_tokens: Maximum tokens to generate (default: 1024)
    
    Example:
        >>> llm = AnthropicLLM('my_claude', 'claude-3-opus-20240229', {'temperature': 0.5})
        >>> response = llm.generate('What is the capital of France?')
        >>> print(response.text)
    """
    
    def __init__(self, name: str, model: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, model, config)
        
        # Resolve API key
        self.api_key = self.config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise LLMError(
                f"Anthropic API key not found for LLM '{name}'. "
                f"Set ANTHROPIC_API_KEY environment variable or provide 'api_key' in config.",
                provider='anthropic',
                model=model
            )
        
        # API configuration
        self.api_base = self.config.get('api_base', 'https://api.anthropic.com')
        self.anthropic_version = self.config.get('anthropic_version', '2023-06-01')
        
        # Initialize HTTP client lazily
        self._http_client = None
    
    def _get_http_client(self):
        """Get or create HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.Client(timeout=self.timeout)
            except ImportError:
                raise LLMError(
                    "httpx is required for Anthropic provider. Install it with: pip install httpx",
                    provider='anthropic',
                    model=self.model
                )
        return self._http_client
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        return {
            'x-api-key': self.api_key,
            'anthropic-version': self.anthropic_version,
            'Content-Type': 'application/json',
        }
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate completion for a prompt.
        
        Anthropic uses the Messages API for all requests, so this converts
        the prompt to a single user message.
        
        Args:
            prompt: The prompt text
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            LLMResponse with generated text
        
        Raises:
            LLMError: If the API call fails
        """
        # Convert prompt to chat format
        messages = [ChatMessage(role='user', content=prompt)]
        return self.generate_chat(messages, **kwargs)
    
    def generate_chat(self, messages: List[ChatMessage], **kwargs) -> LLMResponse:
        """
        Generate chat completion.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional parameters
        
        Returns:
            LLMResponse with generated message
        
        Raises:
            LLMError: If the API call fails
        """
        client = self._get_http_client()
        
        # Convert messages to Anthropic format
        # Anthropic requires alternating user/assistant messages
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                'role': msg.role if msg.role != 'system' else 'user',
                'content': msg.content,
            })
        
        # Build request payload
        payload = {
            'model': self.model,
            'messages': formatted_messages,
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
        }
        
        # Add optional parameters
        if 'top_p' in kwargs:
            payload['top_p'] = kwargs['top_p']
        if 'top_k' in kwargs:
            payload['top_k'] = kwargs['top_k']
        if 'stop_sequences' in kwargs:
            payload['stop_sequences'] = kwargs['stop_sequences']
        if 'system' in kwargs:
            payload['system'] = kwargs['system']
        
        # Make API request
        try:
            response = client.post(
                f"{self.api_base}/v1/messages",
                headers=self._build_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise LLMError(
                f"Anthropic API request failed: {e}",
                provider='anthropic',
                model=self.model,
                status_code=getattr(response, 'status_code', None) if 'response' in locals() else None,
                original_error=e
            )
        
        # Extract response
        if 'content' not in data or not data['content']:
            raise LLMError(
                f"Anthropic API returned no content: {data}",
                provider='anthropic',
                model=self.model
            )
        
        # Anthropic returns content as a list of blocks
        content_blocks = data['content']
        text = ' '.join(block.get('text', '') for block in content_blocks if block.get('type') == 'text')
        
        return LLMResponse(
            text=text,
            raw=data,
            model=data.get('model', self.model),
            usage=data.get('usage'),
            finish_reason=data.get('stop_reason'),
            metadata={
                'provider': 'anthropic',
                'content_blocks': content_blocks,
            }
        )
    
    def supports_streaming(self) -> bool:
        """Anthropic supports streaming."""
        return True
    
    def stream(self, prompt: str, **kwargs) -> Iterable[str]:
        """
        Stream completion for a prompt.
        
        Args:
            prompt: The prompt text
            **kwargs: Additional parameters
        
        Yields:
            Text chunks as they are generated
        
        Raises:
            LLMError: If the API call fails
        """
        # Convert prompt to chat format
        messages = [ChatMessage(role='user', content=prompt)]
        yield from self.stream_chat(messages, **kwargs)
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> Iterable[str]:
        """
        Stream chat completion.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional parameters
        
        Yields:
            Text chunks as they are generated
        
        Raises:
            LLMError: If the API call fails
        """
        client = self._get_http_client()
        
        # Convert messages to Anthropic format
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                'role': msg.role if msg.role != 'system' else 'user',
                'content': msg.content,
            })
        
        # Build request payload
        payload = {
            'model': self.model,
            'messages': formatted_messages,
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'stream': True,
        }
        
        # Add optional parameters
        if 'top_p' in kwargs:
            payload['top_p'] = kwargs['top_p']
        if 'top_k' in kwargs:
            payload['top_k'] = kwargs['top_k']
        if 'stop_sequences' in kwargs:
            payload['stop_sequences'] = kwargs['stop_sequences']
        if 'system' in kwargs:
            payload['system'] = kwargs['system']
        
        # Make streaming request
        try:
            with client.stream(
                'POST',
                f"{self.api_base}/v1/messages",
                headers=self._build_headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        
                        import json
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        
                        # Handle different event types
                        event_type = data.get('type')
                        
                        if event_type == 'content_block_delta':
                            delta = data.get('delta', {})
                            if delta.get('type') == 'text_delta':
                                text = delta.get('text', '')
                                if text:
                                    yield text
                        
                        elif event_type == 'message_stop':
                            break
        except Exception as e:
            raise LLMError(
                f"Anthropic streaming request failed: {e}",
                provider='anthropic',
                model=self.model,
                original_error=e
            )
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return 'anthropic'


# Register the provider
register_provider('anthropic', AnthropicLLM)
