"""Local engine provider implementation for N3Provider.

Production-grade integration with local LLM engines (vLLM, Ollama, etc.).
"""

import os
from typing import Any, AsyncIterable, Dict, List, Optional

from .base import N3Provider, ProviderMessage, ProviderResponse, ProviderError
from .factory import register_provider_class


class LocalProvider(N3Provider):
    """
    Local LLM engine provider implementation.
    
    Supports vLLM, Ollama, and other locally hosted LLM engines with
    OpenAI-compatible APIs.
    
    Configuration (via NAMEL3SS_PROVIDER_LOCAL_* or DSL config):
        - engine_url: Base URL for the local engine (required)
        - engine_type: Engine type ('ollama', 'vllm', or auto-detect)
        - timeout: Request timeout in seconds (default: 120)
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
        
        # Resolve engine URL
        self.engine_url = self.config.get('engine_url') or os.environ.get('LOCAL_ENGINE_URL')
        if not self.engine_url:
            raise ProviderError(
                f"Local engine URL not found for provider '{name}'. "
                f"Set NAMEL3SS_PROVIDER_LOCAL_ENGINE_URL or LOCAL_ENGINE_URL, "
                f"or provide 'engine_url' in config."
            )
        
        # Engine configuration
        self.engine_type = self.config.get('engine_type', 'auto')
        self.timeout = float(self.config.get('timeout', 120))  # Longer timeout for local engines
        
        # Generation parameters
        self.temperature = float(self.config.get('temperature', 0.7))
        self.max_tokens = int(self.config.get('max_tokens', 1024))
        
        # HTTP client (initialized lazily)
        self._http_client = None
        
        # Auto-detect engine type if needed
        if self.engine_type == 'auto':
            self._detect_engine_type()
    
    def _detect_engine_type(self):
        """Auto-detect engine type from URL."""
        url_lower = self.engine_url.lower()
        if 'ollama' in url_lower or ':11434' in url_lower:
            self.engine_type = 'ollama'
        else:
            self.engine_type = 'vllm'  # Default to vLLM/OpenAI-compatible
    
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
                    "httpx is required for Local provider. Install it with: pip install httpx"
                )
        return self._http_client
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        return {'Content-Type': 'application/json'}
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL for the local engine."""
        base_url = self.engine_url.rstrip('/')
        
        if self.engine_type == 'ollama':
            # Ollama uses /api/chat endpoint
            return f"{base_url}/api/chat"
        else:
            # vLLM and OpenAI-compatible engines use /v1/chat/completions
            return f"{base_url}/v1/chat/completions"
    
    def _build_payload_ollama(
        self,
        messages: List[ProviderMessage],
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build request payload for Ollama."""
        return {
            'model': self.model,
            'messages': [msg.to_dict() for msg in messages],
            'stream': stream,
            'options': {
                'temperature': kwargs.get('temperature', self.temperature),
                'num_predict': kwargs.get('max_tokens', self.max_tokens),
            },
        }
    
    def _build_payload_vllm(
        self,
        messages: List[ProviderMessage],
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build request payload for vLLM/OpenAI-compatible engines."""
        return {
            'model': self.model,
            'messages': [msg.to_dict() for msg in messages],
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'stream': stream,
        }
    
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
        url = self._build_url('chat')
        
        # Build payload based on engine type
        if self.engine_type == 'ollama':
            payload = self._build_payload_ollama(messages, stream=False, **kwargs)
        else:
            payload = self._build_payload_vllm(messages, stream=False, **kwargs)
        
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Parse response based on engine type
            if self.engine_type == 'ollama':
                # Ollama format
                message = data.get('message', {})
                content = message.get('content', '')
                
                usage = None
                if 'prompt_eval_count' in data or 'eval_count' in data:
                    usage = {
                        'prompt_tokens': data.get('prompt_eval_count', 0),
                        'completion_tokens': data.get('eval_count', 0),
                        'total_tokens': data.get('prompt_eval_count', 0) + data.get('eval_count', 0),
                    }
                
                return ProviderResponse(
                    model=data.get('model', self.model),
                    output_text=content,
                    raw=data,
                    usage=usage,
                    finish_reason=data.get('done_reason'),
                )
            else:
                # OpenAI-compatible format
                if 'choices' not in data or not data['choices']:
                    raise ProviderError(
                        f"Local engine returned no choices for provider '{self.name}'"
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
                error_msg = f"Local engine API error (status {status_code}): {e}"
            else:
                error_msg = f"Local engine API error: {e}"
            
            raise ProviderError(error_msg) from e
    
    def supports_streaming(self) -> bool:
        """Local engines support streaming."""
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
        url = self._build_url('chat')
        
        # Build payload
        if self.engine_type == 'ollama':
            payload = self._build_payload_ollama(messages, stream=True, **kwargs)
        else:
            payload = self._build_payload_vllm(messages, stream=True, **kwargs)
        
        try:
            async with client.stream('POST', url, json=payload, headers=headers) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line or line.strip() == '':
                        continue
                    
                    # Parse based on engine type
                    if self.engine_type == 'ollama':
                        # Ollama sends JSON per line
                        try:
                            import json
                            chunk_data = json.loads(line)
                            
                            if chunk_data.get('done'):
                                break
                            
                            message = chunk_data.get('message', {})
                            content = message.get('content', '')
                            if content:
                                yield content
                        
                        except json.JSONDecodeError:
                            continue
                    else:
                        # OpenAI-compatible streaming
                        if line.startswith('data: '):
                            line = line[6:]
                        
                        if line.strip() == '[DONE]':
                            break
                        
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
            raise ProviderError(f"Local engine streaming error: {e}") from e
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


# Register this provider with multiple aliases
register_provider_class('local', LocalProvider)
register_provider_class('ollama', LocalProvider)
register_provider_class('vllm', LocalProvider)


__all__ = ['LocalProvider']
