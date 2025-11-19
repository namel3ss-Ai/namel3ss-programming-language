"""Ollama LLM provider implementation for local models."""

import os
from typing import Dict, Any, Optional, List, Iterable
from .base import BaseLLM, LLMResponse, LLMError, ChatMessage
from .factory import register_provider


class OllamaLLM(BaseLLM):
    """
    Ollama LLM provider for local model deployment.
    
    Supports locally hosted models through Ollama.
    
    Configuration:
        - base_url: Ollama API base URL (defaults to OLLAMA_BASE_URL env var or http://localhost:11434)
        - timeout: Request timeout in seconds (default: 120 for local models)
        - temperature: Sampling temperature (default: 0.7)
        - max_tokens: Maximum tokens to generate (default: 1024)
    
    Example:
        >>> llm = OllamaLLM('my_llama', 'llama2', {'temperature': 0.5})
        >>> response = llm.generate('What is the capital of France?')
        >>> print(response.text)
    """
    
    def __init__(self, name: str, model: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, model, config)
        
        # API configuration
        self.base_url = (
            self.config.get('base_url')
            or os.environ.get('OLLAMA_BASE_URL')
            or 'http://localhost:11434'
        )
        
        # Longer default timeout for local models
        if 'timeout' not in self.config:
            self.timeout = 120.0
        
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
                    "httpx is required for Ollama provider. Install it with: pip install httpx",
                    provider='ollama',
                    model=self.model
                )
        return self._http_client
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        return {
            'Content-Type': 'application/json',
        }
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate completion for a prompt.
        
        Args:
            prompt: The prompt text
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            LLMResponse with generated text
        
        Raises:
            LLMError: If the API call fails
        """
        client = self._get_http_client()
        
        # Build request payload (Ollama format)
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': kwargs.get('temperature', self.temperature),
                'num_predict': kwargs.get('max_tokens', self.max_tokens),
            }
        }
        
        # Add optional parameters
        if 'top_p' in kwargs:
            payload['options']['top_p'] = kwargs['top_p']
        if 'top_k' in kwargs:
            payload['options']['top_k'] = kwargs['top_k']
        if 'stop' in kwargs:
            payload['options']['stop'] = kwargs['stop']
        
        # Make API request
        try:
            response = client.post(
                f"{self.base_url}/api/generate",
                headers=self._build_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise LLMError(
                f"Ollama API request failed: {e}",
                provider='ollama',
                model=self.model,
                status_code=getattr(response, 'status_code', None) if 'response' in locals() else None,
                original_error=e
            )
        
        # Extract response
        text = data.get('response', '')
        
        # Build usage info if available
        usage = None
        if 'prompt_eval_count' in data or 'eval_count' in data:
            usage = {
                'prompt_tokens': data.get('prompt_eval_count', 0),
                'completion_tokens': data.get('eval_count', 0),
                'total_tokens': data.get('prompt_eval_count', 0) + data.get('eval_count', 0),
            }
        
        return LLMResponse(
            text=text,
            raw=data,
            model=data.get('model', self.model),
            usage=usage,
            finish_reason='stop' if data.get('done') else None,
            metadata={'provider': 'ollama'}
        )
    
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
        
        # Convert messages to Ollama format
        message_dicts = [msg.to_dict() for msg in messages]
        
        # Build request payload
        payload = {
            'model': self.model,
            'messages': message_dicts,
            'stream': False,
            'options': {
                'temperature': kwargs.get('temperature', self.temperature),
                'num_predict': kwargs.get('max_tokens', self.max_tokens),
            }
        }
        
        # Add optional parameters
        if 'top_p' in kwargs:
            payload['options']['top_p'] = kwargs['top_p']
        if 'top_k' in kwargs:
            payload['options']['top_k'] = kwargs['top_k']
        if 'stop' in kwargs:
            payload['options']['stop'] = kwargs['stop']
        
        # Make API request
        try:
            response = client.post(
                f"{self.base_url}/api/chat",
                headers=self._build_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise LLMError(
                f"Ollama chat API request failed: {e}",
                provider='ollama',
                model=self.model,
                status_code=getattr(response, 'status_code', None) if 'response' in locals() else None,
                original_error=e
            )
        
        # Extract response
        message = data.get('message', {})
        text = message.get('content', '')
        
        # Build usage info if available
        usage = None
        if 'prompt_eval_count' in data or 'eval_count' in data:
            usage = {
                'prompt_tokens': data.get('prompt_eval_count', 0),
                'completion_tokens': data.get('eval_count', 0),
                'total_tokens': data.get('prompt_eval_count', 0) + data.get('eval_count', 0),
            }
        
        return LLMResponse(
            text=text,
            raw=data,
            model=data.get('model', self.model),
            usage=usage,
            finish_reason='stop' if data.get('done') else None,
            metadata={
                'provider': 'ollama',
                'message': message,
            }
        )
    
    def supports_streaming(self) -> bool:
        """Ollama supports streaming."""
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
        client = self._get_http_client()
        
        # Build request payload
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': True,
            'options': {
                'temperature': kwargs.get('temperature', self.temperature),
                'num_predict': kwargs.get('max_tokens', self.max_tokens),
            }
        }
        
        # Add optional parameters
        if 'top_p' in kwargs:
            payload['options']['top_p'] = kwargs['top_p']
        if 'top_k' in kwargs:
            payload['options']['top_k'] = kwargs['top_k']
        if 'stop' in kwargs:
            payload['options']['stop'] = kwargs['stop']
        
        # Make streaming request
        try:
            with client.stream(
                'POST',
                f"{self.base_url}/api/generate",
                headers=self._build_headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                
                import json
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        text = data.get('response', '')
                        if text:
                            yield text
                        if data.get('done'):
                            break
        except Exception as e:
            raise LLMError(
                f"Ollama streaming request failed: {e}",
                provider='ollama',
                model=self.model,
                original_error=e
            )
    
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
        
        # Convert messages to Ollama format
        message_dicts = [msg.to_dict() for msg in messages]
        
        # Build request payload
        payload = {
            'model': self.model,
            'messages': message_dicts,
            'stream': True,
            'options': {
                'temperature': kwargs.get('temperature', self.temperature),
                'num_predict': kwargs.get('max_tokens', self.max_tokens),
            }
        }
        
        # Add optional parameters
        if 'top_p' in kwargs:
            payload['options']['top_p'] = kwargs['top_p']
        if 'top_k' in kwargs:
            payload['options']['top_k'] = kwargs['top_k']
        if 'stop' in kwargs:
            payload['options']['stop'] = kwargs['stop']
        
        # Make streaming request
        try:
            with client.stream(
                'POST',
                f"{self.base_url}/api/chat",
                headers=self._build_headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                
                import json
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        message = data.get('message', {})
                        content = message.get('content', '')
                        if content:
                            yield content
                        if data.get('done'):
                            break
        except Exception as e:
            raise LLMError(
                f"Ollama chat streaming request failed: {e}",
                provider='ollama',
                model=self.model,
                original_error=e
            )
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return 'ollama'


# Register the provider
register_provider('ollama', OllamaLLM)
