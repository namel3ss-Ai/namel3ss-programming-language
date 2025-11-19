"""Azure OpenAI LLM provider implementation."""

import os
from typing import Dict, Any, Optional, List, Iterable
from .base import BaseLLM, LLMResponse, LLMError, ChatMessage
from .factory import register_provider


class AzureOpenAILLM(BaseLLM):
    """
    Azure OpenAI LLM provider.
    
    Uses Azure's deployment of OpenAI models.
    
    Configuration:
        - api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_API_KEY env var)
        - endpoint: Azure OpenAI endpoint URL (defaults to AZURE_OPENAI_ENDPOINT env var)
        - deployment_name: Azure deployment name (required)
        - api_version: API version (default: '2023-05-15')
        - timeout: Request timeout in seconds (default: 60)
        - temperature: Sampling temperature (default: 0.7)
        - max_tokens: Maximum tokens to generate (default: 1024)
    
    Example:
        >>> llm = AzureOpenAILLM(
        ...     'my_azure_gpt4',
        ...     'gpt-4',
        ...     {
        ...         'deployment_name': 'my-gpt4-deployment',
        ...         'temperature': 0.5
        ...     }
        ... )
        >>> response = llm.generate('What is the capital of France?')
        >>> print(response.text)
    """
    
    def __init__(self, name: str, model: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, model, config)
        
        # Resolve API key
        self.api_key = self.config.get('api_key') or os.environ.get('AZURE_OPENAI_API_KEY')
        if not self.api_key:
            raise LLMError(
                f"Azure OpenAI API key not found for LLM '{name}'. "
                f"Set AZURE_OPENAI_API_KEY environment variable or provide 'api_key' in config.",
                provider='azure_openai',
                model=model
            )
        
        # Resolve endpoint
        self.endpoint = self.config.get('endpoint') or os.environ.get('AZURE_OPENAI_ENDPOINT')
        if not self.endpoint:
            raise LLMError(
                f"Azure OpenAI endpoint not found for LLM '{name}'. "
                f"Set AZURE_OPENAI_ENDPOINT environment variable or provide 'endpoint' in config.",
                provider='azure_openai',
                model=model
            )
        
        # Deployment name is required
        self.deployment_name = self.config.get('deployment_name')
        if not self.deployment_name:
            raise LLMError(
                f"Azure OpenAI deployment_name is required for LLM '{name}'. "
                f"Provide 'deployment_name' in config.",
                provider='azure_openai',
                model=model
            )
        
        # API configuration
        self.api_version = self.config.get('api_version', '2023-05-15')
        
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
                    "httpx is required for Azure OpenAI provider. Install it with: pip install httpx",
                    provider='azure_openai',
                    model=self.model
                )
        return self._http_client
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        return {
            'api-key': self.api_key,
            'Content-Type': 'application/json',
        }
    
    def _build_url(self, endpoint_type: str) -> str:
        """Build API URL."""
        base = self.endpoint.rstrip('/')
        return (
            f"{base}/openai/deployments/{self.deployment_name}/{endpoint_type}"
            f"?api-version={self.api_version}"
        )
    
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
        
        # Build request payload
        payload = {
            'prompt': prompt,
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
        }
        
        # Add optional parameters
        if 'top_p' in kwargs:
            payload['top_p'] = kwargs['top_p']
        if 'frequency_penalty' in kwargs:
            payload['frequency_penalty'] = kwargs['frequency_penalty']
        if 'presence_penalty' in kwargs:
            payload['presence_penalty'] = kwargs['presence_penalty']
        if 'stop' in kwargs:
            payload['stop'] = kwargs['stop']
        
        # Make API request
        try:
            response = client.post(
                self._build_url('completions'),
                headers=self._build_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise LLMError(
                f"Azure OpenAI API request failed: {e}",
                provider='azure_openai',
                model=self.model,
                status_code=getattr(response, 'status_code', None) if 'response' in locals() else None,
                original_error=e
            )
        
        # Extract response
        if 'choices' not in data or not data['choices']:
            raise LLMError(
                f"Azure OpenAI API returned no choices: {data}",
                provider='azure_openai',
                model=self.model
            )
        
        choice = data['choices'][0]
        text = choice.get('text', '')
        
        return LLMResponse(
            text=text,
            raw=data,
            model=data.get('model', self.model),
            usage=data.get('usage'),
            finish_reason=choice.get('finish_reason'),
            metadata={'provider': 'azure_openai', 'deployment': self.deployment_name}
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
        
        # Convert messages to OpenAI format
        message_dicts = [msg.to_dict() for msg in messages]
        
        # Build request payload
        payload = {
            'messages': message_dicts,
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
        }
        
        # Add optional parameters
        if 'top_p' in kwargs:
            payload['top_p'] = kwargs['top_p']
        if 'frequency_penalty' in kwargs:
            payload['frequency_penalty'] = kwargs['frequency_penalty']
        if 'presence_penalty' in kwargs:
            payload['presence_penalty'] = kwargs['presence_penalty']
        if 'stop' in kwargs:
            payload['stop'] = kwargs['stop']
        
        # Make API request
        try:
            response = client.post(
                self._build_url('chat/completions'),
                headers=self._build_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise LLMError(
                f"Azure OpenAI chat API request failed: {e}",
                provider='azure_openai',
                model=self.model,
                status_code=getattr(response, 'status_code', None) if 'response' in locals() else None,
                original_error=e
            )
        
        # Extract response
        if 'choices' not in data or not data['choices']:
            raise LLMError(
                f"Azure OpenAI chat API returned no choices: {data}",
                provider='azure_openai',
                model=self.model
            )
        
        choice = data['choices'][0]
        message = choice.get('message', {})
        text = message.get('content', '')
        
        return LLMResponse(
            text=text,
            raw=data,
            model=data.get('model', self.model),
            usage=data.get('usage'),
            finish_reason=choice.get('finish_reason'),
            metadata={
                'provider': 'azure_openai',
                'deployment': self.deployment_name,
                'message': message,
            }
        )
    
    def supports_streaming(self) -> bool:
        """Azure OpenAI supports streaming."""
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
            'prompt': prompt,
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'stream': True,
        }
        
        # Add optional parameters
        if 'top_p' in kwargs:
            payload['top_p'] = kwargs['top_p']
        if 'frequency_penalty' in kwargs:
            payload['frequency_penalty'] = kwargs['frequency_penalty']
        if 'presence_penalty' in kwargs:
            payload['presence_penalty'] = kwargs['presence_penalty']
        if 'stop' in kwargs:
            payload['stop'] = kwargs['stop']
        
        # Make streaming request
        try:
            with client.stream(
                'POST',
                self._build_url('completions'),
                headers=self._build_headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str == '[DONE]':
                            break
                        
                        import json
                        data = json.loads(data_str)
                        if 'choices' in data and data['choices']:
                            text = data['choices'][0].get('text', '')
                            if text:
                                yield text
        except Exception as e:
            raise LLMError(
                f"Azure OpenAI streaming request failed: {e}",
                provider='azure_openai',
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
        
        # Convert messages to OpenAI format
        message_dicts = [msg.to_dict() for msg in messages]
        
        # Build request payload
        payload = {
            'messages': message_dicts,
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'stream': True,
        }
        
        # Add optional parameters
        if 'top_p' in kwargs:
            payload['top_p'] = kwargs['top_p']
        if 'frequency_penalty' in kwargs:
            payload['frequency_penalty'] = kwargs['frequency_penalty']
        if 'presence_penalty' in kwargs:
            payload['presence_penalty'] = kwargs['presence_penalty']
        if 'stop' in kwargs:
            payload['stop'] = kwargs['stop']
        
        # Make streaming request
        try:
            with client.stream(
                'POST',
                self._build_url('chat/completions'),
                headers=self._build_headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str == '[DONE]':
                            break
                        
                        import json
                        data = json.loads(data_str)
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
        except Exception as e:
            raise LLMError(
                f"Azure OpenAI chat streaming request failed: {e}",
                provider='azure_openai',
                model=self.model,
                original_error=e
            )
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return 'azure_openai'


# Register the provider
register_provider('azure_openai', AzureOpenAILLM)
