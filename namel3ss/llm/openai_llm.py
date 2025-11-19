"""OpenAI LLM provider implementation."""

import os
from typing import Dict, Any, Optional, List, Iterable
from .base import BaseLLM, LLMResponse, LLMError, ChatMessage
from .factory import register_provider


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM provider.
    
    Supports GPT-3.5, GPT-4, and other OpenAI models.
    
    Configuration:
        - api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        - api_base: Base URL for API (defaults to https://api.openai.com/v1)
        - organization: Optional organization ID
        - timeout: Request timeout in seconds (default: 60)
        - temperature: Sampling temperature (default: 0.7)
        - max_tokens: Maximum tokens to generate (default: 1024)
    
    Example:
        >>> llm = OpenAILLM('my_gpt4', 'gpt-4', {'temperature': 0.5})
        >>> response = llm.generate('What is the capital of France?')
        >>> print(response.text)
    """
    
    def __init__(self, name: str, model: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, model, config)
        
        # Resolve API key
        self.api_key = self.config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise LLMError(
                f"OpenAI API key not found for LLM '{name}'. "
                f"Set OPENAI_API_KEY environment variable or provide 'api_key' in config.",
                provider='openai',
                model=model
            )
        
        # API configuration
        self.api_base = self.config.get('api_base', 'https://api.openai.com/v1')
        self.organization = self.config.get('organization')
        
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
                    "httpx is required for OpenAI provider. Install it with: pip install httpx",
                    provider='openai',
                    model=self.model
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
            'model': self.model,
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
                f"{self.api_base}/completions",
                headers=self._build_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise LLMError(
                f"OpenAI API request failed: {e}",
                provider='openai',
                model=self.model,
                status_code=getattr(response, 'status_code', None) if 'response' in locals() else None,
                original_error=e
            )
        
        # Extract response
        if 'choices' not in data or not data['choices']:
            raise LLMError(
                f"OpenAI API returned no choices: {data}",
                provider='openai',
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
            metadata={'provider': 'openai'}
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
            'model': self.model,
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
        if 'functions' in kwargs:
            payload['functions'] = kwargs['functions']
        if 'function_call' in kwargs:
            payload['function_call'] = kwargs['function_call']
        
        # Make API request
        try:
            response = client.post(
                f"{self.api_base}/chat/completions",
                headers=self._build_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise LLMError(
                f"OpenAI chat API request failed: {e}",
                provider='openai',
                model=self.model,
                status_code=getattr(response, 'status_code', None) if 'response' in locals() else None,
                original_error=e
            )
        
        # Extract response
        if 'choices' not in data or not data['choices']:
            raise LLMError(
                f"OpenAI chat API returned no choices: {data}",
                provider='openai',
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
                'provider': 'openai',
                'message': message,
            }
        )
    
    def supports_streaming(self) -> bool:
        """OpenAI supports streaming."""
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
                f"{self.api_base}/completions",
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
                f"OpenAI streaming request failed: {e}",
                provider='openai',
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
            'model': self.model,
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
                f"{self.api_base}/chat/completions",
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
                f"OpenAI chat streaming request failed: {e}",
                provider='openai',
                model=self.model,
                original_error=e
            )
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return 'openai'
    
    def supports_structured_output(self) -> bool:
        """OpenAI supports JSON mode and function calling."""
        return True
    
    def generate_structured(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate structured JSON output using OpenAI's JSON mode.
        
        Uses response_format parameter for models that support it (gpt-4-turbo, gpt-3.5-turbo-1106+).
        
        Args:
            prompt: The prompt text
            output_schema: JSON Schema dict
            **kwargs: Additional parameters
        
        Returns:
            LLMResponse with JSON text
        """
        # Convert to chat format for JSON mode support
        messages = [ChatMessage(role="user", content=prompt)]
        return self.generate_structured_chat(messages, output_schema, **kwargs)
    
    def generate_structured_chat(
        self,
        messages: List[ChatMessage],
        output_schema: Dict[str, Any],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate structured output using OpenAI's JSON mode or function calling.
        
        Args:
            messages: List of chat messages
            output_schema: JSON Schema dict
            **kwargs: Additional parameters
        
        Returns:
            LLMResponse with JSON text
        """
        client = self._get_http_client()
        
        # Add format instruction to system message
        enhanced_messages = messages.copy()
        format_instruction = (
            "You must respond with valid JSON matching the specified schema. "
            "Do not include any text before or after the JSON object."
        )
        
        # Prepend system message with JSON instruction
        if enhanced_messages and enhanced_messages[0].role == "system":
            enhanced_messages[0].content = f"{enhanced_messages[0].content}\n\n{format_instruction}"
        else:
            enhanced_messages.insert(0, ChatMessage(role="system", content=format_instruction))
        
        # Convert messages to OpenAI format
        message_dicts = [msg.to_dict() for msg in enhanced_messages]
        
        # Build request payload
        payload = {
            'model': self.model,
            'messages': message_dicts,
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
        }
        
        # Use JSON mode if model supports it (gpt-4-turbo-preview, gpt-3.5-turbo-1106+)
        if any(marker in self.model.lower() for marker in ['gpt-4-turbo', 'gpt-4o', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0125']):
            payload['response_format'] = {"type": "json_object"}
        
        # Add optional parameters
        if 'top_p' in kwargs:
            payload['top_p'] = kwargs['top_p']
        if 'frequency_penalty' in kwargs:
            payload['frequency_penalty'] = kwargs['frequency_penalty']
        if 'presence_penalty' in kwargs:
            payload['presence_penalty'] = kwargs['presence_penalty']
        
        # Make API request
        try:
            response = client.post(
                f"{self.api_base}/chat/completions",
                headers=self._build_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise LLMError(
                f"OpenAI structured chat API request failed: {e}",
                provider='openai',
                model=self.model,
                status_code=getattr(response, 'status_code', None) if 'response' in locals() else None,
                original_error=e
            )
        
        # Extract response
        if 'choices' not in data or not data['choices']:
            raise LLMError(
                f"OpenAI API returned no choices: {data}",
                provider='openai',
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
            metadata={'provider': 'openai', 'structured': True}
        )


# Register the provider
register_provider('openai', OpenAILLM)
