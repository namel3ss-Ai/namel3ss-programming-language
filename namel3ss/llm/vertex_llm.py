"""Google Vertex AI LLM provider implementation."""

import os
from typing import Dict, Any, Optional, List, Iterable
from .base import BaseLLM, LLMResponse, LLMError, ChatMessage
from .factory import register_provider


class VertexLLM(BaseLLM):
    """
    Google Vertex AI LLM provider.
    
    Supports PaLM 2, Gemini, and other Google models.
    
    Configuration:
        - project_id: Google Cloud project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
        - location: Vertex AI location (default: 'us-central1')
        - credentials_path: Path to service account JSON (defaults to GOOGLE_APPLICATION_CREDENTIALS)
        - timeout: Request timeout in seconds (default: 60)
        - temperature: Sampling temperature (default: 0.7)
        - max_tokens: Maximum tokens to generate (default: 1024)
    
    Example:
        >>> llm = VertexLLM('my_gemini', 'gemini-pro', {'temperature': 0.5})
        >>> response = llm.generate('What is the capital of France?')
        >>> print(response.text)
    """
    
    def __init__(self, name: str, model: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, model, config)
        
        # Resolve project ID
        self.project_id = self.config.get('project_id') or os.environ.get('GOOGLE_CLOUD_PROJECT')
        if not self.project_id:
            raise LLMError(
                f"Google Cloud project ID not found for LLM '{name}'. "
                f"Set GOOGLE_CLOUD_PROJECT environment variable or provide 'project_id' in config.",
                provider='vertex',
                model=model
            )
        
        # Configuration
        self.location = self.config.get('location', 'us-central1')
        self.credentials_path = self.config.get('credentials_path') or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        
        # Initialize client lazily
        self._client = None
    
    def _get_client(self):
        """Get or create Vertex AI client."""
        if self._client is None:
            try:
                from google.cloud import aiplatform
                aiplatform.init(project=self.project_id, location=self.location)
                
                # Import the appropriate model class based on model name
                if 'gemini' in self.model.lower():
                    from vertexai.preview.generative_models import GenerativeModel
                    self._client = GenerativeModel(self.model)
                else:
                    from vertexai.preview.language_models import TextGenerationModel
                    self._client = TextGenerationModel.from_pretrained(self.model)
                    
            except ImportError:
                raise LLMError(
                    "google-cloud-aiplatform is required for Vertex AI provider. "
                    "Install it with: pip install google-cloud-aiplatform",
                    provider='vertex',
                    model=self.model
                )
        return self._client
    
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
        client = self._get_client()
        
        try:
            # Generate using Vertex AI
            if 'gemini' in self.model.lower():
                # Gemini models use generate_content
                response = client.generate_content(
                    prompt,
                    generation_config={
                        'temperature': kwargs.get('temperature', self.temperature),
                        'max_output_tokens': kwargs.get('max_tokens', self.max_tokens),
                        'top_p': kwargs.get('top_p', 0.95),
                        'top_k': kwargs.get('top_k', 40),
                    }
                )
                text = response.text
                raw = {'candidates': [{'content': {'parts': [{'text': text}]}}]}
            else:
                # PaLM models use predict
                response = client.predict(
                    prompt,
                    temperature=kwargs.get('temperature', self.temperature),
                    max_output_tokens=kwargs.get('max_tokens', self.max_tokens),
                    top_p=kwargs.get('top_p', 0.95),
                    top_k=kwargs.get('top_k', 40),
                )
                text = response.text
                raw = {'predictions': [{'content': text}]}
            
            return LLMResponse(
                text=text,
                raw=raw,
                model=self.model,
                usage=None,  # Vertex AI doesn't always return token counts
                finish_reason='stop',
                metadata={'provider': 'vertex'}
            )
            
        except Exception as e:
            raise LLMError(
                f"Vertex AI request failed: {e}",
                provider='vertex',
                model=self.model,
                original_error=e
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
        # For now, concatenate messages into a prompt
        # TODO: Use proper chat API when available
        prompt = '\n'.join(f"{msg.role}: {msg.content}" for msg in messages)
        return self.generate(prompt, **kwargs)
    
    def supports_streaming(self) -> bool:
        """Vertex AI supports streaming for some models."""
        return 'gemini' in self.model.lower()
    
    def stream(self, prompt: str, **kwargs) -> Iterable[str]:
        """
        Stream completion for a prompt.
        
        Args:
            prompt: The prompt text
            **kwargs: Additional parameters
        
        Yields:
            Text chunks as they are generated
        
        Raises:
            LLMError: If the API call fails or streaming is not supported
        """
        if not self.supports_streaming():
            raise LLMError(
                f"Streaming is not supported for model '{self.model}'",
                provider='vertex',
                model=self.model
            )
        
        client = self._get_client()
        
        try:
            responses = client.generate_content(
                prompt,
                generation_config={
                    'temperature': kwargs.get('temperature', self.temperature),
                    'max_output_tokens': kwargs.get('max_tokens', self.max_tokens),
                    'top_p': kwargs.get('top_p', 0.95),
                    'top_k': kwargs.get('top_k', 40),
                },
                stream=True
            )
            
            for response in responses:
                if response.text:
                    yield response.text
                    
        except Exception as e:
            raise LLMError(
                f"Vertex AI streaming request failed: {e}",
                provider='vertex',
                model=self.model,
                original_error=e
            )
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return 'vertex'


# Register the provider
register_provider('vertex', VertexLLM)
