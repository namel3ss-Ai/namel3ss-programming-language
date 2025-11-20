"""Google Vertex AI LLM provider implementation."""

import logging
import os
from typing import Dict, Any, Optional, List, Iterable

from .base import BaseLLM, LLMResponse, LLMError, ChatMessage
from .factory import register_provider

logger = logging.getLogger(__name__)


class VertexLLM(BaseLLM):
    """
    Google Vertex AI LLM provider with production-grade chat and streaming support.
    
    Supports Gemini (chat-capable) and PaLM 2 (text-only) models.
    
    Chat Support:
        - For Gemini models: Uses native Vertex AI chat API with multi-turn conversations
        - For non-chat models: Falls back to structured prompt concatenation
    
    Streaming Support:
        - Full streaming support for Gemini models (text and chat)
        - Streaming not available for legacy text models
    
    Configuration:
        - project_id: Google Cloud project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
        - location: Vertex AI location (default: 'us-central1')
        - credentials_path: Path to service account JSON (defaults to GOOGLE_APPLICATION_CREDENTIALS)
        - timeout: Request timeout in seconds (default: 60)
        - temperature: Sampling temperature (default: 0.7)
        - max_tokens: Maximum tokens to generate (default: 1024)
        - top_p: Nucleus sampling parameter (default: 0.95)
        - top_k: Top-k sampling parameter (default: 40)
        - safety_settings: Safety configuration for Gemini models
        - system_instruction: System instruction for Gemini models (if supported)
    
    Example:
        >>> llm = VertexLLM('my_gemini', 'gemini-pro', {'temperature': 0.5})
        >>> response = llm.generate('What is the capital of France?')
        >>> print(response.text)
        >>> 
        >>> # Chat usage
        >>> messages = [
        ...     ChatMessage(role='user', content='Hello!'),
        ...     ChatMessage(role='assistant', content='Hi there!'),
        ...     ChatMessage(role='user', content='How are you?')
        ... ]
        >>> response = llm.generate_chat(messages)
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
        self.top_p = self.config.get('top_p', 0.95)
        self.top_k = self.config.get('top_k', 40)
        self.safety_settings = self.config.get('safety_settings')
        self.system_instruction = self.config.get('system_instruction')
        
        # Initialize client lazily
        self._client = None
        self._is_chat_model = None
    
    def _is_chat_capable(self) -> bool:
        """
        Determine if the model supports native chat API.
        
        Returns:
            True if the model supports chat, False otherwise
        """
        if self._is_chat_model is None:
            # Gemini models support native chat
            # Can be extended based on model capabilities
            model_lower = self.model.lower()
            self._is_chat_model = 'gemini' in model_lower
            
            logger.debug(
                f"Model '{self.model}' chat capability: {self._is_chat_model}"
            )
        
        return self._is_chat_model
    
    def _get_client(self):
        """
        Get or create Vertex AI client.
        
        Returns:
            Vertex AI client (GenerativeModel or TextGenerationModel)
        
        Raises:
            LLMError: If initialization fails
        """
        if self._client is None:
            try:
                from google.cloud import aiplatform
                
                # Initialize Vertex AI with project and location
                aiplatform.init(project=self.project_id, location=self.location)
                logger.info(
                    f"Initialized Vertex AI: project={self.project_id}, location={self.location}"
                )
                
                # Import the appropriate model class based on model capabilities
                if self._is_chat_capable():
                    # Use GenerativeModel for chat-capable models (Gemini)
                    try:
                        from vertexai.generative_models import GenerativeModel
                    except ImportError:
                        # Fallback to preview if stable not available
                        from vertexai.preview.generative_models import GenerativeModel
                    
                    # Create model with optional system instruction
                    if self.system_instruction:
                        self._client = GenerativeModel(
                            self.model,
                            system_instruction=self.system_instruction
                        )
                    else:
                        self._client = GenerativeModel(self.model)
                    
                    logger.info(f"Created GenerativeModel client for '{self.model}'")
                else:
                    # Use TextGenerationModel for legacy text models
                    try:
                        from vertexai.language_models import TextGenerationModel
                    except ImportError:
                        # Fallback to preview if stable not available
                        from vertexai.preview.language_models import TextGenerationModel
                    
                    self._client = TextGenerationModel.from_pretrained(self.model)
                    logger.info(f"Created TextGenerationModel client for '{self.model}'")
                    
            except ImportError as e:
                raise LLMError(
                    "google-cloud-aiplatform is required for Vertex AI provider. "
                    "Install it with: pip install google-cloud-aiplatform",
                    provider='vertex',
                    model=self.model,
                    original_error=e
                )
            except Exception as e:
                raise LLMError(
                    f"Failed to initialize Vertex AI client: {e}",
                    provider='vertex',
                    model=self.model,
                    original_error=e
                )
        
        return self._client
    
    def _build_generation_config(self, **kwargs) -> Dict[str, Any]:
        """
        Build generation configuration from instance config and kwargs.
        
        Args:
            **kwargs: Override parameters
        
        Returns:
            Generation configuration dict
        """
        return {
            'temperature': kwargs.get('temperature', self.temperature),
            'max_output_tokens': kwargs.get('max_tokens', self.max_tokens),
            'top_p': kwargs.get('top_p', self.top_p),
            'top_k': kwargs.get('top_k', self.top_k),
        }
    
    def _extract_gemini_response(self, response, include_usage: bool = True) -> LLMResponse:
        """
        Extract LLMResponse from Gemini API response.
        
        Args:
            response: Vertex AI GenerativeModel response
            include_usage: Whether to extract usage information
        
        Returns:
            Standardized LLMResponse
        """
        try:
            # Extract text from response
            text = response.text if hasattr(response, 'text') else ''
            
            # Extract usage metadata if available
            usage = None
            if include_usage and hasattr(response, 'usage_metadata'):
                usage_meta = response.usage_metadata
                usage = {
                    'prompt_tokens': getattr(usage_meta, 'prompt_token_count', 0),
                    'completion_tokens': getattr(usage_meta, 'candidates_token_count', 0),
                    'total_tokens': getattr(usage_meta, 'total_token_count', 0),
                }
            
            # Extract finish reason
            finish_reason = None
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = str(candidate.finish_reason.name).lower()
            
            # Build metadata
            metadata = {
                'provider': 'vertex',
                'model_version': self.model,
            }
            
            # Include safety ratings if available
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'safety_ratings'):
                    metadata['safety_ratings'] = [
                        {
                            'category': str(rating.category.name),
                            'probability': str(rating.probability.name)
                        }
                        for rating in candidate.safety_ratings
                    ]
            
            return LLMResponse(
                text=text,
                raw=response,
                model=self.model,
                usage=usage,
                finish_reason=finish_reason or 'stop',
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Failed to extract response from Gemini: {e}")
            raise LLMError(
                f"Failed to parse Vertex AI response: {e}",
                provider='vertex',
                model=self.model,
                original_error=e
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
        client = self._get_client()
        generation_config = self._build_generation_config(**kwargs)
        
        try:
            # Generate using Vertex AI
            if self._is_chat_capable():
                # Gemini models use generate_content
                response = client.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                return self._extract_gemini_response(response)
            else:
                # PaLM models use predict
                response = client.predict(
                    prompt,
                    **generation_config
                )
                text = response.text
                
                return LLMResponse(
                    text=text,
                    raw=response,
                    model=self.model,
                    usage=None,  # PaLM doesn't always return token counts
                    finish_reason='stop',
                    metadata={'provider': 'vertex', 'model_version': self.model}
                )
            
        except Exception as e:
            logger.error(f"Vertex AI generate failed: {e}")
            raise LLMError(
                f"Vertex AI request failed: {e}",
                provider='vertex',
                model=self.model,
                original_error=e
            )
    
    def _convert_chat_messages_to_vertex(self, messages: List[ChatMessage]) -> List[Any]:
        """
        Convert ChatMessage list to Vertex AI Content format.
        
        Args:
            messages: List of ChatMessage objects
        
        Returns:
            List of Vertex AI Content objects
        
        Note:
            Gemini uses "user" and "model" roles.
            System messages are handled via system_instruction in model initialization.
        """
        try:
            from vertexai.generative_models import Content, Part
        except ImportError:
            from vertexai.preview.generative_models import Content, Part
        
        vertex_contents = []
        
        for msg in messages:
            # Map roles: assistant -> model, user -> user
            # System messages are filtered (should be in system_instruction)
            if msg.role == 'system':
                logger.warning(
                    "System message in chat history. "
                    "Consider using system_instruction parameter instead."
                )
                continue
            
            role = 'model' if msg.role == 'assistant' else 'user'
            
            # Create Content with Part
            content = Content(
                role=role,
                parts=[Part.from_text(msg.content)]
            )
            vertex_contents.append(content)
        
        return vertex_contents
    
    def generate_chat(self, messages: List[ChatMessage], **kwargs) -> LLMResponse:
        """
        Generate chat completion using native chat API.
        
        For Gemini models, this uses the proper Vertex AI chat API with
        message history support. For legacy models (PaLM), this falls back
        to a structured prompt format that preserves conversation context.
        
        Args:
            messages: List of chat messages with roles (system, user, assistant)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            LLMResponse with generated assistant message
        
        Raises:
            LLMError: If the API call fails
        
        Note:
            System messages for Gemini should ideally be set via the
            system_instruction parameter during VertexLLM initialization.
            System messages in the message list will be logged as warnings.
        """
        client = self._get_client()
        generation_config = self._build_generation_config(**kwargs)
        
        try:
            if self._is_chat_capable():
                # Use native chat API for Gemini
                logger.debug(f"Using native chat API for {self.model}")
                
                # Convert messages to Vertex format
                vertex_contents = self._convert_chat_messages_to_vertex(messages)
                
                if not vertex_contents:
                    raise LLMError(
                        "No valid messages after filtering system messages. "
                        "Provide at least one user or assistant message.",
                        provider='vertex',
                        model=self.model
                    )
                
                # Use start_chat for conversation with history
                if len(vertex_contents) > 1:
                    # Multi-turn conversation: use chat session
                    history = vertex_contents[:-1]  # All but last message
                    last_message = messages[-1].content
                    
                    chat = client.start_chat(history=history)
                    response = chat.send_message(
                        last_message,
                        generation_config=generation_config,
                        safety_settings=self.safety_settings
                    )
                else:
                    # Single message: use generate_content directly
                    response = client.generate_content(
                        vertex_contents[0],
                        generation_config=generation_config,
                        safety_settings=self.safety_settings
                    )
                
                return self._extract_gemini_response(response)
            
            else:
                # Fallback for non-chat models (PaLM)
                # Use structured prompt with role prefixes
                logger.debug(f"Using fallback chat format for {self.model}")
                
                prompt_parts = []
                for msg in messages:
                    role_label = {
                        'system': 'System',
                        'user': 'User',
                        'assistant': 'Assistant'
                    }.get(msg.role, msg.role.capitalize())
                    
                    prompt_parts.append(f"{role_label}: {msg.content}")
                
                # Add assistant prompt to encourage response
                prompt_parts.append("Assistant:")
                prompt = '\n\n'.join(prompt_parts)
                
                # Use generate() for text completion
                response = self.generate(prompt, **kwargs)
                
                # Update metadata to indicate fallback was used
                response.metadata['chat_fallback'] = True
                response.metadata['fallback_reason'] = 'model_does_not_support_native_chat'
                
                return response
                
        except Exception as e:
            logger.error(f"Vertex AI chat generation failed: {e}")
            raise LLMError(
                f"Vertex AI chat request failed: {e}",
                provider='vertex',
                model=self.model,
                original_error=e
            )
    
    def supports_streaming(self) -> bool:
        """
        Check if the model supports streaming.
        
        Returns:
            True if streaming is supported, False otherwise
        """
        # Gemini models support streaming
        return self._is_chat_capable()
    
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
        generation_config = self._build_generation_config(**kwargs)
        
        try:
            responses = client.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings,
                stream=True
            )
            
            for response in responses:
                if hasattr(response, 'text') and response.text:
                    yield response.text
                    
        except Exception as e:
            logger.error(f"Vertex AI streaming failed: {e}")
            raise LLMError(
                f"Vertex AI streaming request failed: {e}",
                provider='vertex',
                model=self.model,
                original_error=e
            )
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> Iterable[str]:
        """
        Stream chat completion using native chat API.
        
        For Gemini models, streams responses from the chat API.
        For legacy models, falls back to structured prompt streaming.
        
        Args:
            messages: List of chat messages
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
        generation_config = self._build_generation_config(**kwargs)
        
        try:
            if self._is_chat_capable():
                # Use native chat API with streaming
                logger.debug(f"Using native chat streaming for {self.model}")
                
                vertex_contents = self._convert_chat_messages_to_vertex(messages)
                
                if not vertex_contents:
                    raise LLMError(
                        "No valid messages for chat streaming",
                        provider='vertex',
                        model=self.model
                    )
                
                # Stream multi-turn conversation
                if len(vertex_contents) > 1:
                    history = vertex_contents[:-1]
                    last_message = messages[-1].content
                    
                    chat = client.start_chat(history=history)
                    responses = chat.send_message(
                        last_message,
                        generation_config=generation_config,
                        safety_settings=self.safety_settings,
                        stream=True
                    )
                else:
                    # Single message
                    responses = client.generate_content(
                        vertex_contents[0],
                        generation_config=generation_config,
                        safety_settings=self.safety_settings,
                        stream=True
                    )
                
                # Yield chunks
                for response in responses:
                    if hasattr(response, 'text') and response.text:
                        yield response.text
            
            else:
                # Fallback: use structured prompt streaming
                logger.debug(f"Using fallback streaming for {self.model}")
                
                prompt_parts = []
                for msg in messages:
                    role_label = {
                        'system': 'System',
                        'user': 'User',
                        'assistant': 'Assistant'
                    }.get(msg.role, msg.role.capitalize())
                    prompt_parts.append(f"{role_label}: {msg.content}")
                
                prompt_parts.append("Assistant:")
                prompt = '\n\n'.join(prompt_parts)
                
                # Stream using generate
                yield from self.stream(prompt, **kwargs)
                
        except Exception as e:
            logger.error(f"Vertex AI chat streaming failed: {e}")
            raise LLMError(
                f"Vertex AI chat streaming request failed: {e}",
                provider='vertex',
                model=self.model,
                original_error=e
            )
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return 'vertex'


# Register the provider
register_provider('vertex', VertexLLM)
