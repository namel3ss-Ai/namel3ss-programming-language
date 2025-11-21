"""Model adapter for LLM API integrations.

Provides unified interface to OpenAI, Anthropic, HuggingFace, and other LLM providers.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, AsyncIterator, Iterator

from pydantic import BaseModel, Field

from .base import (
    AdapterConfig,
    AdapterType,
    BaseAdapter,
    AdapterExecutionError,
    AdapterValidationError,
)


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    VERTEX = "vertex"
    OLLAMA = "ollama"


class ModelAdapterConfig(AdapterConfig):
    """Configuration for model adapter."""
    
    adapter_type: AdapterType = Field(default=AdapterType.MODEL)
    
    # Provider settings
    provider: ModelProvider = Field(..., description="LLM provider")
    api_key: Optional[str] = Field(None, description="API key (if required)")
    api_base: Optional[str] = Field(None, description="Custom API base URL")
    
    # Model settings
    model: str = Field(..., description="Model identifier")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Max tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling")
    
    # Streaming
    stream: bool = Field(default=False, description="Enable response streaming")
    
    # Token tracking
    track_tokens: bool = Field(default=True, description="Track token usage")
    
    # Rate limiting
    max_requests_per_minute: Optional[int] = Field(None, description="Rate limit")


class ModelAdapter(BaseAdapter):
    """Unified LLM adapter supporting multiple providers.
    
    Example:
        Use from N3:
        ```n3
        tool "openai_chat" {
          adapter: "model"
          provider: "openai"
          api_key: env("OPENAI_API_KEY")
          model: "gpt-4"
          temperature: 0.7
          max_tokens: 500
        }
        
        chain "generate_summary" {
          call: "openai_chat"
          inputs: {
            messages: [
              {role: "system", content: "You are a helpful assistant."}
              {role: "user", content: "Summarize: {{text}}"}
            ]
          }
        }
        ```
        
        Programmatic usage:
        >>> config = ModelAdapterConfig(
        ...     name="gpt4",
        ...     provider="openai",
        ...     api_key="sk-...",
        ...     model="gpt-4",
        ...     temperature=0.7
        ... )
        >>> adapter = ModelAdapter(config)
        >>> result = adapter.execute(
        ...     messages=[
        ...         {"role": "user", "content": "Hello!"}
        ...     ]
        ... )
        >>> print(result['content'])
    """
    
    def __init__(self, config: ModelAdapterConfig):
        super().__init__(config)
        self.config: ModelAdapterConfig = config
        self._client = None
        self._total_tokens = 0
        self._total_cost = 0.0
        
        self._setup_provider()
    
    def _setup_provider(self):
        """Setup provider client."""
        if self.config.provider == ModelProvider.OPENAI:
            self._setup_openai()
        elif self.config.provider == ModelProvider.ANTHROPIC:
            self._setup_anthropic()
        elif self.config.provider == ModelProvider.HUGGINGFACE:
            self._setup_huggingface()
        elif self.config.provider == ModelProvider.VERTEX:
            self._setup_vertex()
        elif self.config.provider == ModelProvider.OLLAMA:
            self._setup_ollama()
        else:
            raise AdapterValidationError(
                f"Unsupported provider: {self.config.provider}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
    
    def _setup_openai(self):
        """Setup OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise AdapterExecutionError(
                "OpenAI not installed. Install with: pip install openai",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base,
            timeout=self.config.timeout,
        )
    
    def _setup_anthropic(self):
        """Setup Anthropic client."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise AdapterExecutionError(
                "Anthropic not installed. Install with: pip install anthropic",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        self._client = Anthropic(
            api_key=self.config.api_key,
            base_url=self.config.api_base,
            timeout=self.config.timeout,
        )
    
    def _setup_huggingface(self):
        """Setup HuggingFace client."""
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise AdapterExecutionError(
                "HuggingFace not installed. Install with: pip install huggingface-hub",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        self._client = InferenceClient(
            model=self.config.model,
            token=self.config.api_key,
            timeout=self.config.timeout,
        )
    
    def _setup_vertex(self):
        """Setup Vertex AI client."""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError:
            raise AdapterExecutionError(
                "Vertex AI not installed. Install with: pip install google-cloud-aiplatform",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        # Initialize Vertex AI (assumes GOOGLE_APPLICATION_CREDENTIALS set)
        vertexai.init()
        self._client = GenerativeModel(self.config.model)
    
    def _setup_ollama(self):
        """Setup Ollama client."""
        try:
            from ollama import Client
        except ImportError:
            raise AdapterExecutionError(
                "Ollama not installed. Install with: pip install ollama",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        base_url = self.config.api_base or "http://localhost:11434"
        self._client = Client(host=base_url)
    
    def _execute_impl(self, **inputs: Any) -> Any:
        """Execute model inference."""
        messages = inputs.get("messages")
        if not messages:
            raise AdapterValidationError(
                "Missing required field: messages",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        # Route to provider-specific method
        if self.config.provider == ModelProvider.OPENAI:
            return self._execute_openai(messages, inputs)
        elif self.config.provider == ModelProvider.ANTHROPIC:
            return self._execute_anthropic(messages, inputs)
        elif self.config.provider == ModelProvider.HUGGINGFACE:
            return self._execute_huggingface(messages, inputs)
        elif self.config.provider == ModelProvider.VERTEX:
            return self._execute_vertex(messages, inputs)
        elif self.config.provider == ModelProvider.OLLAMA:
            return self._execute_ollama(messages, inputs)
    
    def _execute_openai(self, messages: List[Dict], inputs: Dict) -> Dict[str, Any]:
        """Execute OpenAI chat completion."""
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                stream=self.config.stream,
            )
            
            if self.config.stream:
                # Return generator for streaming
                return {
                    "stream": True,
                    "chunks": response,
                }
            
            # Extract response
            choice = response.choices[0]
            
            result = {
                "content": choice.message.content,
                "role": choice.message.role,
                "finish_reason": choice.finish_reason,
            }
            
            # Track tokens
            if self.config.track_tokens:
                usage = response.usage
                result["usage"] = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
                self._total_tokens += usage.total_tokens
            
            return result
        
        except Exception as e:
            raise AdapterExecutionError(
                f"OpenAI API error: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
    
    def _execute_anthropic(self, messages: List[Dict], inputs: Dict) -> Dict[str, Any]:
        """Execute Anthropic completion."""
        try:
            # Convert messages format
            system_msg = None
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_messages.append(msg)
            
            response = self._client.messages.create(
                model=self.config.model,
                messages=user_messages,
                system=system_msg,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens or 1024,
                stream=self.config.stream,
            )
            
            if self.config.stream:
                return {
                    "stream": True,
                    "chunks": response,
                }
            
            result = {
                "content": response.content[0].text,
                "role": "assistant",
                "stop_reason": response.stop_reason,
            }
            
            if self.config.track_tokens:
                result["usage"] = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }
                self._total_tokens += result["usage"]["total_tokens"]
            
            return result
        
        except Exception as e:
            raise AdapterExecutionError(
                f"Anthropic API error: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
    
    def _execute_huggingface(self, messages: List[Dict], inputs: Dict) -> Dict[str, Any]:
        """Execute HuggingFace inference."""
        try:
            # Format prompt from messages
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            response = self._client.text_generation(
                prompt,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stream=self.config.stream,
            )
            
            if self.config.stream:
                return {
                    "stream": True,
                    "chunks": response,
                }
            
            return {
                "content": response,
                "role": "assistant",
            }
        
        except Exception as e:
            raise AdapterExecutionError(
                f"HuggingFace API error: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
    
    def _execute_vertex(self, messages: List[Dict], inputs: Dict) -> Dict[str, Any]:
        """Execute Vertex AI generation."""
        try:
            # Format prompt
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            response = self._client.generate_content(
                prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_output_tokens": self.config.max_tokens,
                }
            )
            
            return {
                "content": response.text,
                "role": "assistant",
            }
        
        except Exception as e:
            raise AdapterExecutionError(
                f"Vertex AI error: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
    
    def _execute_ollama(self, messages: List[Dict], inputs: Dict) -> Dict[str, Any]:
        """Execute Ollama chat completion."""
        try:
            response = self._client.chat(
                model=self.config.model,
                messages=messages,
                stream=self.config.stream,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "top_p": self.config.top_p,
                }
            )
            
            if self.config.stream:
                return {
                    "stream": True,
                    "chunks": response,
                }
            
            result = {
                "content": response["message"]["content"],
                "role": response["message"]["role"],
            }
            
            if self.config.track_tokens:
                result["usage"] = {
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
                }
                self._total_tokens += result["usage"]["total_tokens"]
            
            return result
        
        except Exception as e:
            raise AdapterExecutionError(
                f"Ollama error: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage.
        
        Returns:
            Token usage stats
        
        Example:
            >>> adapter.execute(messages=[...])
            >>> usage = adapter.get_token_usage()
            >>> print(f"Total tokens: {usage['total_tokens']}")
        """
        return {
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
        }
    
    def reset_token_usage(self):
        """Reset token usage tracking."""
        self._total_tokens = 0
        self._total_cost = 0.0
