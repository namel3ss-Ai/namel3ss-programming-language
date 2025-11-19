"""
Production-grade LLM provider subsystem for Namel3ss.

This package provides a clean, extensible interface for integrating multiple
LLM providers (OpenAI, Anthropic, Google Vertex AI, Azure OpenAI, Ollama, etc.)
into the Namel3ss runtime.

Key components:
- base: Base LLM interface and response types
- registry: Global LLM instance registry
- factory: Provider instantiation and configuration
- providers: Individual provider implementations
"""

from .base import BaseLLM, LLMResponse, LLMError, ChatMessage
from .registry import LLMRegistry, get_registry
from .factory import create_llm, register_llm, register_provider

__all__ = [
    "BaseLLM",
    "LLMResponse",
    "LLMError",
    "ChatMessage",
    "LLMRegistry",
    "get_registry",
    "create_llm",
    "register_llm",
    "register_provider",
]
