"""
AI model definitions for provider-backed language models.

AIModel provides a declarative handle for referencing external AI models
from providers like OpenAI, Anthropic, Google, Azure, and others.

This module provides production-grade model definitions with validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AIModel:
    """
    Declarative handle for a provider-backed AI model.
    
    Represents a reference to an external AI model that can be used
    in prompts, chains, and other AI-driven constructs. Models are
    resolved at runtime to establish connections to AI providers.
    
    Attributes:
        name: Unique identifier for this model reference in the N3 program
        provider: Provider name (e.g., "openai", "anthropic", "google", "azure")
        model_name: Specific model identifier at the provider
        config: Model-specific configuration (temperature, max_tokens, etc.)
        description: Human-readable description of this model's purpose
        metadata: Additional metadata for introspection and management
        
    Example DSL:
        ai model gpt4 {
            provider: openai
            model: gpt-4-turbo
            config: {
                temperature: 0.7,
                max_tokens: 2048,
                top_p: 0.9
            }
            description: "General-purpose reasoning model"
        }
        
        ai model claude {
            provider: anthropic
            model: claude-3-opus-20240229
            config: {
                temperature: 0.5,
                max_tokens: 4096
            }
        }
        
        ai model gemini {
            provider: google
            model: gemini-1.5-pro
            config: {
                temperature: 0.8,
                top_p: 0.95,
                top_k: 40
            }
        }
        
        ai model embedding_model {
            provider: openai
            model: text-embedding-3-large
            config: {
                dimensions: 1536
            }
            description: "Embedding model for semantic search"
        }
    
    Provider-Specific Notes:
        - OpenAI: Supports gpt-4, gpt-3.5-turbo, text-embedding-* models
        - Anthropic: Supports claude-3-opus, claude-3-sonnet, claude-3-haiku
        - Google: Supports gemini-pro, gemini-1.5-pro, palm2 models
        - Azure: Uses OpenAI models via Azure OpenAI Service
        - Local: Supports Ollama and other local inference servers
        
    Configuration Options:
        - temperature: Sampling temperature (0.0-2.0, higher = more random)
        - max_tokens: Maximum tokens to generate
        - top_p: Nucleus sampling probability mass (0.0-1.0)
        - top_k: Top-k sampling (integer, provider-specific)
        - frequency_penalty: Penalize token frequency (-2.0 to 2.0)
        - presence_penalty: Penalize token presence (-2.0 to 2.0)
        - stop_sequences: List of strings that stop generation
        
    Validation:
        Use validate_ai_model() from .validation to ensure configuration
        is valid before runtime resolution.
        
    Notes:
        - API keys and credentials should be configured via environment
          variables or secure secret management, not in config dict
        - Model availability and pricing varies by provider
        - Configuration options are passed through to provider APIs
        - The runtime handles provider-specific API differences
        - Fallback/retry logic is handled by the runtime layer
    """
    name: str
    provider: str
    model_name: str
    config: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["AIModel"]
