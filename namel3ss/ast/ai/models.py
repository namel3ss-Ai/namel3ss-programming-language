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
        provider: Provider name (e.g., "openai", "anthropic", "google", "azure", "vllm", "ollama", "local_ai")
        model_name: Specific model identifier at the provider
        config: Model-specific configuration (temperature, max_tokens, etc.)
        description: Human-readable description of this model's purpose
        metadata: Additional metadata for introspection and management
        
        # Local deployment extensions
        is_local: Whether this model is locally deployed
        deployment_config: Configuration for local model deployment
        local_model_path: Path to local model files (for file-based models)
        
    Example DSL:
        # Cloud provider model
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
        
        # Local vLLM deployment
        ai model local_llama {
            provider: vllm
            model: meta-llama/Llama-2-7b-chat-hf
            config: {
                temperature: 0.8,
                max_tokens: 4096
            }
            deployment_config: {
                gpu_memory_utilization: 0.95,
                max_model_len: 4096,
                dtype: "float16"
            }
            description: "Local Llama model for private inference"
        }
        
        # Local Ollama model
        ai model local_mistral {
            provider: ollama
            model: mistral:7b
            config: {
                temperature: 0.6,
                num_ctx: 2048
            }
            deployment_config: {
                num_gpu: 1,
                num_thread: 8
            }
            description: "Local Mistral model via Ollama"
        }
        
        # Local file-based model
        ai model custom_model {
            provider: local_ai
            model: custom-model
            local_model_path: "/models/custom-model.gguf"
            config: {
                temperature: 0.7,
                context_length: 2048
            }
            deployment_config: {
                backend: "llama-cpp",
                f16: true,
                threads: 4
            }
            description: "Custom local model via LocalAI"
        }
    
    Provider-Specific Notes:
        - OpenAI: Supports gpt-4, gpt-3.5-turbo, text-embedding-* models
        - Anthropic: Supports claude-3-opus, claude-3-sonnet, claude-3-haiku
        - Google: Supports gemini-pro, gemini-1.5-pro, palm2 models
        - Azure: Uses OpenAI models via Azure OpenAI Service
        - vLLM: Supports HuggingFace models with vLLM inference engine
        - Ollama: Supports local models via Ollama runtime
        - LocalAI: Supports various model formats (GGML, GGUF, etc.)
        
    Local Deployment Configuration:
        - vLLM: gpu_memory_utilization, max_model_len, dtype, tensor_parallel_size
        - Ollama: num_gpu, num_thread, num_ctx, repeat_penalty
        - LocalAI: backend, f16, threads, context_length, batch_size
        
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
        - Local models require appropriate hardware resources
        - Model availability and pricing varies by provider
        - Configuration options are passed through to provider APIs
        - The runtime handles provider-specific API differences
        - Fallback/retry logic is handled by the runtime layer
        - Local deployment configurations are validated against provider capabilities
    """
    name: str
    provider: str
    model_name: str
    config: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Local deployment extensions
    is_local: bool = field(default=False)
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    local_model_path: Optional[str] = None


__all__ = ["AIModel"]
