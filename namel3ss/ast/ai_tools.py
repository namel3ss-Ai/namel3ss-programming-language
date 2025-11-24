"""
LLM and external tool definitions for AI capabilities.

This module contains AST nodes for:
- LLM definitions: Reusable LLM configurations
- Tool definitions: External tools/APIs that LLMs can invoke
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .source_location import SourceLocation

from .base import Expression


@dataclass
class LLMDefinition:
    """
    Reusable LLM configuration definition.
    
    Defines a named LLM setup that can be referenced throughout
    the application, including:
    - Model selection
    - Generation parameters (temperature, max_tokens)
    - Safety and moderation settings
    - Provider-specific configuration
    
    Example DSL:
        llm conservative_writer {
            model: "gpt-4"
            temperature: 0.3
            max_tokens: 1000
            top_p: 0.9
            
            system_prompt: "You are a professional technical writer."
            
            safety: {
                content_filter: "strict",
                pii_detection: true,
                toxicity_threshold: 0.1
            }
            
            metadata: {
                cost_tier: "premium",
                use_case: "documentation"
            }
        }
        
        llm creative_assistant {
            model: "claude-3-opus"
            temperature: 0.8
            max_tokens: 2000
            
            system_prompt: "You are a creative storytelling assistant."
            
            tools: ["web_search", "calculator"]
        }
    """
    name: str
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    system_prompt: Optional[str] = None
    safety: Dict[str, Any] = field(default_factory=dict)
    tools: List[str] = field(default_factory=list)  # References to ToolDefinitions
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = True
    seed: Optional[int] = None
    description: Optional[str] = None
    source_location: Optional['SourceLocation'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """
    External tool/API definition for LLM function calling.
    
    Defines tools that LLMs can invoke to:
    - Access external APIs
    - Query databases
    - Perform computations
    - Retrieve real-time data
    
    Follows OpenAI/Anthropic function calling schema with:
    - Name and description
    - Input parameter schema (JSON Schema)
    - Optional implementation details
    
    Example DSL:
        tool web_search {
            description: "Search the web for current information"
            
            parameters: {
                query: {
                    type: "string",
                    description: "The search query",
                    required: true
                },
                num_results: {
                    type: "integer",
                    description: "Number of results to return",
                    default: 5
                },
                language: {
                    type: "string",
                    description: "Search language (ISO 639-1)",
                    default: "en"
                }
            }
            
            returns: {
                type: "array",
                items: {
                    title: "string",
                    url: "string",
                    snippet: "string"
                }
            }
            
            implementation: {
                endpoint: "https://api.search.example.com/v1/search",
                auth: "bearer_token"
            }
        }
        
        tool calculator {
            description: "Perform mathematical calculations"
            
            parameters: {
                expression: {
                    type: "string",
                    description: "Mathematical expression to evaluate",
                    required: true
                }
            }
            
            returns: {
                type: "number",
                description: "Result of the calculation"
            }
        }
        
        tool database_query {
            description: "Query the customer database"
            
            parameters: {
                table: {
                    type: "string",
                    enum: ["customers", "orders", "products"],
                    required: true
                },
                filters: {
                    type: "object",
                    description: "Filter conditions"
                },
                limit: {
                    type: "integer",
                    default: 100
                }
            }
            
            returns: {
                type: "array",
                items: {
                    type: "object"
                }
            }
            
            implementation: {
                connector: "postgres_db",
                timeout: 30
            }
            
            # Security configuration
            security: {
                permission_level: NETWORK
                required_capabilities: [NETWORK, HTTP_READ]
                timeout_seconds: 10.0
                rate_limit_per_minute: 30
            }
        }
    """
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)  # JSON Schema for inputs
    returns: Optional[Dict[str, Any]] = None  # JSON Schema for outputs
    implementation: Dict[str, Any] = field(default_factory=dict)  # Implementation details
    examples: List[Dict[str, Any]] = field(default_factory=list)  # Example invocations
    
    # Security fields (optional, defaults to READ_ONLY)
    security_config: Optional[Any] = None  # ToolSecurity from ast.security
    permission_level: Optional[str] = None  # PermissionLevel as string
    required_capabilities: List[str] = field(default_factory=list)  # Capability names
    timeout_seconds: Optional[float] = None
    rate_limit_per_minute: Optional[int] = None
    
    source_location: Optional['SourceLocation'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "LLMDefinition",
    "ToolDefinition",
]
