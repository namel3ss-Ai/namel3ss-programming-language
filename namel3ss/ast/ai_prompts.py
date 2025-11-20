"""
Prompt definitions with typed inputs, outputs, and structured schemas.

This module contains all prompt-related AST nodes including:
- PromptField: Schema fields for inputs/outputs
- PromptArgument: Typed arguments for parameterized prompts
- OutputFieldType: Type specifications for structured outputs
- OutputField: Individual output schema fields
- OutputSchema: Complete structured output schema
- Prompt: Main prompt definition with model, template, and schema
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .source_location import SourceLocation, Set


@dataclass
class PromptField:
    """
    Structured schema information for prompt inputs/outputs.
    
    PromptField defines the shape of data flowing into or out of a prompt,
    including type information, validation rules, and documentation.
    
    Example:
        input_field: {
            name: "user_query",
            type: "text",
            required: true,
            description: "The user's question or request"
        }
    """
    name: str
    field_type: str = "text"
    required: bool = True
    description: Optional[str] = None
    default: Optional[Any] = None
    enum: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptArgument:
    """
    Typed argument for parameterized prompt templates.
    
    PromptArguments allow prompts to be parameterized with strongly-typed
    arguments, enabling reusable prompts with different inputs.
    
    Example DSL:
        prompt summarize {
            args: {
                text: string,
                max_length: int = 100
            }
            template: "Summarize the following text in {max_length} words: {text}"
        }
    """
    name: str
    arg_type: str  # string, int, float, bool, list, object
    required: bool = True
    default: Any = None
    description: Optional[str] = None
    location: Optional['SourceLocation'] = None  # Forward reference for future implementation


@dataclass
class EnumType:
    """
    Enum type constraint for output schema fields.
    
    Represents a fixed set of allowed string values for a field,
    enabling the LLM to return structured categorical data.
    
    Example:
        enum["billing", "technical", "account", "other"]
    """
    values: List[str]
    
    def __str__(self) -> str:
        vals = ', '.join(f'"{v}"' for v in self.values)
        return f"enum[{vals}]"


@dataclass
class OutputFieldType:
    """
    Type specification for an output schema field.
    
    Supports rich type definitions including:
    - Primitives: string, int, float, bool
    - Collections: list[T]
    - Nested objects: {field: type, ...}
    - Enums: enum["val1", "val2"]
    - Nullable types: T?
    
    Example types:
        string
        int
        list[string]
        enum["low", "medium", "high"]
        object{field1: string, field2: int}
    """
    base_type: str  # string, int, float, bool, list, object, enum
    element_type: Optional['OutputFieldType'] = None  # For list[T]
    enum_values: Optional[List[str]] = None  # For enum types
    nested_fields: Optional[List['OutputField']] = None  # For object types
    nullable: bool = False
    
    def __str__(self) -> str:
        if self.base_type == "enum" and self.enum_values:
            vals = ', '.join(f'"{v}"' for v in self.enum_values)
            return f"enum[{vals}]"
        elif self.base_type == "list" and self.element_type:
            return f"list[{self.element_type}]"
        elif self.base_type == "object" and self.nested_fields:
            return "object{...}"
        suffix = "?" if self.nullable else ""
        return f"{self.base_type}{suffix}"


@dataclass
class OutputField:
    """
    A single field in an output schema.
    
    Combines a field name with its type specification and metadata,
    defining one component of the structured output.
    
    Example:
        category: enum["billing", "technical"]
        confidence: float
        tags: list[string]
    """
    name: str
    field_type: OutputFieldType
    required: bool = True
    description: Optional[str] = None
    location: Optional['SourceLocation'] = None  # Forward reference


@dataclass
class OutputSchema:
    """
    Structured output schema for a prompt.
    
    Defines the complete structure of data expected from an LLM,
    which can be converted to JSON Schema for provider APIs.
    
    Example DSL:
        output_schema: {
            category: enum["billing", "technical", "account"],
            urgency: enum["low", "medium", "high"],
            needs_handoff: bool,
            confidence: float
        }
    """
    fields: List[OutputField]
    
    def to_json_schema(self) -> Dict[str, Any]:
        """
        Convert to JSON Schema format for LLM providers.
        
        Returns a JSON Schema object that can be sent to LLM APIs
        supporting structured outputs (e.g., OpenAI function calling,
        Anthropic tool use).
        """
        properties = {}
        required = []
        
        for field in self.fields:
            properties[field.name] = self._field_type_to_json_schema(field.field_type)
            if field.description:
                properties[field.name]["description"] = field.description
            if field.required:
                required.append(field.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
    
    def _field_type_to_json_schema(self, ft: OutputFieldType) -> Dict[str, Any]:
        """Convert OutputFieldType to JSON Schema type definition."""
        if ft.base_type == "enum" and ft.enum_values:
            return {"type": "string", "enum": ft.enum_values}
        elif ft.base_type == "list" and ft.element_type:
            return {
                "type": "array",
                "items": self._field_type_to_json_schema(ft.element_type)
            }
        elif ft.base_type == "object" and ft.nested_fields:
            nested_props = {}
            nested_required = []
            for nf in ft.nested_fields:
                nested_props[nf.name] = self._field_type_to_json_schema(nf.field_type)
                if nf.required:
                    nested_required.append(nf.name)
            return {
                "type": "object",
                "properties": nested_props,
                "required": nested_required
            }
        else:
            # Map N3 types to JSON Schema types
            type_map = {
                "string": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
            }
            return {"type": type_map.get(ft.base_type, "string")}


@dataclass
class Prompt:
    """
    A named, reusable prompt with typed inputs and outputs.
    
    Prompts are first-class constructs in N3 that encapsulate:
    - Model selection
    - Template/message content
    - Input schema (what data is needed)
    - Output schema (what structure to expect)
    - Parameters (temperature, max_tokens, etc.)
    - Effects tracking (pure vs side-effecting)
    
    Example DSL:
        prompt classify_ticket {
            model: gpt4
            args: {
                ticket_text: string,
                customer_tier: enum["free", "premium", "enterprise"]
            }
            template: "Classify this support ticket: {{ticket_text}}"
            output_schema: {
                category: enum["billing", "technical", "account"],
                urgency: enum["low", "medium", "high"],
                needs_handoff: bool
            }
            parameters: {
                temperature: 0.3,
                max_tokens: 150
            }
        }
    """
    name: str
    model: str
    template: str
    input_fields: List[PromptField] = field(default_factory=list)
    output_fields: List[PromptField] = field(default_factory=list)
    args: List[PromptArgument] = field(default_factory=list)
    output_schema: Optional[OutputSchema] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    effects: Set[str] = field(default_factory=set)


__all__ = [
    "PromptField",
    "PromptArgument",
    "EnumType",
    "OutputFieldType",
    "OutputField",
    "OutputSchema",
    "Prompt",
]
