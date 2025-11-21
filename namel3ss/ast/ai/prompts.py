"""
Prompt definitions with typed inputs, outputs, and structured schemas.

This module contains all prompt-related AST nodes for building production-grade
prompts with strong typing, validation, and structured outputs.

Components:
- PromptField: Schema fields for inputs/outputs
- PromptArgument: Typed arguments for parameterized prompts
- EnumType: Enum type constraints
- OutputFieldType: Type specifications for structured outputs
- OutputField: Individual output schema fields
- OutputSchema: Complete structured output schema with JSON Schema conversion
- Prompt: Main prompt definition with model, template, and schema

This module provides production-grade prompt definitions with validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..source_location import SourceLocation, Set


@dataclass
class PromptField:
    """
    Structured schema information for prompt inputs/outputs.
    
    PromptField defines the shape of data flowing into or out of a prompt,
    including type information, validation rules, and documentation.
    
    Attributes:
        name: Field name (must be valid Python identifier)
        field_type: Type of the field ("text", "number", "bool", "list", etc.)
        required: Whether this field must be provided
        description: Human-readable description of the field's purpose
        default: Default value if field is not required
        enum: List of allowed values (for categorical fields)
        metadata: Additional field metadata
    
    Example:
        input_field: {
            name: "user_query",
            type: "text",
            required: true,
            description: "The user's question or request"
        }
        
    Validation:
        Use validate_prompt_field() from .validation module.
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
    arguments, enabling reusable prompts with different inputs. Arguments
    are checked at compile-time when possible.
    
    Attributes:
        name: Argument name (must be valid Python identifier)
        arg_type: Type of argument ("string", "int", "float", "bool", "list", "object")
        required: Whether this argument must be provided
        default: Default value if argument is optional
        description: Human-readable description of the argument
        location: Source location for error reporting
    
    Example DSL:
        prompt summarize {
            args: {
                text: string,
                max_length: int = 100,
                style: enum["concise", "detailed"] = "concise"
            }
            template: "Summarize the following text in {max_length} words ({style} style): {text}"
        }
        
    Valid Types:
        - string: Text data
        - int: Integer numbers
        - float: Floating-point numbers
        - bool: Boolean true/false
        - list: Array of values
        - object: Structured object/dict
        
    Validation:
        Use validate_prompt_argument() from .validation module.
    """
    name: str
    arg_type: str  # string, int, float, bool, list, object
    required: bool = True
    default: Any = None
    description: Optional[str] = None
    location: Optional['SourceLocation'] = None


@dataclass
class EnumType:
    """
    Enum type constraint for output schema fields.
    
    Represents a fixed set of allowed string values for a field,
    enabling the LLM to return structured categorical data. This
    improves reliability by constraining LLM outputs to known values.
    
    Attributes:
        values: List of allowed string values (must be non-empty)
    
    Example:
        enum["billing", "technical", "account", "other"]
        enum["low", "medium", "high", "critical"]
        enum["yes", "no", "unknown"]
        
    Usage in Output Schema:
        output_schema: {
            category: enum["bug", "feature", "question"],
            priority: enum["p0", "p1", "p2", "p3"]
        }
    """
    values: List[str]
    
    def __str__(self) -> str:
        """String representation showing enum values."""
        vals = ', '.join(f'"{v}"' for v in self.values)
        return f"enum[{vals}]"


@dataclass
class OutputFieldType:
    """
    Type specification for an output schema field.
    
    Supports rich type definitions including primitives, collections,
    nested objects, enums, and nullable types. Types are converted
    to JSON Schema for LLM provider APIs.
    
    Attributes:
        base_type: Base type ("string", "int", "float", "bool", "list", "object", "enum")
        element_type: Element type for list[T] types
        enum_values: Allowed values for enum types
        nested_fields: Field definitions for object types
        nullable: Whether the field can be null/None
    
    Example Types:
        string                              # Simple string
        int                                 # Integer number
        float?                              # Optional/nullable float
        list[string]                        # Array of strings
        enum["a", "b", "c"]                # Categorical value
        object{name: string, age: int}     # Nested object
        list[enum["red", "green", "blue"]] # Array of enums
        
    Type System:
        The type system is designed to be expressive enough for LLM
        structured outputs while remaining simple and predictable.
        
    Validation:
        Types are validated at parse time. Invalid type specifications
        raise syntax errors during compilation.
    """
    base_type: str  # string, int, float, bool, list, object, enum
    element_type: Optional['OutputFieldType'] = None  # For list[T]
    enum_values: Optional[List[str]] = None  # For enum types
    nested_fields: Optional[List['OutputField']] = None  # For object types
    nullable: bool = False
    
    def __str__(self) -> str:
        """String representation of the type."""
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
    defining one component of the structured output expected from an LLM.
    
    Attributes:
        name: Field name (must be valid JSON key)
        field_type: Type specification for this field
        required: Whether this field must be present in output
        description: Description to include in JSON Schema for the LLM
        location: Source location for error reporting
    
    Example:
        category: enum["billing", "technical"]
        confidence: float
        tags: list[string]
        details: object{reason: string, urgency: int}
        
    Validation:
        Use validate_output_field() from .validation module.
    """
    name: str
    field_type: OutputFieldType
    required: bool = True
    description: Optional[str] = None
    location: Optional['SourceLocation'] = None


@dataclass
class OutputSchema:
    """
    Structured output schema for a prompt.
    
    Defines the complete structure of data expected from an LLM,
    which is converted to JSON Schema and sent to provider APIs.
    This enables reliable structured data extraction from LLMs.
    
    Attributes:
        fields: List of output fields defining the schema structure
    
    Example DSL:
        output_schema: {
            category: enum["billing", "technical", "account"],
            urgency: enum["low", "medium", "high"],
            needs_handoff: bool,
            confidence: float,
            suggested_responses: list[string]
        }
        
    JSON Schema Conversion:
        The schema is automatically converted to JSON Schema format
        compatible with:
        - OpenAI Function Calling
        - OpenAI Structured Outputs
        - Anthropic Tool Use
        - Google Function Calling
        
    Provider Support:
        - OpenAI: Full support via JSON mode or function calling
        - Anthropic: Supported via tool use API
        - Google: Supported via function declarations
        - Azure OpenAI: Full support (same as OpenAI)
        - Local models: Best-effort (depends on model capabilities)
        
    Validation:
        Use validate_output_schema() from .validation module.
        
    Notes:
        - All fields should have clear descriptions for best LLM performance
        - Use enums for categorical outputs when possible
        - Keep schemas simple - complex nesting reduces reliability
        - Test schemas with example inputs/outputs
    """
    fields: List[OutputField]
    
    def to_json_schema(self) -> Dict[str, Any]:
        """
        Convert to JSON Schema format for LLM providers.
        
        Returns a JSON Schema object that can be sent to LLM APIs
        supporting structured outputs (e.g., OpenAI function calling,
        Anthropic tool use).
        
        Returns:
            JSON Schema dict conforming to JSON Schema Draft 7
            
        Example output:
            {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["billing", "technical"],
                        "description": "Ticket category"
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["low", "medium", "high"]
                    },
                    "needs_handoff": {
                        "type": "boolean"
                    }
                },
                "required": ["category", "urgency", "needs_handoff"],
                "additionalProperties": false
            }
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
        """
        Convert OutputFieldType to JSON Schema type definition.
        
        Args:
            ft: OutputFieldType to convert
            
        Returns:
            JSON Schema type definition dict
        """
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
    - Model selection (which LLM to use)
    - Template/message content (what to ask the LLM)
    - Input schema (what data is needed)
    - Output schema (what structure to expect)
    - Parameters (temperature, max_tokens, etc.)
    - Effects tracking (pure vs side-effecting)
    
    Prompts are compiled into provider-specific API calls at runtime
    with automatic retries, error handling, and telemetry.
    
    Attributes:
        name: Unique identifier for this prompt
        model: Reference to an AIModel definition
        template: Prompt template string with {{variable}} placeholders
        input_fields: Input schema fields (legacy, prefer args)
        output_fields: Output schema fields (legacy, prefer output_schema)
        args: Typed arguments for the prompt
        output_schema: Structured output schema
        parameters: Model parameters (temperature, max_tokens, etc.)
        metadata: Additional metadata
        description: Human-readable description
        effects: Set of effects this prompt may have (io, stateful, etc.)
    
    Example DSL:
        prompt classify_ticket {
            model: gpt4
            
            args: {
                ticket_text: string,
                customer_tier: enum["free", "premium", "enterprise"]
            }
            
            template: \"\"\"
            Classify this support ticket:
            
            Ticket: {{ticket_text}}
            Customer tier: {{customer_tier}}
            
            Analyze the ticket and provide structured output.
            \"\"\"
            
            output_schema: {
                category: enum["billing", "technical", "account", "other"],
                urgency: enum["low", "medium", "high", "critical"],
                needs_handoff: bool,
                confidence: float,
                reasoning: string
            }
            
            parameters: {
                temperature: 0.3,
                max_tokens: 250
            }
            
            description: "Classifies support tickets for routing"
        }
        
    Advanced Example with Few-Shot:
        prompt extract_entities {
            model: claude
            
            args: {
                text: string
            }
            
            template: \"\"\"
            Extract named entities from the following text.
            
            Examples:
            Text: "John Smith works at Acme Corp in New York"
            Output: {
                "people": ["John Smith"],
                "organizations": ["Acme Corp"],
                "locations": ["New York"]
            }
            
            Text: "The CEO met with investors in San Francisco"
            Output: {
                "people": ["CEO"],
                "organizations": ["investors"],
                "locations": ["San Francisco"]
            }
            
            Now extract from:
            Text: {{text}}
            Output:
            \"\"\"
            
            output_schema: {
                people: list[string],
                organizations: list[string],
                locations: list[string]
            }
        }
        
    Validation:
        Use validate_prompt() from .validation module to ensure:
        - Name is non-empty
        - Model reference exists
        - Template is non-empty
        - Arguments are valid
        - Output schema is well-formed
        
    Runtime Behavior:
        - Templates are rendered with provided arguments
        - Missing required arguments raise runtime errors
        - Output is validated against schema if provided
        - Retries are automatic for transient failures
        - Telemetry tracks latency, tokens, and cost
        
    Best Practices:
        - Use output schemas for structured extraction
        - Set appropriate temperature (0.0-0.3 for factual, 0.7-1.0 for creative)
        - Include examples in template for complex tasks
        - Keep templates focused on single tasks
        - Version prompts when making significant changes
        - Test with diverse inputs
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
    effects: set = field(default_factory=set)


__all__ = [
    "PromptField",
    "PromptArgument",
    "EnumType",
    "OutputFieldType",
    "OutputField",
    "OutputSchema",
    "Prompt",
]
