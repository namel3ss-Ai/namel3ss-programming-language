"""
Template definitions for reusable prompt templates.

Templates provide a way to define named, reusable prompt strings
that can be referenced throughout the N3 program.

This module provides production-grade template definitions with validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Template:
    """
    Reusable prompt template definition.
    
    Templates allow defining named prompt strings that can be
    referenced by prompts and chains, promoting reusability and
    consistency across an N3 application.
    
    Templates support variable interpolation using {{variable}} syntax
    and can be composed together to build complex prompts.
    
    Attributes:
        name: Unique identifier for this template
        prompt: Template string with {{variable}} placeholders
        metadata: Additional metadata (category, version, author, etc.)
        
    Example DSL:
        define template summarize_text {
            prompt: "Summarize the following text concisely: {{text}}"
            metadata: {
                category: "summarization",
                max_length: "short"
            }
        }
        
        define template classify_sentiment {
            prompt: \"\"\"
            Analyze the sentiment of the following text:
            
            Text: {{text}}
            
            Classify as: positive, negative, or neutral
            Provide a confidence score from 0 to 1.
            \"\"\"
            metadata: {
                category: "classification",
                output_format: "json"
            }
        }
        
        define template system_prompt {
            prompt: \"\"\"
            You are a helpful AI assistant that:
            - Provides accurate, factual information
            - Admits when uncertain
            - Avoids harmful or biased responses
            - Respects user privacy
            \"\"\"
            metadata: {
                type: "system",
                version: "1.0"
            }
        }
        
        define template few_shot_example {
            prompt: \"\"\"
            Example input: {{example_input}}
            Example output: {{example_output}}
            
            Now, for the actual input:
            Input: {{actual_input}}
            Output:
            \"\"\"
        }
    
    Variable Interpolation:
        - Use {{variable_name}} for simple substitution
        - Variables are resolved at prompt execution time
        - Missing variables raise runtime errors (fail-fast)
        - Complex logic should be in prompt code, not templates
        
    Template Composition:
        Templates can reference other templates:
        
        define template base_instruction {
            prompt: "You are an expert {{domain}} assistant."
        }
        
        define template specialized {
            prompt: "{{base_instruction}} Focus on {{specific_task}}."
        }
        
    Validation:
        Use validate_template() from .validation to ensure configuration
        is valid before runtime usage.
        
    Best Practices:
        - Keep templates focused and single-purpose
        - Use metadata to categorize and document templates
        - Prefer templates over string concatenation in code
        - Version templates when making breaking changes
        - Test templates with various input combinations
        - Avoid business logic in templates (keep them declarative)
        
    Notes:
        - Templates are compiled at runtime for efficiency
        - Variable names are case-sensitive
        - Whitespace in templates is preserved (use triple quotes for multiline)
        - Templates are immutable once defined (functional style)
    """
    name: str
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["Template"]
