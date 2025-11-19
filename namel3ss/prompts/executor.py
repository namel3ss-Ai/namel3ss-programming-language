"""
Structured prompt execution - integrating PromptProgram, LLM providers, and validation.

This module provides high-level functions for executing structured prompts
with automatic provider selection, validation, and error handling.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass

from namel3ss.ast import Prompt
from namel3ss.llm.base import BaseLLM, LLMError
from namel3ss.prompts.runtime import PromptProgram, create_prompt_program
from namel3ss.prompts.validator import OutputValidator, ValidationError
from namel3ss.errors import N3Error


class StructuredPromptError(N3Error):
    """Errors during structured prompt execution."""
    pass


@dataclass
class StructuredPromptResult:
    """Result of executing a structured prompt."""
    
    output: Dict[str, Any]
    """Validated output matching the schema"""
    
    raw_response: str
    """Raw LLM response text"""
    
    latency_ms: float
    """Execution time in milliseconds"""
    
    prompt_tokens: int = 0
    """Number of tokens in the prompt"""
    
    completion_tokens: int = 0
    """Number of tokens in the completion"""
    
    model: str = ""
    """Model that generated the output"""
    
    provider: str = ""
    """Provider that was used"""
    
    used_structured_mode: bool = False
    """Whether provider's native structured mode was used"""
    
    validation_passed: bool = True
    """Whether validation succeeded on first try"""


async def execute_structured_prompt(
    prompt_def: Prompt,
    llm: BaseLLM,
    args: Dict[str, Any],
    *,
    retry_on_validation_error: bool = False,
    max_retries: int = 1,
) -> StructuredPromptResult:
    """
    Execute a structured prompt with the given arguments.
    
    This is the main entry point for structured prompt execution. It:
    1. Creates a PromptProgram from the definition
    2. Validates and renders arguments
    3. Calls the LLM (using structured mode if available)
    4. Parses and validates the output
    5. Returns validated result or raises error
    
    Args:
        prompt_def: Prompt AST definition with args and output_schema
        llm: LLM provider instance
        args: Argument values for the prompt
        retry_on_validation_error: Whether to retry if validation fails
        max_retries: Maximum number of retries (default: 1)
    
    Returns:
        StructuredPromptResult with validated output and metadata
    
    Raises:
        StructuredPromptError: If execution fails
        ValidationError: If validation fails and retry is disabled
    """
    # Create prompt program
    program = create_prompt_program(prompt_def)
    
    # Check if prompt has structured output
    if not program.has_structured_output():
        raise StructuredPromptError(
            f"Prompt '{prompt_def.name}' does not have an output_schema defined"
        )
    
    # Render prompt with arguments
    try:
        rendered_prompt = program.render_prompt(args)
    except Exception as e:
        raise StructuredPromptError(
            f"Failed to render prompt '{prompt_def.name}': {e}"
        ) from e
    
    # Get output schema
    output_schema = program.get_output_schema()
    if not output_schema:
        raise StructuredPromptError(
            f"Prompt '{prompt_def.name}' has no output schema"
        )
    
    # Create validator
    validator = OutputValidator(prompt_def.output_schema)  # type: ignore
    
    # Execute with retries
    attempts = 0
    last_error: Optional[Exception] = None
    
    while attempts <= max_retries:
        attempts += 1
        
        try:
            start_time = time.time()
            
            # Call LLM with structured output if supported
            if llm.supports_structured_output():
                llm_response = llm.generate_structured(
                    rendered_prompt,
                    output_schema
                )
                used_structured_mode = True
            else:
                # Fallback to regular generation with format instructions
                llm_response = llm.generate(rendered_prompt)
                used_structured_mode = False
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse JSON response
            raw_text = llm_response.text.strip()
            
            # Try to extract JSON if wrapped in markdown
            if raw_text.startswith('```'):
                # Remove markdown code block markers
                lines = raw_text.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                raw_text = '\n'.join(lines).strip()
            
            # Validate output
            validation_result = validator.validate(raw_text)
            
            if validation_result.valid:
                # Success!
                return StructuredPromptResult(
                    output=validation_result.validated_output,  # type: ignore
                    raw_response=llm_response.text,
                    latency_ms=latency_ms,
                    prompt_tokens=llm_response.prompt_tokens,
                    completion_tokens=llm_response.completion_tokens,
                    model=llm_response.model,
                    provider=llm.get_provider_name(),
                    used_structured_mode=used_structured_mode,
                    validation_passed=attempts == 1
                )
            else:
                # Validation failed
                last_error = validation_result.errors[0]
                
                if not retry_on_validation_error or attempts > max_retries:
                    # Don't retry or max retries reached
                    raise last_error
                
                # Retry with stricter instructions
                error_details = "; ".join(str(e) for e in validation_result.errors[:3])
                rendered_prompt = f"{rendered_prompt}\n\nIMPORTANT: Previous attempt had errors: {error_details}\nPlease correct these issues."
        
        except (ValidationError, json.JSONDecodeError) as e:
            last_error = e
            if not retry_on_validation_error or attempts > max_retries:
                raise StructuredPromptError(
                    f"Prompt '{prompt_def.name}' validation failed: {e}"
                ) from e
        
        except LLMError as e:
            # Don't retry LLM errors
            raise StructuredPromptError(
                f"LLM error executing prompt '{prompt_def.name}': {e}"
            ) from e
        
        except Exception as e:
            raise StructuredPromptError(
                f"Unexpected error executing prompt '{prompt_def.name}': {e}"
            ) from e
    
    # Should not reach here, but just in case
    raise StructuredPromptError(
        f"Failed to execute prompt '{prompt_def.name}' after {attempts} attempts: {last_error}"
    )


def execute_structured_prompt_sync(
    prompt_def: Prompt,
    llm: BaseLLM,
    args: Dict[str, Any],
    **kwargs: Any,
) -> StructuredPromptResult:
    """
    Synchronous wrapper for execute_structured_prompt.
    
    Args:
        prompt_def: Prompt definition
        llm: LLM provider
        args: Argument values
        **kwargs: Additional options (retry_on_validation_error, max_retries)
    
    Returns:
        StructuredPromptResult
    """
    import asyncio
    
    # Run async function in event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        execute_structured_prompt(prompt_def, llm, args, **kwargs)
    )


__all__ = [
    "StructuredPromptError",
    "StructuredPromptResult",
    "execute_structured_prompt",
    "execute_structured_prompt_sync",
]
