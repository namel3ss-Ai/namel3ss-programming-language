"""Prompt definition parsing for structured AI interactions."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from namel3ss.ast import Prompt, PromptArgument, PromptField, OutputSchema

if TYPE_CHECKING:
    from ..base import ParserBase


class PromptsParserMixin:
    """Mixin for parsing structured prompt definitions."""
    
    def _parse_prompt(self: 'ParserBase', line: str, line_no: int, base_indent: int) -> Prompt:
        """
        Parse structured prompt definitions with input/output schemas.
        
        Prompts define AI interactions with strong typing for inputs and
        outputs, supporting both legacy and modern schema syntaxes.
        
        Supported Syntaxes:
            1. Modern (Recommended):
                prompt "Name":
                    model: "GPT4"
                    system: "You are a helpful assistant"
                    user: "Process {input}"
                    output:
                        response: text required
                        confidence: float
            
            2. Legacy (Backward Compatible):
                prompt "Name":
                    model: "GPT4"
                    instructions: "You are helpful"
                    input:
                        query: text required
                    template: "Process {query}"
                    output schema:
                        answer: string required
                        score: number
        
        Features:
            - Input validation with typed arguments
            - Output schema for structured responses
            - Template with variable interpolation
            - Multi-role prompts (system, user, assistant)
            - Few-shot examples
            - Temperature and parameter control
            - Safety policies and evaluation
        """
        stripped = line.strip()
        
        # Strip : or { for backward compatibility
        if stripped.endswith(":") or stripped.endswith("{"):
            stripped = stripped[:-1]
        
        # Try quoted name first: prompt "Name"
        match = re.match(r'(?:define\s+)?prompt\s+"([^"]+)"', stripped, flags=re.IGNORECASE)
        if not match:
            # Try unquoted identifier: prompt name
            match = re.match(r'(?:define\s+)?prompt\s+([A-Za-z_][A-Za-z0-9_]*)', stripped, flags=re.IGNORECASE)
        
        if not match:
            raise self._error(
                'Expected: prompt "Name" or prompt name',
                line_no,
                line
            )
        name = match.group(1)
        config_block = self._parse_kv_block(base_indent)
        description_raw = config_block.pop("description", config_block.pop("desc", None))
        description = str(description_raw) if description_raw is not None else None
        user_raw = None
        model_ref = config_block.pop("model", None)
        if model_ref is None:
            # Handle "using model Foo" style keys captured by kv parser
            for key in list(config_block.keys()):
                if key.lower().startswith("using model"):
                    model_ref = key.split("using model", 1)[1].strip().strip('"').strip("'")
                    value = config_block.pop(key, None)
                    if user_raw is None and value:
                        user_raw = value
                    break
        system_raw = config_block.pop("system", config_block.pop("system_prompt", config_block.pop("instructions", None)))
        system_prompt = str(system_raw) if system_raw is not None else None
        template_candidate = config_block.pop("user", config_block.pop("user_prompt", config_block.pop("template", None)))
        if template_candidate is not None:
            user_raw = template_candidate
        user_prompt = str(user_raw) if user_raw is not None else None
        assistant_raw = config_block.pop("assistant", config_block.pop("assistant_prompt", None))
        assistant_prompt = str(assistant_raw) if assistant_raw is not None else None
        input_raw = config_block.pop("input", config_block.pop("input_schema", config_block.pop("args", None)))
        arguments: List[PromptArgument] = []
        if input_raw is not None:
            if isinstance(input_raw, list):
                for item in input_raw:
                    if isinstance(item, dict):
                        arguments.extend(self._parse_prompt_args_dict(item))
                    else:
                        raise self._error(
                            "Prompt input list elements must be dicts",
                            line_no,
                            line,
                            hint='Use input: with field definitions like name: type'
                        )
            elif isinstance(input_raw, dict):
                arguments = self._parse_prompt_args_dict(input_raw)
            else:
                raise self._error(
                    "Prompt input must be a dict or list of dicts",
                    line_no,
                    line,
                    hint='Define input fields with their types: field_name: text required'
                )
        output_raw = config_block.pop("output", config_block.pop("output_schema", config_block.pop("response", None)))
        output_schema: Optional[OutputSchema] = None
        if output_raw is not None:
            if isinstance(output_raw, dict):
                fields = self._parse_output_schema(output_raw)
                output_schema = OutputSchema(fields=fields)
            elif isinstance(output_raw, list):
                all_fields: List[PromptField] = []
                for entry in output_raw:
                    if isinstance(entry, dict):
                        all_fields.extend(self._parse_output_schema(entry))
                    else:
                        raise self._error(
                            "Prompt output list elements must be dicts",
                            line_no,
                            line,
                            hint='Use output: with field definitions like result: text required'
                        )
                output_schema = OutputSchema(fields=all_fields)
            else:
                raise self._error(
                    "Prompt output must be a dict or list of dicts",
                    line_no,
                    line,
                    hint='Define output fields with their types: field_name: type'
                )
        examples_raw = config_block.pop("examples", config_block.pop("few_shot", None))
        examples: List[Dict[str, Any]] = []
        if examples_raw is not None:
            if isinstance(examples_raw, list):
                examples = [self._transform_config(ex) for ex in examples_raw if isinstance(ex, dict)]
            else:
                raise self._error(
                    "Prompt examples must be a list of dicts",
                    line_no,
                    line,
                    hint='Use examples: [{input: {...}, output: {...}}]'
                )
        safety_policy = config_block.pop("safety_policy", config_block.pop("safety", None))
        if safety_policy is not None:
            safety_policy = str(safety_policy)
        evaluators_raw = config_block.pop("evaluators", config_block.pop("eval", None))
        evaluators: List[str] = []
        if evaluators_raw is not None:
            if isinstance(evaluators_raw, list):
                evaluators = [str(ev) for ev in evaluators_raw]
            elif isinstance(evaluators_raw, str):
                evaluators = [evaluators_raw]
        temperature_raw = config_block.pop("temperature", None)
        temperature: Optional[float] = None
        if temperature_raw is not None:
            try:
                temperature = float(temperature_raw)
            except (ValueError, TypeError):
                raise self._error(
                    f"Prompt temperature must be a number, got {temperature_raw}",
                    line_no,
                    line,
                    hint='Use temperature: 0.7 for controlled randomness'
                )
        max_tokens_raw = config_block.pop("max_tokens", config_block.pop("max_length", None))
        max_tokens: Optional[int] = None
        if max_tokens_raw is not None:
            max_tokens = self._coerce_int(max_tokens_raw)
        metadata_raw = config_block.pop("metadata", {})
        metadata = self._coerce_options_dict(metadata_raw)
        prompt_config: Dict[str, Any] = {}
        for key, val in config_block.items():
            if key not in {
                "model", "system", "user", "assistant", "input", "output",
                "examples", "safety_policy", "evaluators", "temperature",
                "max_tokens", "description", "metadata"
            }:
                prompt_config[key] = self._transform_config(val)
        model_name = str(model_ref) if model_ref is not None else None
        
        # Build template from available prompt components
        template_parts = []
        if system_prompt:
            template_parts.append(("System", system_prompt))
        if user_prompt:
            template_parts.append(("User", user_prompt))
        if assistant_prompt:
            template_parts.append(("Assistant", assistant_prompt))
        
        if len(template_parts) == 1:
            # Preserve the single provided template verbatim (matches legacy expectations)
            _, template = template_parts[0]
        else:
            # When multiple roles are provided, keep lightweight role labels
            template = "\n\n".join(f"{label}: {text}" for label, text in template_parts)
        
        input_fields = [
            PromptField(name=arg.name, field_type=arg.arg_type, required=arg.required)
            for arg in arguments
        ]
        output_fields = output_schema.fields if output_schema else []
        
        return Prompt(
            name=name,
            description=description,
            model=model_name,
            template=template,
            input_fields=input_fields,
            output_fields=output_fields,
            args=arguments if arguments else [],
            output_schema=output_schema,
            metadata=metadata,
        )
