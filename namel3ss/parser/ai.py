from __future__ import annotations

import ast as py_ast
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from namel3ss.ast import AIModel, Chain, ChainStep, Connector, ContextValue, Prompt, PromptField, Template

from .base import ParserBase


_AI_MODEL_PROVIDER_HINTS = {
    "openai",
    "anthropic",
    "azure",
    "azure-openai",
    "azure_openai",
    "google",
    "vertex",
    "bedrock",
    "mistral",
    "cohere",
    "ollama",
}

_GENERAL_MODEL_BLOCK_HINTS = (
    "from ",
    "target:",
    "features:",
    "framework:",
    "objective:",
    "loss:",
    "optimizer:",
    "batch",
    "epochs:",
    "learning rate",
    "learning_rate",
    "datasets:",
    "transform ",
    "hyperparameters:",
    "training metadata:",
    "feature ",
    "monitoring:",
    "serving:",
    "deployments:",
    "task:",
    "tags:",
    "registry:",
)

_AI_MODEL_BLOCK_HINTS = (
    "provider",
    "model",
    "name:",
    "deployment",
    "endpoint",
    "api_",
    "base_url",
    "temperature",
    "top_p",
    "max_tokens",
    "metadata:",
    "headers",
    "params",
)


class AIParserMixin(ParserBase):
    """Parse AI-centric constructs such as connectors, templates, and chains."""

    def _parse_connector(self, line: str, line_no: int, base_indent: int) -> Connector:
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'connector\s+"([^"]+)"\s+type\s+([\w\.\-]+)', stripped)
        if not match:
            raise self._error('Expected: connector "Name" type KIND:', line_no, line)
        name = match.group(1)
        connector_type = match.group(2)
        config: Dict[str, Any] = {}
        description: Any = None
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped_line = nxt.strip()
            if not stripped_line or stripped_line.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            assign = re.match(r'([\w\.\- ]+)\s*(=|:)\s*(.*)$', stripped_line)
            if not assign:
                raise self._error(
                    "Expected 'key = value' inside connector block",
                    self.pos + 1,
                    nxt,
                )
            key = assign.group(1).strip()
            remainder = assign.group(3)
            self._advance()
            if remainder == "":
                nested = self._parse_kv_block(indent)
                value = self._transform_config(nested)
            else:
                value = self._transform_config(self._coerce_scalar(remainder))
            if key.lower() == "description":
                description = value
            else:
                config[key] = value
        description_text = None if description is None else str(description)
        return Connector(name=name, connector_type=connector_type, config=config, description=description_text)

    def _parse_template(self, line: str, line_no: int, base_indent: int) -> Template:
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'define\s+template\s+"([^"]+)"', stripped)
        if not match:
            raise self._error('Expected: define template "Name":', line_no, line)
        name = match.group(1)
        prompt: Any = None
        metadata: Dict[str, Any] = {}
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped_line = nxt.strip()
            if not stripped_line or stripped_line.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            assign = re.match(r'([\w\.\- ]+)\s*(=|:)\s*(.*)$', stripped_line)
            if not assign:
                raise self._error(
                    "Expected 'key = value' inside template block",
                    self.pos + 1,
                    nxt,
                )
            key = assign.group(1).strip()
            remainder = assign.group(3)
            self._advance()
            if remainder == "":
                value = self._transform_config(self._parse_kv_block(indent))
            else:
                value = self._transform_config(self._coerce_scalar(remainder))
            if key.lower() == "prompt":
                prompt = value
            else:
                metadata[key] = value
        if prompt is None:
            prompt = ""
        prompt_text = prompt if isinstance(prompt, str) else str(prompt)
        return Template(name=name, prompt=prompt_text, metadata=metadata)

    def _parse_chain(self, line: str, line_no: int, base_indent: int) -> Chain:
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'define\s+chain\s+"([^"]+)"', stripped)
        if not match:
            raise self._error('Expected: define chain "Name":', line_no, line)
        name = match.group(1)
        input_key = "input"
        steps: List[ChainStep] = []
        metadata: Dict[str, Any] = {}
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped_line = nxt.strip()
            if not stripped_line or stripped_line.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            if '->' in stripped_line:
                pipeline = [segment.strip() for segment in stripped_line.split('->') if segment.strip()]
                self._advance()
                for index, segment in enumerate(pipeline):
                    if index == 0 and segment.lower().startswith('input'):
                        parts = segment.split()
                        if len(parts) > 1:
                            input_key = parts[1]
                        else:
                            input_key = "input"
                        continue
                    tokens = segment.split()
                    if not tokens:
                        continue
                    kind = tokens[0].lower()
                    target = tokens[1] if len(tokens) > 1 else tokens[0]
                    options: Dict[str, Any] = {}
                    if kind == 'python' and len(tokens) > 1:
                        module_spec = tokens[1]
                        module_name, _, method_name = module_spec.partition(':')
                        target = module_name
                        if module_name:
                            options['module'] = module_name
                        if method_name:
                            options['method'] = method_name
                    steps.append(ChainStep(kind=kind, target=target, options=options))
            else:
                assign = re.match(r'([\w\.\- ]+)\s*(=|:)\s*(.*)$', stripped_line)
                if not assign:
                    raise self._error(
                        "Expected chain pipeline or metadata assignment",
                        self.pos + 1,
                        nxt,
                    )
                key = assign.group(1).strip()
                remainder = assign.group(3)
                self._advance()
                if remainder == "":
                    value = self._transform_config(self._parse_kv_block(indent))
                else:
                    value = self._transform_config(self._coerce_scalar(remainder))
                metadata[key] = value
        return Chain(name=name, input_key=input_key, steps=steps, metadata=metadata)

    def _parse_ai_model(self, line: str, line_no: int, base_indent: int) -> AIModel:
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'(?:ai\s+)?model\s+"([^"]+)"(?:\s+using\s+([\w\.\-]+))?', stripped, flags=re.IGNORECASE)
        if not match:
            raise self._error('Expected: model "Name" using PROVIDER', line_no, line)
        name = match.group(1)
        provider = match.group(2) or "custom"
        config_block = self._parse_kv_block(base_indent)
        config = self._transform_config(config_block)
        description_raw = config.pop('description', config.pop('desc', None))
        description = str(description_raw) if description_raw is not None else None
        provider_override = config.pop('provider', None)
        if provider_override:
            provider = str(provider_override)
        metadata_raw = config.pop('metadata', {})
        metadata = self._coerce_options_dict(metadata_raw)
        model_id_raw = config.pop('model', config.pop('name', None))
        if model_id_raw is None:
            raise self._error("AI model block must define 'model:' (provider model id)", line_no, line)
        model_id = self._stringify_value(model_id_raw)
        return AIModel(
            name=name,
            provider=str(provider),
            model_name=str(model_id),
            config=config,
            description=description,
            metadata=metadata,
        )

    def _parse_prompt(self, line: str, line_no: int, base_indent: int) -> Prompt:
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'prompt\s+"([^"]+)"', stripped, flags=re.IGNORECASE)
        if not match:
            raise self._error('Expected: prompt "Name":', line_no, line)
        name = match.group(1)
        input_fields: List[PromptField] = []
        output_fields: List[PromptField] = []
        parameters: Dict[str, Any] = {}
        metadata: Dict[str, Any] = {}
        template_text: Optional[str] = None
        model_name: Optional[str] = None
        description: Optional[str] = None

        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped_line = nxt.strip()
            if not stripped_line or stripped_line.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            lowered = stripped_line.lower()
            if lowered.startswith('input'):
                self._advance()
                schema = self._parse_prompt_schema_block(indent)
                input_fields.extend(schema)
            elif lowered.startswith('output'):
                self._advance()
                schema = self._parse_prompt_schema_block(indent)
                output_fields.extend(schema)
            elif lowered.startswith('metadata:'):
                self._advance()
                metadata_value = self._transform_config(self._parse_kv_block(indent))
                if isinstance(metadata_value, dict):
                    metadata.update(metadata_value)
                else:
                    metadata["value"] = metadata_value
            elif lowered.startswith('parameters:') or lowered.startswith('settings:'):
                self._advance()
                params_value = self._transform_config(self._parse_kv_block(indent))
                if isinstance(params_value, dict):
                    parameters.update(params_value)
                else:
                    parameters["value"] = params_value
            elif lowered.startswith('description:'):
                self._advance()
                desc_raw = stripped_line[len('description:'):].strip()
                description = self._stringify_value(self._coerce_scalar(desc_raw)) if desc_raw else None
            elif lowered.startswith('using model'):
                match_model = re.match(r'using\s+model\s+"([^"]+)"\s*:?\s*(.*)$', stripped_line, flags=re.IGNORECASE)
                if not match_model:
                    raise self._error('Expected: using model "Name":', self.pos + 1, nxt)
                model_name = match_model.group(1)
                inline_text = match_model.group(2).strip()
                self._advance()
                if inline_text:
                    template_text = self._stringify_value(self._coerce_scalar(inline_text))
                else:
                    template_text = self._parse_prompt_template_block(indent)
            else:
                assign = re.match(r'([\w\.\-]+)\s*(=|:)\s*(.*)$', stripped_line)
                if not assign:
                    raise self._error("Unknown directive inside prompt block", self.pos + 1, nxt)
                key = assign.group(1).strip()
                remainder = assign.group(3)
                self._advance()
                value = self._coerce_scalar(remainder)
                parameters[key] = value

        if not input_fields:
            raise self._error(f"Prompt '{name}' must define an input schema", line_no, line)
        if not output_fields:
            raise self._error(f"Prompt '{name}' must define an output schema", line_no, line)
        if model_name is None:
            raise self._error(f"Prompt '{name}' must specify 'using model \"Name\"'", line_no, line)
        if not template_text:
            raise self._error(f"Prompt '{name}' must include a template body", line_no, line)

        parameters_transformed = self._transform_config(parameters)
        parameters_payload = parameters_transformed if isinstance(parameters_transformed, dict) else {"value": parameters_transformed}

        return Prompt(
            name=name,
            model=model_name,
            template=template_text,
            input_fields=input_fields,
            output_fields=output_fields,
            parameters=parameters_payload,
            metadata=metadata,
            description=description,
        )

    def _transform_config(self, value: Any) -> Any:
        if isinstance(value, str):
            reference = self._parse_context_reference(value)
            if reference is not None:
                return reference
            if value.startswith('env.'):
                path = value[4:]
                return self._build_context_value('env', path)
            if value.startswith('ctx.'):
                path = value[4:]
                return self._build_context_value('ctx', path)
            return value
        if isinstance(value, dict):
            return {key: self._transform_config(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._transform_config(item) for item in value]
        return value

    def _build_context_value(self, scope: str, path_text: str) -> ContextValue:
        parts = [segment for segment in path_text.split('.') if segment]
        if not parts:
            return ContextValue(scope=scope, path=[])
        return ContextValue(scope=scope, path=parts)

    def _parse_prompt_schema_block(self, parent_indent: int) -> List[PromptField]:
        fields: List[PromptField] = []
        while self.pos < len(self.lines):
            line = self._peek()
            if line is None:
                break
            indent = self._indent(line)
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            name_part, _, remainder = stripped.partition(':')
            field_name = self._strip_quotes(name_part.strip())
            field_type_text = remainder.strip() or "text"
            self._advance()
            config: Dict[str, Any] = {}
            nxt = self._peek()
            if nxt is not None and self._indent(nxt) > indent:
                config = self._parse_kv_block(indent)
            dtype_override = config.pop('type', config.pop('dtype', None))
            if dtype_override is not None:
                field_type_text = str(dtype_override)
            required_value = config.pop('required', None)
            optional_value = config.pop('optional', None)
            nullable_value = config.pop('nullable', None)
            required = True
            if required_value is not None:
                required = self._to_bool(required_value, True)
            elif optional_value is not None:
                required = not self._to_bool(optional_value, False)
            if nullable_value is not None and self._to_bool(nullable_value, False):
                required = False
            default_value = config.pop('default', None)
            description_raw = config.pop('description', config.pop('desc', None))
            description = str(description_raw) if description_raw is not None else None
            enum_override = config.pop('enum', None)
            field_type, enum_values = self._parse_prompt_field_type(field_type_text)
            if enum_override is not None:
                if isinstance(enum_override, list):
                    enum_values = [self._stringify_value(item) for item in enum_override if item is not None]
                else:
                    enum_values = [self._stringify_value(enum_override)]
            metadata_raw = config.pop('metadata', {})
            metadata = self._coerce_options_dict(metadata_raw)
            if config:
                metadata.update(config)
            fields.append(
                PromptField(
                    name=field_name,
                    field_type=field_type,
                    required=required,
                    description=description,
                    default=default_value,
                    enum=enum_values,
                    metadata=metadata,
                )
            )
        return fields

    def _parse_prompt_template_block(self, parent_indent: int) -> str:
        lines: List[str] = []
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            if indent <= parent_indent:
                break
            lines.append(nxt[parent_indent:])
            self._advance()
        raw = "\n".join(lines).rstrip("\n")
        text = textwrap.dedent(raw).strip("\n")
        stripped = text.strip()
        if stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) >= 6:
            text = stripped[3:-3].strip("\n")
        elif stripped.startswith("'''") and stripped.endswith("'''") and len(stripped) >= 6:
            text = stripped[3:-3].strip("\n")
        if not text:
            raise self._error("Prompt template cannot be empty", self.pos, "")
        return text

    def _parse_prompt_field_type(self, raw: Optional[str]) -> Tuple[str, List[str]]:
        if not raw:
            return "text", []
        text = str(raw).strip()
        lowered = text.lower()
        if lowered.startswith("one_of"):
            start = text.find("(")
            end = text.rfind(")")
            if start != -1 and end != -1 and end > start:
                inner = text[start + 1 : end]
                return "enum", self._parse_prompt_enum_values(inner)
            return "enum", []
        if lowered in {"string", "text"}:
            return "text", []
        if lowered in {"int", "integer"}:
            return "int", []
        if lowered in {"float", "number"}:
            return "number", []
        if lowered in {"bool", "boolean"}:
            return "boolean", []
        if lowered in {"json", "object"}:
            return "json", []
        if lowered in {"list", "array"}:
            return "list", []
        return text, []

    def _parse_prompt_enum_values(self, inner: str) -> List[str]:
        expr = f"[{inner}]"
        try:
            parsed = py_ast.literal_eval(expr)
            if isinstance(parsed, (list, tuple)):
                return [self._stringify_value(item) for item in parsed if item is not None]
        except Exception:
            pass
        tokens = [token.strip() for token in inner.split(',') if token.strip()]
        return [self._strip_quotes(token) for token in tokens]

    def _looks_like_ai_model(self, header_line: str, base_indent: int) -> bool:
        stripped = header_line.strip()
        if stripped.lower().startswith('ai model '):
            return True
        candidate = stripped[:-1] if stripped.endswith(':') else stripped
        match = re.match(r'model\s+"[^"]+"\s+using\s+([\w\.\-]+)', candidate, flags=re.IGNORECASE)
        if match:
            provider = match.group(1).lower()
            if provider in _AI_MODEL_PROVIDER_HINTS:
                return True
        return self._block_contains_ai_hints(base_indent)

    def _block_contains_ai_hints(self, base_indent: int) -> bool:
        idx = self.pos
        saw_ai_hint = False
        while idx < len(self.lines):
            nxt = self.lines[idx]
            indent = self._indent(nxt)
            stripped = nxt.strip()
            idx += 1
            if not stripped or stripped.startswith('#'):
                continue
            if indent <= base_indent:
                break
            lowered = stripped.lower()
            if any(lowered.startswith(token) for token in _GENERAL_MODEL_BLOCK_HINTS):
                return False
            if any(token in lowered for token in _AI_MODEL_BLOCK_HINTS):
                saw_ai_hint = True
        return saw_ai_hint
