from __future__ import annotations

import ast as py_ast
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from namel3ss.ast import (
    AIModel,
    Chain,
    ChainStep,
    Connector,
    ContextValue,
    Memory,
    Prompt,
    PromptField,
    StepEvaluationConfig,
    Template,
    WorkflowForBlock,
    WorkflowIfBlock,
    WorkflowNode,
    WorkflowWhileBlock,
)

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
        provider_value: Optional[Any] = None
        kind_override: Optional[str] = None
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped_line = nxt.strip()
            lowered = stripped_line.lower()
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
            lowered_key = key.lower()
            if lowered_key == "description":
                description = value
            elif lowered_key == "provider":
                provider_value = value
                config[key] = value
            elif lowered_key == "kind":
                kind_override = str(value or "").strip()
            else:
                config[key] = value
        description_text = None if description is None else str(description)
        provider_raw = provider_value if provider_value is not None else config.get("provider")
        if provider_raw is None:
            raise self._error("Connector must define a provider", line_no, line)
        if not isinstance(provider_raw, str):
            raise self._error("Connector provider must be a string literal", line_no, line)
        provider_text = provider_raw.strip()
        if not provider_text:
            raise self._error("Connector provider cannot be empty", line_no, line)
        effective_type = kind_override or connector_type
        if not effective_type:
            raise self._error("Connector type cannot be empty", line_no, line)
        return Connector(
            name=name,
            connector_type=effective_type,
            provider=provider_text,
            config=config,
            description=description_text,
        )

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
            lowered = stripped_line.lower()
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
        match = re.match(r'define\s+chain\s+"([^"]+)"(?:\s+effect\s+([\w\-]+))?', stripped, flags=re.IGNORECASE)
        if not match:
            raise self._error('Expected: define chain "Name":', line_no, line)
        name = match.group(1)
        declared_effect = self._parse_effect_annotation(match.group(2), line_no, line)
        input_key = "input"
        workflow_nodes: List[WorkflowNode] = []
        metadata: Dict[str, Any] = {}
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped_line = nxt.strip()
            lowered = stripped_line.lower()
            if not stripped_line or stripped_line.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            if lowered.startswith('steps:') or lowered.startswith('workflow:'):
                self._advance()
                workflow_nodes.extend(self._parse_workflow_block(indent))
            elif '->' in stripped_line:
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
                    token_offset = 2
                    if kind == 'python' and len(tokens) > 1:
                        module_spec = tokens[1]
                        module_name, _, method_name = module_spec.partition(':')
                        target = module_name
                        if module_name:
                            options['module'] = module_name
                        if method_name:
                            options['method'] = method_name
                    remaining = tokens[token_offset:]
                    if remaining:
                        option_values = self._parse_chain_step_options(remaining)
                        options.update(option_values)
                    workflow_nodes.append(ChainStep(kind=kind, target=target, options=options))
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
        return Chain(name=name, input_key=input_key, steps=workflow_nodes, metadata=metadata, declared_effect=declared_effect)

    def _parse_memory(self, line: str, line_no: int, base_indent: int) -> Memory:
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'memory\s+"([^"]+)"', stripped, flags=re.IGNORECASE)
        if not match:
            raise self._error('Expected: memory "Name":', line_no, line)
        name = match.group(1)
        config = self._parse_kv_block(base_indent)
        scope_raw = config.pop("scope", config.pop("context", "session"))
        scope = str(scope_raw or "session").lower()
        if scope not in _MEMORY_SCOPES:
            allowed = ", ".join(sorted(_MEMORY_SCOPES))
            raise self._error(f"Unknown memory scope '{scope}'. Allowed scopes: {allowed}", line_no, line)
        kind_raw = config.pop("kind", config.pop("mode", "list"))
        kind = str(kind_raw or "list").lower()
        if kind not in _MEMORY_KINDS:
            allowed = ", ".join(sorted(_MEMORY_KINDS))
            raise self._error(f"Unknown memory kind '{kind}'. Allowed kinds: {allowed}", line_no, line)
        max_items_raw = config.pop("max_items", config.pop("limit", None))
        max_items = self._coerce_int(max_items_raw) if max_items_raw is not None else None
        metadata_raw = config.pop("metadata", {})
        metadata = self._transform_config(metadata_raw) if isinstance(metadata_raw, dict) else {}
        normalized_config = {key: self._transform_config(value) for key, value in config.items()}
        return Memory(
            name=name,
            scope=scope,
            kind=kind,
            max_items=max_items,
            config=normalized_config,
            metadata=metadata if isinstance(metadata, dict) else {},
        )

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

        prompt_object = Prompt(
            name=name,
            model=model_name,
            template=template_text,
            input_fields=input_fields,
            output_fields=output_fields,
            parameters=parameters_payload,
            metadata=metadata,
            description=description,
        )
        prompt_object.effects = {"ai"}
        return prompt_object

    def _parse_chain_step_options(self, tokens: List[str]) -> Dict[str, Any]:
        options: Dict[str, Any] = {}
        idx = 0
        while idx < len(tokens):
            key = tokens[idx].strip().lower()
            idx += 1
            if idx >= len(tokens):
                raise self._error("Expected value after chain step option", self.pos - 1, tokens[idx - 1])
            raw_value = tokens[idx].strip()
            idx += 1
            if key in {"read_memory", "memory_read"}:
                names = self._split_memory_names(raw_value)
                existing = options.setdefault("read_memory", [])
                existing.extend(names)
            elif key in {"write_memory", "memory_write"}:
                names = self._split_memory_names(raw_value)
                existing = options.setdefault("write_memory", [])
                existing.extend(names)
            else:
                options[key] = raw_value
        return options

    def _parse_step_evaluation_config(self, value: Any, line_no: int, header: str) -> StepEvaluationConfig:
        if not isinstance(value, dict):
            raise self._error("Step evaluation must be a block", line_no, header)
        evaluators_raw = value.get("evaluators")
        guardrail_raw = value.get("guardrail")
        evaluators: List[str] = []
        if isinstance(evaluators_raw, (list, tuple)):
            evaluators = [str(entry) for entry in evaluators_raw if entry]
        elif isinstance(evaluators_raw, str):
            evaluators = [evaluators_raw]
        if not evaluators:
            raise self._error("Step evaluation must reference at least one evaluator", line_no, header)
        guardrail_name = str(guardrail_raw) if guardrail_raw is not None else None
        return StepEvaluationConfig(evaluators=[str(name) for name in evaluators], guardrail=guardrail_name)

    def _parse_workflow_block(self, parent_indent: int) -> List[WorkflowNode]:
        nodes: List[WorkflowNode] = []
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            if indent <= parent_indent:
                break
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if not stripped.startswith('-'):
                raise self._error("Workflow entries must begin with '-'", self.pos + 1, nxt)
            line_no = self.pos + 1
            entry_line = stripped[1:].strip()
            self._advance()
            nodes.append(self._parse_workflow_entry(entry_line, indent, line_no))
        return nodes

    def _parse_workflow_entry(self, header: str, indent: int, line_no: int) -> WorkflowNode:
        lowered = header.lower()
        if lowered.startswith('step'):
            return self._parse_workflow_step_entry(header, indent, line_no)
        if lowered.startswith('if '):
            return self._parse_workflow_if_entry(header, indent, line_no)
        if lowered.startswith('for '):
            return self._parse_workflow_for_entry(header, indent, line_no)
        if lowered.startswith('while '):
            return self._parse_workflow_while_entry(header, indent, line_no)
        raise self._error("Unsupported workflow entry", line_no, header)

    def _parse_workflow_step_entry(self, header: str, indent: int, line_no: int) -> ChainStep:
        match = re.match(r'step(?:\s+"([^"]+)")?(?:\s*:)?$', header, flags=re.IGNORECASE)
        if not match:
            raise self._error("Expected: - step \"Name\":", line_no, header)
        name = match.group(1)
        config = self._parse_kv_block(indent)
        normalized = {key: self._transform_config(value) for key, value in config.items()}
        kind_value = normalized.pop('kind', normalized.pop('type', None))
        if kind_value is None:
            raise self._error("Workflow step must define 'kind:'", line_no, header)
        target_value = normalized.pop('target', normalized.pop('connector', None))
        if target_value is None:
            target_value = kind_value
        stop_on_error_raw = normalized.pop('stop_on_error', None)
        continue_on_error_raw = normalized.pop('continue_on_error', None)
        if stop_on_error_raw is not None:
            stop_on_error = self._parse_bool(str(stop_on_error_raw))
        elif continue_on_error_raw is not None:
            stop_on_error = not self._parse_bool(str(continue_on_error_raw))
        else:
            stop_on_error = True
        evaluation_value = normalized.pop('evaluation', None)
        evaluation = None
        if evaluation_value is not None:
            evaluation = self._parse_step_evaluation_config(evaluation_value, line_no, header)
        options_value = normalized.pop('options', {})
        options = self._coerce_options_dict(options_value)
        for key, value in normalized.items():
            options[key] = value
        return ChainStep(
            kind=str(kind_value),
            target=str(target_value),
            options=options,
            name=name,
            stop_on_error=stop_on_error,
            evaluation=evaluation,
        )

    def _parse_workflow_if_entry(self, header: str, indent: int, line_no: int) -> WorkflowIfBlock:
        if not header.rstrip().endswith(':'):
            raise self._error("Workflow if entries must end with ':'", line_no, header)
        condition_text = header[:-1].strip()[2:].strip()
        if not condition_text:
            raise self._error("Workflow if condition cannot be empty", line_no, header)
        condition_expr = self._parse_expression(condition_text)
        then_steps: List[WorkflowNode] = []
        elif_branches: List[Tuple[Expression, List[WorkflowNode]]] = []
        else_steps: List[WorkflowNode] = []
        seen_then = False
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            current_indent = self._indent(nxt)
            if current_indent <= indent:
                break
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            lowered = stripped.lower()
            if lowered.startswith('then:'):
                self._advance()
                then_steps = self._parse_workflow_block(current_indent)
                seen_then = True
                continue
            if lowered.startswith('elif '):
                self._advance()
                branch_text = stripped[4:].strip()
                if not branch_text.endswith(':'):
                    raise self._error("Workflow elif must end with ':'", self.pos, stripped)
                branch_expr = self._parse_expression(branch_text[:-1].strip())
                branch_steps = self._parse_workflow_block(current_indent)
                elif_branches.append((branch_expr, branch_steps))
                continue
            if lowered.startswith('else:'):
                self._advance()
                else_steps = self._parse_workflow_block(current_indent)
                continue
            raise self._error("Unexpected entry inside workflow if block", self.pos + 1, nxt)
        if not then_steps:
            raise self._error("Workflow if block must define a 'then:' section", line_no, header)
        return WorkflowIfBlock(condition=condition_expr, then_steps=then_steps, elif_steps=elif_branches, else_steps=else_steps)

    def _parse_workflow_for_entry(self, header: str, indent: int, line_no: int) -> WorkflowForBlock:
        match = re.match(r'for\s+([A-Za-z_][\w]*)\s+in\s+(.+):$', header, flags=re.IGNORECASE)
        if not match:
            raise self._error("Expected: - for item in <expression>:", line_no, header)
        loop_var = match.group(1)
        source_text = match.group(2).strip()
        source_kind = "expression"
        source_name: Optional[str] = None
        source_expression: Optional[Expression] = None
        lowered = source_text.lower()
        if lowered.startswith('dataset '):
            source_kind = "dataset"
            source_name = self._strip_quotes(source_text.split(None, 1)[1].strip())
        else:
            source_expression = self._parse_expression(source_text)
        config = self._parse_workflow_optional_config(indent)
        max_iterations = None
        if config:
            max_iterations = self._coerce_int(config.pop('max_iterations', config.pop('limit', None)))
            if max_iterations is not None and max_iterations <= 0:
                max_iterations = None
            if config:
                unknown = ", ".join(config.keys())
                raise self._error(f"Unsupported options in workflow for block: {unknown}", line_no, header)
        body = self._parse_workflow_block(indent)
        return WorkflowForBlock(
            loop_var=loop_var,
            source_kind=source_kind,
            source_name=source_name,
            source_expression=source_expression,
            body=body,
            max_iterations=max_iterations,
        )

    def _parse_workflow_while_entry(self, header: str, indent: int, line_no: int) -> WorkflowWhileBlock:
        if not header.rstrip().endswith(':'):
            raise self._error("Workflow while entries must end with ':'", line_no, header)
        condition_text = header[:-1].strip()[5:].strip()
        if not condition_text:
            raise self._error("Workflow while condition cannot be empty", line_no, header)
        condition_expr = self._parse_expression(condition_text)
        config = self._parse_workflow_optional_config(indent)
        max_iterations = None
        if config:
            max_iterations = self._coerce_int(config.pop('max_iterations', config.pop('limit', None)))
            if max_iterations is not None and max_iterations <= 0:
                max_iterations = None
            if config:
                unknown = ", ".join(config.keys())
                raise self._error(f"Unsupported options in workflow while block: {unknown}", line_no, header)
        body = self._parse_workflow_block(indent)
        return WorkflowWhileBlock(condition=condition_expr, body=body, max_iterations=max_iterations)

    def _parse_workflow_optional_config(self, parent_indent: int) -> Dict[str, Any]:
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent or stripped.startswith('-'):
                break
            return self._parse_kv_block(parent_indent)
        return {}

    def _split_memory_names(self, raw: str) -> List[str]:
        if not raw:
            return []
        candidates = [part.strip() for part in raw.split(',') if part.strip()]
        normalized: List[str] = []
        for candidate in candidates:
            if (candidate.startswith('"') and candidate.endswith('"')) or (candidate.startswith("'") and candidate.endswith("'")):
                normalized.append(candidate[1:-1])
            else:
                normalized.append(candidate)
        return normalized

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
_MEMORY_SCOPES = {"session", "page", "conversation", "global"}
_MEMORY_KINDS = {"list", "conversation", "key_value", "vector"}
