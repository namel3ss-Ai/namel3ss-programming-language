"""Core AI resource parsers: connectors, templates, chains, memory, and models."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from namel3ss.ast import (
    AIModel,
    Chain,
    ChainStep,
    Connector,
    Memory,
    Prompt,
    PromptArgument,
    PromptField,
    OutputSchema,
    StepEvaluationConfig,
    Template,
    WorkflowNode,
)

if TYPE_CHECKING:
    from ..base import ParserBase

# Import template engine for compile-time validation
try:
    from namel3ss.templates import get_default_engine, TemplateCompilationError
    TEMPLATE_VALIDATION_AVAILABLE = True
except ImportError:
    TEMPLATE_VALIDATION_AVAILABLE = False

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

_MEMORY_SCOPES = {"session", "page", "conversation", "global", "user", "thread"}
_MEMORY_KINDS = {"list", "conversation", "key_value", "vector", "buffer", "kv"}


class ModelsParserMixin:
    """Mixin for parsing connectors, templates, chains, memory, and AI models."""
    
    def _parse_connector(self: 'ParserBase', line: str, line_no: int, base_indent: int) -> Connector:
        """
        Parse connector definition with validation.
        
        Syntax:
            connector "Name" type KIND:
                provider: value
                config: {...}
                description: "..."
        
        Example:
            connector "Database" type postgres:
                provider: "postgresql"
                host: "localhost"
                port: 5432
                database: "myapp"
        """
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'connector\s+"([^"]+)"\s+type\s+([\w\.\-]+)', stripped)
        if not match:
            raise self._error(
                'Expected: connector "Name" type KIND:',
                line_no,
                line,
                hint='Connector definitions must specify a name and type'
            )
        name = match.group(1)
        connector_type = match.group(2)
        config: Dict[str, Any] = {}
        description: Any = None
        provider_value: Optional[Any] = None
        kind_override: Optional[str] = None
        
        # Validate indented block
        indent_info = self._expect_indent_greater_than(
            base_indent,
            context=f'connector "{name}"',
            line_no=line_no
        )
        if not indent_info:
            raise self._error(
                f'Connector "{name}" requires an indented configuration block',
                line_no,
                line,
                hint='Add indented lines with provider, config, etc.'
            )
        
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
                    "Expected 'key = value' or 'key: value' inside connector block",
                    self.pos + 1,
                    nxt,
                    hint='Connector configuration uses key-value pairs'
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

    def _parse_template(self: 'ParserBase', line: str, line_no: int, base_indent: int) -> Template:
        """
        Parse template definition with compile-time validation.
        
        Templates define reusable text patterns with variable interpolation,
        validated at compile-time to catch syntax errors before runtime.
        
        Syntax:
            define template "Name":
                prompt: |
                    Hello {name}!
                    Your order {order_id} is ready.
        
        Validation:
            - Template syntax validated at compile-time if engine available
            - Variable placeholders checked for proper formatting
            - Compilation errors reported with helpful context
        """
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
        
        # Compile-time template validation
        if TEMPLATE_VALIDATION_AVAILABLE and prompt_text:
            try:
                engine = get_default_engine()
                engine.compile(prompt_text, name=f"template_{name}", validate=True)
            except TemplateCompilationError as e:
                raise self._error(
                    f"Template '{name}' compilation error: {str(e)}",
                    line_no,
                    line
                )
        
        return Template(name=name, prompt=prompt_text, metadata=metadata)

    def _parse_memory(self: 'ParserBase', line: str, line_no: int, base_indent: int) -> Memory:
        """
        Parse memory definition for conversational state management.
        
        Memory stores preserve state across interactions, supporting
        various scopes (session, user, global, thread) and kinds
        (list, key-value, summary).
        
        Syntax:
            memory "Name":
                scope: session|user|global|thread
                kind: list|kv|summary
                max_items: 10
        """
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'memory\s+"([^"]+)"', stripped, flags=re.IGNORECASE)
        if not match:
            raise self._error(
                'Expected: memory "Name":',
                line_no,
                line,
                hint='Memory stores require a name, e.g., memory "chat_history":'
            )
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
        normalized_config = {key: self._transform_config(val) for key, val in config.items()}
        return Memory(
            name=name,
            scope=scope,
            kind=kind,
            max_items=max_items,
            config=normalized_config,
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    def _parse_ai_model(self: 'ParserBase', line: str, line_no: int, base_indent: int) -> AIModel:
        """
        Parse AI model reference definition.
        
        AI models define provider-backed LLM references with configuration
        including API keys, endpoints, parameters, and metadata.
        
        Syntax:
            model "Name" using PROVIDER:
                model: gpt-4
                temperature: 0.7
                max_tokens: 2000
                api_key: env:OPENAI_KEY
        
        Supported Providers:
            openai, anthropic, azure, google, vertex, bedrock, mistral, cohere, ollama
        """
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'(?:ai\s+)?model\s+"([^"]+)"(?:\s+using\s+([\w\.\-]+))?', stripped, flags=re.IGNORECASE)
        if not match:
            raise self._error(
                'Expected: model "Name" using PROVIDER',
                line_no,
                line,
                hint='AI models require a name and provider, e.g., model "GPT4" using openai:'
            )
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

    def _parse_chain_step_options(self: 'ParserBase', tokens: List[str]) -> Dict[str, Any]:
        """
        Parse chain step options from token list.
        
        Extracts memory read/write specifications and other options
        from tokenized step configuration.
        
        Supported Options:
            read_memory/memory_read: Memory stores to read from
            write_memory/memory_write: Memory stores to write to
        """
        from .utils import split_memory_names
        
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
                names = split_memory_names(raw_value)
                existing = options.setdefault("read_memory", [])
                existing.extend(names)
            elif key in {"write_memory", "memory_write"}:
                names = split_memory_names(raw_value)
                existing = options.setdefault("write_memory", [])
                existing.extend(names)
            else:
                options[key] = raw_value
        return options

    def _parse_step_evaluation_config(self: 'ParserBase', value: Any, line_no: int, header: str) -> StepEvaluationConfig:
        """
        Parse step evaluation configuration for quality assessment.
        
        Evaluation configs specify evaluators and optional guardrails
        to assess step outputs for quality, safety, and correctness.
        
        Syntax:
            evaluation:
                evaluators: ["eval1", "eval2"]
                guardrail: "SafetyGuard"
        """
        if not isinstance(value, dict):
            raise self._error(
                "Step evaluation must be a block",
                line_no,
                header,
                hint='Use evaluation: block with evaluators and optional guardrail'
            )
        evaluators_raw = value.get("evaluators")
        guardrail_raw = value.get("guardrail")
        evaluators: List[str] = []
        if isinstance(evaluators_raw, (list, tuple)):
            evaluators = [str(entry) for entry in evaluators_raw if entry]
        elif isinstance(evaluators_raw, str):
            evaluators = [evaluators_raw]
        if not evaluators:
            raise self._error(
                "Step evaluation must reference at least one evaluator",
                line_no,
                header,
                hint='Add evaluators: ["eval1", "eval2"] with at least one evaluator'
            )
        guardrail_name = str(guardrail_raw) if guardrail_raw is not None else None
        return StepEvaluationConfig(evaluators=[str(name) for name in evaluators], guardrail=guardrail_name)

    def _looks_like_ai_model(self: 'ParserBase', header_line: str, base_indent: int) -> bool:
        """
        Heuristic to distinguish AI model blocks from general model blocks.
        
        Checks for AI-specific keywords, provider references, and block hints
        to determine if a model declaration refers to an AI/LLM model.
        
        Detection Criteria:
            - Starts with 'ai model'
            - Has 'using' clause with known AI provider
            - Block contains AI-specific hints (api_key, endpoint, temperature)
        """
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

    def _block_contains_ai_hints(self: 'ParserBase', base_indent: int) -> bool:
        """
        Scan block for AI-specific configuration hints.
        
        Looks ahead in the block for keywords that indicate AI model
        configuration (api_key, temperature, max_tokens) vs general
        model configuration (columns, table, query).
        
        Returns:
            True if block contains AI-specific hints, False otherwise
        """
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
