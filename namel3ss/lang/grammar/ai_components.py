"""LLM, tool, and AI wrapper parsing methods."""

from __future__ import annotations
import re
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple, Callable

if TYPE_CHECKING:
    from .helpers import _Line
    from .parser import _GrammarModuleParser

from namel3ss.ast.ai import LLMDefinition, ToolDefinition
from namel3ss.parser.ai import AIParserMixin as _AIParserMixin
from namel3ss.parser.logic import LogicParserMixin as _LogicParserMixin

# Regex patterns for AI component parsing
_LLM_HEADER_RE = re.compile(r'^llm\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$')
_TOOL_HEADER_RE = re.compile(r'^tool\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$')


class AIComponentsParserMixin:
    """Mixin providing llm, tool, and AI wrapper parsing methods."""

    # ------------------------------------------------------------------
    # LLM, Tool, and Prompt block parsing
    # ------------------------------------------------------------------
    def _parse_llm(self, line: _Line) -> None:
        """
        Parse an LLM definition block.
        
        Grammar:
            llm <name>:
                provider: <openai|anthropic|vertex|azure_openai|local>
                model: <model_name>
                temperature: <float>
                max_tokens: <int>
                top_p: <float>
                frequency_penalty: <float>
                presence_penalty: <float>
        """
        match = _LLM_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "llm declaration")
        name = match.group(1)
        base_indent = self._indent(line.text)
        self._advance()
        
        # Parse key-value properties
        properties = self._parse_kv_block(base_indent)
        
        # Extract required fields
        provider = properties.get('provider')
        model = properties.get('model')
        if not provider or not model:
            raise self._error("llm block requires 'provider' and 'model' fields", line)
        
        # Validate provider
        valid_providers = {'openai', 'anthropic', 'vertex', 'azure_openai', 'local'}
        if provider not in valid_providers:
            raise self._error(f"Invalid provider '{provider}'. Must be one of: {', '.join(valid_providers)}", line)
        
        # Extract optional fields
        temperature = float(properties.get('temperature', 0.7))
        max_tokens = int(properties.get('max_tokens', 1024))
        top_p = float(properties['top_p']) if 'top_p' in properties else None
        frequency_penalty = float(properties['frequency_penalty']) if 'frequency_penalty' in properties else None
        presence_penalty = float(properties['presence_penalty']) if 'presence_penalty' in properties else None
        
        # Build config from remaining properties
        config = {k: v for k, v in properties.items() 
                  if k not in {'provider', 'model', 'temperature', 'max_tokens', 'top_p', 
                               'frequency_penalty', 'presence_penalty'}}
        
        llm = LLMDefinition(
            name=name,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            metadata=config,  # Store additional config in metadata field
        )
        
        self._ensure_app(line)
        # Add to both lists for backwards compatibility
        self._app.llms.append(llm)
        
        # Also add as AIModel for resolver validation
        from namel3ss.ast import AIModel
        ai_model = AIModel(
            name=name,
            provider=provider,
            model_name=model,
            config={
                'temperature': temperature,
                'max_tokens': max_tokens,
                **(config or {})
            }
        )
        self._app.ai_models.append(ai_model)

    def _parse_tool(self, line: _Line) -> None:
        """
        Parse a tool definition block.
        
        Grammar:
            tool <name>:
                type: <http|python|database|vector_search>
                endpoint: <url>
                method: <GET|POST|PUT|DELETE>
                input_schema: {...}
                output_schema: {...}
                headers: {...}
                timeout: <float>
        """
        match = _TOOL_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "tool declaration")
        name = match.group(1)
        base_indent = self._indent(line.text)
        self._advance()
        
        # Parse key-value properties
        properties = self._parse_kv_block(base_indent)
        
        # Extract fields
        tool_type = properties.get('type', 'http')
        endpoint = properties.get('endpoint')
        method = properties.get('method', 'POST')
        
        # Parse schemas if present (simple dict parsing for now)
        input_schema = self._parse_schema_field(properties.get('input_schema', {}))
        output_schema = self._parse_schema_field(properties.get('output_schema', {}))
        headers = self._parse_schema_field(properties.get('headers', {}))
        
        timeout = float(properties.get('timeout', 30.0))
        
        # Build config from remaining properties
        config = {k: v for k, v in properties.items() 
                  if k not in {'type', 'endpoint', 'method', 'input_schema', 'output_schema', 
                               'headers', 'timeout'}}
        
        tool = ToolDefinition(
            name=name,
            type=tool_type,
            endpoint=endpoint,
            method=method,
            input_schema=input_schema,
            output_schema=output_schema,
            headers=headers,
            timeout=timeout,
            config=config,
        )
        
        self._ensure_app(line)
        self._app.tools.append(tool)

    def _run_ai_block_parser(
        self,
        line: _Line,
        parser_fn: Callable[["_GrammarModuleParser", str, int, int], object],
    ) -> object:
        """
        Bridge helper that synchronizes the legacy AIParserMixin cursor handling.

        We peek the grammar line, advance past the header so AIParserMixin starts
        on the first body line, and sync both cursor systems before and after
        invoking the mixin parser.
        """
        base_indent = self._indent(line.text)
        self._advance()
        self._sync_pos_to_cursor()
        previous_flag = self._in_ai_block
        self._in_ai_block = True
        try:
            return parser_fn(self, line.text, line.number, base_indent)
        finally:
            self._in_ai_block = previous_flag
            self._sync_cursor_to_pos()

    # ========== AI Parser Wrappers ==========
    # These methods bridge between Grammar's _Line interface and AIParserMixin's (str, int, int) interface

    def _parse_connector_wrapper(self, line: _Line) -> None:
        """Wrapper to parse connector blocks using AIParserMixin."""
        connector = self._run_ai_block_parser(line, _AIParserMixin._parse_connector)
        if connector:
            if not hasattr(self, '_extra_nodes'):
                self._extra_nodes = []
            self._extra_nodes.append(connector)

    def _parse_template_wrapper(self, line: _Line) -> None:
        """Wrapper to parse template definitions using AIParserMixin."""
        template = self._run_ai_block_parser(line, _AIParserMixin._parse_template)
        if template:
            if not hasattr(self, '_extra_nodes'):
                self._extra_nodes = []
            self._extra_nodes.append(template)

    def _parse_chain_wrapper(self, line: _Line) -> None:
        """Wrapper to parse chain blocks using AIParserMixin."""
        chain = self._run_ai_block_parser(line, _AIParserMixin._parse_chain)
        if chain:
            self._ensure_app(line)
            self._app.chains.append(chain)

    def _parse_memory_wrapper(self, line: _Line) -> None:
        """Wrapper to parse memory configurations using AIParserMixin."""
        memory = self._run_ai_block_parser(line, _AIParserMixin._parse_memory)
        if memory:
            if not hasattr(self, '_extra_nodes'):
                self._extra_nodes = []
            self._extra_nodes.append(memory)

    def _parse_ai_model_wrapper(self, line: _Line) -> None:
        """Wrapper to parse AI model blocks using AIParserMixin."""
        model = self._run_ai_block_parser(line, _AIParserMixin._parse_ai_model)
        if model:
            if not hasattr(self, '_extra_nodes'):
                self._extra_nodes = []
            self._extra_nodes.append(model)

    def _parse_prompt_wrapper(self, line: _Line) -> None:
        """Wrapper to parse structured prompts using AIParserMixin."""
        prompt = self._run_ai_block_parser(line, _AIParserMixin._parse_prompt)
        if prompt:
            self._ensure_app(line)
            self._app.prompts.append(prompt)

    def _parse_training_job_wrapper(self, line: _Line) -> None:
        """Wrapper to parse training job definitions using AIParserMixin."""
        training_job = self._run_ai_block_parser(line, _AIParserMixin._parse_training_job)
        if training_job:
            if not hasattr(self, '_extra_nodes'):
                self._extra_nodes = []
            self._extra_nodes.append(training_job)

    def _parse_tuning_job_wrapper(self, line: _Line) -> None:
        """Wrapper to parse tuning job definitions using AIParserMixin."""
        tuning_job = self._run_ai_block_parser(line, _AIParserMixin._parse_tuning_job)
        if tuning_job:
            if not hasattr(self, '_extra_nodes'):
                self._extra_nodes = []
            self._extra_nodes.append(tuning_job)

    def _parse_rlhf_job_wrapper(self, line: _Line) -> None:
        """Wrapper to parse RLHF job definitions using AIParserMixin."""
        rlhf_job = self._run_ai_block_parser(line, _AIParserMixin._parse_rlhf_job)
        if rlhf_job:
            if not hasattr(self, '_extra_nodes'):
                self._extra_nodes = []
            self._extra_nodes.append(rlhf_job)

    def _parse_knowledge_wrapper(self, line: _Line) -> None:
        """Wrapper to parse knowledge module definitions using LogicParserMixin."""
        knowledge_module = self._run_ai_block_parser(line, _LogicParserMixin._parse_knowledge_module)
        if knowledge_module:
            if not hasattr(self, '_extra_nodes'):
                self._extra_nodes = []
            self._extra_nodes.append(knowledge_module)

    def _parse_query_wrapper(self, line: _Line) -> None:
        """Wrapper to parse query definitions using LogicParserMixin."""
        query = self._run_ai_block_parser(line, _LogicParserMixin._parse_query)
        if query:
            if not hasattr(self, '_extra_nodes'):
                self._extra_nodes = []
            self._extra_nodes.append(query)

    # ========== Legacy Prompt Parser (Deprecated - Use AIParserMixin) ==========

    def _parse_prompt_legacy(self, line: _Line) -> None:
        """Legacy prompt parser - now handled by AIParserMixin."""
        pass


__all__ = ['AIComponentsParserMixin']
