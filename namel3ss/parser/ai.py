from __future__ import annotations

import re
from typing import Any, Dict, List

from namel3ss.ast import Chain, ChainStep, Connector, ContextValue, Template

from .base import ParserBase


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
