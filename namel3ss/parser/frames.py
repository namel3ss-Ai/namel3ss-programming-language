from __future__ import annotations

import shlex
from typing import Any, Dict, List, Optional

from namel3ss.ast import (
    Frame,
    FrameAccessPolicy,
    FrameColumn,
    FrameColumnConstraint,
    FrameConstraint,
    FrameIndex,
    FrameRelationship,
    FrameSourceDef,
)

from .datasets import DatasetParserMixin


class FrameParserMixin(DatasetParserMixin):
    """Parsing helpers for N3Frame declarations."""

    def _parse_frame(self, line: str, line_no: int, base_indent: int) -> Frame:
        stripped = line.strip()
        if stripped.endswith(':'):
            stripped = stripped[:-1].rstrip()
        try:
            tokens = shlex.split(stripped)
        except ValueError as exc:
            raise self._error(f"Unable to parse frame declaration: {exc}", line_no, line)
        if len(tokens) < 2 or tokens[0].lower() != 'frame':
            raise self._error('Expected: frame "Name" [from dataset <source>]:', line_no, line)

        name = self._strip_quotes(tokens[1])
        source_type = 'dataset'
        source: Optional[str] = None
        idx = 2
        while idx < len(tokens):
            token = tokens[idx].lower()
            if token == 'from' and idx + 1 < len(tokens):
                source_type = tokens[idx + 1].lower()
                idx += 2
                if idx >= len(tokens):
                    raise self._error("Frame source is missing after 'from'", line_no, line)
                source = self._strip_quotes(tokens[idx])
                idx += 1
                continue
            if token == 'using' and idx + 1 < len(tokens):
                idx += 1
                continue
            # Unrecognised trailing tokens are treated as part of the source name.
            if source is None:
                source = self._strip_quotes(tokens[idx])
            idx += 1

        if source is None:
            source = name

        frame = Frame(name=name, source_type=source_type, source=source)

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
            if lowered.startswith('description:'):
                self._advance()
                desc_raw = stripped_line[len('description:'):].strip()
                frame.description = self._stringify_value(self._coerce_scalar(desc_raw)) if desc_raw else None
            elif lowered.startswith('tags:'):
                frame.tags = self._parse_tag_list(stripped_line[len('tags:'):].strip())
                self._advance()
            elif lowered.startswith('metadata:'):
                block_indent = indent
                self._advance()
                frame.metadata.update(self._parse_kv_block(block_indent))
            elif lowered.startswith('options:'):
                block_indent = indent
                self._advance()
                frame.options.update(self._parse_kv_block(block_indent))
            elif lowered.startswith('column '):
                header_line = self._advance()
                if header_line is None:
                    break
                column = self._parse_frame_column(header_line, indent)
                frame.columns.append(column)
            elif lowered.startswith('index '):
                header_line = self._advance()
                if header_line is None:
                    break
                frame.indexes.append(self._parse_frame_index(header_line, indent))
            elif lowered.startswith('relationship '):
                header_line = self._advance()
                if header_line is None:
                    break
                frame.relationships.append(self._parse_frame_relationship(header_line, indent))
            elif lowered.startswith('constraint '):
                header_line = self._advance()
                if header_line is None:
                    break
                frame.constraints.append(self._parse_frame_constraint(header_line, indent))
            elif lowered.startswith('access:'):
                block_indent = indent
                self._advance()
                policy_data = self._parse_kv_block(block_indent)
                frame.access = self._build_frame_access_policy(policy_data)
            elif lowered.startswith('sample:') or lowered.startswith('example:'):
                block_indent = indent
                self._advance()
                sample_row = self._parse_kv_block(block_indent)
                if sample_row:
                    frame.examples.append(sample_row)
            elif lowered.startswith('with option '):
                option_text = stripped_line[len('with option '):].strip()
                key, value = self._parse_connector_option(option_text, self.pos + 1, nxt)
                self._assign_frame_option(frame, key, value)
                self._advance()
            elif lowered.startswith('key:'):
                self._advance()
                key_text = stripped_line[len('key:'):].strip()
                frame.key = self._parse_frame_key_list(key_text)
            elif lowered.startswith('splits:'):
                block_indent = indent
                self._advance()
                frame.splits = self._parse_frame_splits(block_indent)
            elif lowered.startswith('source:'):
                block_indent = indent
                self._advance()
                source_config = self._parse_frame_source_config(block_indent)
                frame.source_config = source_config
            else:
                raise self._error("Expected frame property, column, index, relationship, or constraint", self.pos + 1, nxt)
        return frame

    def _parse_frame_column(self, header_line: str, header_indent: int) -> FrameColumn:
        stripped = header_line.strip()
        has_block = stripped.endswith(':')
        header = stripped[:-1] if has_block else stripped
        try:
            tokens = shlex.split(header)
        except ValueError as exc:
            raise self._error(f"Unable to parse column declaration: {exc}", self.pos, header_line)
        if len(tokens) < 2 or tokens[0].lower() != 'column':
            raise self._error("Expected: column name [dtype]", self.pos, header_line)
        name = self._strip_quotes(tokens[1])
        dtype = 'string'
        nullable = True
        idx = 2
        if idx < len(tokens):
            dtype = self._strip_quotes(tokens[idx])
            idx += 1
        inline_tokens = [token.lower() for token in tokens[idx:]]
        if any(token in {'required', 'not_null', 'not-null', '!'} for token in inline_tokens):
            nullable = False
        if any(token in {'nullable', 'optional'} for token in inline_tokens):
            nullable = True

        config: Dict[str, Any] = {}
        if has_block:
            config = self._parse_kv_block(header_indent)

        dtype_raw = config.pop('dtype', None)
        if dtype_raw is not None:
            dtype = self._stringify_value(dtype_raw)
        nullable = self._to_bool(config.pop('nullable', nullable))
        description_raw = config.pop('description', config.pop('desc', None))
        description = self._stringify_value(description_raw) if description_raw is not None else None
        default = config.pop('default', None)
        source_raw = config.pop('source', config.pop('field', None))
        source = self._stringify_value(source_raw) if source_raw is not None else None
        expr_raw = config.pop('expression', config.pop('expr', None))
        expression = self._coerce_expression(expr_raw) if expr_raw is not None else None
        tags = self._ensure_string_list(config.pop('tags', []))
        metadata = self._coerce_options_dict(config.pop('metadata', {}))
        validations = self._parse_column_validations(config.pop('validations', config.pop('validators', [])))
        role_raw = config.pop('role', None)
        role = self._stringify_value(role_raw) if role_raw is not None else None

        if config:
            metadata.update(self._coerce_options_dict(config))

        return FrameColumn(
            name=name,
            dtype=dtype,
            nullable=nullable,
            description=description,
            default=default,
            expression=expression,
            source=source,
            role=role,
            tags=tags,
            metadata=metadata,
            validations=validations,
        )

    def _parse_column_validations(self, raw: Any) -> List[FrameColumnConstraint]:
        if raw is None:
            return []
        entries: List[Dict[str, Any]] = []
        if isinstance(raw, dict):
            for key, value in raw.items():
                entry = dict(value) if isinstance(value, dict) else {'expression': value}
                entry.setdefault('name', key)
                entries.append(entry)
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    entries.append(dict(item))
                else:
                    entries.append({'expression': item})
        else:
            entries.append({'expression': raw})

        validations: List[FrameColumnConstraint] = []
        for entry in entries:
            name_value = entry.pop('name', None)
            message_value = entry.pop('message', entry.pop('msg', None))
            expr_raw = entry.pop('expression', entry.pop('expr', None))
            validation = FrameColumnConstraint(
                name=self._stringify_value(name_value) if name_value is not None else None,
                expression=self._coerce_expression(expr_raw) if expr_raw is not None else None,
                message=self._stringify_value(message_value) if message_value is not None else None,
                severity=str(entry.pop('severity', 'error')),
                config=self._coerce_options_dict(entry) if entry else {},
            )
            validations.append(validation)
        return validations

    def _parse_frame_index(self, header_line: str, header_indent: int) -> FrameIndex:
        stripped = header_line.strip()
        has_block = stripped.endswith(':')
        header = stripped[:-1] if has_block else stripped
        try:
            tokens = shlex.split(header)
        except ValueError as exc:
            raise self._error(f"Unable to parse index declaration: {exc}", self.pos, header_line)
        if len(tokens) < 2 or tokens[0].lower() != 'index':
            raise self._error('Expected: index "Name" [on columns]', self.pos, header_line)
        name = self._strip_quotes(tokens[1])
        inline_columns: List[str] = []
        for idx, token in enumerate(tokens[2:], start=2):
            if token.lower() == 'on':
                column_text = ' '.join(tokens[idx + 1 :])
                inline_columns = [self._strip_quotes(part.strip()) for part in column_text.split(',') if part.strip()]
                break

        config: Dict[str, Any] = {}
        if has_block:
            config = self._parse_kv_block(header_indent)

        columns_raw = config.pop('columns', config.pop('fields', inline_columns))
        columns = self._ensure_string_list(columns_raw)
        unique = self._to_bool(config.pop('unique', False))
        method_raw = config.pop('method', config.pop('using', None))
        method = self._stringify_value(method_raw) if method_raw is not None else None
        options = self._coerce_options_dict(config.pop('options', {}))
        if config:
            options.update(self._coerce_options_dict(config))

        return FrameIndex(name=name, columns=columns, unique=unique, method=method, options=options)

    def _parse_frame_relationship(self, header_line: str, header_indent: int) -> FrameRelationship:
        stripped = header_line.strip()
        has_block = stripped.endswith(':')
        header = stripped[:-1] if has_block else stripped
        try:
            tokens = shlex.split(header)
        except ValueError as exc:
            raise self._error(f"Unable to parse relationship declaration: {exc}", self.pos, header_line)
        if len(tokens) < 2 or tokens[0].lower() != 'relationship':
            raise self._error('Expected: relationship "Name"', self.pos, header_line)

        name = self._strip_quotes(tokens[1])
        target_frame: Optional[str] = None
        target_dataset: Optional[str] = None
        for idx, token in enumerate(tokens[2:], start=2):
            lowered = token.lower()
            if lowered == 'to' and idx + 2 < len(tokens):
                target_kind = tokens[idx + 1].lower()
                target_value = self._strip_quotes(tokens[idx + 2])
                if target_kind == 'frame':
                    target_frame = target_value
                elif target_kind == 'dataset':
                    target_dataset = target_value
                break

        config: Dict[str, Any] = {}
        if has_block:
            config = self._parse_kv_block(header_indent)

        target_frame_raw = config.pop('frame', config.pop('target_frame', None))
        if target_frame_raw is not None:
            target_frame = self._stringify_value(target_frame_raw)
        target_dataset_raw = config.pop('dataset', config.pop('target_dataset', None))
        if target_dataset_raw is not None:
            target_dataset = self._stringify_value(target_dataset_raw)
        local_key_raw = config.pop('local_key', config.pop('local', None))
        local_key = self._stringify_value(local_key_raw) if local_key_raw is not None else None
        remote_key_raw = config.pop('remote_key', config.pop('remote', None))
        remote_key = self._stringify_value(remote_key_raw) if remote_key_raw is not None else None
        cardinality = str(config.pop('cardinality', 'many_to_one'))
        join_type = str(config.pop('join_type', config.pop('join', 'left')))
        description_raw = config.pop('description', config.pop('desc', None))
        description = self._stringify_value(description_raw) if description_raw is not None else None
        metadata = self._coerce_options_dict(config)

        return FrameRelationship(
            name=name,
            target_frame=target_frame,
            target_dataset=target_dataset,
            local_key=local_key,
            remote_key=remote_key,
            cardinality=cardinality,
            join_type=join_type,
            description=description,
            metadata=metadata,
        )

    def _parse_frame_constraint(self, header_line: str, header_indent: int) -> FrameConstraint:
        stripped = header_line.strip()
        has_block = stripped.endswith(':')
        header = stripped[:-1] if has_block else stripped
        try:
            tokens = shlex.split(header)
        except ValueError as exc:
            raise self._error(f"Unable to parse constraint declaration: {exc}", self.pos, header_line)
        if len(tokens) < 2 or tokens[0].lower() != 'constraint':
            raise self._error('Expected: constraint "Name"', self.pos, header_line)
        name = self._strip_quotes(tokens[1])

        config: Dict[str, Any] = {}
        if has_block:
            config = self._parse_kv_block(header_indent)

        expr_raw = config.pop('expression', config.pop('expr', None))
        expression = self._coerce_expression(expr_raw) if expr_raw is not None else None
        message_raw = config.pop('message', config.pop('msg', None))
        message = self._stringify_value(message_raw) if message_raw is not None else None
        severity = str(config.pop('severity', 'error'))
        metadata = self._coerce_options_dict(config)

        return FrameConstraint(name=name, expression=expression, message=message, severity=severity, metadata=metadata)

    def _build_frame_access_policy(self, data: Dict[str, Any]) -> FrameAccessPolicy:
        payload = dict(data or {})
        public = self._to_bool(payload.pop('public', False), default=False)
        roles = self._ensure_string_list(payload.pop('roles', []))
        allow_anonymous = self._to_bool(payload.pop('allow_anonymous', payload.pop('anonymous', False)), default=False)
        rate_limit = self._coerce_int(payload.pop('rate_limit_per_minute', payload.pop('rate_limit', None)))
        cache_seconds = self._coerce_int(payload.pop('cache_seconds', payload.pop('cache_ttl', None)))
        metadata = self._coerce_options_dict(payload.pop('metadata', {}))
        if payload:
            metadata.update(self._coerce_options_dict(payload))
        return FrameAccessPolicy(
            public=public,
            roles=roles,
            allow_anonymous=allow_anonymous,
            rate_limit_per_minute=rate_limit,
            cache_seconds=cache_seconds,
            metadata=metadata,
        )

    def _assign_frame_option(self, frame: Frame, key: str, value: Any) -> None:
        if not key:
            raise self._error("Option key cannot be empty", self.pos + 1)
        parts = [part.strip() for part in key.split('.') if part.strip()]
        if not parts:
            raise self._error("Option key cannot be empty", self.pos + 1)
        target = frame.options
        for part in parts[:-1]:
            existing = target.get(part)
            if not isinstance(existing, dict):
                existing = {}
                target[part] = existing
            target = existing
        target[parts[-1]] = value

    def _parse_frame_key_list(self, raw: str) -> List[str]:
        if not raw:
            return []
        parts = [segment.strip() for segment in raw.split(',') if segment.strip()]
        return [self._strip_quotes(part) for part in parts]

    def _parse_frame_splits(self, base_indent: int) -> Dict[str, float]:
        data = self._parse_kv_block(base_indent)
        splits: Dict[str, float] = {}
        for key, value in data.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                raise self._error(f"Split '{key}' must be numeric", self.pos, str(value))
            splits[str(key)] = numeric
        return splits

    def _parse_frame_source_config(self, base_indent: int) -> Optional[FrameSourceDef]:
        raw = self._parse_kv_block(base_indent)
        if not raw:
            return None
        kind = str(raw.pop('kind', raw.pop('type', 'file')) or 'file').lower()
        if kind not in {'file', 'sql'}:
            raise self._error("Frame source kind must be 'file' or 'sql'", self.pos)
        connection_raw = raw.pop('connection', raw.pop('conn', None))
        table_raw = raw.pop('table', raw.pop('dataset', None))
        path_raw = raw.pop('path', raw.pop('file', None))
        format_raw = raw.pop('format', raw.pop('fmt', None))
        if raw:
            # Merge remaining entries under metadata-like structure to avoid silent typos.
            extra_keys = ', '.join(raw.keys())
            raise self._error(f"Unknown keys in frame source block: {extra_keys}", self.pos)
        connection = self._stringify_value(connection_raw) if connection_raw is not None else None
        table = self._stringify_value(table_raw) if table_raw is not None else None
        path = self._stringify_value(path_raw) if path_raw is not None else None
        fmt = self._stringify_value(format_raw) if format_raw is not None else None
        if kind == 'sql':
            if not connection or not table:
                raise self._error("SQL frame sources require 'connection' and 'table'", self.pos)
            return FrameSourceDef(kind='sql', connection=connection, table=table)
        if not path:
            raise self._error("File frame sources require 'path'", self.pos)
        fmt_value = (fmt or 'csv').lower()
        if fmt_value not in {'csv', 'parquet'}:
            raise self._error("File frame sources support 'csv' or 'parquet'", self.pos)
        return FrameSourceDef(kind='file', path=path, format=fmt_value)


__all__ = ["FrameParserMixin"]
