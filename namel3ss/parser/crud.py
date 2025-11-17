"""Parser support for CRUD resource declarations."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from namel3ss.ast import CrudResource

from .base import ParserBase

_ALLOWED_OPERATIONS = ("list", "retrieve", "create", "update", "delete")
_ALLOWED_OPERATION_SET = set(_ALLOWED_OPERATIONS)

_OPERATION_ALIASES = {
    "read": {"list", "retrieve"},
    "all": set(_ALLOWED_OPERATIONS),
    "*": set(_ALLOWED_OPERATIONS),
    "write": {"create", "update", "delete"},
}


class CrudParserMixin(ParserBase):
    """Parse ``enable crud`` declarations at the top level."""

    _RESOURCE_PATTERN = re.compile(
    r"^enable\s+crud\s+for\s+(table|dataset)\s+([\w\.]+?)(?:\s+as\s+\"([^\"]+)\")?([:.])?$",
        flags=re.IGNORECASE,
    )

    def _parse_crud_resource(self, line: str, line_no: int, base_indent: int) -> CrudResource:
        match = self._RESOURCE_PATTERN.match(line.strip())
        if not match:
            raise self._error(
                "Expected: enable crud for table <name> [as \"Slug\"]:",
                line_no,
                line,
            )

        source_type = match.group(1).lower()
        source_name = self._validate_identifier(match.group(2), line_no, line, allow_dot=True)
        alias = match.group(3)
        terminator = match.group(4) or ""
        has_block = terminator == ":"

        resource_slug = self._slugify_resource(alias or source_name)
        resource_label = alias.strip() if alias else None

        options: Dict[str, object] = {}
        if has_block:
            options = self._parse_crud_options_block(base_indent)
        options = self._normalise_option_keys(options)

        slug_override = self._option_value(options, "slug", "name")
        if isinstance(slug_override, str) and slug_override.strip():
            resource_slug = self._slugify_resource(slug_override)

        label_option = self._option_value(options, "label", "title")
        if isinstance(label_option, str):
            cleaned_label = label_option.strip()
            if cleaned_label:
                resource_label = cleaned_label

        primary_key_raw = self._option_value(options, "primary key", "pk")
        primary_key = self._validate_identifier(primary_key_raw or "id", line_no, line)

        select_fields = self._coerce_identifier_list(
            self._option_value(options, "select", "columns", "fields"),
            line_no,
            line,
        )
        mutable_fields = self._coerce_identifier_list(
            self._option_value(options, "mutable", "write", "editable"),
            line_no,
            line,
        )

        tenant_column_value = self._option_value(options, "tenant column", "tenant")
        tenant_column = (
            self._validate_identifier(tenant_column_value, line_no, line)
            if tenant_column_value
            else None
        )

        default_limit = self._coerce_limit(
            self._option_value(options, "page size", "default limit"),
            line_no,
            line,
            fallback=100,
        )
        max_limit = self._coerce_limit(
            self._option_value(options, "max limit", "maximum limit"),
            line_no,
            line,
            fallback=500,
        )

        read_only = self._coerce_bool_option(options, line_no, line)

        allowed_operations = self._coerce_operations(options, line_no, line)
        if read_only:
            allowed_operations = [op for op in allowed_operations if op in {"list", "retrieve"}]

        return CrudResource(
            name=resource_slug,
            source_type=source_type,
            source_name=source_name,
            primary_key=primary_key or "id",
            select_fields=select_fields,
            mutable_fields=mutable_fields,
            allowed_operations=allowed_operations,
            tenant_column=tenant_column,
            default_limit=default_limit,
            max_limit=max(default_limit, max_limit),
            read_only=read_only,
            label=resource_label,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parse_crud_options_block(self, parent_indent: int) -> Dict[str, object]:
        options: Dict[str, object] = {}
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith("#"):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            match = re.match(r"([\w\s]+):\s*(.*)$", stripped)
            if not match:
                raise self._error("Expected 'key: value' inside CRUD block", self.pos + 1, nxt)
            key = match.group(1).strip().lower()
            remainder = match.group(2)
            self._advance()
            if remainder == "":
                nested = self._parse_kv_block(indent)
                nested_normalised = self._normalise_option_keys(nested)
                if key == "options":
                    options.update(nested_normalised)
                else:
                    options[key] = nested_normalised
            else:
                options[key] = self._coerce_scalar(remainder)
        return options

    def _slugify_resource(self, value: str) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return "resource"
        slug = re.sub(r"[^a-z0-9]+", "-", text)
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug or "resource"

    def _validate_identifier(
        self,
        value: Optional[object],
        line_no: int,
        line_text: str,
        *,
        allow_dot: bool = False,
    ) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        pattern = r"^[A-Za-z_][A-Za-z0-9_]*$"
        if allow_dot and "." in text:
            segments = text.split(".")
            if all(re.match(pattern, segment) for segment in segments):
                return text
        elif re.match(pattern, text):
            return text
        raise self._error(f"Invalid identifier '{text}'", line_no, line_text)

    def _coerce_identifier_list(
        self,
        value: Optional[object],
        line_no: int,
        line_text: str,
    ) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            items = list(value)
        else:
            items = [segment.strip() for segment in str(value).split(",")]
        output: List[str] = []
        for item in items:
            if not item:
                continue
            identifier = self._validate_identifier(item, line_no, line_text)
            if identifier and identifier not in output:
                output.append(identifier)
        return output

    def _coerce_limit(
        self,
        value: Optional[object],
        line_no: int,
        line_text: str,
        *,
        fallback: int,
    ) -> int:
        try:
            coerced = self._coerce_int(value)
        except Exception:
            coerced = None
        if coerced is None or coerced <= 0:
            return fallback
        return coerced

    def _coerce_operations(
        self,
        options: Dict[str, object],
        line_no: int,
        line_text: str,
    ) -> List[str]:
        raw = self._option_value(options, "allow", "allowed", "operations", "ops")
        denied = self._option_value(options, "deny", "denied", "except", "without")
        operations: List[str]
        if raw is None:
            operations = list(_ALLOWED_OPERATIONS)
        else:
            operations = []
            candidates = self._coerce_identifier_list(raw, line_no, line_text)
            for candidate in candidates:
                lowered = candidate.lower()
                expanded = set(_OPERATION_ALIASES.get(lowered, {lowered}))
                for op in _ALLOWED_OPERATIONS:
                    if op in expanded and op not in operations:
                        operations.append(op)
        if denied is not None:
            deny_list = self._coerce_identifier_list(denied, line_no, line_text)
            denied_ops = {self._normalise_operation(value) for value in deny_list}
            operations = [op for op in operations if op not in denied_ops]
        return operations or ["list", "retrieve"]

    def _normalise_operation(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        text = str(value).strip().lower()
        if text in _ALLOWED_OPERATION_SET:
            return text
        if text == "read":
            return "retrieve"
        if text == "list":
            return "list"
        if text == "view":
            return "retrieve"
        return None

    def _coerce_bool_option(
        self,
        options: Dict[str, object],
        line_no: int,
        line_text: str,
    ) -> bool:
        raw = self._option_value(options, "read only", "readonly")
        if raw is None:
            return False
        if isinstance(raw, bool):
            return raw
        try:
            return self._parse_bool(str(raw))
        except Exception as exc:
            raise self._error(f"Invalid boolean for 'read only': {raw}", line_no, line_text) from exc

    def _normalise_option_keys(self, data: Dict[str, object]) -> Dict[str, object]:
        normalised: Dict[str, object] = {}
        for key, value in data.items():
            normal_key = str(key).strip().lower()
            normalised[normal_key] = value
        return normalised

    def _option_value(self, options: Dict[str, object], *candidates: str) -> Optional[object]:
        for name in candidates:
            if name in options:
                return options[name]
        return None


__all__ = ["CrudParserMixin"]

