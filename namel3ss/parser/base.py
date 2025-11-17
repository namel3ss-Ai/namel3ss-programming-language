from __future__ import annotations

import ast
import io
import re
import tokenize
from tokenize import TokenInfo
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union, Literal as TypingLiteral

_CONTEXT_SENTINEL = "__n3_ctx__"
_WHITESPACE_TOKENS: Set[int] = {
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
}
_BOOL_NORMALISATIONS: Dict[str, str] = {
    "true": "True",
    "false": "False",
    "null": "None",
    "none": "None",
}
_LIKE_TOKEN_MAP: Dict[str, str] = {
    "like": "<<",
    "ilike": ">>",
}

from namel3ss.ast import (
    AttributeRef,
    BinaryOp,
    CallExpression,
    CachePolicy,
    ContextValue,
    Expression,
    LayoutMeta,
    LayoutSpec,
    Literal,
    NameRef,
    PaginationPolicy,
    StreamingPolicy,
    UnaryOp,
    WindowFrame,
)


class N3SyntaxError(Exception):
    """Custom exception raised when encountering invalid syntax."""

    def __init__(self, message: str, line_no: int, line: str):
        super().__init__(f"Syntax error on line {line_no}: {message}\n{line}")
        self.line_no = line_no
        self.line = line


class ParserBase:
    """Shared parser state and helper utilities."""

    def __init__(self, source: str):
        self.lines: List[str] = source.splitlines()
        self.pos: int = 0
        self.app = None
        self._loop_depth: int = 0

    # ------------------------------------------------------------------
    # Cursor helpers
    # ------------------------------------------------------------------
    def _peek(self) -> Optional[str]:
        """Return the current line without consuming it."""
        if self.pos < len(self.lines):
            return self.lines[self.pos]
        return None

    def _advance(self) -> Optional[str]:
        """Return the current line and move the cursor forward."""
        line = self._peek()
        self.pos += 1
        return line

    def _indent(self, line: str) -> int:
        """Compute the indentation level (leading spaces) for *line*."""
        return len(line) - len(line.lstrip(' '))

    def _error(
        self,
        message: str,
        line_no: Optional[int] = None,
        line: Optional[str] = None,
    ) -> N3SyntaxError:
        """Create a consistent :class:`N3SyntaxError` instance."""

        if line_no is None:
            line_no = min(self.pos, len(self.lines))
        if line is None and 0 <= self.pos - 1 < len(self.lines):
            line = self.lines[self.pos - 1]
        elif line is None:
            line = ''
        return N3SyntaxError(message, line_no, line)

    # ------------------------------------------------------------------
    # Scalar coercion helpers
    # ------------------------------------------------------------------
    def _parse_bool(self, raw: str) -> bool:
        """Parse a boolean-like string into a bool."""
        value = raw.strip().lower()
        if value in {"true", "yes", "1", "on"}:
            return True
        if value in {"false", "no", "0", "off"}:
            return False
        raise self._error(f"Expected boolean value, found '{raw}'")

    def _parse_context_reference(self, token: str) -> Optional[ContextValue]:
        match = re.match(r'^(ctx|env):([A-Za-z0-9_\.]+)$', token)
        if not match:
            return None
        scope = match.group(1)
        path_text = match.group(2)
        if not path_text:
            return None
        path = [segment for segment in path_text.split('.') if segment]
        if not path:
            return None
        return ContextValue(scope=scope, path=path)

    def _coerce_scalar(self, raw: str) -> Any:
        """Attempt to coerce a scalar configuration value."""
        text = raw.strip()
        if not text:
            return text

        context_ref = self._parse_context_reference(text)
        if context_ref is not None:
            return context_ref
        lower = text.lower()
        if lower in {"true", "false", "null", "none"}:
            if lower in {"true", "false"}:
                return self._parse_bool(text)
            return None
        if re.fullmatch(r"[-+]?\d+", text):
            try:
                return int(text)
            except ValueError:
                pass
        if re.fullmatch(r"[-+]?\d*\.\d+", text):
            try:
                return float(text)
            except ValueError:
                pass
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            inner = text[1:-1]
            context_ref_inner = self._parse_context_reference(inner)
            if context_ref_inner is not None:
                return context_ref_inner
            return inner
        if text.startswith('[') or text.startswith('{') or text.startswith('('):
            try:
                parsed = ast.literal_eval(text)
                return parsed
            except (SyntaxError, ValueError):
                pass
        return text

    def _coerce_expression(self, value: Any) -> Expression:
        if isinstance(value, Expression):
            return value
        if isinstance(value, str):
            try:
                return self._parse_expression(value)
            except N3SyntaxError:
                return Literal(value)
        return Literal(value)

    def _coerce_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        try:
            text = str(value).strip()
            if not text:
                return None
            match = re.match(r"(-?\d+)", text)
            if not match:
                return None
            return int(match.group(1))
        except (ValueError, TypeError):
            return None

    def _stringify_value(self, value: Any) -> str:
        if value is None:
            return ''
        if isinstance(value, ContextValue):
            path = '.'.join(value.path)
            return f"{value.scope}:{path}" if path else value.scope
        if isinstance(value, Literal):
            inner = value.value
            return '' if inner is None else str(inner)
        if isinstance(value, Expression):
            return str(value)
        return str(value)

    # ------------------------------------------------------------------
    # Block helpers
    # ------------------------------------------------------------------
    def _parse_kv_block(self, parent_indent: int) -> Dict[str, Any]:
        """Parse an indented key/value block and return a dictionary."""
        config: Dict[str, Any] = {}
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
            match = re.match(r'([\w\.\s]+):\s*(.*)$', stripped)
            if not match:
                raise self._error("Expected 'key: value' inside block", self.pos + 1, line)
            key = match.group(1).strip()
            remainder = match.group(2)
            self._advance()
            if remainder == "":
                nested = self._parse_kv_block(indent)
                config[key] = nested
            else:
                value = self._coerce_scalar(remainder)
                config[key] = value
        return config

    # ------------------------------------------------------------------
    # Windowing & layout helpers
    # ------------------------------------------------------------------
    def _parse_window_frame(self, raw: str) -> WindowFrame:
        """Parse textual window specification into a WindowFrame."""
        text = raw.strip()
        lower = text.lower()
        mode: TypingLiteral["rolling", "expanding", "cumulative"] = "rolling"
        interval_value: Optional[int] = None
        interval_unit: Optional[str] = None

        match_last = re.match(r'last\s+(\d+)\s+([\w]+)', lower)
        if match_last:
            interval_value = int(match_last.group(1))
            interval_unit = match_last.group(2)
        elif lower.startswith('over all') or lower in {'all', 'overall', 'cumulative'}:
            mode = "cumulative"
        elif lower.startswith('expanding'):
            mode = "expanding"

        return WindowFrame(mode=mode, interval_value=interval_value, interval_unit=interval_unit)

    def _build_cache_policy(self, data: Dict[str, Any]) -> CachePolicy:
        if not data:
            return CachePolicy(strategy="none")
        strategy = str(data.get('strategy', 'memory') or 'memory').lower()
        ttl_raw = data.get('ttl_seconds') or data.get('ttl') or data.get('ttl_s')
        ttl_seconds: Optional[int] = None
        if ttl_raw is not None:
            if isinstance(ttl_raw, (int, float)):
                ttl_seconds = int(ttl_raw)
            else:
                ttl_clean = str(ttl_raw).strip()
                match_val = re.match(r'(\d+)', ttl_clean)
                if match_val:
                    ttl_seconds = int(match_val.group(1))
        max_entries = data.get('max_entries') or data.get('max rows') or data.get('max')
        if max_entries is not None and not isinstance(max_entries, int):
            try:
                max_entries = int(str(max_entries))
            except ValueError:
                max_entries = None
        return CachePolicy(strategy=strategy, ttl_seconds=ttl_seconds, max_entries=max_entries)

    def _build_pagination_policy(self, data: Dict[str, Any]) -> PaginationPolicy:
        if not data:
            return PaginationPolicy(enabled=False)
        enabled = self._parse_bool(str(data.get('enabled', 'true'))) if 'enabled' in data else True
        page_size = data.get('page_size') or data.get('page size') or data.get('size')
        if page_size is not None and not isinstance(page_size, int):
            try:
                page_size = int(str(page_size))
            except ValueError:
                page_size = None
        max_pages = data.get('max_pages') or data.get('max pages')
        if max_pages is not None and not isinstance(max_pages, int):
            try:
                max_pages = int(str(max_pages))
            except ValueError:
                max_pages = None
        return PaginationPolicy(enabled=enabled, page_size=page_size, max_pages=max_pages)

    def _build_streaming_policy(self, data: Dict[str, Any]) -> StreamingPolicy:
        if not data:
            return StreamingPolicy(enabled=True)
        enabled = self._parse_bool(str(data.get('enabled', 'true'))) if 'enabled' in data else True
        chunk_size = data.get('chunk_size') or data.get('chunk size') or data.get('batch')
        if chunk_size is not None and not isinstance(chunk_size, int):
            try:
                chunk_size = int(str(chunk_size))
            except ValueError:
                chunk_size = None
        return StreamingPolicy(enabled=enabled, chunk_size=chunk_size)

    def _build_layout_spec(self, data: Dict[str, Any]) -> LayoutSpec:
        layout = LayoutSpec()

        def _to_int(val: Any) -> Optional[int]:
            if isinstance(val, int):
                return val
            if isinstance(val, float):
                return int(val)
            try:
                if isinstance(val, str) and not val.strip():
                    return None
                return int(val)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                try:
                    return int(float(str(val)))
                except (ValueError, TypeError):
                    return None

        for key, value in data.items():
            lower = key.replace(' ', '_').lower()
            if lower == 'width':
                maybe = _to_int(value)
                if maybe is not None:
                    layout.width = maybe
            elif lower == 'height':
                maybe = _to_int(value)
                if maybe is not None:
                    layout.height = maybe
            elif lower == 'variant':
                layout.variant = str(value)
            elif lower == 'order':
                maybe = _to_int(value)
                if maybe is not None:
                    layout.order = maybe
            elif lower == 'area':
                layout.area = str(value)
            elif lower == 'breakpoint':
                layout.breakpoint = str(value)
            else:
                layout.props[key] = value
        return layout

    def _build_layout_meta(self, data: Dict[str, Any]) -> LayoutMeta:
        meta = LayoutMeta()

        def _to_int(val: Any) -> Optional[int]:
            if isinstance(val, int):
                return val
            if isinstance(val, float):
                return int(val)
            try:
                if isinstance(val, str) and not val.strip():
                    return None
                return int(val)
            except (ValueError, TypeError):
                try:
                    return int(float(str(val)))
                except (ValueError, TypeError):
                    return None

        extras: Dict[str, Any] = {}
        for key, value in data.items():
            lower = key.replace(' ', '_').lower()
            if lower == 'width':
                maybe = _to_int(value)
                if maybe is not None:
                    meta.width = maybe
            elif lower == 'height':
                maybe = _to_int(value)
                if maybe is not None:
                    meta.height = maybe
            elif lower == 'variant':
                meta.variant = str(value)
            elif lower == 'align':
                meta.align = str(value)
            elif lower == 'emphasis':
                meta.emphasis = str(value)
            else:
                extras[key] = value
        meta.extras = extras
        return meta

    # Placeholder methods expected in subclasses/mixins
    def _parse_expression(self, text: str) -> Expression:
        """Parse an expression string into an Expression AST node."""

        source = (text or "").strip()
        if not source:
            raise self._error("Expression cannot be empty", self.pos, text)

        def _raise(message: str) -> None:
            raise self._error(message, self.pos, text)

        prepared = _prepare_expression_source(source, _raise)
        try:
            parsed = ast.parse(prepared, mode="eval")
        except SyntaxError as exc:
            details = exc.msg or "Invalid expression syntax"
            if exc.offset is not None:
                details = f"{details} (column {exc.offset})"
            _raise(details)

        builder = _ExpressionBuilder(_raise)
        try:
            expression = builder.convert(parsed)
        except N3SyntaxError:
            raise
        except Exception as exc:  # pragma: no cover - defensive safeguard
            _raise(f"Unsupported expression element: {exc}")

        if not isinstance(expression, Expression):  # pragma: no cover - defensive
            _raise("Parsed expression did not produce an Expression node")
        return expression

def _prepare_expression_source(source: str, raise_error: Callable[[str], None]) -> str:
    """Convert N3 expression syntax into Python-compatible source for AST parsing."""

    reader = io.StringIO(source).readline
    try:
        tokens = list(tokenize.generate_tokens(reader))
    except tokenize.TokenError as exc:
        raise_error(f"Invalid expression: {exc}")

    result: List[TokenInfo] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        token_type = token.type
        token_str = token.string

        if token_type == tokenize.ENDMARKER:
            result.append(token)
            break

        if token_type == tokenize.NAME:
            lowered = token_str.lower()

            if lowered in _BOOL_NORMALISATIONS:
                normalised = _BOOL_NORMALISATIONS[lowered]
                result.append(TokenInfo(tokenize.NAME, normalised, token.start, token.end, token.line))
                i += 1
                continue

            if lowered in {"ctx", "env"}:
                j = i + 1
                while j < len(tokens) and tokens[j].type in _WHITESPACE_TOKENS:
                    j += 1
                if j < len(tokens) and tokens[j].type == tokenize.OP and tokens[j].string == ":":
                    j += 1
                    path_parts: List[str] = []
                    expect_segment = True
                    while j < len(tokens):
                        lookahead = tokens[j]
                        if lookahead.type in _WHITESPACE_TOKENS:
                            j += 1
                            continue
                        if expect_segment and lookahead.type in {tokenize.NAME, tokenize.NUMBER}:
                            path_parts.append(lookahead.string)
                            j += 1
                            expect_segment = False
                            continue
                        if not expect_segment and lookahead.type == tokenize.OP and lookahead.string == '.':
                            expect_segment = True
                            j += 1
                            continue
                        break
                    if expect_segment:
                        raise_error("Expected context path after prefix")
                    if not path_parts:
                        raise_error("Context path cannot be empty")
                    result.extend(_build_context_tokens(lowered, path_parts, token))
                    i = j
                    continue

            if lowered in _LIKE_TOKEN_MAP:
                j = i + 1
                while j < len(tokens) and tokens[j].type in _WHITESPACE_TOKENS:
                    j += 1
                if j < len(tokens) and tokens[j].type == tokenize.OP and tokens[j].string == '(':
                    result.append(token)
                    i += 1
                    continue
                replacement = _LIKE_TOKEN_MAP[lowered]
                result.append(TokenInfo(tokenize.OP, replacement, token.start, token.end, token.line))
                i += 1
                continue

            result.append(token)
            i += 1
            continue

        if token_type == tokenize.OP:
            if token_str == '=':
                result.append(TokenInfo(tokenize.OP, '==', token.start, token.end, token.line))
                i += 1
                continue
            if token_str == '<>':
                result.append(TokenInfo(tokenize.OP, '!=', token.start, token.end, token.line))
                i += 1
                continue
            if token_str == '<' and i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if nxt.type == tokenize.OP and nxt.string == '>':
                    result.append(TokenInfo(tokenize.OP, '!=', token.start, nxt.end, token.line))
                    i += 2
                    continue

        result.append(token)
        i += 1

    return tokenize.untokenize(result)


def _build_context_tokens(scope: str, path: List[str], template: TokenInfo) -> List[TokenInfo]:
    """Create the token sequence representing a context reference call."""

    line = template.line
    tokens: List[TokenInfo] = [
        TokenInfo(tokenize.NAME, _CONTEXT_SENTINEL, template.start, template.end, line),
        TokenInfo(tokenize.OP, '(', template.start, template.end, line),
        TokenInfo(tokenize.STRING, repr(scope), template.start, template.end, line),
    ]
    for segment in path:
        tokens.append(TokenInfo(tokenize.OP, ',', template.start, template.end, line))
        tokens.append(TokenInfo(tokenize.STRING, repr(segment), template.start, template.end, line))
    tokens.append(TokenInfo(tokenize.OP, ')', template.start, template.end, line))
    return tokens


class _ExpressionBuilder(ast.NodeVisitor):
    """Convert Python AST nodes into Namel3ss Expression instances."""

    def __init__(self, raise_error: Callable[[str], None]) -> None:
        self._raise = raise_error

    def convert(self, node: ast.AST) -> Expression:
        if isinstance(node, ast.Expression):
            return self.visit(node)
        return self.visit(ast.Expression(body=node))  # pragma: no cover - defensive

    def visit_Expression(self, node: ast.Expression) -> Expression:  # type: ignore[override]
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Expression:  # type: ignore[override]
        return Literal(node.value)

    def visit_NameConstant(self, node: ast.NameConstant) -> Expression:  # pragma: no cover - py<3.8
        return Literal(node.value)

    def visit_Num(self, node: ast.Num) -> Expression:  # pragma: no cover - py<3.8
        return Literal(node.n)

    def visit_Str(self, node: ast.Str) -> Expression:  # pragma: no cover - py<3.8
        return Literal(node.s)

    def visit_List(self, node: ast.List) -> Expression:  # type: ignore[override]
        return Literal(self._literal_eval(node))

    def visit_Tuple(self, node: ast.Tuple) -> Expression:  # type: ignore[override]
        return Literal(self._literal_eval(node))

    def visit_Set(self, node: ast.Set) -> Expression:  # type: ignore[override]
        return Literal(self._literal_eval(node))

    def visit_Dict(self, node: ast.Dict) -> Expression:  # type: ignore[override]
        return Literal(self._literal_eval(node))

    def visit_Name(self, node: ast.Name) -> Expression:  # type: ignore[override]
        identifier = node.id
        if identifier == _CONTEXT_SENTINEL:
            self._raise("Context reference placeholder cannot appear directly")
        if identifier.startswith('__') and identifier not in {'__builtins__'}:
            self._raise(f"Name '{identifier}' is not permitted in expressions")
        return NameRef(name=identifier)

    def visit_Attribute(self, node: ast.Attribute) -> Expression:  # type: ignore[override]
        return self._convert_attribute(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Expression:  # type: ignore[override]
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return UnaryOp(op='not', operand=operand)
        if isinstance(node.op, ast.USub):
            return UnaryOp(op='-', operand=operand)
        if isinstance(node.op, ast.UAdd):
            return UnaryOp(op='+', operand=operand)
        self._raise(f"Unsupported unary operator '{type(node.op).__name__}'")
        raise AssertionError  # pragma: no cover - unreachable

    def visit_BoolOp(self, node: ast.BoolOp) -> Expression:  # type: ignore[override]
        if not node.values:
            self._raise("Boolean expression is empty")
        if isinstance(node.op, ast.And):
            op = 'and'
        elif isinstance(node.op, ast.Or):
            op = 'or'
        else:
            self._raise("Unsupported boolean operator")
        current = self.visit(node.values[0])
        for value in node.values[1:]:
            current = BinaryOp(left=current, op=op, right=self.visit(value))
        return current

    def visit_BinOp(self, node: ast.BinOp) -> Expression:  # type: ignore[override]
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            op = '+'
        elif isinstance(node.op, ast.Sub):
            op = '-'
        elif isinstance(node.op, ast.Mult):
            op = '*'
        elif isinstance(node.op, ast.Div):
            op = '/'
        elif isinstance(node.op, ast.Mod):
            op = '%'
        elif isinstance(node.op, ast.LShift):
            op = 'like'
        elif isinstance(node.op, ast.RShift):
            op = 'ilike'
        else:
            self._raise(f"Unsupported binary operator '{type(node.op).__name__}'")
        return BinaryOp(left=left, op=op, right=right)

    def visit_Compare(self, node: ast.Compare) -> Expression:  # type: ignore[override]
        if not node.ops:
            return self.visit(node.left)
        left_expr = self.visit(node.left)
        result: Optional[Expression] = None
        current_left = left_expr
        for op_node, comparator in zip(node.ops, node.comparators):
            right_expr = self.visit(comparator)
            op = self._map_compare_operator(op_node)
            comparison = BinaryOp(left=current_left, op=op, right=right_expr)
            if op == 'in':
                if not isinstance(right_expr, Literal) or not isinstance(right_expr.value, (list, tuple, set)):
                    self._raise("'in' operator requires a literal list, tuple, or set on the right-hand side")
            if result is None:
                result = comparison
            else:
                result = BinaryOp(left=result, op='and', right=comparison)
            current_left = right_expr
        return result if result is not None else left_expr

    def visit_Call(self, node: ast.Call) -> Expression:  # type: ignore[override]
        if isinstance(node.func, ast.Name) and node.func.id == _CONTEXT_SENTINEL:
            return self._convert_context_call(node)
        if node.keywords:
            self._raise("Function calls in expressions do not support keyword arguments")
        arguments: List[Expression] = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                self._raise("Varargs are not supported in expressions")
            arguments.append(self.visit(arg))
        function = self._convert_callable(node.func)
        return CallExpression(function=function, arguments=arguments)

    def visit_Subscript(self, node: ast.Subscript) -> Expression:  # type: ignore[override]
        self._raise("Subscript expressions are not supported")
        raise AssertionError  # pragma: no cover - unreachable

    def visit_Lambda(self, node: ast.Lambda) -> Expression:  # pragma: no cover - unsupported
        self._raise("Lambda expressions are not supported")
        raise AssertionError

    def visit_ListComp(self, node: ast.ListComp) -> Expression:  # pragma: no cover - unsupported
        self._raise("Comprehensions are not supported")
        raise AssertionError

    def visit_DictComp(self, node: ast.DictComp) -> Expression:  # pragma: no cover - unsupported
        self._raise("Comprehensions are not supported")
        raise AssertionError

    def visit_SetComp(self, node: ast.SetComp) -> Expression:  # pragma: no cover - unsupported
        self._raise("Comprehensions are not supported")
        raise AssertionError

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Expression:  # pragma: no cover - unsupported
        self._raise("Generator expressions are not supported")
        raise AssertionError

    def generic_visit(self, node: ast.AST) -> Expression:  # type: ignore[override]
        self._raise(f"Unsupported expression element '{type(node).__name__}'")
        raise AssertionError  # pragma: no cover - unreachable

    def _literal_eval(self, node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except Exception as exc:
            self._raise(f"Unsupported literal value: {exc}")
            raise AssertionError  # pragma: no cover - unreachable

    def _convert_attribute(self, node: ast.Attribute) -> Expression:
        base = node.value
        if isinstance(base, ast.Name):
            return AttributeRef(base=base.id, attr=node.attr)
        if isinstance(base, ast.Attribute):
            parent = self._convert_attribute(base)
            prefix = self._flatten_attribute_name(parent)
            return AttributeRef(base=prefix, attr=node.attr)
        self._raise("Attribute access must start from a name or attribute chain")
        raise AssertionError  # pragma: no cover - unreachable

    def _convert_callable(self, node: ast.AST) -> Union[NameRef, AttributeRef]:
        if isinstance(node, ast.Name):
            return NameRef(name=node.id)
        if isinstance(node, ast.Attribute):
            attr_expr = self._convert_attribute(node)
            if isinstance(attr_expr, NameRef):  # pragma: no cover - not expected
                return attr_expr
            return attr_expr
        self._raise("Unsupported function reference in call expression")
        raise AssertionError  # pragma: no cover - unreachable

    def _flatten_attribute_name(self, expr: Expression) -> str:
        if isinstance(expr, NameRef):
            return expr.name
        if isinstance(expr, AttributeRef):
            if expr.base:
                return f"{expr.base}.{expr.attr}"
            return expr.attr
        self._raise("Invalid attribute chain")
        raise AssertionError  # pragma: no cover - unreachable

    def _convert_context_call(self, node: ast.Call) -> ContextValue:
        if node.keywords:
            self._raise("Context references do not support keyword arguments")
        if not node.args:
            self._raise("Context reference requires scope and path segments")
        scope_node = node.args[0]
        if not isinstance(scope_node, ast.Constant) or not isinstance(scope_node.value, str):
            self._raise("Context scope must be a string literal")
        scope = scope_node.value.lower()
        if scope not in {"ctx", "env"}:
            self._raise("Context scope must be 'ctx' or 'env'")
        path: List[str] = []
        for arg in node.args[1:]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                path.append(arg.value)
                continue
            self._raise("Context path segments must be string literals")
        if not path:
            self._raise("Context reference must include at least one path segment")
        return ContextValue(scope=scope, path=path)

    def _map_compare_operator(self, op: ast.cmpop) -> str:
        if isinstance(op, ast.Eq):
            return '=='
        if isinstance(op, ast.NotEq):
            return '!='
        if isinstance(op, ast.Lt):
            return '<'
        if isinstance(op, ast.LtE):
            return '<='
        if isinstance(op, ast.Gt):
            return '>'
        if isinstance(op, ast.GtE):
            return '>='
        if isinstance(op, ast.In):
            return 'in'
        self._raise(f"Unsupported comparison operator '{type(op).__name__}'")
        raise AssertionError  # pragma: no cover - unreachable
