"""Basic declaration parsing (app, theme, dataset, frame)."""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .helpers import _Line

from namel3ss.ast import (
    App,
    Dataset,
    FilterOp,
    Frame,
    FrameColumn,
    GroupByOp,
    Theme,
)


class DeclarationsParserMixin:
    """Mixin providing basic declaration parsing methods."""

    def _parse_app(self, line: _Line) -> None:
        """Parse app declaration: app "Name" [connects to postgres "ALIAS"] or app:"""
        from .constants import APP_HEADER_RE
        
        stripped = line.text.strip()
        
        # Handle "app:" syntax (name in body)
        if stripped == 'app:':
            if self._app is not None and self._explicit_app_declared:
                raise self._error('Only one app declaration is allowed', line)
            
            base_indent = self._indent(line.text)
            self._advance()
            
            # Parse name from body
            name = None
            database = None
            while True:
                nxt = self._peek_line()
                if nxt is None:
                    break
                indent = self._indent(nxt.text)
                nxt_stripped = nxt.text.strip()
                
                if nxt_stripped and indent <= base_indent:
                    break
                
                if not nxt_stripped or nxt_stripped.startswith('#') or nxt_stripped.startswith('//'):
                    self._advance()
                    continue
                
                # Parse name: value
                if ':' in nxt_stripped:
                    key, _, value = nxt_stripped.partition(':')
                    key = key.strip()
                    value = value.strip()
                    if key == 'name':
                        name = value.strip('"').strip("'")
                    elif key == 'database':
                        database = value.strip('"').strip("'")
                
                self._advance()
            
            if not name:
                name = 'app'  # Default name
            
            self._app = App(name=name, database=database)
            self._explicit_app_declared = True
            return
        
        # Handle "app Name" syntax
        match = APP_HEADER_RE.match(stripped)
        if not match:
            raise self._error('Expected: app "Name" [connects to postgres "ALIAS"] or app:', line)
        name = match.group(1)
        database = match.group(2)
        if self._app is not None and self._explicit_app_declared:
            raise self._error('Only one app declaration is allowed', line)
        self._app = App(name=name, database=database)
        self._explicit_app_declared = True
        self._advance()

    def _parse_theme(self, line: _Line) -> None:
        """Parse theme block with key-value pairs."""
        base_indent = self._indent(line.text)
        if not line.text.rstrip().endswith(':'):
            raise self._error("Theme declaration must end with ':'", line)
        self._advance()
        entries = self._parse_kv_block(base_indent)
        self._ensure_app(line)
        theme = self._app.theme if self._app and self._app.theme else Theme()
        theme.values.update(entries)
        if self._app:
            self._app.theme = theme

    def _parse_dataset(self, line: _Line) -> None:
        """Parse dataset declaration: dataset "name" from source_type "source":"""
        from .constants import DATASET_HEADER_RE
        
        match = DATASET_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "dataset declaration")
        name = match.group(1)
        source_type = match.group(2)
        raw_source = match.group(3)
        source = raw_source.strip('"')
        base_indent = self._indent(line.text)
        self._advance()
        operations = []
        while True:
            next_line = self._peek_line()
            if next_line is None:
                break
            stripped = next_line.text.strip()
            indent = self._indent(next_line.text)
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            if stripped.startswith('filter by:'):
                expr_text = stripped.split(':', 1)[1].strip()
                expr = self._expression_helper.parse(expr_text, line_no=next_line.number, line=next_line.text)
                operations.append(FilterOp(condition=expr))
                self._advance()
                continue
            if stripped.startswith('group by:'):
                columns_text = stripped.split(':', 1)[1]
                columns = [col.strip() for col in columns_text.split(',') if col.strip()]
                operations.append(GroupByOp(columns=columns))
                self._advance()
                continue
            # Unknown clause - advance before raising to avoid infinite loop
            self._advance()
            self._unsupported(next_line, "dataset clause")
        dataset = Dataset(name=name, source_type=source_type, source=source, operations=operations)
        self._ensure_app(line)
        if self._app:
            self._app.datasets.append(dataset)

    def _parse_frame(self, line: _Line) -> None:
        """Parse frame declaration: frame "name" [from source_type "source"]:"""
        from .constants import FRAME_HEADER_RE
        
        match = FRAME_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "frame declaration")
        name = match.group(1)
        source_type = match.group(2) or 'dataset'
        source = match.group(3).strip('"') if match.group(3) else None
        base_indent = self._indent(line.text)
        self._advance()
        columns: List[FrameColumn] = []
        description: Optional[str] = None
        while True:
            next_line = self._peek_line()
            if next_line is None:
                break
            stripped = next_line.text.strip()
            indent = self._indent(next_line.text)
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            if stripped.startswith('columns:'):
                column_text = stripped.split(':', 1)[1]
                column_names = [col.strip() for col in column_text.split(',') if col.strip()]
                columns.extend(FrameColumn(name=col) for col in column_names)
                self._advance()
                continue
            if stripped.startswith('description:'):
                description = stripped.split(':', 1)[1].strip().strip('"')
                self._advance()
                continue
            self._unsupported(next_line, "frame clause")
        frame = Frame(name=name, source_type=source_type, source=source, description=description, columns=columns)
        self._ensure_app(line)
        if self._app:
            self._app.frames.append(frame)


__all__ = ['DeclarationsParserMixin']
