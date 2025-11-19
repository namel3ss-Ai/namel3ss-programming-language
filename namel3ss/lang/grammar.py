"""Grammar-driven parser entry points for Namel3ss `.n3` programs.

This module introduces a lightweight EBNF description of the supported
surface syntax and a reference implementation that produces the existing
AST structures used throughout the runtime.  The grammar intentionally
captures the high-level layout of a module while delegating certain
expression details to the established expression parser mixin.

```
module         ::= directive* declaration*
directive      ::= module_decl | import_stmt | language_version
module_decl    ::= "module" dotted_name NEWLINE
import_stmt    ::= "import" dotted_name ("as" NAME)? (":" import_target ("," import_target)*)? NEWLINE
import_target  ::= NAME ("as" NAME)?
language_version ::= "language_version" STRING NEWLINE

declaration    ::= app_decl | theme_block | dataset_def | frame_def | page_block
                 | model_block | ai_model_block | prompt_block | memory_block
                 | template_block | chain_block | experiment_block | crud_block
                 | evaluator_block | metric_block | guardrail_block

app_decl       ::= "app" STRING ("connects to" NAME STRING)? "."?
theme_block    ::= "theme" ":" INDENT theme_entry+ DEDENT

dataset_def    ::= "dataset" STRING "from" source_ref ":" INDENT dataset_stmt+ DEDENT
dataset_stmt   ::= "filter by" ":" expression
                 | "group by" ":" name_list
                 | "order by" ":" name_list
                 | "transform" NAME ":" expression

frame_def      ::= "frame" STRING ("from" source_ref)? ":" INDENT frame_stmt+ DEDENT
frame_stmt     ::= "columns" ":" column_list
                 | "description" ":" STRING

page_block     ::= "page" STRING "at" STRING ":" INDENT page_stmt+ DEDENT
page_stmt      ::= show_stmt | form_stmt | action_stmt | control_flow_stmt
control_flow_stmt ::= if_stmt | for_stmt
if_stmt        ::= "if" expression ":" INDENT page_stmt+ DEDENT (elif_stmt)* (else_stmt)?
for_stmt       ::= "for" NAME "in" ("dataset"|"table"|"frame") NAME ":" INDENT page_stmt+ DEDENT
show_stmt      ::= "show" ("text"|"table"|"chart") ...
```

Only a subset of the listed productions are implemented at this stage;
the focus of Phase 1 is to support realistic modules that declare an app,
datasets, frames, pages, and page-level control-flow.  The grammar text
serves both as documentation and as a roadmap for future incremental
coverage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Callable

from namel3ss.ast import (
    App,
    Dataset,
    FilterOp,
    Frame,
    FrameColumn,
    GroupByOp,
    Page,
    PageStatement,
    ShowChart,
    ShowTable,
    ShowText,
    Theme,
)
from namel3ss.ast.ai import (
    LLMDefinition, 
    ToolDefinition, 
    PromptArgument, 
    Prompt, 
    OutputSchema, 
    OutputField, 
    OutputFieldType, 
    Chain, 
    ChainStep,
    Connector,
    Template,
    Memory,
    AIModel,
    TrainingJob,
    TuningJob,
)
from namel3ss.ast.rag import IndexDefinition, RagPipelineDefinition
from namel3ss.ast.agents import AgentDefinition, GraphDefinition, GraphEdge, MemoryConfig
from namel3ss.ast.policy import PolicyDefinition
from namel3ss.ast.pages import ElifBlock, ForLoop, IfBlock
from namel3ss.ast.program import Module
from namel3ss.ast.modules import Import, ImportedName
from namel3ss.parser.base import N3SyntaxError, ParserBase
from namel3ss.parser.expressions import ExpressionParserMixin
# Import AIParserMixin from specific module to avoid circular import
from namel3ss.parser.ai import AIParserMixin as _AIParserMixin
from namel3ss.parser.logic import LogicParserMixin as _LogicParserMixin


_DATASET_HEADER_RE = re.compile(r'^dataset\s+"([^"]+)"\s+from\s+(\w+)\s+([A-Za-z0-9_"\.]+)\s*:\s*$')
_FRAME_HEADER_RE = re.compile(r'^frame\s+"([^"]+)"(?:\s+from\s+(\w+)\s+([A-Za-z0-9_"\.]+))?\s*:\s*$')
_PAGE_HEADER_RE = re.compile(r'^page\s+"([^"]+)"\s+at\s+"([^"]+)"\s*:\s*$')
_LLM_HEADER_RE = re.compile(r'^llm\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$')
_TOOL_HEADER_RE = re.compile(r'^tool\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$')
_PROMPT_HEADER_RE = re.compile(r'^prompt\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$')
_CHAIN_HEADER_RE = re.compile(r'^define\s+chain\s+"([^"]+)"(?:\s+effect\s+([\w\-]+))?\s*:\s*$')
_INDEX_HEADER_RE = re.compile(r'^index\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$')
_RAG_PIPELINE_HEADER_RE = re.compile(r'^rag_pipeline\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$')
_AGENT_HEADER_RE = re.compile(r'^agent\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*$')
_GRAPH_HEADER_RE = re.compile(r'^graph\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*$')
_POLICY_HEADER_RE = re.compile(r'^policy\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*$')
_APP_HEADER_RE = re.compile(
    r'^app\s+"([^"]+)"(?:\s+connects\s+to\s+[A-Za-z_][A-Za-z0-9_]*\s+"([^"]+)")?\s*\.?$'
)
_MODULE_DECL_RE = re.compile(r'^module\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*$')
_LANGUAGE_VERSION_RE = re.compile(r'^language_version\s+"([0-9]+\.[0-9]+\.[0-9]+)"\s*\.?$')
_IMPORT_TARGET_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)(?:\s+as\s+([A-Za-z_][A-Za-z0-9_]*))?\s*$')


class GrammarUnsupportedError(Exception):
    """Raised when the grammar-backed parser encounters an unsupported construct."""


class _ExpressionHelper(ExpressionParserMixin):
    """Adapter around the legacy expression parser for reuse in the grammar parser."""

    def __init__(self) -> None:
        # The base parser expects an initial source buffer; we keep it empty
        # and populate `lines`/`pos` before every parse call.
        ParserBase.__init__(self, "", path="")

    def parse(self, text: str, *, line_no: int, line: str) -> object:
        self.lines = [line]
        self.pos = line_no
        return self._parse_expression(text)


def parse_module(source: str, path: str = "", module_name: Optional[str] = None) -> Module:
    """Parse *source* using the grammar-backed parser and return a Module AST."""

    parser = _GrammarModuleParser(source=source, path=path, module_name=module_name)
    try:
        return parser.parse()
    except GrammarUnsupportedError:
        from namel3ss.parser.program import LegacyProgramParser

        legacy = LegacyProgramParser(source, module_name=module_name, path=path)
        return legacy.parse()


@dataclass
class _Line:
    text: str
    number: int


class _GrammarModuleParser(_LogicParserMixin, _AIParserMixin):
    """Recursive-descent parser that follows the EBNF outlined above.
    
    Inherits from LogicParserMixin and AIParserMixin to support full DSL parsing including
    knowledge bases, logic queries, structured prompts, AI models, training jobs, etc.
    """

    def __init__(self, source: str, *, path: str = "", module_name: Optional[str] = None) -> None:
        # Grammar-specific state
        self._source = source
        self._lines_raw = source.splitlines()
        self._lines = [_Line(text=text, number=i + 1) for i, text in enumerate(self._lines_raw)]
        self._path = path
        self._cursor = 0
        self._app: Optional[App] = None
        self._explicit_app_declared = False
        self._module_name: Optional[str] = None
        self._module_name_override = module_name
        self._language_version: Optional[str] = None
        self._imports: List[Import] = []
        self._extra_nodes: List[object] = []
        self._expression_helper = _ExpressionHelper()
        self._directives_locked = False
        self._in_ai_block = False
        
        # Initialize ParserBase infrastructure required by AIParserMixin
        # AIParserMixin expects self.lines to be raw strings, not _Line objects
        ParserBase.__init__(self, source, path=path)
        self.lines = self._lines_raw  # AIParserMixin needs raw strings
        self.pos = 0

    # ------------------------------------------------------------------
    # Synchronization helpers for AIParserMixin integration
    # ------------------------------------------------------------------
    def _sync_cursor_to_pos(self) -> None:
        """Sync _cursor from pos after AIParserMixin operations."""
        self._cursor = self.pos

    def _sync_pos_to_cursor(self) -> None:
        """Sync pos to _cursor before calling AIParserMixin methods."""
        self.pos = self._cursor
    
    def _advance(self) -> None:
        """Override to keep cursor and pos in sync."""
        self._cursor += 1
        self.pos = self._cursor

    # ------------------------------------------------------------------
    # High-level driver
    # ------------------------------------------------------------------
    def parse(self) -> Module:
        while self._cursor < len(self._lines):
            line = self._peek_line()
            if line is None:
                break
            stripped = line.text.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            indent = self._indent(line.text)
            if indent != 0:
                raise self._error("Top level statements must not be indented", line)
            if stripped.startswith('module '):
                if self._directives_locked:
                    raise self._error("Module declaration must appear before other statements", line)
                self._parse_module_declaration(line)
                continue
            if stripped.startswith('import '):
                if self._directives_locked:
                    raise self._error("Import statements must appear before other declarations", line)
                self._parse_import(line)
                continue
            if stripped.lower().startswith('language_version'):
                if self._directives_locked:
                    raise self._error("language_version must appear before other declarations", line)
                if self._language_version is not None:
                    raise self._error('language_version directive may only appear once', line)
                self._parse_language_version(line)
                continue
            self._directives_locked = True
            if stripped.startswith('app ') or stripped == 'app:':
                self._parse_app(line)
                continue
            if stripped.startswith('theme'):
                self._parse_theme(line)
                continue
            if stripped.startswith('dataset '):
                self._parse_dataset(line)
                continue
            if stripped.startswith('frame '):
                self._parse_frame(line)
                continue
            if stripped.startswith('page '):
                self._parse_page(line)
                continue
            if stripped.startswith('llm '):
                self._parse_llm(line)
                continue
            if stripped.startswith('tool '):
                self._parse_tool(line)
                continue
            if stripped.startswith('prompt '):
                self._parse_prompt_wrapper(line)
                continue
            if stripped.startswith('define chain '):
                self._parse_chain_wrapper(line)
                continue
            if stripped.startswith('connector '):
                self._parse_connector_wrapper(line)
                continue
            if stripped.startswith('define template '):
                self._parse_template_wrapper(line)
                continue
            if stripped.startswith('memory '):
                self._parse_memory_wrapper(line)
                continue
            if stripped.startswith('model ') or stripped.startswith('ai model '):
                self._parse_ai_model_wrapper(line)
                continue
            if stripped.startswith('training '):
                self._parse_training_job_wrapper(line)
                continue
            if stripped.startswith('tuning '):
                self._parse_tuning_job_wrapper(line)
                continue
            if stripped.startswith('index '):
                self._parse_index(line)
                continue
            if stripped.startswith('rag_pipeline '):
                self._parse_rag_pipeline(line)
                continue
            if stripped.startswith('agent '):
                self._parse_agent(line)
                continue
            if stripped.startswith('knowledge '):
                self._parse_knowledge_wrapper(line)
                continue
            if stripped.startswith('query '):
                self._parse_query_wrapper(line)
                continue
            if stripped.startswith('graph '):
                self._parse_graph(line)
                continue
            if stripped.startswith('policy '):
                self._parse_policy(line)
                continue
            if stripped.startswith('fn '):
                self._parse_function_def(line)
                continue
            if stripped.startswith('rule '):
                self._parse_rule_def(line)
                continue
            self._unsupported(line, f"top-level statement '{stripped.split()[0]}'")

        body: List[object] = []
        if self._app is not None:
            # Collect knowledge modules and queries from extra nodes
            from namel3ss.ast import KnowledgeModule, LogicQuery
            knowledge_modules = [n for n in self._extra_nodes if isinstance(n, KnowledgeModule)]
            queries = [n for n in self._extra_nodes if isinstance(n, LogicQuery)]
            
            # Populate App fields
            if knowledge_modules:
                self._app.knowledge_modules = knowledge_modules
            if queries:
                self._app.queries = queries
            
            body.append(self._app)
        body.extend(self._extra_nodes)
        return Module(
            name=self._module_name or self._module_name_override,
            language_version=self._language_version,
            path=self._path,
            imports=list(self._imports),
            body=body,
            has_explicit_app=self._explicit_app_declared,
        )

    # ------------------------------------------------------------------
    # Directive parsing
    # ------------------------------------------------------------------
    def _parse_module_declaration(self, line: _Line) -> None:
        match = _MODULE_DECL_RE.match(line.text.strip())
        if not match:
            raise self._error("Expected: module <name>", line)
        if self._module_name is not None:
            raise self._error("Only one module declaration is allowed", line)
        if self._imports:
            raise self._error("Module declaration must appear before imports", line)
        if self._app is not None:
            raise self._error("Module declaration must appear before app declaration", line)
        self._module_name = match.group(1)
        self._advance()

    def _parse_import(self, line: _Line) -> None:
        remainder = line.text.strip()[len('import ') :].strip()
        if not remainder:
            raise self._error("Expected module path after 'import'", line)
        module_part = remainder
        names_part: Optional[str] = None
        colon_index = remainder.find(':')
        if colon_index != -1:
            module_part = remainder[:colon_index].strip()
            names_part = remainder[colon_index + 1 :].strip()
            if not names_part:
                raise self._error("Expected imported names after ':'", line)
        target_match = _IMPORT_TARGET_RE.match(module_part)
        if not target_match:
            raise self._error("Invalid module import target", line)
        alias = target_match.group(2)
        module_name = target_match.group(1)
        names: Optional[List[ImportedName]] = None
        if names_part is not None:
            names = self._parse_import_names(names_part, line)
            if alias:
                raise self._error("Module alias is not allowed when selecting specific names", line)
        self._imports.append(Import(module=module_name, names=names, alias=alias))
        self._advance()

    def _parse_import_names(self, segment: str, line: _Line) -> List[ImportedName]:
        entries = [piece.strip() for piece in segment.split(',') if piece.strip()]
        results: List[ImportedName] = []
        for entry in entries:
            match = _IMPORT_TARGET_RE.match(entry)
            if not match or match.group(1).count('.'):
                raise self._error("Invalid imported name", line)
            results.append(ImportedName(name=match.group(1), alias=match.group(2)))
        if not results:
            raise self._error("Import list cannot be empty", line)
        return results

    def _parse_language_version(self, line: _Line) -> None:
        match = _LANGUAGE_VERSION_RE.match(line.text.strip())
        if not match:
            raise self._error('Expected: language_version "<semver>"', line)
        if self._language_version is not None:
            raise self._error('language_version directive may only appear once', line)
        self._language_version = match.group(1)
        self._advance()

    # ------------------------------------------------------------------
    # Top-level declarations
    # ------------------------------------------------------------------
    def _parse_app(self, line: _Line) -> None:
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
                
                if not nxt_stripped or nxt_stripped.startswith('#'):
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
        match = _APP_HEADER_RE.match(stripped)
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
        match = _DATASET_HEADER_RE.match(line.text.strip())
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
            if not stripped or stripped.startswith('#'):
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
            self._unsupported(next_line, "dataset clause")
        dataset = Dataset(name=name, source_type=source_type, source=source, operations=operations)
        self._ensure_app(line)
        if self._app:
            self._app.datasets.append(dataset)

    def _parse_frame(self, line: _Line) -> None:
        match = _FRAME_HEADER_RE.match(line.text.strip())
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
            if not stripped or stripped.startswith('#'):
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

    def _parse_page(self, line: _Line) -> None:
        match = _PAGE_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "page declaration")
        page_name = match.group(1)
        route = match.group(2)
        base_indent = self._indent(line.text)
        self._advance()
        statements = self._parse_page_statements(base_indent)
        page = Page(name=page_name, route=route, statements=statements)
        self._ensure_app(line)
        if self._app:
            self._app.pages.append(page)

    # ------------------------------------------------------------------
    # Page statements and control flow
    # ------------------------------------------------------------------
    def _parse_page_statements(self, parent_indent: int) -> List[PageStatement]:
        statements: List[PageStatement] = []
        while True:
            line = self._peek_line()
            if line is None:
                break
            stripped = line.text.strip()
            indent = self._indent(line.text)
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            if stripped.startswith('show text '):
                statements.append(self._parse_show_text(line))
                continue
            if stripped.startswith('show table '):
                statements.append(self._parse_show_table(line))
                continue
            if stripped.startswith('show chart '):
                statements.append(self._parse_show_chart(line))
                continue
            if stripped.startswith('if '):
                statements.append(self._parse_if_block(line, indent))
                continue
            if stripped.startswith('for '):
                statements.append(self._parse_for_loop(line, indent))
                continue
            self._unsupported(line, "page statement")
        return statements

    def _parse_show_text(self, line: _Line) -> ShowText:
        match = re.match(r'^\s*show\s+text\s+"([^"]+)"\s*$', line.text)
        if not match:
            raise self._error('Expected: show text "message"', line)
        self._advance()
        return ShowText(text=match.group(1))

    def _parse_show_table(self, line: _Line) -> ShowTable:
        match = re.match(
            r'^\s*show\s+table\s+"([^"]+)"\s+from\s+(dataset|table|frame)\s+([A-Za-z_][A-Za-z0-9_]*)\s*$',
            line.text,
        )
        if not match:
            self._unsupported(line, "show table statement")
        self._advance()
        return ShowTable(title=match.group(1), source_type=match.group(2), source=match.group(3))

    def _parse_show_chart(self, line: _Line) -> ShowChart:
        match = re.match(
            r'^\s*show\s+chart\s+"([^"]+)"\s+from\s+(dataset|table)\s+([A-Za-z_][A-Za-z0-9_]*)\s*$',
            line.text,
        )
        if not match:
            self._unsupported(line, "show chart statement")
        self._advance()
        return ShowChart(heading=match.group(1), source_type=match.group(2), source=match.group(3))

    def _parse_if_block(self, line: _Line, indent: int) -> IfBlock:
        condition_text = line.text.strip()
        if not condition_text.endswith(':'):
            raise self._error("if statement must end with ':'", line)
        condition_src = condition_text[len('if') : -1].strip()
        condition = self._expression_helper.parse(condition_src, line_no=line.number, line=line.text)
        self._advance()
        body = self._parse_page_statements(indent)
        elifs: List[ElifBlock] = []
        else_body: Optional[List[PageStatement]] = None
        while True:
            next_line = self._peek_line()
            if next_line is None:
                break
            stripped = next_line.text.strip()
            next_indent = self._indent(next_line.text)
            if not stripped:
                self._advance()
                continue
            if next_indent != indent:
                break
            if stripped.startswith('elif '):
                if not stripped.endswith(':'):
                    raise self._error("elif statement must end with ':'", next_line)
                expr_text = stripped[len('elif') : -1].strip()
                condition_expr = self._expression_helper.parse(expr_text, line_no=next_line.number, line=next_line.text)
                self._advance()
                elif_body = self._parse_page_statements(indent)
                elifs.append(ElifBlock(condition=condition_expr, body=elif_body))
                continue
            if stripped.startswith('else:'):
                self._advance()
                else_body = self._parse_page_statements(indent)
                break
            break
        return IfBlock(condition=condition, body=body, elifs=elifs, else_body=else_body)

    def _parse_for_loop(self, line: _Line, indent: int) -> ForLoop:
        match = re.match(r'^\s*for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(dataset|table|frame)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$', line.text)
        if not match:
            raise self._error('Expected: for name in dataset foo:', line)
        loop_var = match.group(1)
        source_kind = match.group(2)
        source_name = match.group(3)
        self._advance()
        body = self._parse_page_statements(indent)
        return ForLoop(loop_var=loop_var, source_kind=source_kind, source_name=source_name, body=body)

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def _parse_kv_block(self, parent_indent: int) -> dict[str, str]:
        if self._in_ai_block:
            # Delegate to ParserBase implementation when AIParserMixin is driving parsing.
            return ParserBase._parse_kv_block(self, parent_indent)
        entries: dict[str, str] = {}
        while True:
            line = self._peek_line()
            if line is None:
                break
            stripped = line.text.strip()
            indent = self._indent(line.text)
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            match = re.match(r'^([A-Za-z0-9_\-\s]+):\s*(.+)$', stripped)
            if not match:
                raise self._error("Expected 'key: value' inside block", line)
            key = match.group(1).strip()
            value = match.group(2).strip()
            entries[key] = value
            self._advance()
        return entries

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
            config=config,
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
        """
        Parse a prompt definition block with typed arguments and output schema.
        
        Grammar:
            prompt <name>:
                args:
                    <arg_name>: <type> [= <default>]
                    ...
                output_schema:
                    <field_name>: <type>
                    ...
                template: <string>
                model: <llm_name>
        """
        match = _PROMPT_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "prompt declaration")
        name = match.group(1)
        base_indent = self._indent(line.text)
        self._advance()
        
        # Initialize fields
        args: List[PromptArgument] = []
        output_schema: Optional[OutputSchema] = None
        template: Optional[str] = None
        model: Optional[str] = None
        config: Dict[str, Any] = {}
        
        # Parse block content
        while True:
            nxt = self._peek_line()
            if nxt is None:
                break
            indent = self._indent(nxt.text)
            stripped = nxt.text.strip()
            
            # Stop if we've dedented
            if stripped and indent <= base_indent:
                break
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            
            lowered = stripped.lower()
            
            # Handle args: block
            if lowered.startswith('args:'):
                self._advance()
                args = self._parse_prompt_args_block(indent)
            # Handle output_schema: block
            elif lowered.startswith('output_schema:'):
                self._advance()
                output_schema = self._parse_output_schema_block(indent)
            # Handle template: field
            elif lowered.startswith('template:'):
                self._advance()
                template = self._parse_prompt_template(indent, stripped)
            # Handle model: field
            elif lowered.startswith('model:'):
                model = stripped.split(':', 1)[1].strip()
                self._advance()
            # Handle other config fields
            else:
                if ':' in stripped:
                    key, val = stripped.split(':', 1)
                    config[key.strip()] = val.strip()
                self._advance()
        
        if not template:
            raise self._error("prompt block requires 'template' field", line)
        
        prompt = Prompt(
            name=name,
            model=model or '',
            template=template,
            args=args,
            output_schema=output_schema,
            parameters=config,
        )
        
        self._ensure_app(line)
        self._app.prompts.append(prompt)

    def _parse_prompt_args_block(self, parent_indent: int) -> List[PromptArgument]:
        """Parse the args: block for a structured prompt."""
        args: List[PromptArgument] = []
        
        while True:
            nxt = self._peek_line()
            if nxt is None:
                break
            
            indent = self._indent(nxt.text)
            stripped = nxt.text.strip()
            
            # Stop if dedented
            if stripped and indent <= parent_indent:
                break
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            
            # Parse arg line: name: type [= default]
            if ':' in stripped:
                parts = stripped.split(':', 1)
                arg_name = parts[0].strip()
                arg_spec = parts[1].strip()
                
                # Check for default value
                arg_type = 'string'
                default = None
                required = True
                
                if '=' in arg_spec:
                    type_part, default_part = arg_spec.split('=', 1)
                    arg_type = type_part.strip()
                    default = default_part.strip()
                    # Remove quotes from default if present
                    if default.startswith('"') and default.endswith('"'):
                        default = default[1:-1]
                    elif default.startswith("'") and default.endswith("'"):
                        default = default[1:-1]
                    required = False
                else:
                    arg_type = arg_spec
                
                args.append(PromptArgument(
                    name=arg_name,
                    arg_type=arg_type,
                    required=required,
                    default=default,
                ))
            
            self._advance()
        
        return args

    def _parse_output_schema_block(self, parent_indent: int) -> OutputSchema:
        """Parse the output_schema: block for a structured prompt."""
        fields: List[OutputField] = []
        
        while True:
            nxt = self._peek_line()
            if nxt is None:
                break
            
            indent = self._indent(nxt.text)
            stripped = nxt.text.strip()
            
            # Stop if dedented
            if stripped and indent <= parent_indent:
                break
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            
            # Parse field line: name: type
            if ':' in stripped:
                parts = stripped.split(':', 1)
                field_name = parts[0].strip()
                type_spec = parts[1].strip()
                
                # Check for optional marker (?)
                required = True
                if field_name.endswith('?'):
                    field_name = field_name[:-1]
                    required = False
                
                # Parse the type specification
                field_type = self._parse_output_field_type(type_spec)
                
                fields.append(OutputField(
                    name=field_name,
                    field_type=field_type,
                    required=required,
                ))
            
            self._advance()
        
        return OutputSchema(fields=fields)

    def _parse_output_field_type(self, type_spec: str, line_no: int = None, line: str = None) -> OutputFieldType:
        """
        Parse a type specification like 'string', 'enum(\"a\", \"b\")', 'list[string]', 'object {...}'.
        
        Args compatible with both Grammar (1 arg) and AIParserMixin (3 args) call patterns.
        """
        type_spec = type_spec.strip()
        
        # Handle enum: enum(\"val1\", \"val2\", ...)
        if type_spec.startswith('enum(') or type_spec.startswith('enum['):
            # Extract enum values
            start_char = '(' if '(' in type_spec else '['
            end_char = ')' if start_char == '(' else ']'
            start_idx = type_spec.index(start_char) + 1
            end_idx = type_spec.rindex(end_char)
            values_str = type_spec[start_idx:end_idx]
            
            # Split and clean values
            enum_values = []
            for val in values_str.split(','):
                val = val.strip()
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                elif val.startswith("'") and val.endswith("'"):
                    val = val[1:-1]
                if val:
                    enum_values.append(val)
            
            return OutputFieldType(base_type='enum', enum_values=enum_values)
        
        # Handle list: list[element_type]
        if type_spec.startswith('list[') or type_spec.startswith('list<'):
            start_idx = type_spec.index('[') if '[' in type_spec else type_spec.index('<')
            end_idx = type_spec.rindex(']') if '[' in type_spec else type_spec.rindex('>')
            element_type_str = type_spec[start_idx+1:end_idx].strip()
            element_type = self._parse_output_field_type(element_type_str)
            return OutputFieldType(base_type='list', element_type=element_type)
        
        # Handle object: object { ... } (simplified - just mark as object for now)
        if type_spec.startswith('object'):
            # For now, we'll create a simple object type
            # Full nested object parsing would require more complex logic
            return OutputFieldType(base_type='object')
        
        # Handle primitives: string, int, float, bool
        if type_spec in ('string', 'int', 'float', 'bool'):
            return OutputFieldType(base_type=type_spec)
        
        # Default to string
        return OutputFieldType(base_type='string')

    def _parse_prompt_template(self, parent_indent: int, first_line: str) -> str:
        """Parse template field - can be inline or multiline block."""
        # Check if there's inline content after template:
        if 'template:' in first_line.lower():
            inline = first_line.split(':', 1)[1].strip()
            if inline:
                # Remove quotes if present
                if inline.startswith('"""') or inline.startswith("'''"):
                    # Multiline string starts on same line - collect until end marker
                    quote = inline[:3]
                    content = inline[3:]
                    lines = [content]
                    
                    while True:
                        nxt = self._peek_line()
                        if nxt is None:
                            break
                        line_text = nxt.text.rstrip()
                        self._advance()
                        
                        if quote in line_text:
                            # Found end quote
                            end_idx = line_text.index(quote)
                            lines.append(line_text[:end_idx])
                            break
                        else:
                            lines.append(line_text)
                    
                    return '\n'.join(lines)
                elif inline.startswith('"') and inline.endswith('"'):
                    return inline[1:-1]
                elif inline.startswith("'") and inline.endswith("'"):
                    return inline[1:-1]
                else:
                    return inline
        
        # Otherwise, expect a multiline block
        lines = []
        while True:
            nxt = self._peek_line()
            if nxt is None:
                break
            
            indent = self._indent(nxt.text)
            stripped = nxt.text.strip()
            
            # Stop if dedented
            if stripped and indent <= parent_indent:
                break
            
            # Add line content (preserving indentation within the block)
            if stripped:
                lines.append(nxt.text[parent_indent+2:] if len(nxt.text) > parent_indent+2 else stripped)
            else:
                lines.append('')
            
            self._advance()
        
        return '\n'.join(lines)

    def _parse_chain_legacy(self, line: _Line) -> None:
        """
        Legacy chain parser (Deprecated - Use AIParserMixin version)
        
        Parse a chain definition.
        
        Grammar:
            define chain "<name>" [effect <effect_name>]:
                policy: <policy_name>
                <key>: <value>
                input -> step1 | step2 | ...
        
        Example:
            define chain "support_agent" effect read:
                policy: strict_safety
                input -> template.customer_support(query = input.query) | llm.chat_model
        """
        match = _CHAIN_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "chain declaration")
        
        name = match.group(1)
        effect = match.group(2) if match.lastindex >= 2 else None
        
        base_indent = self._indent(line.text)
        self._advance()
        
        # Initialize fields
        policy_name: Optional[str] = None
        metadata: Dict[str, Any] = {}
        steps: List[ChainStep] = []
        input_key: str = "input"
        
        # Parse block content
        while True:
            nxt = self._peek_line()
            if nxt is None:
                break
            indent = self._indent(nxt.text)
            stripped = nxt.text.strip()
            
            # Stop if we've dedented
            if stripped and indent <= base_indent:
                break
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            
            lowered = stripped.lower()
            
            # Handle policy reference
            if lowered.startswith('policy:'):
                policy_name = stripped.split(':', 1)[1].strip().strip('"').strip("'")
                self._advance()
            # Handle pipeline with ->
            elif '->' in stripped:
                # Simple pipeline parsing: input -> step1 | step2
                pipeline_parts = stripped.split('->')
                if len(pipeline_parts) >= 2:
                    # First part is input
                    input_part = pipeline_parts[0].strip()
                    if input_part.lower().startswith('input'):
                        parts = input_part.split()
                        if len(parts) > 1:
                            input_key = parts[1]
                    
                    # Rest is the pipeline
                    pipeline_str = '->'.join(pipeline_parts[1:]).strip()
                    # Split by | for steps
                    step_strs = [s.strip() for s in pipeline_str.split('|')]
                    
                    for step_str in step_strs:
                        if not step_str:
                            continue
                        
                        # Parse step: "template.name(args)" or "llm.model"
                        tokens = step_str.split()
                        if tokens:
                            # Simple step parsing - just capture the step string
                            # Full parsing would need expression parser integration
                            kind = tokens[0].split('.')[0] if '.' in tokens[0] else tokens[0]
                            target = tokens[0]
                            steps.append(ChainStep(kind=kind, target=target, options={}))
                
                self._advance()
            # Handle other metadata
            elif ':' in stripped:
                key, val = stripped.split(':', 1)
                metadata[key.strip()] = val.strip()
                self._advance()
            else:
                self._advance()
        
        chain = Chain(
            name=name,
            input_key=input_key,
            steps=steps,
            metadata=metadata,
            declared_effect=effect,
            policy_name=policy_name,
        )
        
        self._ensure_app(line)
        self._app.chains.append(chain)

    def _parse_index(self, line: _Line) -> None:
        """
        Parse an index definition block.
        
        Grammar:
            index <name>:
                source_dataset: <dataset_name>
                embedding_model: <model_name>
                chunk_size: <int>
                overlap: <int>
                backend: <pgvector|qdrant|weaviate>
                namespace: <string>
                collection: <string>
                table_name: <string>
                metadata_fields: [<field1>, <field2>, ...]
        """
        match = _INDEX_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "index declaration")
        name = match.group(1)
        base_indent = self._indent(line.text)
        self._advance()
        
        # Parse key-value properties
        properties = self._parse_kv_block(base_indent)
        
        # Extract required fields
        source_dataset = properties.get('source_dataset')
        embedding_model = properties.get('embedding_model')
        if not source_dataset or not embedding_model:
            raise self._error("index block requires 'source_dataset' and 'embedding_model' fields", line)
        
        # Extract optional fields with defaults
        chunk_size = int(properties.get('chunk_size', 512))
        overlap = int(properties.get('overlap', 64))
        backend = properties.get('backend', 'pgvector')
        namespace = properties.get('namespace')
        collection = properties.get('collection')
        table_name = properties.get('table_name')
        
        # Parse metadata_fields if present (can be list or comma-separated string)
        metadata_fields = None
        if 'metadata_fields' in properties:
            mf = properties['metadata_fields']
            if isinstance(mf, list):
                metadata_fields = mf
            elif isinstance(mf, str):
                # Parse string like "[field1, field2]" or "field1, field2"
                mf = mf.strip()
                if mf.startswith('[') and mf.endswith(']'):
                    mf = mf[1:-1]
                metadata_fields = [f.strip() for f in mf.split(',') if f.strip()]
        
        # Build config from remaining properties
        config = {k: v for k, v in properties.items() 
                  if k not in {'source_dataset', 'embedding_model', 'chunk_size', 'overlap', 
                               'backend', 'namespace', 'collection', 'table_name', 'metadata_fields'}}
        
        index = IndexDefinition(
            name=name,
            source_dataset=source_dataset,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            overlap=overlap,
            backend=backend,
            namespace=namespace,
            collection=collection,
            table_name=table_name,
            metadata_fields=metadata_fields,
            config=config,
        )
        
        self._ensure_app(line)
        self._app.indices.append(index)

    def _parse_rag_pipeline(self, line: _Line) -> None:
        """
        Parse a RAG pipeline definition block.
        
        Grammar:
            rag_pipeline <name>:
                query_encoder: <embedding_model_name>
                index: <index_name>
                top_k: <int>
                reranker: <reranker_model_name>
                distance_metric: <cosine|euclidean|dot>
                filters: {<key>: <value>, ...}
        """
        match = _RAG_PIPELINE_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "rag_pipeline declaration")
        name = match.group(1)
        base_indent = self._indent(line.text)
        self._advance()
        
        # Parse key-value properties
        properties = self._parse_kv_block(base_indent)
        
        # Extract required fields
        query_encoder = properties.get('query_encoder')
        index = properties.get('index')
        if not query_encoder or not index:
            raise self._error("rag_pipeline block requires 'query_encoder' and 'index' fields", line)
        
        # Extract optional fields with defaults
        top_k = int(properties.get('top_k', 5))
        reranker = properties.get('reranker')
        distance_metric = properties.get('distance_metric', 'cosine')
        
        # Parse filters if present
        filters = None
        if 'filters' in properties:
            f = properties['filters']
            if isinstance(f, dict):
                filters = f
            elif isinstance(f, str):
                # Try to parse as dict-like string
                import json
                try:
                    filters = json.loads(f)
                except json.JSONDecodeError:
                    pass
        
        # Build config from remaining properties
        config = {k: v for k, v in properties.items() 
                  if k not in {'query_encoder', 'index', 'top_k', 'reranker', 
                               'distance_metric', 'filters'}}
        
        rag_pipeline = RagPipelineDefinition(
            name=name,
            query_encoder=query_encoder,
            index=index,
            top_k=top_k,
            reranker=reranker,
            distance_metric=distance_metric,
            filters=filters,
            config=config,
        )
        
        self._ensure_app(line)
        self._app.rag_pipelines.append(rag_pipeline)

    def _parse_agent(self, line: _Line) -> None:
        """
        Parse an agent definition block.
        
        Grammar:
            agent <name> {
                llm: <llm_name>
                tools: [<tool1>, <tool2>, ...]
                memory: "<policy>" or {config}
                goal: "<description>"
                system_prompt: "<prompt>"  # optional
                max_turns: <int>  # optional
                temperature: <float>  # optional
            }
        """
        match = _AGENT_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "agent declaration")
        name = match.group(1)
        base_indent = self._indent(line.text)
        self._advance()
        
        # Parse key-value properties within braces
        properties = self._parse_kv_block_braces(base_indent)
        
        # Extract required fields
        llm_name = properties.get('llm')
        if not llm_name:
            raise self._error("agent block requires 'llm' field", line)
        
        # Parse tools list
        tools_raw = properties.get('tools', '[]')
        tool_names = self._parse_list_field(tools_raw)
        
        # Parse memory config
        memory_raw = properties.get('memory')
        memory_config = None
        if memory_raw:
            if isinstance(memory_raw, str):
                # Simple string policy like "conversation" or "none"
                memory_config = memory_raw
            elif isinstance(memory_raw, dict):
                memory_config = MemoryConfig(**memory_raw)
        
        goal = properties.get('goal', '')
        system_prompt = properties.get('system_prompt')
        max_turns = int(properties['max_turns']) if 'max_turns' in properties else None
        max_tokens = int(properties['max_tokens']) if 'max_tokens' in properties else None
        temperature = float(properties['temperature']) if 'temperature' in properties else None
        top_p = float(properties['top_p']) if 'top_p' in properties else None
        
        # Build config from remaining properties
        config = {k: v for k, v in properties.items() 
                  if k not in {'llm', 'tools', 'memory', 'goal', 'system_prompt', 
                               'max_turns', 'max_tokens', 'temperature', 'top_p'}}
        
        agent = AgentDefinition(
            name=name,
            llm_name=llm_name,
            tool_names=tool_names,
            memory_config=memory_config,
            goal=goal,
            system_prompt=system_prompt,
            max_turns=max_turns,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            config=config,
        )
        
        self._ensure_app(line)
        self._app.agents.append(agent)

    def _parse_graph(self, line: _Line) -> None:
        """
        Parse a graph definition block.
        
        Grammar:
            graph <name> {
                start: <agent_name>
                edges: [
                    { from: <agent1>, to: <agent2>, when: "<condition>" },
                    ...
                ]
                termination: <agent_name> or [<agent1>, <agent2>]
                max_hops: <int>  # optional
                timeout_ms: <int>  # optional
            }
        """
        match = _GRAPH_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "graph declaration")
        name = match.group(1)
        base_indent = self._indent(line.text)
        self._advance()
        
        # Parse key-value properties within braces
        properties = self._parse_kv_block_braces(base_indent)
        
        # Extract required fields
        start_agent = properties.get('start')
        if not start_agent:
            raise self._error("graph block requires 'start' field", line)
        
        # Parse edges list
        edges_raw = properties.get('edges', '[]')
        edges = self._parse_graph_edges(edges_raw)
        
        # Parse termination (can be single agent or list)
        termination_raw = properties.get('termination')
        termination_agents = []
        termination_condition = None
        if termination_raw:
            if isinstance(termination_raw, str):
                # Check if it's a condition expression or agent name
                if termination_raw.startswith('"') or '==' in termination_raw or 'and' in termination_raw:
                    termination_condition = termination_raw.strip('"')
                else:
                    termination_agents = [termination_raw]
            elif isinstance(termination_raw, list):
                termination_agents = termination_raw
        
        max_hops = int(properties['max_hops']) if 'max_hops' in properties else None
        timeout_ms = int(properties['timeout_ms']) if 'timeout_ms' in properties else None
        timeout_s = float(properties['timeout_s']) if 'timeout_s' in properties else None
        
        # Build config from remaining properties
        config = {k: v for k, v in properties.items() 
                  if k not in {'start', 'edges', 'termination', 'max_hops', 'timeout_ms', 'timeout_s'}}
        
        graph = GraphDefinition(
            name=name,
            start_agent=start_agent,
            edges=edges,
            termination_agents=termination_agents,
            termination_condition=termination_condition,
            max_hops=max_hops,
            timeout_ms=timeout_ms,
            timeout_s=timeout_s,
            config=config,
        )
        
        self._ensure_app(line)
        self._app.graphs.append(graph)

    def _parse_policy(self, line: _Line) -> None:
        """
        Parse a policy definition block.
        
        Grammar:
            policy <name> {
                block_categories: ["self-harm", "hate", "sexual_minors"]
                allow_categories: ["educational"]
                alert_only_categories: ["profanity"]
                redact_pii: true
                max_tokens: 512
                fallback_message: "I can't help with that."
                log_level: "full"
            }
        """
        match = _POLICY_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "policy declaration")
        name = match.group(1)
        base_indent = self._indent(line.text)
        self._advance()
        
        # Parse key-value properties within braces
        properties = self._parse_kv_block_braces(base_indent)
        
        # Extract policy fields
        block_categories = properties.get('block_categories', [])
        if isinstance(block_categories, str):
            # Parse string representation of list
            block_categories = self._parse_string_list(block_categories)
        
        allow_categories = properties.get('allow_categories', [])
        if isinstance(allow_categories, str):
            allow_categories = self._parse_string_list(allow_categories)
        
        alert_only_categories = properties.get('alert_only_categories', [])
        if isinstance(alert_only_categories, str):
            alert_only_categories = self._parse_string_list(alert_only_categories)
        
        redact_pii = properties.get('redact_pii', False)
        if isinstance(redact_pii, str):
            redact_pii = redact_pii.lower() in ('true', 'yes', '1')
        
        max_tokens = properties.get('max_tokens')
        if max_tokens is not None:
            max_tokens = int(max_tokens)
        
        fallback_message = properties.get('fallback_message')
        if fallback_message and isinstance(fallback_message, str):
            # Remove quotes if present
            if fallback_message.startswith('"') and fallback_message.endswith('"'):
                fallback_message = fallback_message[1:-1]
            elif fallback_message.startswith("'") and fallback_message.endswith("'"):
                fallback_message = fallback_message[1:-1]
        
        log_level = properties.get('log_level', 'full')
        if isinstance(log_level, str):
            log_level = log_level.strip('"').strip("'")
        
        # Build config from remaining properties
        config = {
            k: v for k, v in properties.items()
            if k not in {
                'block_categories', 'allow_categories', 'alert_only_categories',
                'redact_pii', 'max_tokens', 'fallback_message', 'log_level'
            }
        }
        
        policy = PolicyDefinition(
            name=name,
            block_categories=block_categories,
            allow_categories=allow_categories,
            alert_only_categories=alert_only_categories,
            redact_pii=redact_pii,
            max_tokens=max_tokens,
            fallback_message=fallback_message,
            log_level=log_level,
            config=config,
        )
        
        self._ensure_app(line)
        self._app.policies.append(policy)

    def _parse_kv_block_braces(self, parent_indent: int) -> dict[str, any]:
        """Parse a key-value block enclosed in braces."""
        entries: dict[str, any] = {}
        depth = 1  # Start with one open brace
        
        while True:
            line = self._peek_line()
            if line is None:
                break
            stripped = line.text.strip()
            
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            
            # Check for closing brace
            if stripped == '}':
                depth -= 1
                self._advance()
                if depth == 0:
                    break
                continue
            
            # Parse key-value pair
            match = re.match(r'^([A-Za-z0-9_\-\s]+):\s*(.*)$', stripped)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                
                # Check if value is a multi-line list
                if value.startswith('[') and not value.endswith(']'):
                    # Collect multi-line list (this will advance past all list lines)
                    entries[key] = self._collect_multiline_list()
                    continue  # Skip the normal advance at the end
                # Parse value (handle lists, strings, numbers)
                elif value.startswith('['):
                    # Single-line list
                    entries[key] = self._parse_list_value(value)
                elif value.startswith('"') and value.endswith('"'):
                    entries[key] = value[1:-1]
                elif value.replace('.', '').replace('-', '').isdigit():
                    entries[key] = float(value) if '.' in value else int(value)
                elif value == '':
                    # Empty value, might be multi-line, skip for now
                    pass
                else:
                    entries[key] = value
            
            self._advance()
        
        return entries

    def _collect_multiline_list(self) -> str:
        """Collect a multi-line list value into a single string."""
        lines = []
        bracket_depth = 1  # Already saw opening [
        
        self._advance()  # Move past the key: [ line
        
        while True:
            line = self._peek_line()
            if line is None:
                break
            
            stripped = line.text.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            
            lines.append(stripped)
            
            # Track bracket depth
            bracket_depth += stripped.count('[')
            bracket_depth -= stripped.count(']')
            
            self._advance()
            
            if bracket_depth == 0:
                break
        
        return '[' + ' '.join(lines) + ']'

    def _parse_list_field(self, value) -> List[str]:
        """Parse a list field like [item1, item2, item3]."""
        # If already a list, return it
        if isinstance(value, list):
            return value
        if not isinstance(value, str):
            return []
        if not value.startswith('['):
            return []
        value = value.strip()[1:-1]  # Remove [ ]
        if not value:
            return []
        items = [item.strip().strip('"').strip("'") for item in value.split(',')]
        return [item for item in items if item]

    def _parse_list_value(self, value: str) -> List[any]:
        """Parse a list value, potentially spanning multiple lines."""
        # Simple implementation - in production would need full bracket matching
        return self._parse_list_field(value)

    def _parse_graph_edges(self, edges_raw: str) -> List[GraphEdge]:
        """Parse graph edges from list of edge dictionaries."""
        if not edges_raw or edges_raw == '[]':
            return []
        
        # Simple parser for edge dictionaries
        # In production, would use proper JSON/dict parsing
        edges = []
        
        # Extract individual edge blocks
        edge_pattern = re.findall(r'\{([^}]+)\}', edges_raw)
        for edge_str in edge_pattern:
            edge_dict = {}
            # Parse key-value pairs within edge
            pairs = re.findall(r'([A-Za-z_]+):\s*["\']?([^,"\']+)["\']?', edge_str)
            for key, value in pairs:
                edge_dict[key.strip()] = value.strip()
            
            if 'from' in edge_dict and 'to' in edge_dict:
                edges.append(GraphEdge(
                    from_agent=edge_dict['from'],
                    to_agent=edge_dict['to'],
                    condition=edge_dict.get('when', 'default'),
                ))
        
        return edges

    def _parse_schema_field(self, value):
        """Helper to parse schema fields (dict or string)."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            # Simple string-to-dict conversion for now
            # In production, this would use proper JSON/dict parsing
            return {'_raw': value}
        return {}

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _ensure_app(self, line: _Line) -> None:
        if self._app is None:
            fallback_name = self._module_name or self._module_name_override
            fallback = fallback_name.split('.')[-1] if fallback_name else 'app'
            self._app = App(name=fallback)

    def _peek_line(self) -> Optional[_Line]:
        """Return the current line as a _Line object for Grammar parsing."""
        if self._cursor < len(self._lines):
            return self._lines[self._cursor]
        return None

    @staticmethod
    def _indent(text: str) -> int:
        """
        Compute indent for either a raw string or a _Line wrapper.

        AIParserMixin sometimes passes _Line objects back into the grammar
        helpers, so we normalize here to avoid type errors.
        """
        if isinstance(text, _Line):
            text = text.text
        return len(text) - len(text.lstrip(' '))

    def _error(self, message: str, line_or_line_no=None, line_text: str = None) -> N3SyntaxError:
        """
        Create a syntax error. Supports two call patterns:
        1. Grammar style: _error(message, line: _Line)
        2. AIParserMixin style: _error(message, line_no: int, line: str)
        """
        # Pattern 1: Grammar style with _Line object
        if isinstance(line_or_line_no, _Line):
            line = line_or_line_no
            return N3SyntaxError(
                f"Syntax error: {message}",
                path=self._path or None,
                line=line.number,
                code="SYNTAX_GRAMMAR",
                hint=line.text.strip() or None,
            )
        # Pattern 2: AIParserMixin style with line_no and line_text
        elif isinstance(line_or_line_no, int):
            line_no = line_or_line_no
            return N3SyntaxError(
                f"Syntax error: {message}",
                path=self._path or None,
                line=line_no,
                code="SYNTAX_GRAMMAR",
                hint=line_text.strip() if line_text else None,
            )
        # Fallback for no line info
        else:
            return N3SyntaxError(
                f"Syntax error: {message}",
                path=self._path or None,
                line=None,
                code="SYNTAX_GRAMMAR",
                hint=None,
            )
    
    def _parse_string_list(self, value: str) -> List[str]:
        """Parse a string representation of a list like '["a", "b", "c"]'."""
        import json
        try:
            # Try JSON parsing first
            result = json.loads(value)
            if isinstance(result, list):
                return [str(item) for item in result]
            return [str(result)]
        except (json.JSONDecodeError, ValueError):
            # Fall back to basic parsing
            if value.startswith('[') and value.endswith(']'):
                content = value[1:-1]
                items = []
                for item in content.split(','):
                    item = item.strip().strip('"').strip("'")
                    if item:
                        items.append(item)
                return items
            return [value.strip().strip('"').strip("'")]
    
    def _parse_function_def(self, line: _Line) -> None:
        """Parse function definition: fn name(params) => body"""
        from namel3ss.parser.symbolic import SymbolicExpressionParser
        from namel3ss.ast.expressions import FunctionDef
        
        # For now, only handle single-line function definitions
        func_text = line.text
        
        # Create symbolic parser
        parser = SymbolicExpressionParser(func_text, path=self._path)
        
        try:
            func_def = parser.parse_function_def()
            
            # Functions are attached to the active app; create a default if needed.
            self._ensure_app(line)
            
            # Add function to app functions collection
            if self._app:
                self._app.functions.append(func_def)
            self._extra_nodes.append(func_def)

            # Move past this line so the main loop can continue
            self._advance()
            
        except Exception as e:
            raise self._error(f"Failed to parse function definition: {e}", line)
    
    def _parse_rule_def(self, line: _Line) -> None:
        """Parse rule definition: rule head :- body."""
        from namel3ss.parser.symbolic import SymbolicExpressionParser
        from namel3ss.ast.expressions import RuleDef
        
        # For now, only handle single-line rule definitions
        rule_text = line.text
        
        # Create symbolic parser
        parser = SymbolicExpressionParser(rule_text, path=self._path)
        
        try:
            rule_def = parser.parse_rule_def()
            
            # Ensure app exists
            if self._app is None:
                self._app = App(name="", body=[])
            
            # Add rule to app rules collection
            self._app.rules.append(rule_def)
            self._extra_nodes.append(rule_def)
            
        except Exception as e:
            raise self._error(f"Failed to parse rule definition: {e}", line)

    def _unsupported(self, line: _Line, feature: str) -> None:
        location = f"{self._path}:{line.number}" if self._path else f"line {line.number}"
        raise GrammarUnsupportedError(f"Unsupported {feature} near {location}")
