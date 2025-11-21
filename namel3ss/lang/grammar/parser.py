"""Main grammar parser class composing all mixin parsers."""

from __future__ import annotations
from typing import List, Optional

from namel3ss.ast import App, KnowledgeModule, LogicQuery
from namel3ss.ast.modules import Import
from namel3ss.ast.program import Module
from namel3ss.parser.base import ParserBase
from namel3ss.parser.expressions import ExpressionParserMixin
from namel3ss.parser.ai import AIParserMixin as _AIParserMixin
from namel3ss.parser.logic import LogicParserMixin as _LogicParserMixin

from .helpers import _Line
from .directives import DirectiveParserMixin
from .declarations import DeclarationsParserMixin
from .pages import PagesParserMixin
from .ai_components import AIComponentsParserMixin
from .prompts import PromptsParserMixin
from .rag import RAGParserMixin
from .agents import AgentsParserMixin
from .policy import PolicyParserMixin
from .utility_parsers import UtilityParsersMixin
from .functions import FunctionsParserMixin
from .utility_methods import UtilityMethodsMixin


class _ExpressionHelper(ExpressionParserMixin):
    """Helper class to parse standalone expressions."""
    
    def __init__(self):
        """Initialize with empty source - will be reset before each use."""
        ParserBase.__init__(self, "", path="")
    
    def parse(self, text: str, *, line_no: int, line: str) -> object:
        """Parse an expression string."""
        self.lines = [line]
        self.pos = line_no
        return self._parse_expression(text)


class _GrammarModuleParser(
    UtilityMethodsMixin,
    DirectiveParserMixin,
    DeclarationsParserMixin,
    PagesParserMixin,
    AIComponentsParserMixin,
    PromptsParserMixin,
    RAGParserMixin,
    AgentsParserMixin,
    PolicyParserMixin,
    UtilityParsersMixin,
    FunctionsParserMixin,
    _LogicParserMixin,
    _AIParserMixin,
):
    """Recursive-descent parser that follows the EBNF outlined above.
    
    Inherits from LogicParserMixin and AIParserMixin to support full DSL parsing including
    knowledge bases, logic queries, structured prompts, AI models, training jobs, etc.
    """

    def __init__(self, source: str, *, path: str = "", module_name: Optional[str] = None) -> None:
        # Initialize ParserBase infrastructure required by AIParserMixin
        # Must be done FIRST before setting custom attributes
        super().__init__(source, path=path, module_name=module_name)
        
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
        
        # AIParserMixin expects self.lines to be raw strings, not _Line objects
        self.lines = self._lines_raw  # AIParserMixin needs raw strings
        self.pos = 0

    # Synchronization helpers for AIParserMixin integration
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

    # High-level driver
    def parse(self) -> Module:
        """Parse the entire module."""
        while self._cursor < len(self._lines):
            line = self._peek_line()
            if line is None:
                break
            stripped = line.text.strip()
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
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
            if stripped.startswith('train rlhf ') or stripped.startswith('rlhf '):
                self._parse_rlhf_job_wrapper(line)
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
            # Collect knowledge modules, queries, and training jobs from extra nodes
            knowledge_modules = [n for n in self._extra_nodes if isinstance(n, KnowledgeModule)]
            queries = [n for n in self._extra_nodes if isinstance(n, LogicQuery)]
            training_jobs = [n for n in self._extra_nodes if n.__class__.__name__ == 'TrainingJob']
            tuning_jobs = [n for n in self._extra_nodes if n.__class__.__name__ == 'TuningJob']
            rlhf_jobs = [n for n in self._extra_nodes if n.__class__.__name__ == 'RLHFJob']
            
            # Populate App fields
            if knowledge_modules:
                self._app.knowledge_modules = knowledge_modules
            if queries:
                self._app.queries = queries
            if training_jobs:
                self._app.training_jobs = training_jobs
            if tuning_jobs:
                self._app.tuning_jobs = tuning_jobs
            if rlhf_jobs:
                self._app.rlhf_jobs = rlhf_jobs
            
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


__all__ = ['_GrammarModuleParser']
