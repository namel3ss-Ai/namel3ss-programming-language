from __future__ import annotations

import re

from typing import Any, Dict, List, Optional

from namel3ss.ast import (
    App,
    Dataset,
    Import,
    ImportedName,
    Insight,
    Model,
    Module,
    Page,
    Theme,
    VariableAssignment,
)
from namel3ss.lang import (
    TOP_LEVEL_KEYWORDS,
    suggest_keyword,
)

from .ai import AIParserMixin
from .eval import EvaluationParserMixin
from .insights import InsightParserMixin
from .experiments import ExperimentParserMixin
from .models import ModelParserMixin
from .pages import PageParserMixin
from .crud import CrudParserMixin
from .frames import FrameParserMixin
from .logic import LogicParserMixin


_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_MODULE_DECL_RE = re.compile(rf"^module\s+({_IDENTIFIER}(?:\.{_IDENTIFIER})*)\s*$")
_IMPORT_TARGET_RE = re.compile(rf"^({_IDENTIFIER}(?:\.{_IDENTIFIER})*)(?:\s+as\s+({_IDENTIFIER}))?\s*$")
_IMPORT_NAME_RE = re.compile(rf"^({_IDENTIFIER})(?:\s+as\s+({_IDENTIFIER}))?\s*$")
_LANG_VERSION_RE = re.compile(r'^language_version\s+"([0-9]+\.[0-9]+\.[0-9]+)"\s*\.?$', flags=re.IGNORECASE)


class LegacyProgramParser(
    LogicParserMixin,
    FrameParserMixin,
    PageParserMixin,
    CrudParserMixin,
    ModelParserMixin,
    InsightParserMixin,
    ExperimentParserMixin,
    AIParserMixin,
    EvaluationParserMixin,
):
    """
    Top-level program parser with keyword validation and helpful error messages.
    
    Parses N3 programs consisting of:
    - Module declarations and imports
    - App configuration
    - Top-level constructs (page, model, dataset, etc.)
    - AI/ML components (prompts, connectors, experiments)
    - Logic programming (knowledge, query)
    - Evaluation components (evaluator, metric, guardrail)
    
    Features:
    - Keyword validation with fuzzy matching suggestions
    - Context-aware error messages
    - Maintains backward compatibility
    - Production-ready error handling
    """

    def parse(self) -> Module:
        imports_allowed = True
        language_version_declared = False
        extra_nodes: List[Any] = []
        while self.pos < len(self.lines):
            raw = self._advance()
            line_no = self.pos
            if raw is None:
                break
            line = raw.rstrip('\n')
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                continue
            indent = self._indent(line)
            if indent != 0:
                raise self._error(
                    "Top-level statements must not be indented",
                    line_no,
                    line,
                    hint="Top-level constructs (app, page, model, etc.) should start at column 0"
                )
            if stripped.startswith('module '):
                if not imports_allowed:
                    raise self._error("Module declaration must appear before other statements", line_no, line)
                self._parse_module_declaration(line, line_no)
                continue
            if stripped.startswith('import '):
                if not imports_allowed:
                    raise self._error("Import statements must appear before other declarations", line_no, line)
                self._parse_import_statement(line, line_no)
                continue
            if stripped.lower().startswith('language_version'):
                if not imports_allowed:
                    raise self._error("language_version must appear before other declarations", line_no, line)
                if language_version_declared:
                    raise self._error("language_version directive may only appear once", line_no, line)
                self._parse_language_version(line, line_no)
                language_version_declared = True
                continue
            imports_allowed = False
            if stripped.startswith('app '):
                if self.app is not None:
                    raise self._error("Only one app definition is allowed", line_no, line)
                self.app = self._parse_app(line, line_no)
                self._explicit_app_declared = True
                imports_allowed = False
            elif stripped.startswith('evaluator '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                evaluator = self._parse_evaluator(line, line_no, indent)
                self.app.evaluators.append(evaluator)
                extra_nodes.append(evaluator)
            elif stripped.startswith('metric '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                metric = self._parse_metric(line, line_no, indent)
                self.app.metrics.append(metric)
                extra_nodes.append(metric)
            elif stripped.startswith('guardrail '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                guardrail = self._parse_guardrail(line, line_no, indent)
                self.app.guardrails.append(guardrail)
                extra_nodes.append(guardrail)
            elif stripped.startswith('eval_suite '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                eval_suite = self._parse_eval_suite(line, line_no, indent)
                self.app.eval_suites.append(eval_suite)
                extra_nodes.append(eval_suite)
            elif stripped.startswith('theme'):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                self._parse_theme(indent)
            elif stripped.startswith('dataset '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                dataset = self._parse_dataset(line, line_no, indent)
                self.app.datasets.append(dataset)
            elif stripped.startswith('frame '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                frame = self._parse_frame(line, line_no, indent)
                self.app.frames.append(frame)
            elif stripped.startswith('insight '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                insight = self._parse_insight(line, line_no, indent)
                self.app.insights.append(insight)
            elif stripped.startswith('ai model '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                ai_model = self._parse_ai_model(line, line_no, indent)
                self.app.ai_models.append(ai_model)
            elif stripped.startswith('prompt '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                prompt = self._parse_prompt(line, line_no, indent)
                self.app.prompts.append(prompt)
            elif stripped.startswith('model '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                if self._looks_like_ai_model(line, indent):
                    ai_model = self._parse_ai_model(line, line_no, indent)
                    self.app.ai_models.append(ai_model)
                else:
                    model = self._parse_model(line, line_no, indent)
                    self.app.models.append(model)
            elif stripped.startswith('connector '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                connector = self._parse_connector(line, line_no, indent)
                self.app.connectors.append(connector)
            elif stripped.startswith('memory '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                memory = self._parse_memory(line, line_no, indent)
                self.app.memories.append(memory)
            elif stripped.startswith('define template'):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                template = self._parse_template(line, line_no, indent)
                self.app.templates.append(template)
            elif stripped.startswith('define chain'):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                chain = self._parse_chain(line, line_no, indent)
                self.app.chains.append(chain)
            elif stripped.startswith('training '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                training_job = self._parse_training_job(line, line_no, indent)
                self.app.training_jobs.append(training_job)
            elif stripped.startswith('tuning '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                tuning_job = self._parse_tuning_job(line, line_no, indent)
                self.app.tuning_jobs.append(tuning_job)
            elif stripped.startswith('train rlhf ') or stripped.startswith('rlhf '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                rlhf_job = self._parse_rlhf_job(line, line_no, indent)
                self.app.rlhf_jobs.append(rlhf_job)
            elif stripped.startswith('experiment '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                experiment = self._parse_experiment(line, line_no, indent)
                self.app.experiments.append(experiment)
            elif stripped.startswith('knowledge '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                knowledge_module = self._parse_knowledge_module(line, line_no, indent)
                self.app.knowledge_modules.append(knowledge_module)
            elif stripped.startswith('query '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                query = self._parse_query(line, line_no, indent)
                self.app.queries.append(query)
            elif stripped.startswith('enable crud'):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                crud_resource = self._parse_crud_resource(line, line_no, indent)
                self.app.crud_resources.append(crud_resource)
            elif stripped.startswith('page '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                page = self._parse_page(line, line_no, indent)
                self.app.pages.append(page)
            elif stripped.startswith('set '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                assignment = self._parse_variable_assignment(line, line_no, indent)
                self.app.variables.append(assignment)
            elif stripped.startswith('llm '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                # Parse LLM definition
                from namel3ss.ast.ai_tools import LLMDefinition
                llm_def = self._parse_llm_definition(line, line_no, indent)
                self.app.llms.append(llm_def)
            elif stripped.startswith('fn '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                # Parse function definition using symbolic parser
                from namel3ss.parser.symbolic import SymbolicExpressionParser
                parser = SymbolicExpressionParser(line)
                func_def = parser.parse_function_def()
                if not hasattr(self.app, 'functions'):
                    self.app.functions = []
                self.app.functions.append(func_def)
            elif stripped.startswith('chain '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                # Parse chain definition
                chain_def = self._parse_chain_definition(line, line_no, indent)
                if not hasattr(self.app, 'chains'):
                    self.app.chains = []
                self.app.chains.append(chain_def)
            elif stripped.startswith('memory '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                # Parse memory definition
                memory_def = self._parse_memory_definition(line, line_no, indent)
                if not hasattr(self.app, 'memories'):
                    self.app.memories = []
                self.app.memories.append(memory_def)
            elif stripped.startswith('rag_pipeline '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                # Parse RAG pipeline definition
                rag_def = self._parse_rag_pipeline_definition(line, line_no, indent)
                if not hasattr(self.app, 'rag_pipelines'):
                    self.app.rag_pipelines = []
                self.app.rag_pipelines.append(rag_def)
            else:
                # Unknown top-level construct - provide helpful suggestion
                first_word = stripped.split()[0] if stripped.split() else stripped
                suggestion = suggest_keyword(first_word, 'top-level')
                
                # Build list of common top-level keywords
                common_keywords = [
                    'app', 'page', 'model', 'dataset', 'connector', 'insight',
                    'prompt', 'experiment', 'evaluator', 'theme', 'knowledge', 'query'
                ]
                
                error_msg = f"Unknown top-level construct: '{first_word}'"
                if suggestion and suggestion != first_word and suggestion in TOP_LEVEL_KEYWORDS:
                    hint = f"Did you mean '{suggestion}'? Common keywords: {', '.join(common_keywords)}"
                else:
                    hint = f"Valid top-level keywords: {', '.join(common_keywords)} (and others)"
                
                raise self._error(error_msg, line_no, line, hint=hint)
        body: List[Any] = []
        if self.app is not None:
            body.append(self.app)
        body.extend(extra_nodes)
        module = Module(
            name=self.module_name,
            language_version=self.language_version,
            path=self.source_path,
            imports=list(self.module_imports),
            body=body,
            has_explicit_app=self._explicit_app_declared,
        )
        return module

    def _parse_module_declaration(self, line: str, line_no: int) -> None:
        match = _MODULE_DECL_RE.match(line.strip())
        if not match:
            raise self._error("Expected: module <name>", line_no, line)
        if self._module_declared:
            raise self._error("Only one module declaration is allowed per file", line_no, line)
        if self.module_imports:
            raise self._error("Module declaration must appear before imports", line_no, line)
        if self.app is not None:
            raise self._error("Module declaration must appear before app declaration", line_no, line)
        name = match.group(1)
        self._module_declared = True
        self.module_name = name

    def _parse_language_version(self, line: str, line_no: int) -> None:
        match = _LANG_VERSION_RE.match(line.strip())
        if not match:
            raise self._error('Expected: language_version "<semver>"', line_no, line)
        version = match.group(1)
        self.language_version = version

    def _parse_import_statement(self, line: str, line_no: int) -> None:
        remainder = line.strip()[len("import ") :].strip()
        if not remainder:
            raise self._error("Expected module path after 'import'", line_no, line)
        module_part = remainder
        names_part: Optional[str] = None
        colon_index = remainder.find(":")
        if colon_index != -1:
            module_part = remainder[:colon_index].strip()
            names_part = remainder[colon_index + 1 :].strip()
            if not names_part:
                raise self._error("Expected imported names after ':'", line_no, line)
        match = _IMPORT_TARGET_RE.match(module_part)
        if not match:
            raise self._error("Invalid module import target", line_no, line)
        module_name = match.group(1)
        alias = match.group(2)
        names: Optional[List[ImportedName]] = None
        if names_part is not None:
            if alias:
                raise self._error("Module alias is not allowed when selecting names", line_no, line)
            names = self._parse_imported_names(names_part, line_no, line)
        self.module_imports.append(Import(module=module_name, names=names, alias=alias))

    def _parse_imported_names(self, names_text: str, line_no: int, line: str) -> List[ImportedName]:
        entries = [segment.strip() for segment in names_text.split(",")]
        names: List[ImportedName] = []
        for entry in entries:
            if not entry:
                raise self._error("Expected identifier in import list", line_no, line)
            match = _IMPORT_NAME_RE.match(entry)
            if not match:
                raise self._error("Invalid imported name", line_no, line)
            names.append(ImportedName(name=match.group(1), alias=match.group(2)))
        if not names:
            raise self._error("Import list cannot be empty", line_no, line)
        return names

    def _parse_app(self, line: str, line_no: int) -> App:
        match = re.match(r'app\s+"([^"]+)"(?:\s+connects\s+to\s+\w+\s+"([^"]+)")?\.?$', line.strip())
        if not match:
            raise self._error(
                'Expected: app "Name" [connects to postgres "ALIAS"].',
                line_no,
                line,
            )
        name = match.group(1)
        database = match.group(2) if match.group(2) else None
        return App(name=name, database=database)

    def _ensure_app_initialized(self, line_no: int, line: str) -> None:
        if self.app is not None:
            return
        name = self.module_name or self._module_name_override or self._default_app_name()
        self.app = App(name=name)

    def _parse_theme(self, base_indent: int) -> None:
        theme_values: Dict[str, str] = {}
        while self.pos < len(self.lines):
            line = self._peek()
            if line is None:
                break
            indent = self._indent(line)
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            match = re.match(r'([\w\s]+):\s*(.+)$', stripped)
            if not match:
                raise self._error("Expected 'key: value' inside theme block", self.pos + 1, line)
            key = match.group(1).strip()
            value = match.group(2).strip()
            theme_values[key] = value
            self._advance()
        if self.app.theme is None:
            self.app.theme = Theme(values=theme_values)
        else:
            self.app.theme.values.update(theme_values)
    
    def _parse_llm_definition(self, line: str, line_no: int, base_indent: int) -> 'LLMDefinition':
        """Parse LLM definition: llm name { ... } or llm "name" { ... } or llm name: ..."""
        from namel3ss.ast.ai_tools import LLMDefinition
        import re
        
        # Parse: llm <name> { or llm "name" { or llm <name>: or llm "name":
        # Try quoted name first
        match = re.match(r'llm\s+"([^"]+)"\s*[:{]', line.strip())
        if not match:
            # Try unquoted identifier
            match = re.match(r'llm\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:{]', line.strip())
        
        if not match:
            raise self._error('Expected: llm <name> { or llm "name": or llm name:', line_no, line)
        
        name = match.group(1)
        uses_braces = '{' in line
        
        # Parse configuration block
        config = {}
        model = None
        provider = None
        temperature = None
        max_tokens = None
        
        self._advance()
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            stripped = nxt.strip()
            
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                self._advance()
                continue
            
            # Handle closing brace for brace-style blocks
            if uses_braces and stripped == '}':
                self._advance()
                break
            
            indent = self._indent(nxt)
            # For colon-style blocks, break on dedent; for braces, continue until }
            if not uses_braces and indent <= base_indent:
                break
            
            # Parse key: value pairs
            if ':' in stripped:
                key, value = stripped.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle env references (don't strip quotes from env.VAR)
                if not value.startswith('env.'):
                    value = value.strip('"').strip("'")
                
                if key == 'model':
                    model = value
                elif key == 'provider':
                    provider = value
                elif key == 'temperature':
                    temperature = float(value) if value else None
                elif key == 'max_tokens':
                    max_tokens = int(value) if value else None
                elif key == 'api_key':
                    api_key = value
                else:
                    config[key] = value
            
            self._advance()
        
        return LLMDefinition(
            name=name,
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=config
        )
    
    def _parse_chain_definition(self, line: str, line_no: int, base_indent: int) -> Dict[str, Any]:
        """Parse chain definition: chain name { ... } or chain name: ..."""
        import re
        
        # Parse: chain <name> { or chain <name>:
        match = re.match(r'chain\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:{]', line.strip())
        if not match:
            raise self._error('Expected: chain <name> { or chain <name>:', line_no, line)
        
        name = match.group(1)
        uses_braces = '{' in line
        
        # Parse configuration as key-value dict
        config = {'name': name}
        
        self._advance()
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            stripped = nxt.strip()
            
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                self._advance()
                continue
            
            if uses_braces and stripped == '}':
                self._advance()
                break
            
            indent = self._indent(nxt)
            if not uses_braces and indent <= base_indent:
                break
            
            # Parse key: value pairs
            if ':' in stripped:
                key, value = stripped.split(':', 1)
                config[key.strip()] = value.strip().strip('"').strip("'")
            
            self._advance()
        
        return config
    
    def _parse_memory_definition(self, line: str, line_no: int, base_indent: int) -> Dict[str, Any]:
        """Parse memory definition: memory name { ... } or memory "name" { ... }"""
        import re
        
        # Parse: memory <name> { or memory "name" {
        match = re.match(r'memory\s+"([^"]+)"\s*[:{]', line.strip())
        if not match:
            match = re.match(r'memory\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:{]', line.strip())
        
        if not match:
            raise self._error('Expected: memory <name> { or memory "name":', line_no, line)
        
        name = match.group(1)
        uses_braces = '{' in line
        
        # Parse configuration as key-value dict
        config = {'name': name}
        
        self._advance()
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            stripped = nxt.strip()
            
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                self._advance()
                continue
            
            if uses_braces and stripped == '}':
                self._advance()
                break
            
            indent = self._indent(nxt)
            if not uses_braces and indent <= base_indent:
                break
            
            # Parse key: value or nested blocks
            if ':' in stripped:
                key, value = stripped.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle nested metadata blocks
                if not value or value == '{':
                    nested = {}
                    self._advance()
                    while self.pos < len(self.lines):
                        nxt2 = self._peek()
                        if not nxt2 or nxt2.strip() == '}':
                            if nxt2 and nxt2.strip() == '}':
                                self._advance()
                            break
                        if self._indent(nxt2) <= indent:
                            break
                        if ':' in nxt2:
                            k2, v2 = nxt2.strip().split(':', 1)
                            nested[k2.strip()] = v2.strip().strip('"').strip("'")
                        self._advance()
                    config[key] = nested
                else:
                    config[key] = value.strip('"').strip("'")
            
            self._advance()
        
        return config
    
    def _parse_rag_pipeline_definition(self, line: str, line_no: int, base_indent: int) -> Dict[str, Any]:
        """Parse RAG pipeline definition: rag_pipeline name { ... } or rag_pipeline name: ..."""
        import re
        
        # Parse: rag_pipeline <name> { or rag_pipeline <name>:
        match = re.match(r'rag_pipeline\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:{]', line.strip())
        if not match:
            raise self._error('Expected: rag_pipeline <name> { or rag_pipeline <name>:', line_no, line)
        
        name = match.group(1)
        uses_braces = '{' in line
        
        # Parse configuration as key-value dict
        config = {'name': name}
        
        self._advance()
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            stripped = nxt.strip()
            
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                self._advance()
                continue
            
            if uses_braces and stripped == '}':
                self._advance()
                break
            
            indent = self._indent(nxt)
            if not uses_braces and indent <= base_indent:
                break
            
            # Parse key: value pairs
            if ':' in stripped:
                key, value = stripped.split(':', 1)
                config[key.strip()] = value.strip().strip('"').strip("'")
            
            self._advance()
        
        return config


__all__ = ["LegacyProgramParser"]
