from __future__ import annotations

import re

from typing import Dict, List, Optional

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

from .ai import AIParserMixin
from .eval import EvaluationParserMixin
from .insights import InsightParserMixin
from .experiments import ExperimentParserMixin
from .models import ModelParserMixin
from .pages import PageParserMixin
from .crud import CrudParserMixin
from .frames import FrameParserMixin


_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_MODULE_DECL_RE = re.compile(rf"^module\s+({_IDENTIFIER}(?:\.{_IDENTIFIER})*)\s*$")
_IMPORT_TARGET_RE = re.compile(rf"^({_IDENTIFIER}(?:\.{_IDENTIFIER})*)(?:\s+as\s+({_IDENTIFIER}))?\s*$")
_IMPORT_NAME_RE = re.compile(rf"^({_IDENTIFIER})(?:\s+as\s+({_IDENTIFIER}))?\s*$")
_LANG_VERSION_RE = re.compile(r'^language_version\s+"([0-9]+\.[0-9]+\.[0-9]+)"\s*\.?$', flags=re.IGNORECASE)


class LegacyProgramParser(
    FrameParserMixin,
    PageParserMixin,
    CrudParserMixin,
    ModelParserMixin,
    InsightParserMixin,
    ExperimentParserMixin,
    AIParserMixin,
    EvaluationParserMixin,
):
    """Compatibility parser that retains the legacy line-by-line behavior."""

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
            if not stripped or stripped.startswith('#'):
                continue
            indent = self._indent(line)
            if indent != 0:
                raise self._error("Top level statements must not be indented", line_no, line)
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
            elif stripped.startswith('experiment '):
                imports_allowed = False
                self._ensure_app_initialized(line_no, line)
                experiment = self._parse_experiment(line, line_no, indent)
                self.app.experiments.append(experiment)
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
            else:
                raise self._error(
                    "Expected 'app', 'theme', 'dataset', 'connector', 'insight', 'model', 'experiment', 'evaluator', 'metric', 'guardrail', 'eval_suite', 'enable crud', or 'page'",
                    line_no,
                    line,
                )
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


__all__ = ["LegacyProgramParser"]
