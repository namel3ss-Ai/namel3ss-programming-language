from __future__ import annotations

import re

from typing import Dict

from namel3ss.ast import App, Dataset, Insight, Model, Page, Theme, VariableAssignment

from .ai import AIParserMixin
from .insights import InsightParserMixin
from .experiments import ExperimentParserMixin
from .models import ModelParserMixin
from .pages import PageParserMixin
from .crud import CrudParserMixin
from .frames import FrameParserMixin


class ProgramParserMixin(
    FrameParserMixin,
    PageParserMixin,
    CrudParserMixin,
    ModelParserMixin,
    InsightParserMixin,
    ExperimentParserMixin,
    AIParserMixin,
):
    """Top-level entry point for parsing programs."""

    def parse(self) -> App:
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
            if stripped.startswith('app '):
                if self.app is not None:
                    raise self._error("Only one app definition is allowed", line_no, line)
                self.app = self._parse_app(line, line_no)
            elif stripped.startswith('theme'):
                if self.app is None:
                    raise self._error("Theme must appear after app declaration", line_no, line)
                self._parse_theme(indent)
            elif stripped.startswith('dataset '):
                if self.app is None:
                    raise self._error("Dataset must appear after app declaration", line_no, line)
                dataset = self._parse_dataset(line, line_no, indent)
                self.app.datasets.append(dataset)
            elif stripped.startswith('frame '):
                if self.app is None:
                    raise self._error("Frame must appear after app declaration", line_no, line)
                frame = self._parse_frame(line, line_no, indent)
                self.app.frames.append(frame)
            elif stripped.startswith('insight '):
                if self.app is None:
                    raise self._error("Insight must appear after app declaration", line_no, line)
                insight = self._parse_insight(line, line_no, indent)
                self.app.insights.append(insight)
            elif stripped.startswith('ai model '):
                if self.app is None:
                    raise self._error("Model must appear after app declaration", line_no, line)
                ai_model = self._parse_ai_model(line, line_no, indent)
                self.app.ai_models.append(ai_model)
            elif stripped.startswith('prompt '):
                if self.app is None:
                    raise self._error("Prompt must appear after app declaration", line_no, line)
                prompt = self._parse_prompt(line, line_no, indent)
                self.app.prompts.append(prompt)
            elif stripped.startswith('model '):
                if self.app is None:
                    raise self._error("Model must appear after app declaration", line_no, line)
                if self._looks_like_ai_model(line, indent):
                    ai_model = self._parse_ai_model(line, line_no, indent)
                    self.app.ai_models.append(ai_model)
                else:
                    model = self._parse_model(line, line_no, indent)
                    self.app.models.append(model)
            elif stripped.startswith('connector '):
                if self.app is None:
                    raise self._error("Connector must appear after app declaration", line_no, line)
                connector = self._parse_connector(line, line_no, indent)
                self.app.connectors.append(connector)
            elif stripped.startswith('define template'):
                if self.app is None:
                    raise self._error("Template must appear after app declaration", line_no, line)
                template = self._parse_template(line, line_no, indent)
                self.app.templates.append(template)
            elif stripped.startswith('define chain'):
                if self.app is None:
                    raise self._error("Chain must appear after app declaration", line_no, line)
                chain = self._parse_chain(line, line_no, indent)
                self.app.chains.append(chain)
            elif stripped.startswith('experiment '):
                if self.app is None:
                    raise self._error("Experiment must appear after app declaration", line_no, line)
                experiment = self._parse_experiment(line, line_no, indent)
                self.app.experiments.append(experiment)
            elif stripped.startswith('enable crud'):
                if self.app is None:
                    raise self._error("CRUD declaration must appear after app declaration", line_no, line)
                crud_resource = self._parse_crud_resource(line, line_no, indent)
                self.app.crud_resources.append(crud_resource)
            elif stripped.startswith('page '):
                if self.app is None:
                    raise self._error("Page must appear after app declaration", line_no, line)
                page = self._parse_page(line, line_no, indent)
                self.app.pages.append(page)
            elif stripped.startswith('set '):
                if self.app is None:
                    raise self._error(
                        "Variable assignment must appear after app declaration",
                        line_no,
                        line,
                    )
                assignment = self._parse_variable_assignment(line, line_no, indent)
                self.app.variables.append(assignment)
            else:
                raise self._error(
                    "Expected 'app', 'theme', 'dataset', 'connector', 'insight', 'model', 'experiment', 'enable crud', or 'page'",
                    line_no,
                    line,
                )
        if self.app is None:
            raise self._error("App declaration ('app \"Name\" ...') is required", 0, '')
        return self.app

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
