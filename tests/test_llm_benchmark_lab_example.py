"""Tests for the LLM Benchmark & Experiment Lab example."""

from pathlib import Path

from namel3ss.codegen.frontend.react.pages import generate_page_component
from namel3ss.ir.builder import IRBuilder
from namel3ss.parser import Parser


EXAMPLE_PATH = Path("examples/llm_benchmark_lab/benchmark.ai")


def _load_app():
  source = EXAMPLE_PATH.read_text()
  module = Parser(source, path=str(EXAMPLE_PATH)).parse()
  return module.body[0]


def test_benchmark_example_parses_and_has_components():
  """Parser coverage: ensure the example loads and key components exist."""
  app = _load_app()
  assert app.name == "LLM Benchmark & Experiment Lab"
  page_names = {page.name for page in app.pages}
  assert "LLM Benchmark Dashboard" in page_names
  dashboard = next(page for page in app.pages if page.name == "LLM Benchmark Dashboard")
  statement_types = {type(stmt).__name__ for stmt in dashboard.body}
  assert {"EvaluationResult", "DiffView", "ToolCallView", "LogView"}.issubset(statement_types)


def test_ir_and_codegen_include_observability_components():
  """IR/codegen coverage: design tokens and AI components survive to React."""
  app = _load_app()
  ir = IRBuilder().build(app)
  assert ir["design_tokens"]["theme"] == "dark"
  page = ir["pages"][0]
  code = generate_page_component(page, ir)
  assert "EvaluationResult" in code
  assert "ToolCallView" in code
  assert "LogView" in code
