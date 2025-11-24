from __future__ import annotations

import uuid

import pytest

from namel3ss.codegen.backend import generate_backend
from namel3ss.parser import Parser
from namel3ss.plugins.base import PLUGIN_CATEGORY_EVALUATOR
from namel3ss.plugins import registry as registry_module
from namel3ss.plugins.registry import register_plugin

from tests.backend_test_utils import load_backend_module


@pytest.fixture(autouse=True)
def restore_registry():
    snapshot = {category: dict(entries) for category, entries in registry_module._PLUGINS.items()}
    try:
        yield
    finally:
        registry_module._PLUGINS.clear()
        registry_module._PLUGINS.update({category: dict(entries) for category, entries in snapshot.items()})


class RecordingEvaluatorPlugin:
    name = "recording_eval"
    instances: list["RecordingEvaluatorPlugin"] = []

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        RecordingEvaluatorPlugin.instances.append(self)

    def configure(self, config):
        self.config = dict(config)

    async def call(self, context, payload):
        self.calls.append(dict(payload))
        result = dict(payload)
        result["violation"] = self.config.get("violation", False)
        result["label"] = "violation" if result["violation"] else "ok"
        result["score"] = self.config.get("score", 0.0)
        return result


class ErrorEvaluatorPlugin:
    name = "error_eval"

    def configure(self, config):
        self.config = dict(config)

    async def call(self, context, payload):
        raise RuntimeError("evaluator failure")


def _build_runtime(source: str, tmp_path, monkeypatch):
    app = Parser(source).parse_app()
    backend_dir = tmp_path / f"backend_{uuid.uuid4().hex}"
    generate_backend(app, backend_dir)
    return load_backend_module(tmp_path, backend_dir, monkeypatch)


def _register_eval_plugin(provider: str, plugin_cls):
    register_plugin(PLUGIN_CATEGORY_EVALUATOR, provider, plugin_cls)


EVAL_SOURCE = '''
app "EvalApp".

evaluator "record_eval":
  kind: "safety"
  provider: "recording_eval"

guardrail "block_guard":
  evaluators: ["record_eval"]
  action: "block"
  message: "blocked"

guardrail "log_guard":
  evaluators: ["record_eval"]
  action: "log_only"
  message: "logged"

define template "response":
  prompt = "ok"

define chain "check":
  steps:
    - step "generate":
        kind: template
        target: response
        evaluation:
          evaluators: ["record_eval"]

define chain "block_check":
  steps:
    - step "generate":
        kind: template
        target: response
        evaluation:
          evaluators: ["record_eval"]
          guardrail: "block_guard"

define chain "log_check":
  steps:
    - step "generate":
        kind: template
        target: response
        evaluation:
          evaluators: ["record_eval"]
          guardrail: "log_guard"

page "Home" at "/":
  show text "ok"
'''


def test_chain_step_runs_evaluator(tmp_path, monkeypatch):
    _register_eval_plugin("recording_eval", RecordingEvaluatorPlugin)
    with _build_runtime(EVAL_SOURCE, tmp_path, monkeypatch) as module:
        context = module.runtime.build_context("home")
        result = module.run_chain("check", {}, context=context)
        assert result["status"] == "ok"
        entry = result["steps"][0]
        assert "record_eval" in entry.get("evaluation", {})
        plugin_instance = context["evaluators"]["record_eval"]
        assert plugin_instance.calls


def test_guardrail_blocks_output(tmp_path, monkeypatch):
    _register_eval_plugin("recording_eval", RecordingEvaluatorPlugin)
    RecordingEvaluatorPlugin.instances.clear()
    plugin_cls = RecordingEvaluatorPlugin
    plugin_cls.instances = []
    with _build_runtime(EVAL_SOURCE, tmp_path, monkeypatch) as module:
        context = module.runtime.build_context("home")
        context["evaluators"]["record_eval"].config["violation"] = True
        result = module.run_chain("block_check", {}, context=context)
        assert result["status"] == "error"
        payload = result["result"]
        assert payload["status"] == "blocked"
        assert payload["guardrail"] == "block_guard"


def test_guardrail_log_only(tmp_path, monkeypatch):
    _register_eval_plugin("recording_eval", RecordingEvaluatorPlugin)
    with _build_runtime(EVAL_SOURCE, tmp_path, monkeypatch) as module:
        context = module.runtime.build_context("home")
        context["evaluators"]["record_eval"].config["violation"] = True
        result = module.run_chain("log_check", {}, context=context)
        assert result["status"] == "ok"
        assert result["result"] == "ok"


def test_evaluator_failure_surfaces(tmp_path, monkeypatch):
    source = '''
app "EvalApp".

evaluator "error_eval":
  kind: "safety"
  provider: "error_eval"

define template "response":
  prompt = "ok"

define chain "check":
  steps:
    - step "generate":
        kind: template
        target: response
        evaluation:
          evaluators: ["error_eval"]

page "Home" at "/":
  show text "ok"
'''
    _register_eval_plugin("error_eval", ErrorEvaluatorPlugin)
    with _build_runtime(source, tmp_path, monkeypatch) as module:
        context = module.runtime.build_context("home")
        result = module.run_chain("check", {}, context=context)
        assert result["status"] == "error"
