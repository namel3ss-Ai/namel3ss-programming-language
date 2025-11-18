"""Runtime tests covering workflow chains with branching and loops."""

from __future__ import annotations

from namel3ss.codegen.backend import generate_backend
from namel3ss.parser import Parser

from tests.backend_test_utils import load_backend_module


WORKFLOW_SOURCE = (
    'app "WorkflowApp".\n'
    '\n'
    'memory "scratchpad":\n'
    '  scope: session\n'
    '  kind: list\n'
    '  max_items: 10\n'
    '\n'
    'define template "echo":\n'
    '  prompt = "{input}"\n'
    '\n'
    'define template "escalate_template":\n'
    '  prompt = "escalated"\n'
    '\n'
    'define template "auto_template":\n'
    '  prompt = "auto"\n'
    '\n'
    'define template "seed_template":\n'
    '  prompt = "{input}"\n'
    '\n'
    'define template "append_template":\n'
    '  prompt = "{input}{loop:item};"\n'
    '\n'
    'define template "start_value":\n'
    '  prompt = "start"\n'
    '\n'
    'define template "finish_value":\n'
    '  prompt = "done"\n'
    '\n'
    'define chain "router":\n'
    '  steps:\n'
    '    - step "classifier":\n'
    '        kind: template\n'
    '        target: echo\n'
    '        continue_on_error: true\n'
    '    - if ctx:steps.classifier.result == "escalate":\n'
    '        then:\n'
    '          - step "route_escalate":\n'
    '              kind: template\n'
    '              target: escalate_template\n'
    '        else:\n'
    '          - step "route_auto":\n'
    '              kind: template\n'
    '              target: auto_template\n'
    '\n'
    'define chain "looper":\n'
    '  steps:\n'
    '    - step "seed":\n'
    '        kind: template\n'
    '        target: seed_template\n'
    '    - for item in ctx:payload.items:\n'
    '        max_iterations: 10\n'
    '        - step "append":\n'
    '            kind: template\n'
    '            target: append_template\n'
    '            write_memory: scratchpad\n'
    '\n'
    'define chain "loop_until_done":\n'
    '  steps:\n'
    '    - step "seed":\n'
    '        kind: template\n'
    '        target: start_value\n'
    '    - while value != "done":\n'
    '        max_iterations: 5\n'
    '        - step "finish":\n'
    '            kind: template\n'
    '            target: finish_value\n'
    '\n'
    'page "Home" at "/":\n'
    '  show text "ok"\n'
)


def _build_workflow_backend(tmp_path, monkeypatch):
    app = Parser(WORKFLOW_SOURCE).parse_app()
    backend_dir = tmp_path / "backend_workflows"
    generate_backend(app, backend_dir)
    return load_backend_module(tmp_path, backend_dir, monkeypatch)


def test_workflow_if_branch_and_context_steps(tmp_path, monkeypatch):
    with _build_workflow_backend(tmp_path, monkeypatch) as module:
        context = module.runtime.build_context("home")
        result = module.run_chain("router", {"input": "escalate"}, context=context)
        assert result["result"] == "escalated"
        assert [step["name"] for step in result["steps"]] == ["classifier", "route_escalate"]
        assert context["steps"]["classifier"]["result"] == "escalate"

        second = module.run_chain("router", {"input": "auto"}, context=context)
        assert second["result"] == "auto"
        assert "route_escalate" not in context["steps"]
        assert context["steps"]["route_auto"]["status"] == "ok"


def test_workflow_for_loop_appends_and_writes_memory(tmp_path, monkeypatch):
    with _build_workflow_backend(tmp_path, monkeypatch) as module:
        context = module.runtime.build_context("home")
        payload = {"input": "", "items": ["alpha", "beta", "gamma"]}
        result = module.run_chain("looper", payload, context=context)
        assert result["result"] == "alpha;beta;gamma;"
        # one seed step plus one iteration per item
        assert len(result["steps"]) == len(payload["items"]) + 1
        entries = context["memory_state"]["scratchpad"]["entries"]
        assert entries[-1]["value"] == result["result"]


def test_workflow_while_loop_terminates(tmp_path, monkeypatch):
    with _build_workflow_backend(tmp_path, monkeypatch) as module:
        context = module.runtime.build_context("home")
        result = module.run_chain("loop_until_done", {"input": "ignored"}, context=context)
        assert result["result"] == "done"
        assert result["status"] == "ok"
