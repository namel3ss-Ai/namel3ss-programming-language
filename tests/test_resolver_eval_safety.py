import pytest

from namel3ss.ast.program import Program
from namel3ss.parser import Parser
from namel3ss.resolver import ModuleResolutionError, resolve_program


def _parse_module(source: str, path: str) -> object:
    module = Parser(source, path=path).parse()
    if not module.name:
        module.name = path
    module.path = path
    return module


def test_resolver_exposes_evaluators_metrics_guardrails():
    source = '''
module eval.root

app "EvalApp".

evaluator "toxicity_checker":
  kind: "safety"
  provider: "acme.toxicity"

metric "toxicity_rate":
  evaluator: "toxicity_checker"
  aggregation: "mean"

guardrail "safety_guard":
  evaluators: ["toxicity_checker"]
  action: "block"
  message: "Blocked."
'''
    module = _parse_module(source, "eval_root.n3")
    program = Program(modules=[module])
    resolved = resolve_program(program)
    app = resolved.app
    assert app.evaluators[0].name == "toxicity_checker"
    assert app.metrics[0].evaluator == "toxicity_checker"
    assert app.guardrails[0].evaluators == ["toxicity_checker"]


def test_resolver_validates_metric_references():
    module_source = '''
module eval.metrics

app "EvalApp".

metric "invalid":
  evaluator: "missing"
'''
    module = _parse_module(module_source, "metric_fail.n3")
    program = Program(modules=[module])
    with pytest.raises(ModuleResolutionError):
        resolve_program(program)


def test_resolver_validates_guardrail_references_across_modules():
    shared = '''
module eval.shared

evaluator "toxicity_checker":
  kind: "safety"
  provider: "acme.toxicity"
'''
    consumer = '''
module eval.consumer
import eval.shared

app "Consumer".

guardrail "safety_guard":
  evaluators: ["toxicity_checker"]
  action: "log_only"
'''
    shared_module = _parse_module(shared, "shared.n3")
    consumer_module = _parse_module(consumer, "consumer.n3")
    program = Program(modules=[shared_module, consumer_module])
    resolved = resolve_program(program)
    names = [guardrail.name for guardrail in resolved.app.guardrails]
    assert "safety_guard" in names


def test_guardrail_reference_error():
    source = '''
module eval.guard

guardrail "safety_guard":
  evaluators: ["missing"]
  action: "block"
'''
    module = _parse_module(source, "guard_fail.n3")
    program = Program(modules=[module])
    with pytest.raises(ModuleResolutionError):
        resolve_program(program)
