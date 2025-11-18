import pytest

from namel3ss.parser import Parser, N3SyntaxError


def _parse_module(source: str):
    return Parser(source).parse()


def test_parse_evaluator_block():
    source = '''
app "EvalApp".

evaluator "toxicity_checker":
  kind: "safety"
  provider: "acme.toxicity"
  config:
    threshold: 0.9
'''
    module = _parse_module(source)
    evaluator = module.body[0].evaluators[0]
    assert evaluator.name == "toxicity_checker"
    assert evaluator.kind == "safety"
    assert evaluator.provider == "acme.toxicity"
    assert evaluator.config["threshold"] == 0.9


def test_evaluator_requires_kind_and_provider():
    source = '''
app "EvalApp".

evaluator "broken":
  provider: "missing.kind"
'''
    with pytest.raises(N3SyntaxError):
        _parse_module(source)


def test_parse_metric_block():
    source = '''
app "EvalApp".

metric "toxicity_rate":
  evaluator: "toxicity_checker"
  aggregation: "mean"
'''
    module = _parse_module(source)
    metric = module.body[0].metrics[0]
    assert metric.name == "toxicity_rate"
    assert metric.evaluator == "toxicity_checker"
    assert metric.aggregation == "mean"


def test_metric_requires_evaluator():
    source = '''
app "EvalApp".

metric "broken":
  aggregation: "mean"
'''
    with pytest.raises(N3SyntaxError):
        _parse_module(source)


def test_parse_guardrail_block():
    source = '''
app "EvalApp".

guardrail "safety_guard":
  evaluators: ["toxicity_checker", "policy_checker"]
  action: "block"
  message: "Blocked due to policy violation."
'''
    module = _parse_module(source)
    guardrail = module.body[0].guardrails[0]
    assert guardrail.name == "safety_guard"
    assert guardrail.evaluators == ["toxicity_checker", "policy_checker"]
    assert guardrail.action == "block"
    assert guardrail.message == "Blocked due to policy violation."


def test_guardrail_requires_evaluators_and_action():
    source = '''
app "EvalApp".

guardrail "broken":
  evaluators: []
  message: "noop"
'''
    with pytest.raises(N3SyntaxError):
        _parse_module(source)

