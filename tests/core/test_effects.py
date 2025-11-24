"""Tests for the effect analysis pass."""

from namel3ss.ast import Action, ShowForm
from namel3ss.effects import EffectAnalyzer, EffectError
from namel3ss.parser import Parser


def _analyze(source: str):
    app = Parser(source).parse_app()
    analyzer = EffectAnalyzer(app)
    analyzer.analyze()
    return app


def test_action_infers_ai_effect() -> None:
    source = (
        'app "AI".\n'
        '\n'
        'connector "openai" type llm:\n'
        '  provider = "openai"\n'
        '  model = "gpt-4o-mini"\n'
        '\n'
        'define chain "summaries":\n'
        '  input -> connector openai\n'
        '\n'
        'page "Home" at "/":\n'
        '  action "Responder":\n'
        '    when button.click:\n'
        '      run chain summaries with:\n'
        '        text = "hi"\n'
        '  show form "Ask":\n'
        '    fields: prompt\n'
        '    on submit:\n'
        '      ask connector openai with:\n'
        '        prompt = form.prompt\n'
    )

    app = _analyze(source)
    action = next(stmt for stmt in app.pages[0].statements if isinstance(stmt, Action))
    assert "ai" in action.effects
    form = next(stmt for stmt in app.pages[0].statements if isinstance(stmt, ShowForm))
    assert "ai" in form.effects
    chain = app.chains[0]
    assert "ai" in chain.effects


def test_action_declared_pure_rejects_ai_operations() -> None:
    source = (
        'app "AI".\n'
        '\n'
        'connector "openai" type llm:\n'
        '  provider = "openai"\n'
        '  model = "gpt-4o-mini"\n'
        '\n'
        'page "Home" at "/":\n'
        '  action "Responder" effect pure:\n'
        '    when submit.click:\n'
        '      ask connector openai with:\n'
        '        prompt = "hello"\n'
    )

    app = Parser(source).parse_app()
    analyzer = EffectAnalyzer(app)
    try:
        analyzer.analyze()
        raise AssertionError("Expected EffectError")
    except EffectError as exc:
        assert "declared pure" in str(exc)


def test_chain_declared_pure_must_not_call_ai() -> None:
    source = (
        'app "AI".\n'
        '\n'
        'connector "openai" type llm:\n'
        '  provider = "openai"\n'
        '  model = "gpt-4o-mini"\n'
        '\n'
        'define chain "summaries" effect pure:\n'
        '  input -> connector openai\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "ok"\n'
    )

    app = Parser(source).parse_app()
    analyzer = EffectAnalyzer(app)
    try:
        analyzer.analyze()
        raise AssertionError("Expected EffectError")
    except EffectError as exc:
        assert "Chain 'summaries'" in str(exc)


def test_workflow_chain_effects_detect_ai_usage() -> None:
    source = (
        'app "AI".\n'
        '\n'
        'connector "openai" type llm:\n'
        '  provider = "openai"\n'
        '  model = "gpt-4o-mini"\n'
        '\n'
        'define template "copy":\n'
        '  prompt = "{input}"\n'
        '\n'
        'define chain "workflow":\n'
        '  steps:\n'
        '    - if ctx:payload.intent == "escalate":\n'
        '        then:\n'
        '          - step "llm_call":\n'
        '              kind: connector\n'
        '              target: openai\n'
        '        else:\n'
        '          - step "fallback":\n'
        '              kind: template\n'
        '              target: copy\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "ready"\n'
    )

    app = Parser(source).parse_app()
    analyzer = EffectAnalyzer(app)
    analyzer.analyze()
    chain = next(c for c in app.chains if c.name == "workflow")
    assert "ai" in chain.effects
