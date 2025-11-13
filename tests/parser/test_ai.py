"""Parsing tests for AI-specific constructs in Namel3ss."""

from namel3ss.ast import Chain, Connector, ContextValue, Template
from namel3ss.parser import Parser


def test_parse_connectors_templates_and_chains() -> None:
    source = (
        'app "AI".\n'
        '\n'
        'connector "openai" type llm:\n'
        '  provider = "openai"\n'
        '  model = "gpt-4"\n'
        '  api_key = env.OPENAI_KEY\n'
        '\n'
        'define template "summary":\n'
        '  prompt = "Summarize: {input}"\n'
        '  tone = "friendly"\n'
        '\n'
        'define chain "summarize_chain":\n'
        '  input -> template summary -> connector openai\n'
        '  output = "text"\n'
    )

    app = Parser(source).parse()

    assert len(app.connectors) == 1
    connector = app.connectors[0]
    assert isinstance(connector, Connector)
    assert connector.connector_type == "llm"
    assert connector.config["provider"] == "openai"
    assert connector.config["model"] == "gpt-4"
    api_key = connector.config["api_key"]
    assert isinstance(api_key, ContextValue)
    assert api_key.scope == "env"
    assert api_key.path == ["OPENAI_KEY"]

    assert len(app.templates) == 1
    template = app.templates[0]
    assert isinstance(template, Template)
    assert template.prompt == "Summarize: {input}"
    assert template.metadata["tone"] == "friendly"

    assert len(app.chains) == 1
    chain = app.chains[0]
    assert isinstance(chain, Chain)
    assert chain.input_key == "input"
    assert len(chain.steps) == 2
    assert chain.steps[0].kind == "template"
    assert chain.steps[0].target == "summary"
    assert chain.steps[1].kind == "connector"
    assert chain.steps[1].target == "openai"
    assert chain.metadata["output"] == "text"
