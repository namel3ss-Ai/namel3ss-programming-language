"""Parser tests for AI models and prompt declarations."""

from namel3ss.ast import AIModel, Prompt
from namel3ss.parser import Parser


def test_parse_ai_model_and_prompt_blocks() -> None:
    source = (
        'app "AI Demo".\n'
        "\n"
        'model "chat_model" using openai:\n'
        '  name: "gpt-4o-mini"\n'
        "  temperature: 0.2\n"
        "\n"
        'prompt "SummarizeTicket":\n'
        "  input:\n"
        "    ticket: text\n"
        "  output:\n"
        "    summary: text\n"
        "  metadata:\n"
        '    tags: ["support"]\n'
        '  using model "chat_model":\n'
        '    "Summarize: {{ticket}}"\n'
        "\n"
        'page "Home" at "/":\n'
        '  show text "ready"\n'
    )

    app = Parser(source).parse()

    assert len(app.ai_models) == 1
    model = app.ai_models[0]
    assert isinstance(model, AIModel)
    assert model.name == "chat_model"
    assert model.provider == "openai"
    assert model.model_name == "gpt-4o-mini"
    assert model.config["temperature"] == 0.2

    assert len(app.prompts) == 1
    prompt = app.prompts[0]
    assert isinstance(prompt, Prompt)
    assert prompt.name == "SummarizeTicket"
    assert prompt.model == "chat_model"
    assert prompt.input_fields[0].name == "ticket"
    assert prompt.output_fields[0].field_type == "text"
    assert prompt.metadata["tags"] == ["support"]
    assert "Summarize" in prompt.template
