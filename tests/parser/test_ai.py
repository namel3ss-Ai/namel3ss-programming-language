"""Parser tests for AI models, prompts, and training blocks."""

from namel3ss.ast import (
    AIModel,
    Prompt,
    TrainingJob,
    ChainStep,
    WorkflowForBlock,
    WorkflowIfBlock,
    WorkflowWhileBlock,
)
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

    app = Parser(source).parse_app()

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


def test_parse_chain_with_declared_effect() -> None:
    source = (
        'app "Chains".\n'
        '\n'
        'define template "echo":\n'
        '  prompt = "{input}"\n'
        '\n'
        'define chain "echo_chain" effect pure:\n'
        '  input -> template echo\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "ready"\n'
    )

    app = Parser(source).parse_app()
    assert app.chains[0].declared_effect == "pure"


def test_parse_memory_block() -> None:
    source = (
        'app "MemoryApp".\n'
        '\n'
        'memory "chat_history":\n'
        '  scope: session\n'
        '  kind: conversation\n'
        '  max_items: 25\n'
        '  metadata:\n'
        '    retention: short\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "hi"\n'
    )

    app = Parser(source).parse_app()
    assert len(app.memories) == 1
    memory = app.memories[0]
    assert memory.name == "chat_history"
    assert memory.scope == "session"
    assert memory.kind == "conversation"
    assert memory.max_items == 25
    assert memory.metadata["retention"] == "short"


def test_parse_chain_with_workflow_nodes() -> None:
    source = (
        'app "Workflows".\n'
        '\n'
        'define chain "flow":\n'
        '  steps:\n'
        '    - step "classifier":\n'
        '        kind: template\n'
        '        target: classify\n'
        '        continue_on_error: true\n'
        '    - if ctx:input == "route":\n'
        '        then:\n'
        '          - step "branch_a":\n'
        '              kind: template\n'
        '              target: branch_a\n'
        '        else:\n'
        '          - step "branch_b":\n'
        '              kind: template\n'
        '              target: branch_b\n'
        '    - for item in ctx:payload.items:\n'
        '        max_iterations: 5\n'
        '        - step "loop_step":\n'
        '            kind: template\n'
        '            target: loop\n'
        '    - while value != "done":\n'
        '        max_iterations: 3\n'
        '        - step "finisher":\n'
        '            kind: template\n'
        '            target: finish\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "ok"\n'
    )

    app = Parser(source).parse_app()
    chain = app.chains[0]
    assert len(chain.steps) == 4

    first = chain.steps[0]
    assert isinstance(first, ChainStep)
    assert first.name == "classifier"
    assert first.stop_on_error is False

    conditional = chain.steps[1]
    assert isinstance(conditional, WorkflowIfBlock)
    assert len(conditional.then_steps) == 1
    assert len(conditional.else_steps) == 1

    loop_block = chain.steps[2]
    assert isinstance(loop_block, WorkflowForBlock)
    assert loop_block.loop_var == "item"
    assert loop_block.max_iterations == 5
    assert len(loop_block.body) == 1

    while_block = chain.steps[3]
    assert isinstance(while_block, WorkflowWhileBlock)
    assert while_block.max_iterations == 3
    assert len(while_block.body) == 1


def test_parse_training_job_block() -> None:
    source = (
        'app "Trainer".\n'
        '\n'
        'dataset "training_data" from table training_table.\n'
        '\n'
        'training "baseline":\n'
        '  model: "image_classifier"\n'
        '  dataset: "training_data"\n'
        '  objective: "minimize_loss"\n'
        '  hyperparameters:\n'
        '    epochs: 10\n'
        '    learning_rate: 0.01\n'
        '  compute:\n'
        '    backend: "vertex-ai"\n'
        '    resources:\n'
        '      accelerator: "a100"\n'
        '  metadata:\n'
        '    owner: "ml-team"\n'
        '  metrics:\n'
        '    - accuracy\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "ok"\n'
    )

    app = Parser(source).parse_app()

    assert len(app.training_jobs) == 1
    job = app.training_jobs[0]
    assert isinstance(job, TrainingJob)
    assert job.name == "baseline"
    assert job.model == "image_classifier"
    assert job.dataset == "training_data"
    assert job.objective == "minimize_loss"
    assert job.hyperparameters["epochs"] == 10
    assert job.hyperparameters["learning_rate"] == 0.01
    assert job.compute.backend == "vertex-ai"
    assert job.compute.resources["accelerator"] == "a100"
    assert job.metadata["owner"] == "ml-team"
    assert job.metrics == ["accuracy"]
