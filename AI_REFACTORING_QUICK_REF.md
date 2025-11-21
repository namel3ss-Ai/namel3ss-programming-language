# AI Subpackage Quick Reference

## Import Cheat Sheet

### Recommended (New)
```python
from namel3ss.ast.ai import (
    # Core
    Connector, AIModel, Memory, Template,
    # Prompts
    Prompt, PromptArgument, OutputSchema,
    # Workflows
    Chain, ChainStep,
    # Tools
    LLMDefinition, ToolDefinition,
    # Training
    TrainingJob, TuningJob,
    # Validation & Errors
    AIValidationError,
    validate_prompt,
    validate_chain,
)
```

### Alternative (Also Works)
```python
from namel3ss.ast import Connector, Prompt, Chain
```

### Legacy (Still Supported)
```python
from namel3ss.ast.ai_prompts import Prompt
from namel3ss.ast.ai_workflows import Chain
```

## Validation Quick Reference

```python
from namel3ss.ast.ai.validation import *
from namel3ss.ast.ai.errors import AIValidationError

# Validate any AI construct before use
validate_connector(connector)
validate_memory(memory)
validate_ai_model(model)
validate_template(template)
validate_prompt(prompt)
validate_llm_definition(llm)
validate_tool_definition(tool)
validate_chain(chain)
validate_training_job(job)
validate_tuning_job(tuning)
```

## Error Handling

```python
from namel3ss.ast.ai.errors import (
    AIValidationError,      # Validation failures
    AIConfigurationError,   # Runtime config issues
    AIExecutionError,       # Execution failures
)

try:
    validate_prompt(prompt)
except AIValidationError as e:
    print(f"Error: {e.format()}")
    print(f"Construct: {e.construct_type}")
    print(f"Field: {e.field}")
    if e.hint:
        print(f"Hint: {e.hint}")
```

## Module Structure

```
namel3ss/ast/ai/
├── __init__.py       → All public exports
├── errors.py         → AIValidationError, AIConfigurationError, AIExecutionError
├── validation.py     → 15 validation functions
├── connectors.py     → Connector
├── memory.py         → Memory
├── models.py         → AIModel
├── templates.py      → Template
├── prompts.py        → Prompt, PromptArgument, OutputSchema, etc.
├── tools.py          → LLMDefinition, ToolDefinition
├── training.py       → TrainingJob, TuningJob, etc.
└── workflows.py      → Chain, ChainStep, workflow nodes
```

## Common Patterns

### Create and Validate Connector
```python
from namel3ss.ast.ai import Connector
from namel3ss.ast.ai.validation import validate_connector

conn = Connector(
    name="db",
    connector_type="postgres",
    config={"host": "localhost"}
)
validate_connector(conn)  # Raises AIValidationError if invalid
```

### Create and Validate Prompt
```python
from namel3ss.ast.ai import Prompt, PromptArgument
from namel3ss.ast.ai.validation import validate_prompt

prompt = Prompt(
    name="classify",
    model="gpt4",
    template="Classify: {{text}}",
    args=[PromptArgument(name="text", arg_type="string")]
)
validate_prompt(prompt)
```

### Create and Validate Chain
```python
from namel3ss.ast.ai import Chain, ChainStep
from namel3ss.ast.ai.validation import validate_chain

chain = Chain(
    name="workflow",
    steps=[
        ChainStep(kind="prompt", target="classify", options={})
    ]
)
validate_chain(chain)
```

## Error Codes Reference

| Code | Type | Description |
|------|------|-------------|
| AI001 | Connector | Name cannot be empty |
| AI002 | Connector | Type must be specified |
| AI003 | Connector | Config must be dict |
| AI010 | Memory | Name cannot be empty |
| AI011 | Memory | Invalid scope |
| AI012 | Memory | Invalid kind |
| AI013 | Memory | max_items must be positive |
| AI020 | AIModel | Name cannot be empty |
| AI021 | AIModel | Provider must be specified |
| AI022 | AIModel | Model name must be specified |
| AI023 | AIModel | Config must be dict |
| AI030 | Template | Name cannot be empty |
| AI031 | Template | Prompt cannot be empty |
| AI040 | PromptField | Field name cannot be empty |
| AI041 | PromptField | Field type cannot be empty |
| AI042 | PromptArgument | Argument name cannot be empty |
| AI043 | PromptArgument | Invalid argument type |
| AI044 | OutputField | Output field name cannot be empty |
| AI045 | OutputSchema | Must have at least one field |
| AI046 | Prompt | Prompt name cannot be empty |
| AI047 | Prompt | Template cannot be empty |
| AI048 | Prompt | Model cannot be empty |
| AI050 | LLMDefinition | LLM name cannot be empty |
| AI051 | LLMDefinition | Temperature out of range (0-2) |
| AI052 | LLMDefinition | max_tokens must be positive |
| AI053 | LLMDefinition | top_p out of range (0-1) |
| AI054 | ToolDefinition | Tool name cannot be empty |
| AI055 | ToolDefinition | Description cannot be empty |
| AI056 | ToolDefinition | Parameters must be dict |
| AI060 | ChainStep | Step kind cannot be empty |
| AI061 | ChainStep | Step target cannot be empty |
| AI062 | ChainStep | Options must be dict |
| AI063 | Chain | Chain name cannot be empty |
| AI064 | Chain | Must have at least one step |
| AI070 | TrainingJob | Job name cannot be empty |
| AI071 | TrainingJob | Model cannot be empty |
| AI072 | TrainingJob | Dataset cannot be empty |
| AI073 | TrainingJob | Objective cannot be empty |
| AI074 | TrainingJob | Split ratios must sum to 1.0 |
| AI075 | TuningJob | Tuning name cannot be empty |
| AI076 | TuningJob | Must reference base training job |
| AI077 | TuningJob | Must specify hyperparameters |

## Validation Function Summary

| Function | Purpose | Checks |
|----------|---------|--------|
| `validate_connector()` | External service config | Name, type, config dict |
| `validate_memory()` | Memory store config | Name, scope, kind, max_items |
| `validate_ai_model()` | AI model reference | Name, provider, model_name |
| `validate_template()` | Prompt template | Name, prompt text |
| `validate_prompt()` | Complete prompt | Name, model, template, args, schema |
| `validate_prompt_field()` | Schema field | Name, type |
| `validate_prompt_argument()` | Typed argument | Name, valid type |
| `validate_output_field()` | Output field | Name |
| `validate_output_schema()` | Output schema | Has fields, all fields valid |
| `validate_llm_definition()` | LLM config | Name, param ranges |
| `validate_tool_definition()` | Tool spec | Name, description, params |
| `validate_chain()` | Workflow chain | Name, has steps, all steps valid |
| `validate_chain_step()` | Chain step | Kind, target, options |
| `validate_training_job()` | Training spec | Name, model, dataset, objective |
| `validate_tuning_job()` | Hyperparameter tuning | Name, base job, search space |

## Tips

1. **Always validate** before runtime execution
2. **Use type hints** for better IDE support
3. **Check error codes** for specific issues
4. **Read hints** in error messages
5. **Use centralized validators** instead of ad-hoc checks
6. **Catch AIValidationError** for graceful handling
7. **Import from ai** for new code, legacy imports work
8. **Check docstrings** for detailed examples

## Complete Example

```python
from namel3ss.ast.ai import (
    Connector, AIModel, Prompt, Chain, ChainStep,
    PromptArgument, validate_connector, validate_prompt, validate_chain,
    AIValidationError
)

# 1. Create and validate connector
try:
    db = Connector(name="db", connector_type="postgres")
    validate_connector(db)
    print("✓ Connector valid")
except AIValidationError as e:
    print(f"✗ {e.format()}")

# 2. Create and validate model
try:
    model = AIModel(name="gpt4", provider="openai", model_name="gpt-4")
    validate_ai_model(model)
    print("✓ Model valid")
except AIValidationError as e:
    print(f"✗ {e.format()}")

# 3. Create and validate prompt
try:
    prompt = Prompt(
        name="classify",
        model="gpt4",
        template="Classify: {{text}}",
        args=[PromptArgument(name="text", arg_type="string")]
    )
    validate_prompt(prompt)
    print("✓ Prompt valid")
except AIValidationError as e:
    print(f"✗ {e.format()}")

# 4. Create and validate chain
try:
    chain = Chain(
        name="workflow",
        steps=[ChainStep(kind="prompt", target="classify", options={})]
    )
    validate_chain(chain)
    print("✓ Chain valid")
except AIValidationError as e:
    print(f"✗ {e.format()}")

print("\n✓ All constructs valid!")
```
