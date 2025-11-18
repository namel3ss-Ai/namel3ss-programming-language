# Namel3ss Language Specification

**Specification Version:** 0.1.0  
**Date:** 2024-11-18  
**Status:** Draft (implementation aligned with Namel3ss compiler release)

> The language specification version is distinct from the Namel3ss compiler implementation version. Compilers declare which language spec versions they support.

## 1. Overview

Namel3ss (N3) is a declarative, English-like DSL for describing full-stack AI applications. A program is composed of modules that define apps, pages, datasets, connectors, chains (workflows), models, templates, experiments, and safety/evaluation metadata. This specification describes the syntax and semantics of the language independent of any specific compiler.

## 2. Lexical Structure

### 2.1 Identifiers

Identifiers consist of letters, digits, and underscores, starting with a letter. They are case-sensitive.

### 2.2 Literals

- Strings: enclosed in double quotes (`"..."`).
- Numbers: integer or decimal (e.g., `42`, `3.14`).
- Booleans: `true`, `false`.
- Context references: `env.VAR`, `ctx.value`.

### 2.3 Comments

Lines beginning with `#` are treated as comments.

## 3. Modules & Imports

```
language_version "0.1.0"
module my_app.main
import shared.ui
```

- `language_version` is optional but, when present, must precede other declarations.
- `module` declares the fully qualified module name.
- `import` statements allow referencing declarations from other modules.

## 4. Top-Level Declarations

### 4.1 App

```
app "Support Portal" connects to postgres "SUPPORT_DB".
```

Defines the entry-point application: name, optional database binding, theme, variables, etc.

### 4.2 Pages

```
page "Home" at "/":
  show text "Welcome"
```

Pages contain statements such as `show text`, `show table`, conditional logic, and actions.

### 4.3 Datasets

```
dataset "tickets" from sql "db.tickets":
  filter status = "open"
```

Datasets describe data sources, transformations, schemas, and caching policies.

### 4.4 Connectors

```
connector "support_llm" type llm:
  provider = "openai"
  model = "gpt-4o-mini"
```

Connectors bind names to plugin-backed tools (LLM providers, vector stores, etc.). Required fields include `type/kind`, `provider`, and configuration.

### 4.5 Templates & Prompts

```
define template "ticket_summary":
  prompt = "Summarize: {input}"
```
```
prompt "SummarizeTicket" using model "support_llm":
  input:
    ticket: text
  output:
    summary: text
  using template ticket_summary
```

Templates define reusable text fragments; prompts bind templates to models with typed inputs/outputs.

### 4.6 Chains

```
define chain "AnswerTicket":
  steps:
    - step "draft":
        kind: template
        target: ticket_summary
    - step "polish":
        kind: connector
        target: support_llm
        options:
          prompt: ctx:steps.draft.result
```

Chains are workflows composed of steps (`kind`, `target`, `options`) and control-flow nodes (`if`, `for`, `while`).

### 4.7 Models

```
ai model "ticket_classifier" using openai:
  model: "gpt-4o-mini"
```

AI models declare provider-specific configuration for inference.

### 4.8 Memory

```
memory "scratchpad":
  scope: session
  kind: list
  max_items: 10
```

### 4.9 Experiments

```
experiment "PromptTest":
  variants:
    - "baseline"
    - "prompt_v2"
```

Experiments compare variants across metrics defined elsewhere.

### 4.10 Evaluation & Safety

```
evaluator "toxicity_checker":
  kind: "safety"
  provider: "acme.toxicity"

metric "toxicity_rate":
  evaluator: "toxicity_checker"
  aggregation: "mean"

guardrail "safety_guard":
  evaluators: ["toxicity_checker"]
  action: "block"
  message: "Response blocked due to policy violation."
```

Evaluators describe plugins that score outputs. Metrics derive from evaluator results. Guardrails define enforcement policies applied to chain steps.

## 5. Type System

Primitive types: `text`, `number`, `boolean`. Collections: lists (`[T]`), dictionaries (`{ key: T }`). Schema annotations appear in prompts, datasets, and models to describe expected structures.

## 6. Execution Model

1. Modules are resolved following import order.
2. The root app composes datasets, connectors, pages, chains, etc.
3. Chains execute sequentially unless control-flow nodes dictate otherwise.
4. Each step receives the current working value and context (payload, vars, memory, prior steps).
5. Steps may invoke connectors, templates, prompts, or Python hooks.
6. Evaluation blocks run post-step; guardrails inspect evaluator outputs.

## 7. Modules & Import Resolution

- Module names use dotted notation (`a.b.c`).
- Imports may reference entire modules or specific symbols.
- Duplicate definitions across modules cause resolve-time errors.

## 8. Plugins & Connectors

- Built-in categories include `llm_provider`, `vector_store`, `embedding_provider`, `custom_tool`, `evaluator`.
- Each connector/evaluator references a provider name resolved via the plugin registry.
- Configuration blocks are passed verbatim to plugin implementations after context resolution.

## 9. Language Versioning

- The language spec follows semantic versioning (MAJOR.MINOR.PATCH).
- Compilers declare the spec versions they support.
- `language_version "X.Y.Z"` directives let projects opt into specific versions; mixing versions within a project is invalid.
- Breaking syntax/semantics changes increment the MAJOR version; additive changes increment MINOR; clarifications or bug fixes increment PATCH.

## 10. Change History

- **0.1.0** – Initial public specification covering modules, connectors, chains, evaluation, and guardrails.

Future revisions will extend these sections with more detailed grammar and semantics as the language evolves.
