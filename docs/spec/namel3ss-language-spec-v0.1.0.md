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

Lines beginning with `# ` (optionally `# <emoji> `) are treated as comments. Valid form: `# <emoji?> <text>`.

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

## 11. N3Frame Data Model (Preview Spec 0.1.x)

Frames introduce a first-class, immutable tabular abstraction that bridges typed schema definitions, runtime execution, and downstream ML use cases.

### 11.1 Architectural role

```
Frame AST ──> FrameExpressionAnalyzer (type + schema inference)
            └─> Backend state (FRAMES registry)
                └─> Runtime execution (Polars-backed N3Frame)
                    └─> FastAPI frame endpoints / experiments / builders
```

Frames sit alongside datasets and connectors at the AST level, but differ in that they enforce a static schema (`N3FrameType`), support a dedicated operations DSL, and are executed via the Polars engine (with a minimal Pandas fallback when Polars is unavailable).

### 11.2 Core concepts

* **Immutability** – every frame evaluation returns a new `N3Frame` instance; no in-place mutation is allowed. This ensures deterministic planner output and re-usable pipelines.
* **Schema-first** – each column declares `name`, `dtype`, `nullable`, optional `role`, metadata, and validations. Frame-level metadata covers tags, access policies, key columns, and dataset splits.
* **Roles** – roles annotate semantic meaning (`id`, `feature`, `target`, `time`, `weight`, `group`, etc.) and are propagated into `N3FrameType`. Roles enable ML builders to infer canonical feature/label/time columns without additional configuration.
* **Polars runtime** – frames are executed with Polars expressions wherever possible. When Polars is missing, a constrained Pandas fallback is used so that development/testing can proceed, but production builds are expected to ship with Polars.
* **FRAMES registry** – compiled frames are stored inside `FRAMES: Dict[str, Dict[str, Any]]`. Runtime helpers resolve frame specs, load source data, apply computed columns/constraints, and expose fetch/export APIs.

### 11.3 Defining frames

```
frame "OrdersFrame" from dataset orders:
  description: "Normalized orders view"
  column order_id string:
    role: id
    nullable: false
  column country string:
    role: feature
  column amount number:
    role: feature
  column is_vip bool:
    role: target
  column purchased_at datetime:
    role: time
  key: order_id
  splits:
    train: 0.7
    test: 0.3
  sample:
    order_id: "ord_1"
    country: "DE"
    amount: 120.5
    is_vip: false
    purchased_at: "2024-01-01"
```

Key elements:

| Property          | Description |
|-------------------|-------------|
| `source_type` / `source` | When omitted, a frame reads from `dataset <name>`; `frame "Foo" from sql view_name` or `from frame OtherFrame` reuse other sources. |
| `column <name> <dtype>` | Defines typed columns. Accepted dtypes mirror dataset dtypes (`string`, `text`, `bool`, `int`, `number`, `decimal`, `datetime`, etc.). |
| `role`            | Optional semantic hint. Roles flow into type checking, runtime metadata, and ML builders. |
| `default`, `expression`, `source` | Allow derived columns and value coercion. Expressions are evaluated per row using the same sandbox as dataset computed columns. |
| `key`, `splits`, `examples` | Provide logical primary keys, train/val/test split hints, and optional sample rows (used for documentation and fallback data when a source is empty). |
| `source:` block  | Overrides file/SQL loading (described in §11.6). |

Frames inherit dataset parser conveniences such as `with option ...`, `metadata`, and `access` blocks.

### 11.4 Frame operations DSL

Frame pipelines can be expressed in variable assignments, model configs, or experiment metadata by chaining DSL methods on a frame reference:

```
set vip_by_country =
  orders_frame
    .filter(is_vip == true and amount > 100)
    .select(country, order_id)
    .order_by(purchased_at, descending = true)

set revenue_summary =
  daily_sales
    .group_by(country)
    .summarise(
      total = sum(revenue),
      avg = mean(revenue),
      buyers = nunique(customer_id)
    )

set joined =
  orders_frame
    .join(customers_frame, on = customer_id, how = "left")
```

Supported operations (in order of application):

| Operation | Syntax | Semantics |
|-----------|--------|-----------|
| `filter(predicate)` | `frame.filter(amount > 100 and country == "US")` | Boolean expression over columns. Supports `==`, `!=`, `<`, `<=`, `>`, `>=`, `and`, `or`, `not`, and membership (`in`). |
| `select(col1, col2, alias = expr)` | `frame.select(id, total_spend = spend * 1.2)` | Projects/renames columns and allows computed expressions. |
| `order_by(col1, descending=true)` | `frame.order_by(created_at, descending = true)` | Stable sort by one or more columns. |
| `group_by(col1, col2)` | `frame.group_by(country).summarise(...)` | Declares grouping keys and must be followed by `summarise`. |
| `summarise(name = agg(expr), …)` | `summarise(total = sum(amount))` | Aggregations: `sum`, `mean`, `avg`, `min`, `max`, `count`, `nunique`, `std`. |
| `join(frame_name, on = column, how = "inner")` | `users.join(events, on = user_id, how = "left")` | Joins two frames/datasets. Allowed `how`: `inner`, `left`, `right`, `outer`. |

Chaining order matters; each method consumes the prior frame expression and yields a new immutable pipeline node. The parser enforces valid method names, while the analyzer (below) validates column references and type rules.

### 11.5 Type system and schema propagation

`N3FrameType` captures:

* `columns: Dict[str, FrameColumnType]` – name, dtype, nullable flag, and role.
* `order: List[str]` – column order for deterministic planning.
* `key: List[str]` – logical key columns.
* `splits: Dict[str, float]` – normalized split hints.

The `FrameExpressionAnalyzer` walks DSL pipelines:

1. Resolves the root frame reference to an `N3FrameType`.
2. Ensures referenced columns exist (`filter`, `select`, `group_by`, `order_by`, `join`).
3. Validates predicate operand compatibility (no string vs. int comparisons, `in` requires list-like RHS).
4. Checks aggregation signatures (e.g., `sum` requires numeric input).
5. Ensures join key columns exist on both sides and have compatible canonical dtypes.
6. Produces a new `N3FrameType` describing the pipeline output schema (select projection, group-by keys + aggregation outputs, join merge rules, etc.).

Typical errors surfaced during analysis:

* `Frame 'OrdersFrame' does not define columns: discount`: missing columns in `select`, `group_by`, etc.
* `Filter on frame 'OrdersFrame' cannot compare amount (number) with "north"`: incompatible predicate types.
* `Aggregation 'avg_amount' on frame 'OrdersFrame' requires numeric input but status is 'string'`: invalid aggregation input.
* `join between 'orders' and 'customers' expects matching dtypes for column 'customer_id'`: join key mismatch.

These errors are raised at resolve-time (before runtime execution) and halt compilation unless addressed.

### 11.6 Source loading

Frames default to reading from an existing dataset, but custom file or SQL sources can be declared:

```
frame "Countries":
  source:
    kind: file
    path: ${DATA_ROOT}/countries.parquet
    format: parquet

frame "DailyFacts" from sql facts_view:
  source:
    kind: sql
    connection: analytics_db
    table: daily_facts
```

Rules:

* `kind: file` supports `format: csv` (default) or `parquet`. Paths can use `${ENV_VAR}` placeholders resolved at runtime. CSV loading supports header inference and column coercion based on frame schema.
* `kind: sql` requires `connection` (matches a configured SQLAlchemy DSN) and `table` (or view/query reference). SQL sources are executed asynchronously and converted to Polars frames.
* When a `source:` block is absent, the runtime obtains rows from the referenced dataset or frame. Frame source recursion is allowed but must be acyclic.
* Missing external data triggers `frame_source_failed` errors, and the runtime falls back to `sample` rows only when a source returns no data.

Loading occurs lazily: `fetch_frame_rows` (REST API / runtime helper) resolves the frame spec, loads the data, applies computed columns, and returns a window respecting limit/offset/order defaults. Export helpers (`export_frame_csv`, `export_frame_parquet`) reuse the same evaluated rows.

### 11.7 Runtime execution

* Every pipeline is executed via Polars expressions in `namel3ss.codegen.backend.core.runtime.frames`. If Polars is unavailable, a narrow Pandas fallback is used (only for basic filtering/selection) and a warning is emitted.
* Runtime evaluation enforces immutability by always cloning rows when shaping results or joining.
* Ordering, pagination, and CSV/Parquet export reuse the evaluated `N3Frame` rows; no operation mutates the shared registry.
* Runtime errors (missing columns, failed expressions, source failures) are recorded in the request context and surfaced in frame API responses.

### 11.8 ML / Experiment integration

Experiments can opt into frame-backed datasets by placing a `data` block in their metadata:

```
experiment "churn_eval":
  metadata:
    data:
      frame: CustomersFrame
      target: churned
      features: tenure, spend
      time: observed_at
      groups: household_id
      splits:
        train: 0.6
        validation: 0.2
        test: 0.2
```

At runtime:

1. `evaluate_experiment` loads the referenced frame (or pipeline) and materializes it into an immutable dataset payload containing schema, rows, `X` (feature matrix), `y` (target series), optional weights, and concrete splits.
2. Variant payloads receive `args["dataset"]`, `args["y_true"]`, and split metadata. Built-in metric evaluators use this information to compute accuracy/precision/etc. without requiring additional user input.
3. Roles declared on columns provide defaults for missing `features`/`target`/`time` entries. For example, any column with `role: feature` is picked up if the metadata omits `features`.
4. Frame-level `splits` seed train/validation/test splits; experiment metadata can override them per experiment.

This integration makes frames the canonical data source for ML workflows: they deliver validated schema, typed rows, and context-aware splits to the experiment builder while keeping the runtime execution path (Polars) consistent with API consumers.
