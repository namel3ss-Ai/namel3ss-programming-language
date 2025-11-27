# LLM Benchmark & Experiment Lab

Production-grade example that benchmarks multiple LLM profiles, runs repeatable experiments, and surfaces evaluation, diffs, tool calls, and logs through Namel3ss UI components.

## Highlights
- **Two model profiles**: `fast_eval` (speed-first) vs `grounded_eval` (accuracy-first) with per-1k token pricing metadata.
- **Tool + log transparency**: Tool-call and log feeds captured in JSON and wired to `tool_call_view`/`log_view` for real observability data.
- **Real datasets**: `datasets/benchmark_cases.json` (8 curated cases), `datasets/case_category_counts.json`, `runs/sample_run_history.json`, per-model metrics, tool-call, and log feeds.
- **Full observability UI**: `evaluation_result`, `diff_view`, `chart`, `data_table`, `tool_call_view`, and `log_view` wired to live data; layout uses stack/grid/split/tabs primitives.

## Files
- `benchmark.ai` — Main Namel3ss source with datasets, models, chain, and four pages (dashboard, dataset browser, experiment runner, run details).
- `datasets/benchmark_cases.json` — Ground-truth prompts, contexts, and expected outputs.
- `datasets/case_category_counts.json` — Aggregated coverage by category.
- `runs/sample_run_history.json` — Full baseline run with per-case outputs, metrics, tool calls, and logs.
- `runs/sample_model_metrics.json` — Flattened metrics per model per run.
- `runs/sample_tool_calls.json` / `runs/sample_logs.json` — Feeds for observability components.
- `runs/latest_run.json` — Current run payload that drives live components.

## UI Pages (in `benchmark.ai`)
- **LLM Benchmark Dashboard**: Stat summaries, accuracy/cost charts, recent runs table, `evaluation_result` with comparison, `diff_view` for model outputs, `tool_call_view` and `log_view` for transparency.
- **Dataset Browser**: Tabs with the raw benchmark cases plus category coverage charts/tables.
- **Experiment Runner**: Token-aware form (design tokens: elevated/primary/lg/compact), current `evaluation_result`, unified `diff_view`, latency/cost charts, and live logs/tool calls from `latest_run`.
- **Run Details**: Tokenized `show table` with design tokens, case-level tool calls, diff regression check, and detailed logs.

## Running
```bash
# Generate app code
namel3ss generate examples/llm_benchmark_lab/benchmark.ai benchmark_output

# Update `runs/latest_run.json` with your own results (optional)
# (edit the JSON directly or replace it with generated data)
```

## Dataset Snapshot
- 8 cases spanning factoid, math, translation, summarization, code review, temporal reasoning, billing math, and fact-check tasks.
- Expected outputs include real-world facts (e.g., Voyager 1 launch year, renewable energy share) with reference URLs.

## How this pushes Namel3ss
- Demonstrates design tokens (theme/color scheme, form variant/tone/size/density) across layouts.
- Combines evaluation (`evaluation_result`), observability (`log_view`, `tool_call_view`), and comparison (`diff_view`) in one workflow.
- Uses datasets + tooling to show cost/latency/accuracy trade-offs, not static screenshots.
