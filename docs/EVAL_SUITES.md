# Evaluation Suites in Namel3ss

## Overview

Evaluation suites (`eval_suite`) provide a first-class way to evaluate AI chains and RAG pipelines in Namel3ss applications. They enable systematic, metrics-driven development by running chains over datasets and computing quality metrics.

## Syntax

```n3
eval_suite <identifier> {
  dataset: "<dataset_name>"
  target_chain: "<chain_name>"
  
  metrics: [
    { name: "<metric_name>", type: "<metric_type>" },
    { name: "<metric_name_2>", type: "<metric_type_2>" },
    ...
  ]
  
  # Optional: LLM-based judging
  judge_llm: "<llm_name>"
  rubric: """
  <rubric_text>
  """
  
  # Optional
  description: "Description of this eval suite"
}
```

## Fields

### Required Fields

- **dataset**: Name of a dataset containing evaluation examples. Each row should have:
  - An input field (e.g., `question`, `query`, `input`)
  - Optionally, ground-truth fields (e.g., `answer`, `contexts`)

- **target_chain**: Name of the chain to evaluate

- **metrics**: List of metric specifications. Each metric has:
  - `name`: Logical name of the metric (for reporting)
  - `type`: Metric type identifier (see below)
  - Additional metric-specific configuration (optional)

### Optional Fields

- **judge_llm**: Reference to an LLM definition to use as a judge
- **rubric**: Text describing scoring criteria for the judge
- **description**: Human-readable description of the eval suite

## Supported Metric Types

### Built-in Metrics

- **`builtin_latency`**: Measures total execution latency in milliseconds
- **`builtin_cost`**: Measures cost in USD (from token usage or direct cost tracking)

### RAGAS Metrics

RAGAS metrics require the `ragas` library (`pip install ragas`):

- **`ragas_relevance`** / **`ragas_answer_relevancy`**: Answer relevance to question
- **`ragas_context_precision`**: Precision of retrieved contexts
- **`ragas_context_recall`**: Recall of retrieved contexts
- **`ragas_faithfulness`**: Faithfulness of answer to retrieved contexts
- **`ragas_answer_similarity`**: Similarity to reference answer
- **`ragas_answer_correctness`**: Correctness compared to reference answer

### Custom Metrics

You can register custom metrics using the `register_metric()` function in Python:

```python
from namel3ss.eval import register_metric, EvalMetric, EvalContext, EvalMetricResult

class MyCustomMetric(EvalMetric):
    async def compute(self, ctx: EvalContext) -> EvalMetricResult:
        # Your metric logic here
        score = ...
        return EvalMetricResult(name=self.name, value=score)

register_metric("custom_my_metric", MyCustomMetric)
```

## Complete Example

```n3
app "Customer Support Eval".

# Dataset with evaluation examples
dataset "support_eval_set" from csv "data/support_eval.csv":
  columns: question, ground_truth, contexts

# LLM for the chain
llm "gpt4" using openai:
  model = "gpt-4"

# RAG pipeline
index "support_docs" from dataset "knowledge_base":
  embedding: "text-embedding-ada-002"

define chain "answer_with_rag":
  input -> retrieve from "support_docs" -> llm "gpt4"

# Evaluation suite
eval_suite support_eval {
  dataset: "support_eval_set"
  target_chain: "answer_with_rag"
  
  metrics: [
    { name: "answer_relevance", type: "ragas_relevance" },
    { name: "context_precision", type: "ragas_context_precision" },
    { name: "latency_ms", type: "builtin_latency" },
    { name: "cost_usd", type: "builtin_cost" }
  ]
  
  judge_llm: "gpt4"
  rubric: """
  Score the answer on three dimensions (1-5 scale):
  - Helpfulness: Is the answer helpful to the customer?
  - Correctness: Is the information factually correct?
  - Safety: Is the answer safe and appropriate?
  """
  
  description: "Evaluate RAG pipeline on support queries"
}
```

## Running Eval Suites

### CLI Command

```bash
# Run eval suite
namel3ss eval-suite support_eval -f app.n3

# Limit number of examples
namel3ss eval-suite support_eval --limit 50

# Save results to file
namel3ss eval-suite support_eval --output results.json

# Verbose output with per-example metrics
namel3ss eval-suite support_eval --verbose

# Batch processing for parallelism
namel3ss eval-suite support_eval --batch-size 5
```

### Output Format

```json
{
  "status": "ok",
  "suite": "support_eval",
  "num_examples": 100,
  "examples_per_second": 2.5,
  "total_time_ms": 40000,
  "summary_metrics": {
    "answer_relevance": {
      "mean": 0.85,
      "median": 0.88,
      "std": 0.12,
      "min": 0.45,
      "max": 0.98,
      "count": 100
    },
    "context_precision": {
      "mean": 0.78,
      "median": 0.80,
      "std": 0.15,
      "min": 0.30,
      "max": 0.95,
      "count": 100
    },
    "latency_ms": {
      "mean": 350.5,
      "median": 320.0,
      "std": 85.3,
      "min": 180.0,
      "max": 890.0,
      "count": 100
    },
    "cost_usd": {
      "mean": 0.015,
      "median": 0.014,
      "std": 0.003,
      "min": 0.008,
      "max": 0.025,
      "count": 100
    },
    "judge_helpfulness": {
      "mean": 4.2,
      "median": 4.0,
      "std": 0.8,
      "min": 2.0,
      "max": 5.0,
      "count": 100
    }
  },
  "errors": [],
  "metadata": {
    "batch_size": 1,
    "continue_on_error": true,
    "limit": null
  }
}
```

## Best Practices

1. **Start Small**: Begin with a small dataset (10-50 examples) to validate your eval suite setup

2. **Mix Metric Types**: Use a combination of:
   - Quality metrics (RAGAS, judge scores)
   - Performance metrics (latency)
   - Cost metrics

3. **Version Your Eval Datasets**: Keep eval datasets under version control alongside your code

4. **CI Integration**: Run eval suites in CI to catch regressions:
   ```bash
   namel3ss eval-suite support_eval --limit 20 --output results.json
   # Assert metrics meet thresholds
   ```

5. **Judge Rubrics**: When using LLM judges, be specific in rubrics:
   - Define clear dimensions
   - Specify numeric scales
   - Provide examples of each score level (in future iterations)

6. **Ground Truth Data**: For reference-based metrics (RAGAS correctness, similarity):
   - Ensure your dataset has `ground_truth` or `reference_answer` fields
   - For RAG metrics, include `contexts` field with expected retrieved docs

## Dataset Format

Eval suite datasets should be structured with these fields:

```csv
question,ground_truth,contexts
"How do I reset my password?","Click 'Forgot Password' on the login page","[""Users can reset passwords..."", ""Password reset emails...""]"
"What is your refund policy?","We offer 30-day refunds for all purchases","[""Our refund policy..."", ""Refund processing...""]"
```

Or as a Namel3ss dataset:

```n3
dataset "eval_data" from inline:
  rows: [
    {
      question: "How do I reset my password?",
      ground_truth: "Click 'Forgot Password' on the login page",
      contexts: ["Password reset documentation..."]
    }
  ]
```

## Limitations and Future Work

Current limitations:
- No built-in threshold assertions (use external scripts)
- Judge rubrics don't yet support few-shot examples
- No automatic comparison between eval suite runs

Planned enhancements:
- Threshold assertions in eval_suite blocks
- Automatic regression detection
- Integration with experiment tracking tools (MLflow, Weights & Biases)
- Support for custom aggregation functions
- Comparative evaluation (A/B testing multiple chains)
