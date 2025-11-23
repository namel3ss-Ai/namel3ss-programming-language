"""
Evaluation command implementations.

This module handles the 'eval' and 'eval-suite' subcommands for running
experiments and evaluation suites.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from ..context import get_cli_context
from ..errors import CLIRuntimeError, handle_cli_exception
from ..loading import load_n3_app, load_runtime_module
from ..output import print_experiment_result
from ..utils import find_experiment, find_first_n3_file
from ..validation import validate_int


def _format_error_detail(exc: BaseException) -> str:
    """Format exception as concise error detail string."""
    message = f"{exc.__class__.__name__}: {exc}"
    return message if len(message) <= 280 else f"{message[:277]}..."


def _traceback_excerpt(limit: int = 2000) -> str:
    """Get truncated traceback for error reporting."""
    trace = traceback.format_exc().strip()
    return trace if len(trace) <= limit else f"{trace[:limit - 3]}..."


def cmd_eval(args: argparse.Namespace) -> None:
    """
    Handle the 'eval' subcommand for running experiments.
    
    Evaluates an experiment by running it through the runtime and comparing
    variants according to defined metrics.
    
    Args:
        args: Parsed command-line arguments containing:
            - experiment: Name of experiment to evaluate
            - file: Path to .n3 source file (optional, auto-discovers)
            - format: Output format (json or text, default: json)
    
    Raises:
        SystemExit: On any error during evaluation
    
    Examples:
        >>> args = argparse.Namespace(  # doctest: +SKIP
        ...     experiment='prompt_comparison',
        ...     file='app.n3',
        ...     format='text'
        ... )
        >>> cmd_eval(args)
        Experiment: prompt_comparison
        Status: success
        Winner: variant_a
    """
    try:
        experiment_name = args.experiment
        source_arg = args.file
        
        # Auto-discover .n3 file if not specified
        if source_arg is None:
            default_file = find_first_n3_file()
            if default_file is None:
                raise CLIRuntimeError(
                    "No .n3 file found to evaluate experiments",
                    hint="Specify a .n3 file with --file or create one in the current directory",
                )
            source_arg = str(default_file)
        
        # Load app and find experiment
        app = load_n3_app(Path(source_arg))
        experiment = find_experiment(app, experiment_name)
        if experiment is None:
            available = ", ".join(sorted(exp.name for exp in app.experiments)) or "none"
            raise CLIRuntimeError(
                f"Experiment '{experiment_name}' not found",
                hint=f"Available experiments: {available}",
            )
        
        # Execute experiment
        payload: Optional[Dict[str, Any]] = None
        cache_key = str(Path(source_arg).resolve())
        
        try:
            runtime = load_runtime_module(app, cache_key)
            runtime_result = runtime.evaluate_experiment(experiment_name, payload or {})
            if not isinstance(runtime_result, dict):
                raise TypeError("evaluate_experiment returned non-dict result")
            result = dict(runtime_result)
            result.setdefault("status", "ok")
            result.setdefault("inputs", payload or {})
        except Exception as exc:
            result = {
                "status": "error",
                "error": "experiment_execution_failed",
                "detail": _format_error_detail(exc),
                "traceback": _traceback_excerpt(),
                "experiment": experiment_name,
            }
        
        result.setdefault("experiment", experiment_name)
        result.setdefault("inputs", payload or {})
        
        # Output result
        if getattr(args, 'format', 'json') == 'text':
            print_experiment_result(result)
        else:
            print(json.dumps(result, indent=2))
    
    except Exception as exc:
        handle_cli_exception(exc, verbose=getattr(args, "verbose", False))


def cmd_eval_suite(args: argparse.Namespace) -> None:
    """
    Handle the 'eval-suite' subcommand for running evaluation suites.
    
    Runs comprehensive evaluation suites that test chains against datasets
    with multiple metrics and optional LLM judges.
    
    Args:
        args: Parsed command-line arguments containing:
            - suite: Name of evaluation suite to run
            - file: Path to .n3 source file (optional, auto-discovers)
            - limit: Maximum number of examples to evaluate (optional)
            - batch_size: Batch size for evaluation (default: 1)
            - output: Output file path for results (optional, default: stdout)
            - verbose: Include per-example metrics in output (optional)
    
    Raises:
        SystemExit: On any error during evaluation
    
    Examples:
        >>> args = argparse.Namespace(  # doctest: +SKIP
        ...     suite='accuracy_test',
        ...     file='app.n3',
        ...     limit=100,
        ...     verbose=True
        ... )
        >>> cmd_eval_suite(args)
        Running eval suite 'accuracy_test'...
        === Evaluation Summary ===
        Suite: accuracy_test
        Examples: 100
        Metrics (mean ± std):
          accuracy: 0.9500 ± 0.0200
    """
    try:
        import asyncio
        from namel3ss.eval import EvalSuiteRunner, create_metric
        from namel3ss.eval.judge import LLMJudge
        
        suite_name = args.suite
        source_arg = args.file
        
        # Auto-discover .n3 file if not specified
        if source_arg is None:
            default_file = find_first_n3_file()
            if default_file is None:
                raise CLIRuntimeError(
                    "No .n3 file found to run eval suite",
                    hint="Specify a .n3 file with --file or create one in the current directory",
                )
            source_arg = str(default_file)
        
        source_path = Path(source_arg)
        app = load_n3_app(source_path)
        
        # Find the eval suite
        eval_suite = None
        for suite in app.eval_suites:
            if suite.name == suite_name:
                eval_suite = suite
                break
        
        if eval_suite is None:
            available = ", ".join(sorted(s.name for s in app.eval_suites)) or "none"
            raise CLIRuntimeError(
                f"Eval suite '{suite_name}' not found",
                hint=f"Available suites: {available}",
            )
        
        # Load runtime module
        cache_key = str(source_path.resolve())
        try:
            runtime = load_runtime_module(app, cache_key)
        except Exception as exc:
            raise CLIRuntimeError(
                f"Failed to load runtime: {exc}",
                hint="Ensure backend is generated and dependencies are installed",
            ) from exc
        
        # Get dataset rows
        try:
            dataset_name = eval_suite.dataset_name
            datasets_data = getattr(runtime, "DATASETS_DATA", {})
            if dataset_name not in datasets_data:
                raise CLIRuntimeError(
                    f"Dataset '{dataset_name}' not loaded in runtime",
                    hint="Check eval suite configuration and dataset definition",
                )
            
            dataset_rows = datasets_data[dataset_name]
            if not isinstance(dataset_rows, list):
                dataset_rows = list(dataset_rows)
        except Exception as exc:
            raise CLIRuntimeError(
                f"Failed to load dataset '{eval_suite.dataset_name}': {exc}",
            ) from exc
        
        # Create chain executor
        chain_name = eval_suite.target_chain_name
        run_chain = getattr(runtime, "run_chain", None)
        if not callable(run_chain):
            raise CLIRuntimeError(
                "Runtime does not support run_chain",
                hint="Regenerate backend to enable chain execution",
            )
        
        def chain_executor(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Execute chain synchronously."""
            return run_chain(chain_name, input_data)
        
        # Create metrics
        metrics = []
        for metric_spec in eval_suite.metrics:
            try:
                metric = create_metric(metric_spec.name, metric_spec.type, metric_spec.config)
                metrics.append(metric)
            except Exception as exc:
                raise CLIRuntimeError(
                    f"Failed to create metric '{metric_spec.name}': {exc}",
                ) from exc
        
        # Create judge if specified
        judge = None
        if eval_suite.judge_llm_name and eval_suite.rubric:
            try:
                llm_instances = getattr(runtime, "_LLM_INSTANCES", {})
                if eval_suite.judge_llm_name not in llm_instances:
                    raise CLIRuntimeError(
                        f"Judge LLM '{eval_suite.judge_llm_name}' not found",
                        hint="Check LLM configuration in N3 source",
                    )
                
                judge_llm = llm_instances[eval_suite.judge_llm_name]
                judge = LLMJudge(judge_llm, eval_suite.rubric)
            except Exception as exc:
                raise CLIRuntimeError(
                    f"Failed to create judge: {exc}",
                ) from exc
        
        # Create runner
        runner = EvalSuiteRunner(
            suite_name=suite_name,
            dataset_rows=dataset_rows,
            chain_executor=chain_executor,
            metrics=metrics,
            judge=judge,
        )
        
        # Run evaluation
        try:
            limit = args.limit if hasattr(args, 'limit') else None
            batch_size = args.batch_size if hasattr(args, 'batch_size') else 1
            
            print(f"Running eval suite '{suite_name}'...", file=sys.stderr)
            result = runner.run_sync(limit=limit, batch_size=batch_size)
            
            # Build result dictionary
            result_dict = {
                "status": "ok",
                "suite": result.suite_name,
                "num_examples": result.num_examples,
                "examples_per_second": result.examples_per_second,
                "total_time_ms": result.total_time_ms,
                "summary_metrics": result.summary_metrics,
                "errors": result.errors,
                "metadata": result.metadata,
            }
            
            if args.verbose:
                result_dict["metrics_per_example"] = [
                    {
                        "example_id": ex.example_id,
                        "metrics": {
                            name: {"value": m.value, "details": m.details}
                            for name, m in ex.metrics.items()
                        },
                        "judge_scores": ex.judge_scores,
                        "error": ex.error,
                        "execution_time_ms": ex.execution_time_ms,
                    }
                    for ex in result.metrics_per_example
                ]
            
            result_json = json.dumps(result_dict, indent=2)
            
            # Write output
            output = args.output if hasattr(args, 'output') else None
            if output:
                with open(output, 'w') as f:
                    f.write(result_json)
                print(f"Results written to {output}", file=sys.stderr)
            else:
                print(result_json)
            
            # Print summary to stderr
            print("\n=== Evaluation Summary ===", file=sys.stderr)
            print(f"Suite: {result.suite_name}", file=sys.stderr)
            print(f"Examples: {result.num_examples}", file=sys.stderr)
            print(
                f"Time: {result.total_time_ms:.2f}ms ({result.examples_per_second:.2f} ex/s)",
                file=sys.stderr
            )
            
            if result.errors:
                print(f"Errors: {len(result.errors)}", file=sys.stderr)
            
            print("\nMetrics (mean ± std):", file=sys.stderr)
            for metric_name, stats in result.summary_metrics.items():
                mean_val = stats.get("mean", 0)
                std_val = stats.get("std", 0)
                print(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f}", file=sys.stderr)
        
        except Exception as exc:
            raise CLIRuntimeError(
                f"Evaluation failed: {exc}",
                hint="Check dataset and chain configuration",
            ) from exc
    
    except Exception as exc:
        handle_cli_exception(exc, verbose=getattr(args, "verbose", False))
