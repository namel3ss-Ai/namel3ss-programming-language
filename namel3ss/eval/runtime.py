"""Runtime for executing evaluation suites over datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import asyncio
import logging
from statistics import mean, median, stdev

from .metrics import EvalContext, EvalMetric, EvalMetricResult, create_metric
from .judge import LLMJudge

logger = logging.getLogger(__name__)


@dataclass
class ExampleResult:
    """Result for a single evaluation example."""
    example_id: int
    example: Dict[str, Any]
    output: Dict[str, Any]
    metrics: Dict[str, EvalMetricResult] = field(default_factory=dict)
    judge_scores: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class EvalSuiteResult:
    """
    Complete result of running an evaluation suite.
    
    Attributes:
        suite_name: Name of the eval suite
        num_examples: Total number of examples evaluated
        examples_per_second: Throughput metric
        total_time_ms: Total execution time
        metrics_per_example: List of per-example results
        summary_metrics: Aggregated metrics (mean, median, std, etc.)
        errors: List of errors encountered
    """
    suite_name: str
    num_examples: int
    examples_per_second: float
    total_time_ms: float
    metrics_per_example: List[ExampleResult] = field(default_factory=list)
    summary_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvalSuiteRunner:
    """
    Runner for executing evaluation suites.
    
    Executes a chain over a dataset, collecting metrics for each example.
    """
    
    def __init__(
        self,
        suite_name: str,
        dataset_rows: List[Dict[str, Any]],
        chain_executor: Any,
        metrics: List[EvalMetric],
        judge: Optional[LLMJudge] = None,
    ) -> None:
        """
        Initialize eval suite runner.
        
        Args:
            suite_name: Name of the evaluation suite
            dataset_rows: List of dataset rows to evaluate
            chain_executor: Callable that executes the chain on one input
            metrics: List of EvalMetric instances to compute
            judge: Optional LLMJudge for rubric-based scoring
        """
        self.suite_name = suite_name
        self.dataset_rows = dataset_rows
        self.chain_executor = chain_executor
        self.metrics = metrics
        self.judge = judge
    
    async def run(
        self, 
        limit: Optional[int] = None,
        batch_size: int = 1,
        continue_on_error: bool = True,
    ) -> EvalSuiteResult:
        """
        Run the evaluation suite.
        
        Args:
            limit: Maximum number of examples to evaluate (None = all)
            batch_size: Number of examples to process concurrently
            continue_on_error: Whether to continue if an example fails
            
        Returns:
            EvalSuiteResult with all metrics and summary statistics
        """
        start_time = time.time()
        
        # Limit dataset if requested
        rows_to_eval = self.dataset_rows[:limit] if limit else self.dataset_rows
        num_examples = len(rows_to_eval)
        
        logger.info(f"Starting eval suite '{self.suite_name}' with {num_examples} examples")
        
        example_results: List[ExampleResult] = []
        errors: List[str] = []
        
        # Process examples (with optional batching)
        if batch_size > 1:
            # Batch processing
            for batch_start in range(0, num_examples, batch_size):
                batch_end = min(batch_start + batch_size, num_examples)
                batch = rows_to_eval[batch_start:batch_end]
                
                tasks = [
                    self._evaluate_example(idx + batch_start, row, continue_on_error)
                    for idx, row in enumerate(batch)
                ]
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        error_msg = f"Batch example failed: {result}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                        if not continue_on_error:
                            raise result
                    else:
                        example_results.append(result)
                        if result.error:
                            errors.append(result.error)
        else:
            # Sequential processing
            for idx, row in enumerate(rows_to_eval):
                try:
                    result = await self._evaluate_example(idx, row, continue_on_error)
                    example_results.append(result)
                    if result.error:
                        errors.append(result.error)
                except Exception as exc:
                    error_msg = f"Example {idx} failed: {exc}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    if not continue_on_error:
                        raise
        
        # Compute summary statistics
        summary = self._compute_summary(example_results)
        
        elapsed_ms = (time.time() - start_time) * 1000
        examples_per_sec = num_examples / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        
        logger.info(
            f"Completed eval suite '{self.suite_name}': "
            f"{num_examples} examples in {elapsed_ms:.2f}ms "
            f"({examples_per_sec:.2f} examples/sec)"
        )
        
        return EvalSuiteResult(
            suite_name=self.suite_name,
            num_examples=num_examples,
            examples_per_second=examples_per_sec,
            total_time_ms=elapsed_ms,
            metrics_per_example=example_results,
            summary_metrics=summary,
            errors=errors,
            metadata={
                "batch_size": batch_size,
                "continue_on_error": continue_on_error,
                "limit": limit,
            }
        )
    
    def run_sync(
        self,
        limit: Optional[int] = None,
        batch_size: int = 1,
        continue_on_error: bool = True,
    ) -> EvalSuiteResult:
        """Synchronous wrapper for run."""
        return asyncio.run(self.run(limit, batch_size, continue_on_error))
    
    async def _evaluate_example(
        self, 
        idx: int, 
        example: Dict[str, Any],
        continue_on_error: bool,
    ) -> ExampleResult:
        """
        Evaluate a single example.
        
        Args:
            idx: Example index
            example: Dataset row
            continue_on_error: Whether to return partial result on error
            
        Returns:
            ExampleResult with metrics
        """
        example_start = time.time()
        result = ExampleResult(
            example_id=idx,
            example=example,
            output={},
        )
        
        try:
            # Extract input for chain
            chain_input = self._prepare_chain_input(example)
            
            # Execute chain and collect timing/cost
            exec_start = time.time()
            chain_output = await self._execute_chain(chain_input)
            exec_time_ms = (time.time() - exec_start) * 1000
            
            result.output = chain_output
            
            # Extract timings and cost from chain output
            timings = {"total_latency_ms": exec_time_ms}
            if isinstance(chain_output, dict):
                if "metadata" in chain_output and isinstance(chain_output["metadata"], dict):
                    if "elapsed_ms" in chain_output["metadata"]:
                        timings["total_latency_ms"] = chain_output["metadata"]["elapsed_ms"]
                    timings.update(chain_output["metadata"].get("timings", {}))
            
            cost = {}
            if isinstance(chain_output, dict) and "metadata" in chain_output:
                meta = chain_output["metadata"]
                if isinstance(meta, dict):
                    if "usage" in meta:
                        usage = meta["usage"]
                        if isinstance(usage, dict):
                            cost.update(usage)
            
            # Build evaluation context
            ctx = EvalContext(
                example=example,
                output=chain_output,
                rag_contexts=self._extract_rag_contexts(chain_output),
                timings=timings,
                cost=cost,
            )
            
            # Compute metrics
            for metric in self.metrics:
                try:
                    metric_result = await metric.compute(ctx)
                    result.metrics[metric.name] = metric_result
                except Exception as exc:
                    error_msg = f"Metric '{metric.name}' failed on example {idx}: {exc}"
                    logger.error(error_msg)
                    if not continue_on_error:
                        raise
                    result.error = error_msg if not result.error else f"{result.error}; {error_msg}"
            
            # Run judge if configured
            if self.judge:
                try:
                    judge_result = await self.judge.score(ctx)
                    result.judge_scores = judge_result.get("scores", {})
                except Exception as exc:
                    error_msg = f"Judge failed on example {idx}: {exc}"
                    logger.error(error_msg)
                    if not continue_on_error:
                        raise
                    result.error = error_msg if not result.error else f"{result.error}; {error_msg}"
        
        except Exception as exc:
            error_msg = f"Example {idx} evaluation failed: {exc}"
            logger.error(error_msg)
            result.error = error_msg
            if not continue_on_error:
                raise
        
        result.execution_time_ms = (time.time() - example_start) * 1000
        return result
    
    def _prepare_chain_input(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare chain input from dataset example.
        
        Looks for common input field names.
        """
        # Common input field names
        input_candidates = ["question", "query", "input", "prompt", "text"]
        
        for candidate in input_candidates:
            if candidate in example:
                return {candidate: example[candidate]}
        
        # If no standard field, pass entire example
        return example
    
    async def _execute_chain(self, chain_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the chain with given input.
        
        Handles both sync and async chain executors.
        """
        if asyncio.iscoroutinefunction(self.chain_executor):
            return await self.chain_executor(chain_input)
        else:
            # Sync executor - run in executor to avoid blocking
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.chain_executor, chain_input)
    
    def _extract_rag_contexts(self, chain_output: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Extract RAG contexts from chain output if available.
        
        Looks for common patterns where retrieved docs are stored.
        """
        if not isinstance(chain_output, dict):
            return None
        
        # Check for contexts in various locations
        if "contexts" in chain_output:
            return chain_output["contexts"]
        
        if "retrieved_docs" in chain_output:
            return chain_output["retrieved_docs"]
        
        if "documents" in chain_output:
            return chain_output["documents"]
        
        # Check in metadata
        if "metadata" in chain_output and isinstance(chain_output["metadata"], dict):
            meta = chain_output["metadata"]
            if "contexts" in meta:
                return meta["contexts"]
            if "retrieved_docs" in meta:
                return meta["retrieved_docs"]
        
        return None
    
    def _compute_summary(self, results: List[ExampleResult]) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics across all examples.
        
        Returns dict mapping metric names to summary stats (mean, median, std, etc.).
        """
        summary: Dict[str, Dict[str, float]] = {}
        
        # Group values by metric name
        metric_values: Dict[str, List[float]] = {}
        
        for result in results:
            for metric_name, metric_result in result.metrics.items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                metric_values[metric_name].append(metric_result.value)
            
            # Include judge scores
            for dimension, score in result.judge_scores.items():
                judge_metric_name = f"judge_{dimension}"
                if judge_metric_name not in metric_values:
                    metric_values[judge_metric_name] = []
                metric_values[judge_metric_name].append(score)
        
        # Compute statistics for each metric
        for metric_name, values in metric_values.items():
            if not values:
                continue
            
            stats: Dict[str, float] = {
                "mean": mean(values),
                "median": median(values),
                "min": min(values),
                "max": max(values),
                "count": float(len(values)),
            }
            
            if len(values) > 1:
                stats["std"] = stdev(values)
            else:
                stats["std"] = 0.0
            
            summary[metric_name] = stats
        
        return summary
