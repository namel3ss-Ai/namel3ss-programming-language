"""Metric abstraction layer for eval suites."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvalContext:
    """
    Context containing all information for evaluating a single example.
    
    Attributes:
        example: The original dataset row (with ground truth if available)
        output: Chain/model output and intermediate states
        rag_contexts: Retrieved documents/contexts (for RAG pipelines)
        timings: Timing measurements in milliseconds
        cost: Cost measurements (typically in USD)
        metadata: Additional context-specific metadata
    """
    example: Dict[str, Any]
    output: Dict[str, Any]
    rag_contexts: Optional[List[Dict[str, Any]]] = None
    timings: Dict[str, float] = field(default_factory=dict)
    cost: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalMetricResult:
    """
    Result of computing a single metric on one example.
    
    Attributes:
        name: Metric name
        value: Numeric score
        details: Additional information about the metric computation
    """
    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)


class EvalMetric(ABC):
    """Base class for evaluation metrics."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.type = self.__class__.__name__
        self.config = config or {}
    
    @abstractmethod
    async def compute(self, ctx: EvalContext) -> EvalMetricResult:
        """
        Compute the metric for a given evaluation context.
        
        Args:
            ctx: Evaluation context with all necessary information
            
        Returns:
            EvalMetricResult with computed score and details
        """
        pass
    
    def compute_sync(self, ctx: EvalContext) -> EvalMetricResult:
        """
        Synchronous wrapper for compute.
        
        Override this if your metric has a purely synchronous implementation.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # We're in an async context - cannot use asyncio.run
            raise RuntimeError(
                f"Metric {self.name} called synchronously from async context. Use await compute() instead."
            )
        
        return asyncio.run(self.compute(ctx))


class BuiltinLatencyMetric(EvalMetric):
    """Built-in metric that measures total latency in milliseconds."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self.type = "builtin_latency"
    
    async def compute(self, ctx: EvalContext) -> EvalMetricResult:
        """Extract latency from timing measurements."""
        latency_ms = ctx.timings.get("total_latency_ms", 0.0)
        if not latency_ms and "latency_ms" in ctx.timings:
            latency_ms = ctx.timings["latency_ms"]
        
        return EvalMetricResult(
            name=self.name,
            value=latency_ms,
            details={
                "unit": "milliseconds",
                "timings": ctx.timings,
            }
        )


class BuiltinCostMetric(EvalMetric):
    """Built-in metric that measures cost in USD."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self.type = "builtin_cost"
    
    async def compute(self, ctx: EvalContext) -> EvalMetricResult:
        """Extract cost from cost measurements."""
        cost_usd = ctx.cost.get("usd", 0.0)
        if not cost_usd and "total_cost" in ctx.cost:
            cost_usd = ctx.cost["total_cost"]
        
        # Calculate from token counts if direct cost not available
        if not cost_usd and "prompt_tokens" in ctx.cost and "completion_tokens" in ctx.cost:
            # Default pricing (can be overridden in config)
            prompt_price = self.config.get("prompt_token_price_per_1k", 0.0)
            completion_price = self.config.get("completion_token_price_per_1k", 0.0)
            
            prompt_tokens = ctx.cost["prompt_tokens"]
            completion_tokens = ctx.cost["completion_tokens"]
            cost_usd = (prompt_tokens / 1000.0 * prompt_price + 
                       completion_tokens / 1000.0 * completion_price)
        
        return EvalMetricResult(
            name=self.name,
            value=cost_usd,
            details={
                "unit": "USD",
                "cost_data": ctx.cost,
            }
        )


class RagasMetricWrapper(EvalMetric):
    """
    Wrapper for RAGAS metrics.
    
    Requires ragas library to be installed. Gracefully handles missing library.
    """
    
    def __init__(self, name: str, metric_type: str, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self.type = metric_type
        self.ragas_metric_name = metric_type.replace("ragas_", "")
        self._ragas_available = False
        self._ragas_metric = None
        
        try:
            import ragas
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
                answer_similarity,
                answer_correctness,
            )
            
            self._ragas_available = True
            
            # Map metric type to RAGAS metric instance
            metric_map = {
                "relevance": answer_relevancy,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "faithfulness": faithfulness,
                "answer_similarity": answer_similarity,
                "answer_correctness": answer_correctness,
            }
            
            self._ragas_metric = metric_map.get(self.ragas_metric_name)
            if not self._ragas_metric:
                logger.warning(f"Unknown RAGAS metric: {self.ragas_metric_name}")
        
        except ImportError:
            logger.warning(
                f"RAGAS library not installed. Metric '{name}' will fail. "
                "Install with: pip install ragas"
            )
    
    async def compute(self, ctx: EvalContext) -> EvalMetricResult:
        """Compute RAGAS metric."""
        if not self._ragas_available:
            raise RuntimeError(
                f"RAGAS library not available for metric '{self.name}'. "
                "Install with: pip install ragas"
            )
        
        if not self._ragas_metric:
            raise RuntimeError(f"RAGAS metric '{self.ragas_metric_name}' not found")
        
        try:
            # Prepare inputs for RAGAS
            # RAGAS expects specific field names
            question = ctx.example.get("question") or ctx.example.get("query") or ctx.example.get("input")
            answer = ctx.output.get("text") or ctx.output.get("answer") or ctx.output.get("result")
            
            # Ground truth / reference
            ground_truth = ctx.example.get("ground_truth") or ctx.example.get("reference_answer") or ctx.example.get("answer")
            
            # Retrieved contexts
            contexts = None
            if ctx.rag_contexts:
                contexts = [
                    doc.get("text") or doc.get("content") or str(doc)
                    for doc in ctx.rag_contexts
                ]
            elif "contexts" in ctx.example:
                contexts = ctx.example["contexts"]
            
            # Build RAGAS input dict
            ragas_input = {}
            if question:
                ragas_input["question"] = str(question)
            if answer:
                ragas_input["answer"] = str(answer)
            if ground_truth:
                ragas_input["ground_truth"] = str(ground_truth)
            if contexts:
                ragas_input["contexts"] = contexts
            
            # Check required fields for specific metrics
            if self.ragas_metric_name in ("context_precision", "context_recall") and not contexts:
                raise ValueError(f"Metric '{self.name}' requires contexts but none found")
            
            if self.ragas_metric_name in ("answer_correctness", "answer_similarity") and not ground_truth:
                raise ValueError(f"Metric '{self.name}' requires ground_truth but none found")
            
            # Compute metric
            # Note: RAGAS metrics are async
            score = await self._ragas_metric.ascore(**ragas_input)
            
            return EvalMetricResult(
                name=self.name,
                value=float(score),
                details={
                    "metric_type": self.ragas_metric_name,
                    "inputs": {k: v if not isinstance(v, list) else f"[{len(v)} items]" for k, v in ragas_input.items()},
                }
            )
        
        except Exception as exc:
            logger.error(f"Error computing RAGAS metric '{self.name}': {exc}")
            raise RuntimeError(f"Failed to compute RAGAS metric '{self.name}': {exc}") from exc


# Metric factory registry
_METRIC_REGISTRY: Dict[str, type] = {
    "builtin_latency": BuiltinLatencyMetric,
    "builtin_cost": BuiltinCostMetric,
}


def register_metric(metric_type: str, metric_class: type) -> None:
    """
    Register a custom metric type.
    
    Args:
        metric_type: Type identifier (e.g., "custom_accuracy")
        metric_class: Metric class (must inherit from EvalMetric)
    """
    if not issubclass(metric_class, EvalMetric):
        raise TypeError(f"Metric class must inherit from EvalMetric, got {metric_class}")
    
    _METRIC_REGISTRY[metric_type] = metric_class
    logger.info(f"Registered metric type: {metric_type}")


def create_metric(name: str, metric_type: str, config: Optional[Dict[str, Any]] = None) -> EvalMetric:
    """
    Factory function to create metric instances from specifications.
    
    Args:
        name: Logical name of the metric
        metric_type: Type identifier (e.g., "builtin_latency", "ragas_relevance")
        config: Optional configuration dict
        
    Returns:
        EvalMetric instance
        
    Raises:
        ValueError: If metric_type is not recognized
    """
    config = config or {}
    
    # Check registry first
    if metric_type in _METRIC_REGISTRY:
        metric_class = _METRIC_REGISTRY[metric_type]
        return metric_class(name, config)
    
    # Handle RAGAS metrics
    if metric_type.startswith("ragas_"):
        return RagasMetricWrapper(name, metric_type, config)
    
    # Handle custom metrics
    if metric_type.startswith("custom_"):
        raise ValueError(
            f"Custom metric type '{metric_type}' not registered. "
            f"Use register_metric() to register custom metrics."
        )
    
    raise ValueError(f"Unknown metric type: {metric_type}")
