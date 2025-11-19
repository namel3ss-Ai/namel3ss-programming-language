"""LLM-based judging and rubric scoring for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class RubricScore:
    """
    Structured result from rubric-based LLM judging.
    
    Attributes:
        scores: Dict mapping dimension names to numeric scores
        reasoning: Optional explanation from the judge
        raw_response: Raw LLM response for debugging
    """
    scores: Dict[str, float] = field(default_factory=dict)
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None


class LLMJudge:
    """
    LLM-based judge that scores outputs using a rubric.
    
    Uses an LLM to evaluate outputs on multiple dimensions specified in a rubric.
    """
    
    def __init__(self, llm: Any, rubric: str) -> None:
        """
        Initialize LLM judge.
        
        Args:
            llm: LLM instance (must have generate or generate_chat method)
            rubric: Rubric text describing scoring criteria
        """
        self.llm = llm
        self.rubric = rubric
    
    def _build_judge_prompt(self, ctx: Any) -> str:
        """
        Build the prompt for the judge LLM.
        
        Args:
            ctx: EvalContext with example and output
            
        Returns:
            Formatted prompt string
        """
        from .metrics import EvalContext
        
        if not isinstance(ctx, EvalContext):
            raise TypeError("Expected EvalContext")
        
        # Extract relevant fields
        question = ctx.example.get("question") or ctx.example.get("query") or ctx.example.get("input", "")
        answer = ctx.output.get("text") or ctx.output.get("answer") or ctx.output.get("result", "")
        reference = ctx.example.get("ground_truth") or ctx.example.get("reference_answer") or ctx.example.get("answer")
        
        prompt_parts = [
            "You are an expert evaluator. Your task is to score the following AI-generated output based on the rubric provided.\n",
            f"RUBRIC:\n{self.rubric}\n",
            f"\nQUESTION/INPUT:\n{question}\n",
            f"\nAI GENERATED ANSWER:\n{answer}\n",
        ]
        
        if reference:
            prompt_parts.append(f"\nREFERENCE ANSWER (for comparison):\n{reference}\n")
        
        prompt_parts.append(
            "\nProvide your evaluation as a JSON object with the following structure:\n"
            "{\n"
            '  "scores": {"dimension1": <score>, "dimension2": <score>, ...},\n'
            '  "reasoning": "<brief explanation of your scoring>"\n'
            "}\n"
            "\nScores should be numeric values as specified in the rubric (typically 1-5).\n"
            "\nYour response:"
        )
        
        return "".join(prompt_parts)
    
    async def score(self, ctx: Any) -> Dict[str, Any]:
        """
        Score an output using the judge LLM and rubric.
        
        Args:
            ctx: EvalContext with example and output information
            
        Returns:
            Dict with scores, reasoning, and raw response
            
        Raises:
            RuntimeError: If LLM call fails or response cannot be parsed
        """
        from .metrics import EvalContext
        
        if not isinstance(ctx, EvalContext):
            raise TypeError("Expected EvalContext")
        
        prompt = self._build_judge_prompt(ctx)
        
        try:
            # Try to call LLM with chat interface first
            if hasattr(self.llm, "generate_chat"):
                from namel3ss.llm import ChatMessage
                messages = [ChatMessage(role="user", content=prompt)]
                response = await self.llm.generate_chat_async(messages)
                text = response.text
            elif hasattr(self.llm, "generate"):
                response = await self.llm.generate_async(prompt)
                text = response.text
            else:
                raise RuntimeError(f"LLM does not have generate or generate_chat method: {type(self.llm)}")
            
            # Parse JSON response
            result = self._parse_judge_response(text)
            result["raw_response"] = text
            
            return result
        
        except Exception as exc:
            logger.error(f"LLM judge failed: {exc}")
            raise RuntimeError(f"LLM judge scoring failed: {exc}") from exc
    
    def score_sync(self, ctx: Any) -> Dict[str, Any]:
        """Synchronous wrapper for score."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            raise RuntimeError("LLMJudge.score_sync called from async context. Use await score() instead.")
        
        return asyncio.run(self.score(ctx))
    
    def _parse_judge_response(self, text: str) -> Dict[str, Any]:
        """
        Parse the judge's JSON response.
        
        Args:
            text: Raw LLM response text
            
        Returns:
            Dict with parsed scores and reasoning
            
        Raises:
            ValueError: If response cannot be parsed
        """
        # Try to extract JSON from response
        text = text.strip()
        
        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
        
        # Try to find JSON object
        if not text.startswith("{"):
            start_idx = text.find("{")
            if start_idx != -1:
                text = text[start_idx:]
        
        if not text.endswith("}"):
            end_idx = text.rfind("}")
            if end_idx != -1:
                text = text[:end_idx + 1]
        
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse judge response as JSON: {text[:200]}")
            raise ValueError(f"Judge response is not valid JSON: {exc}") from exc
        
        if not isinstance(parsed, dict):
            raise ValueError(f"Judge response must be a JSON object, got {type(parsed)}")
        
        if "scores" not in parsed:
            raise ValueError("Judge response missing 'scores' field")
        
        if not isinstance(parsed["scores"], dict):
            raise ValueError("Judge 'scores' must be a dictionary")
        
        # Convert scores to floats
        scores = {}
        for key, value in parsed["scores"].items():
            try:
                scores[key] = float(value)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert score '{key}': {value} to float, using 0.0")
                scores[key] = 0.0
        
        return {
            "scores": scores,
            "reasoning": parsed.get("reasoning"),
        }


class RubricMetric:
    """
    Evaluation metric based on LLM judge with rubric.
    
    This wraps LLMJudge to provide a metric that extracts a specific
    dimension score from rubric-based judging.
    """
    
    def __init__(
        self, 
        name: str, 
        judge: LLMJudge, 
        dimension: str,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize rubric metric.
        
        Args:
            name: Metric name
            judge: LLMJudge instance
            dimension: Dimension to extract from judge scores (e.g., "helpfulness")
            config: Optional configuration
        """
        self.name = name
        self.type = "rubric"
        self.judge = judge
        self.dimension = dimension
        self.config = config or {}
    
    async def compute(self, ctx: Any) -> Any:
        """Compute metric by running judge and extracting dimension score."""
        from .metrics import EvalMetricResult, EvalContext
        
        if not isinstance(ctx, EvalContext):
            raise TypeError("Expected EvalContext")
        
        result = await self.judge.score(ctx)
        scores = result.get("scores", {})
        
        if self.dimension not in scores:
            available = ", ".join(scores.keys()) if scores else "none"
            raise ValueError(
                f"Judge did not return score for dimension '{self.dimension}'. "
                f"Available dimensions: {available}"
            )
        
        return EvalMetricResult(
            name=self.name,
            value=scores[self.dimension],
            details={
                "dimension": self.dimension,
                "all_scores": scores,
                "reasoning": result.get("reasoning"),
            }
        )
    
    def compute_sync(self, ctx: Any) -> Any:
        """Synchronous wrapper for compute."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            raise RuntimeError("RubricMetric.compute_sync called from async context. Use await compute() instead.")
        
        return asyncio.run(self.compute(ctx))
