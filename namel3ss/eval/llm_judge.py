"""LLM-based judge for evaluating RAG answer quality."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional
from types import SimpleNamespace
import json

try:
    from openai import AsyncOpenAI as _RealAsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover - fallback when openai not installed
    _RealAsyncOpenAI = None


class _StubAsyncOpenAI:
    """Lightweight stub to avoid network/API requirements in tests."""
    def __init__(self, *_, **__):
        async def _stub(*args, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
            )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_stub))


# Expose AsyncOpenAI symbol for tests to patch
AsyncOpenAI = _RealAsyncOpenAI or _StubAsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class FaithfulnessResult:
    """Result from LLM faithfulness evaluation."""
    faithfulness: float  # 0-1
    relevance: float  # 0-1
    correctness: Optional[float]  # 0-1 or None
    reasoning: str
    raw_response: str = ""


class LLMJudge:
    """
    LLM-based judge for evaluating RAG answers.
    
    Evaluates:
    - Faithfulness: Is the answer grounded in the retrieved contexts?
    - Relevance: Does the answer address the query?
    - Correctness: Is the answer factually correct (if ground truth provided)?
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """
        Initialize LLM judge.
        
        Args:
            model: LLM model to use
            api_key: API key (or set OPENAI_API_KEY env var)
            temperature: Temperature for generation
        """
        self.model = model
        self.temperature = temperature
        
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client: Optional[AsyncOpenAI] = None
        
        client_cls = AsyncOpenAI
        if not self.api_key and AsyncOpenAI is _RealAsyncOpenAI:
            client_cls = _StubAsyncOpenAI
        
        if self.api_key:
            try:
                self.client = client_cls(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client, using stub: {e}")
                self.client = _StubAsyncOpenAI()
        else:
            # In tests we often patch the client directly
            self.client = client_cls()
    
    async def evaluate_answer(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> FaithfulnessResult:
        """
        Evaluate answer quality using LLM.
        
        Args:
            query: User query
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer
            
        Returns:
            FaithfulnessResult with scores and reasoning
        """
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
        
        # Call LLM
        client = self.client or AsyncOpenAI()
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator for question-answering systems. "
                               "Evaluate answers based on faithfulness, relevance, and correctness."
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
        )
        
        raw_response = response.choices[0].message.content
        
        # Parse response
        result = self._parse_evaluation_response(raw_response)
        
        return result
    
    def _build_evaluation_prompt(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str],
    ) -> str:
        """Build evaluation prompt."""
        contexts_str = "\n\n".join([
            f"Context {i+1}:\n{ctx}"
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""Evaluate the following question-answering system output.

**Query**: {query}

**Retrieved Contexts**:
{contexts_str}

**Generated Answer**: {answer}
"""
        
        if ground_truth:
            prompt += f"\n**Ground Truth Answer**: {ground_truth}\n"
        
        prompt += """
Please evaluate the answer on the following criteria (score from 0 to 1):

1. **Faithfulness** (0-1): Is the answer grounded in the retrieved contexts? Does it contain hallucinations or unsupported claims?
   - 1.0: Fully grounded, no hallucinations
   - 0.5: Partially grounded, some unsupported claims
   - 0.0: Not grounded, mostly hallucinations

2. **Relevance** (0-1): Does the answer address the query appropriately?
   - 1.0: Directly and completely addresses the query
   - 0.5: Partially relevant
   - 0.0: Not relevant to the query

"""
        
        if ground_truth:
            prompt += """3. **Correctness** (0-1): Is the answer factually correct compared to the ground truth?
   - 1.0: Fully correct
   - 0.5: Partially correct
   - 0.0: Incorrect

"""
        
        prompt += """Provide your evaluation in the following JSON format:
```json
{
  "faithfulness_score": <float 0-1>,
  "relevance_score": <float 0-1>,
  "correctness_score": <float 0-1>,
  "reasoning": "<explanation of your scores>"
}
```
"""
        
        return prompt
    
    def _parse_evaluation_response(self, response: str) -> FaithfulnessResult:
        """Parse LLM evaluation response."""
        # Try to extract JSON from response
        import re
        
        # Look for JSON block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find any JSON object
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning("Failed to parse JSON from LLM response")
                return FaithfulnessResult(
                    faithfulness=0.0,
                    relevance=0.0,
                    correctness=0.0,
                    reasoning="error: failed to parse evaluation",
                    raw_response=response,
                )
        
        try:
            data = json.loads(json_str)
            
            return FaithfulnessResult(
                faithfulness=float(data.get("faithfulness", data.get("faithfulness_score", 0.0))),
                relevance=float(data.get("relevance", data.get("relevance_score", 0.0))),
                correctness=data.get("correctness", data.get("correctness_score", 0.0)),
                reasoning=data.get("reasoning", ""),
                raw_response=response,
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode JSON: {e}")
            return FaithfulnessResult(
                faithfulness=0.0,
                relevance=0.0,
                correctness=0.0,
                reasoning="error: failed to decode evaluation JSON",
                raw_response=response,
            )
