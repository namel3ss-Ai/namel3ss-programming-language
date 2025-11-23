"""RAG-specific evaluation metrics and utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Union
import numpy as np

from .llm_judge import LLMJudge

logger = logging.getLogger(__name__)


@dataclass
class RAGEvaluationResult:
    """Result from RAG evaluation."""
    query: str
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    hit_rate: float
    mrr: float
    retrieved_docs: List[str] = field(default_factory=list)
    relevant_docs: List[str] = field(default_factory=list)
    faithfulness: Optional[float] = None
    relevance_score: Optional[float] = None
    correctness: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_precision_at_k(
    retrieved: List[str],
    relevant: Union[Set[str], List[str]],
    k: int,
) -> float:
    """
    Compute Precision@K.
    
    Args:
        retrieved: List of retrieved document IDs (ordered by relevance)
        relevant: Set of relevant document IDs
        k: Cutoff rank
        
    Returns:
        Precision@K score
    """
    if k <= 0:
        return 0.0
    
    relevant_set = set(relevant)
    retrieved_at_k = retrieved[:k]
    relevant_count = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
    
    return relevant_count / k


def compute_recall_at_k(
    retrieved: List[str],
    relevant: Union[Set[str], List[str]],
    k: int,
) -> float:
    """
    Compute Recall@K.
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: Set of relevant document IDs
        k: Cutoff rank
        
    Returns:
        Recall@K score
    """
    relevant_set = set(relevant)
    
    if len(relevant_set) == 0:
        return 0.0
    
    if k <= 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_count = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
    
    return relevant_count / len(relevant_set)


def compute_hit_rate(
    retrieved: List[str],
    relevant: Union[Set[str], List[str]],
) -> float:
    """
    Compute hit rate (whether any relevant doc was retrieved).
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: Set of relevant document IDs
        
    Returns:
        1.0 if any relevant doc retrieved, 0.0 otherwise
    """
    relevant_set = set(relevant)
    for doc_id in retrieved:
        if doc_id in relevant_set:
            return 1.0
    return 0.0


def compute_mrr(
    retrieved: List[str],
    relevant: Union[Set[str], List[str]],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: Set of relevant document IDs
        
    Returns:
        Reciprocal rank of first relevant document
    """
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    retrieved: List[str],
    relevant_or_scores: Union[Set[str], Dict[str, float], List[str]],
    relevance_scores: Optional[Dict[str, float]] = None,
    k: int = 10,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).
    
    Args:
        retrieved: List of retrieved document IDs (ordered by system ranking)
        relevant: Set of relevant document IDs
        relevance_scores: Optional dict mapping doc_id to relevance score (0-1)
        k: Cutoff rank
        
    Returns:
        NDCG@K score
    """
    if k <= 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    
    # Allow second arg to be relevance_scores dict (test style) or set/list of relevant docs
    if isinstance(relevant_or_scores, dict):
        relevance_scores = relevant_or_scores
        relevant = set(relevant_or_scores.keys())
    else:
        relevant = set(relevant_or_scores)
        relevance_scores = relevance_scores or {doc_id: 1.0 for doc_id in relevant}
    
    # Compute DCG
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_at_k, start=1):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += rel / np.log2(rank + 1)
    
    # Compute IDCG (ideal DCG)
    ideal_ranking = sorted(
        [relevance_scores.get(doc_id, 0.0) for doc_id in retrieved_at_k],
        reverse=True,
    )
    idcg = 0.0
    for rank, rel in enumerate(ideal_ranking, start=1):
        idcg += rel / np.log2(rank + 1)
    
    # Avoid division by zero
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


class RAGEvaluator:
    """
    Evaluator for RAG systems.
    
    Computes retrieval metrics and optionally generation quality metrics.
    """
    
    def __init__(
        self,
        k_values: List[int] = None,
        use_llm_judge: bool = False,
        llm_judge_model: str = "gpt-4",
        llm_api_key: Optional[str] = None,
    ):
        """
        Initialize RAG evaluator.
        
        Args:
            k_values: List of k values for @k metrics
            use_llm_judge: Whether to use LLM for answer quality evaluation
            llm_judge_model: LLM model for judging
            llm_api_key: API key for LLM
        """
        self.k_values = k_values or [1, 3, 5, 10]
        self.use_llm_judge = use_llm_judge
        self.llm_judge_model = llm_judge_model
        self.llm_api_key = llm_api_key
        
        self.llm_judge = None
        if use_llm_judge:
            from .llm_judge import LLMJudge
            self.llm_judge = LLMJudge(
                model=llm_judge_model,
                api_key=llm_api_key,
            )
    
    async def evaluate_query(
        self,
        query: str,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        generated_answer: Optional[str] = None,
        ground_truth_answer: Optional[str] = None,
        retrieved_contents: Optional[List[str]] = None,
    ) -> RAGEvaluationResult:
        """
        Evaluate a single query.
        
        Args:
            query: Query text
            retrieved_doc_ids: Retrieved document IDs (ordered)
            relevant_doc_ids: Ground truth relevant document IDs
            relevance_scores: Optional relevance scores for docs
            generated_answer: Generated answer (for generation eval)
            ground_truth_answer: Ground truth answer (for generation eval)
            retrieved_contents: Retrieved document contents (for LLM judge)
            
        Returns:
            RAGEvaluationResult with all metrics
        """
        relevant_set = set(relevant_doc_ids)
        
        # Compute retrieval metrics
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        
        for k in self.k_values:
            precision_at_k[k] = compute_precision_at_k(
                retrieved_doc_ids,
                relevant_set,
                k,
            )
            recall_at_k[k] = compute_recall_at_k(
                retrieved_doc_ids,
                relevant_set,
                k,
            )
            ndcg_at_k[k] = compute_ndcg_at_k(
                retrieved_doc_ids,
                relevant_set,
                relevance_scores,
                k,
            )
        
        hit_rate = compute_hit_rate(retrieved_doc_ids, relevant_set)
        mrr = compute_mrr(retrieved_doc_ids, relevant_set)
        
        metadata = {}
        faithfulness = None
        relevance_score = None
        correctness = None
        
        # Optionally evaluate generated answer
        if self.use_llm_judge and generated_answer and self.llm_judge:
            judge_result = await self.llm_judge.evaluate_answer(
                query=query,
                answer=generated_answer,
                contexts=retrieved_contents or [],
                ground_truth=ground_truth_answer,
            )
            faithfulness = getattr(judge_result, "faithfulness", None)
            relevance_score = getattr(judge_result, "relevance", None)
            correctness = getattr(judge_result, "correctness", None)
            metadata["judge_reasoning"] = getattr(judge_result, "reasoning", "")
        
        return RAGEvaluationResult(
            query=query,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            hit_rate=hit_rate,
            mrr=mrr,
            retrieved_docs=retrieved_doc_ids,
            relevant_docs=relevant_doc_ids,
            faithfulness=faithfulness,
            relevance_score=relevance_score,
            correctness=correctness,
            metadata=metadata,
        )
    
    async def evaluate_dataset(
        self,
        eval_examples: List[Dict[str, Any]],
        retriever_fn: callable,
        generator_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate on a dataset.
        
        Args:
            eval_examples: List of evaluation examples with keys:
                - query: Query text
                - relevant_docs: List of relevant document IDs
                - relevance_scores: Optional dict of relevance scores
                - ground_truth_answer: Optional ground truth answer
            retriever_fn: Async function(query) -> List[retrieved_doc_ids]
            generator_fn: Optional async function(query, contexts) -> answer
            
        Returns:
            Dict with aggregated metrics
        """
        results = []
        
        for example in eval_examples:
            query = example["query"]
            relevant_docs = example["relevant_docs"]
            relevance_scores = example.get("relevance_scores")
            ground_truth = example.get("ground_truth_answer")
            
            # Retrieve documents
            retrieved = await retriever_fn(query)
            retrieved_doc_ids = []
            retrieved_contents = []
            if retrieved and isinstance(retrieved[0], dict):
                retrieved_doc_ids = [doc.get("id", doc) for doc in retrieved]
                retrieved_contents = [doc.get("content", "") for doc in retrieved]
            else:
                retrieved_doc_ids = list(retrieved)
                retrieved_contents = [str(doc) for doc in retrieved]
            
            # Optionally generate answer
            generated_answer = None
            if generator_fn:
                generated_answer = await generator_fn(query, retrieved_contents)
            
            # Evaluate
            result = await self.evaluate_query(
                query=query,
                retrieved_doc_ids=retrieved_doc_ids,
                relevant_doc_ids=relevant_docs,
                relevance_scores=relevance_scores,
                generated_answer=generated_answer,
                ground_truth_answer=ground_truth,
                retrieved_contents=retrieved_contents,
            )
            
            results.append(result)
        
        # Aggregate metrics
        aggregated = self._aggregate_results(results)
        
        return aggregated
    
    def _aggregate_results(
        self,
        results: List[RAGEvaluationResult],
    ) -> RAGEvaluationResult:
        """Aggregate results across queries."""
        if not results:
            return RAGEvaluationResult(
                query="aggregated",
                precision_at_k={},
                recall_at_k={},
                ndcg_at_k={},
                hit_rate=0.0,
                mrr=0.0,
            )
        
        # Aggregate retrieval metrics
        precision_at_k: Dict[int, float] = {}
        recall_at_k: Dict[int, float] = {}
        ndcg_at_k: Dict[int, float] = {}
        
        for k in self.k_values:
            precision_at_k[k] = float(np.mean([r.precision_at_k.get(k, 0.0) for r in results]))
            recall_at_k[k] = float(np.mean([r.recall_at_k.get(k, 0.0) for r in results]))
            ndcg_at_k[k] = float(np.mean([r.ndcg_at_k.get(k, 0.0) for r in results]))
        
        hit_rate = float(np.mean([r.hit_rate for r in results]))
        mrr = float(np.mean([r.mrr for r in results]))
        
        faithfulness = None
        relevance_score = None
        correctness = None
        if self.use_llm_judge:
            faithfulness_vals = [r.faithfulness for r in results if r.faithfulness is not None]
            relevance_vals = [r.relevance_score for r in results if r.relevance_score is not None]
            correctness_vals = [r.correctness for r in results if r.correctness is not None]
            if faithfulness_vals:
                faithfulness = float(np.mean(faithfulness_vals))
            if relevance_vals:
                relevance_score = float(np.mean(relevance_vals))
            if correctness_vals:
                correctness = float(np.mean(correctness_vals))
        
        return RAGEvaluationResult(
            query="aggregated",
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            hit_rate=hit_rate,
            mrr=mrr,
            faithfulness=faithfulness,
            relevance_score=relevance_score,
            correctness=correctness,
            metadata={"num_queries": len(results)},
        )
    
    def format_results(self, aggregated: RAGEvaluationResult) -> str:
        """Format aggregated results as markdown."""
        lines = ["# RAG Evaluation Results\n"]
        lines.append(f"**Number of queries**: {aggregated.metadata.get('num_queries', 1)}\n")
        
        lines.append("## Retrieval Metrics\n")
        lines.append(f"- **Hit Rate**: {aggregated.hit_rate:.4f}")
        lines.append(f"- **MRR**: {aggregated.mrr:.4f}\n")
        
        lines.append("### Precision@K\n")
        for k, score in aggregated.precision_at_k.items():
            lines.append(f"- **Precision@{k}**: {score:.4f}")
        
        lines.append("\n### Recall@K\n")
        for k, score in aggregated.recall_at_k.items():
            lines.append(f"- **Recall@{k}**: {score:.4f}")
        
        lines.append("\n### NDCG@K\n")
        for k, score in aggregated.ndcg_at_k.items():
            lines.append(f"- **NDCG@{k}**: {score:.4f}")
        
        if aggregated.faithfulness is not None or aggregated.relevance_score is not None:
            lines.append("\n## Generation Metrics\n")
            if aggregated.faithfulness is not None:
                lines.append(f"- **Faithfulness**: {aggregated.faithfulness:.4f}")
            if aggregated.relevance_score is not None:
                lines.append(f"- **Relevance**: {aggregated.relevance_score:.4f}")
            if aggregated.correctness is not None:
                lines.append(f"- **Correctness**: {aggregated.correctness:.4f}")
        
        return "\n".join(lines)
