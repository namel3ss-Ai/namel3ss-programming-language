"""Unit tests for RAG evaluation metrics."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, patch

from namel3ss.eval.rag_eval import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_ndcg_at_k,
    compute_hit_rate,
    compute_mrr,
    RAGEvaluator,
    RAGEvaluationResult,
)


class TestMetricFunctions:
    """Test individual metric computation functions."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc2", "doc4", "doc6"]
        
        p_at_3 = compute_precision_at_k(retrieved, relevant, k=3)
        assert p_at_3 == 1/3  # doc2 is relevant out of first 3
        
        p_at_5 = compute_precision_at_k(retrieved, relevant, k=5)
        assert p_at_5 == 2/5  # doc2, doc4 are relevant out of 5
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc2", "doc4", "doc6"]
        
        r_at_3 = compute_recall_at_k(retrieved, relevant, k=3)
        assert r_at_3 == 1/3  # Found 1 out of 3 relevant docs
        
        r_at_10 = compute_recall_at_k(retrieved, relevant, k=10)
        assert r_at_10 == 1/3  # Still only found 1
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevance_scores = {"doc1": 0.0, "doc2": 3.0, "doc3": 2.0, "doc4": 3.0}
        
        ndcg = compute_ndcg_at_k(retrieved, relevance_scores, k=4)
        
        assert 0.0 <= ndcg <= 1.0
        # With perfect ranking (doc2, doc4, doc3, doc1), NDCG would be 1.0
        assert ndcg < 1.0  # Current ranking is not perfect
    
    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        retrieved = ["doc3", "doc2", "doc1"]  # High to low relevance
        relevance_scores = {"doc1": 1.0, "doc2": 2.0, "doc3": 3.0}
        
        ndcg = compute_ndcg_at_k(retrieved, relevance_scores, k=3)
        assert ndcg == pytest.approx(1.0, rel=1e-6)
    
    def test_hit_rate(self):
        """Test hit rate calculation."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc2", "doc4"]
        
        hit = compute_hit_rate(retrieved, relevant)
        assert hit == 1.0  # At least one relevant doc found
        
        retrieved_no_hit = ["doc5", "doc6"]
        hit_no = compute_hit_rate(retrieved_no_hit, relevant)
        assert hit_no == 0.0  # No relevant docs found
    
    def test_mrr(self):
        """Test Mean Reciprocal Rank calculation."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = ["doc3", "doc5"]
        
        mrr = compute_mrr(retrieved, relevant)
        assert mrr == 1/3  # First relevant doc at position 3
        
        retrieved_first = ["doc5", "doc1", "doc2"]
        mrr_first = compute_mrr(retrieved_first, relevant)
        assert mrr_first == 1.0  # First relevant doc at position 1
        
        retrieved_none = ["doc1", "doc2"]
        mrr_none = compute_mrr(retrieved_none, relevant)
        assert mrr_none == 0.0  # No relevant docs found
    
    def test_empty_inputs(self):
        """Test metrics with empty inputs."""
        assert compute_precision_at_k([], ["doc1"], k=5) == 0.0
        assert compute_recall_at_k(["doc1"], [], k=5) == 0.0
        assert compute_hit_rate([], ["doc1"]) == 0.0
        assert compute_mrr([], ["doc1"]) == 0.0


class TestRAGEvaluator:
    """Test RAGEvaluator class."""
    
    @pytest.mark.asyncio
    async def test_init(self):
        """Test evaluator initialization."""
        evaluator = RAGEvaluator(
            k_values=[1, 3, 5],
            use_llm_judge=False,
        )
        
        assert evaluator.k_values == [1, 3, 5]
        assert evaluator.use_llm_judge is False
        assert evaluator.llm_judge is None
    
    @pytest.mark.asyncio
    async def test_evaluate_query(self):
        """Test evaluating a single query."""
        evaluator = RAGEvaluator(k_values=[1, 3, 5])
        
        query = "test query"
        retrieved_doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant_doc_ids = ["doc2", "doc4", "doc6"]
        relevance_scores = {"doc2": 3.0, "doc4": 2.0, "doc6": 3.0}
        
        result = await evaluator.evaluate_query(
            query=query,
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
            relevance_scores=relevance_scores,
        )
        
        assert isinstance(result, RAGEvaluationResult)
        assert result.query == query
        assert 1 in result.precision_at_k
        assert 3 in result.precision_at_k
        assert 5 in result.precision_at_k
        assert result.mrr > 0
        assert 0.0 <= result.hit_rate <= 1.0
    
    @pytest.mark.asyncio
    async def test_evaluate_dataset(self):
        """Test evaluating multiple queries."""
        evaluator = RAGEvaluator(k_values=[3, 5])
        
        # Mock retriever function
        async def mock_retriever(query):
            if "first" in query:
                return ["doc1", "doc2", "doc3"]
            else:
                return ["doc4", "doc5", "doc6"]
        
        eval_examples = [
            {
                "query": "first query",
                "relevant_docs": ["doc2", "doc7"],
                "relevance_scores": {"doc2": 2.0, "doc7": 3.0},
            },
            {
                "query": "second query",
                "relevant_docs": ["doc4", "doc8"],
                "relevance_scores": {"doc4": 3.0, "doc8": 1.0},
            },
        ]
        
        aggregated = await evaluator.evaluate_dataset(
            eval_examples=eval_examples,
            retriever_fn=mock_retriever,
        )
        
        assert isinstance(aggregated, RAGEvaluationResult)
        assert aggregated.query == "aggregated"
        assert 3 in aggregated.precision_at_k
        assert 5 in aggregated.precision_at_k
        # Check that metrics are averages
        assert 0.0 <= aggregated.precision_at_k[3] <= 1.0
    
    @pytest.mark.asyncio
    @patch("namel3ss.eval.rag_eval.LLMJudge")
    async def test_evaluate_with_llm_judge(self, mock_judge_class):
        """Test evaluation with LLM judge."""
        mock_judge = AsyncMock()
        mock_judge.evaluate_answer.return_value = AsyncMock(
            faithfulness=0.9,
            relevance=0.85,
            correctness=0.95,
            reasoning="Good answer",
        )
        mock_judge_class.return_value = mock_judge
        
        evaluator = RAGEvaluator(
            k_values=[3],
            use_llm_judge=True,
            llm_judge_model="gpt-4",
        )
        evaluator.llm_judge = mock_judge
        
        result = await evaluator.evaluate_query(
            query="test",
            retrieved_doc_ids=["doc1"],
            relevant_doc_ids=["doc1"],
            generated_answer="Answer text",
            ground_truth_answer="Truth text",
        )
        
        assert result.faithfulness == 0.9
        assert result.relevance_score == 0.85
        assert result.correctness == 0.95
        mock_judge.evaluate_answer.assert_called_once()
    
    def test_format_results(self):
        """Test formatting results as markdown."""
        evaluator = RAGEvaluator(k_values=[1, 3, 5])
        
        result = RAGEvaluationResult(
            query="aggregated",
            precision_at_k={1: 0.8, 3: 0.7, 5: 0.6},
            recall_at_k={1: 0.3, 3: 0.5, 5: 0.7},
            ndcg_at_k={1: 0.75, 3: 0.70, 5: 0.65},
            hit_rate=0.95,
            mrr=0.82,
        )
        
        markdown = evaluator.format_results(result)
        
        assert "Precision@1" in markdown
        assert "Recall@5" in markdown
        assert "NDCG@3" in markdown
        assert "Hit Rate" in markdown
        assert "MRR" in markdown
        assert "0.8" in markdown or "0.80" in markdown


class TestRAGEvaluationResult:
    """Test RAGEvaluationResult dataclass."""
    
    def test_create_result(self):
        """Test creating evaluation result."""
        result = RAGEvaluationResult(
            query="test query",
            precision_at_k={1: 0.8, 5: 0.6},
            recall_at_k={1: 0.4, 5: 0.8},
            ndcg_at_k={1: 0.75, 5: 0.65},
            hit_rate=1.0,
            mrr=0.9,
        )
        
        assert result.query == "test query"
        assert result.precision_at_k[1] == 0.8
        assert result.mrr == 0.9
    
    def test_optional_fields(self):
        """Test result with optional LLM judge fields."""
        result = RAGEvaluationResult(
            query="test",
            precision_at_k={1: 0.5},
            recall_at_k={1: 0.5},
            ndcg_at_k={1: 0.5},
            hit_rate=1.0,
            mrr=0.5,
            faithfulness=0.95,
            relevance_score=0.88,
            correctness=0.92,
        )
        
        assert result.faithfulness == 0.95
        assert result.relevance_score == 0.88
        assert result.correctness == 0.92


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
