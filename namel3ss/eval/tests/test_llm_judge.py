"""Unit tests for LLM judge evaluation."""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock

from namel3ss.eval.llm_judge import (
    LLMJudge,
    FaithfulnessResult,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_evaluation_data():
    """Sample data for evaluation."""
    return {
        "query": "What is machine learning?",
        "answer": "Machine learning is a subset of AI that enables systems to learn from data.",
        "contexts": [
            "Machine learning is a branch of artificial intelligence.",
            "ML systems improve through experience.",
        ],
        "ground_truth": "Machine learning is a type of AI that learns from data.",
    }


class TestLLMJudge:
    """Test suite for LLMJudge."""
    
    async def test_init(self):
        """Test LLM judge initialization."""
        judge = LLMJudge(model="gpt-4", api_key="test-key")
        
        assert judge.model == "gpt-4"
        assert judge.client is not None
    
    @patch("namel3ss.eval.llm_judge.AsyncOpenAI")
    async def test_evaluate_answer(self, mock_openai_class, sample_evaluation_data):
        """Test evaluating an answer."""
        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "faithfulness": 0.95,
                        "relevance": 0.90,
                        "correctness": 0.88,
                        "reasoning": "The answer is accurate and grounded in the context."
                    })
                )
            )
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        judge = LLMJudge(model="gpt-4", api_key="test-key")
        judge.client = mock_client
        
        result = await judge.evaluate_answer(
            query=sample_evaluation_data["query"],
            answer=sample_evaluation_data["answer"],
            contexts=sample_evaluation_data["contexts"],
            ground_truth=sample_evaluation_data["ground_truth"],
        )
        
        assert isinstance(result, FaithfulnessResult)
        assert result.faithfulness == 0.95
        assert result.relevance == 0.90
        assert result.correctness == 0.88
        assert "accurate" in result.reasoning.lower()
        mock_client.chat.completions.create.assert_called_once()
    
    @patch("namel3ss.eval.llm_judge.AsyncOpenAI")
    async def test_evaluate_without_ground_truth(self, mock_openai_class):
        """Test evaluation without ground truth."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "faithfulness": 0.92,
                        "relevance": 0.88,
                        "correctness": None,
                        "reasoning": "No ground truth provided."
                    })
                )
            )
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        judge = LLMJudge(model="gpt-4", api_key="test-key")
        judge.client = mock_client
        
        result = await judge.evaluate_answer(
            query="test query",
            answer="test answer",
            contexts=["context1"],
            ground_truth=None,
        )
        
        assert result.faithfulness == 0.92
        assert result.relevance == 0.88
        assert result.correctness is None
    
    def test_build_evaluation_prompt(self):
        """Test building evaluation prompt."""
        judge = LLMJudge(model="gpt-4")
        
        prompt = judge._build_evaluation_prompt(
            query="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["AI stands for artificial intelligence."],
            ground_truth="Artificial intelligence.",
        )
        
        assert "What is AI?" in prompt
        assert "AI is artificial intelligence." in prompt
        assert "faithfulness" in prompt.lower()
        assert "relevance" in prompt.lower()
        assert "correctness" in prompt.lower()
    
    def test_parse_evaluation_response(self):
        """Test parsing LLM evaluation response."""
        judge = LLMJudge(model="gpt-4")
        
        response_text = '''
        Here's my evaluation:
        ```json
        {
            "faithfulness": 0.85,
            "relevance": 0.90,
            "correctness": 0.82,
            "reasoning": "The answer is mostly accurate."
        }
        ```
        '''
        
        result = judge._parse_evaluation_response(response_text)
        
        assert result.faithfulness == 0.85
        assert result.relevance == 0.90
        assert result.correctness == 0.82
        assert "mostly accurate" in result.reasoning.lower()
    
    def test_parse_invalid_json(self):
        """Test handling invalid JSON response."""
        judge = LLMJudge(model="gpt-4")
        
        # Response without valid JSON
        response_text = "This is not JSON at all."
        
        result = judge._parse_evaluation_response(response_text)
        
        # Should return default values
        assert result.faithfulness == 0.0
        assert result.relevance == 0.0
        assert result.correctness == 0.0
        assert "error" in result.reasoning.lower()
    
    def test_parse_incomplete_json(self):
        """Test handling incomplete JSON response."""
        judge = LLMJudge(model="gpt-4")
        
        response_text = '{"faithfulness": 0.9}'  # Missing other fields
        
        result = judge._parse_evaluation_response(response_text)
        
        assert result.faithfulness == 0.9
        # Missing fields should default to 0.0
        assert result.relevance == 0.0
        assert result.correctness == 0.0


class TestFaithfulnessResult:
    """Test FaithfulnessResult dataclass."""
    
    def test_create_result(self):
        """Test creating faithfulness result."""
        result = FaithfulnessResult(
            faithfulness=0.95,
            relevance=0.88,
            correctness=0.92,
            reasoning="All metrics are high.",
        )
        
        assert result.faithfulness == 0.95
        assert result.relevance == 0.88
        assert result.correctness == 0.92
        assert result.reasoning == "All metrics are high."
    
    def test_optional_correctness(self):
        """Test result with optional correctness."""
        result = FaithfulnessResult(
            faithfulness=0.90,
            relevance=0.85,
            correctness=None,
            reasoning="No ground truth.",
        )
        
        assert result.faithfulness == 0.90
        assert result.correctness is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
