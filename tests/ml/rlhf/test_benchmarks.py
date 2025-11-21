"""
Unit tests for RLHF benchmark evaluation.

Tests benchmark runners:
- MTBenchRunner
- AlpacaEvalRunner
- TruthfulQARunner
- BenchmarkSuite
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from namel3ss.ml.rlhf.evaluation.benchmarks import (
    MTBenchRunner,
    AlpacaEvalRunner,
    TruthfulQARunner,
    BenchmarkSuite,
    BenchmarkResult,
    BenchmarkType,
)


class MockModel:
    """Mock model for testing benchmarks."""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
    
    def generate(self, prompt, **kwargs):
        """Generate mock response."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return "Default response"
    
    def generate_batch(self, prompts, **kwargs):
        """Generate batch of mock responses."""
        return [self.generate(p, **kwargs) for p in prompts]


class TestMTBenchRunner:
    """Test MT-Bench evaluation."""

    def test_initialization(self):
        """Test MT-Bench runner initialization."""
        runner = MTBenchRunner()
        
        assert runner.benchmark_name == "MT-Bench"
        assert runner.benchmark_type == BenchmarkType.MT_BENCH
        assert len(runner.categories) == 8

    def test_categories(self):
        """Test that all expected categories are present."""
        runner = MTBenchRunner()
        
        expected_categories = [
            "writing", "roleplay", "reasoning", "math",
            "coding", "extraction", "stem", "humanities"
        ]
        
        assert all(cat in runner.categories for cat in expected_categories)

    def test_run_with_mock_model(self):
        """Test running MT-Bench with mock model."""
        model = MockModel(responses=[
            "This is a well-written response.",
            "I will roleplay as requested.",
            "Here is my reasoning: ...",
        ])
        
        runner = MTBenchRunner()
        result = runner.run(model, num_samples=3)
        
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "MT-Bench"
        assert result.benchmark_type == BenchmarkType.MT_BENCH
        assert result.model_name == "mock_model"
        assert 1.0 <= result.score <= 10.0
        assert len(result.individual_scores) == 3

    def test_judge_model_configuration(self):
        """Test configuring judge model."""
        runner = MTBenchRunner(judge_model="gpt-3.5-turbo")
        
        assert runner.judge_model == "gpt-3.5-turbo"

    def test_score_range(self):
        """Test that scores are in valid range (1-10)."""
        model = MockModel(responses=["Good response"] * 5)
        runner = MTBenchRunner()
        result = runner.run(model, num_samples=5)
        
        assert 1.0 <= result.score <= 10.0
        for score_dict in result.individual_scores:
            assert 1.0 <= score_dict["score"] <= 10.0

    def test_category_distribution(self):
        """Test that categories are distributed across samples."""
        model = MockModel(responses=["Response"] * 8)
        runner = MTBenchRunner()
        result = runner.run(model, num_samples=8)
        
        categories = [s["category"] for s in result.individual_scores]
        # Should have variety of categories
        assert len(set(categories)) >= 3

    def test_metadata_includes_categories(self):
        """Test that metadata includes category breakdown."""
        model = MockModel()
        runner = MTBenchRunner()
        result = runner.run(model, num_samples=4)
        
        assert "category_scores" in result.metadata
        assert isinstance(result.metadata["category_scores"], dict)


class TestAlpacaEvalRunner:
    """Test AlpacaEval evaluation."""

    def test_initialization(self):
        """Test AlpacaEval runner initialization."""
        runner = AlpacaEvalRunner()
        
        assert runner.benchmark_name == "AlpacaEval"
        assert runner.benchmark_type == BenchmarkType.ALPACA_EVAL

    def test_reference_model_configuration(self):
        """Test configuring reference model."""
        runner = AlpacaEvalRunner(reference_model="gpt-4")
        
        assert runner.reference_model == "gpt-4"

    def test_run_with_mock_model(self):
        """Test running AlpacaEval with mock model."""
        model = MockModel(responses=[
            "I will follow these instructions carefully.",
            "Here is the requested information.",
            "Task completed as specified.",
        ])
        
        runner = AlpacaEvalRunner()
        result = runner.run(model, num_samples=3)
        
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "AlpacaEval"
        assert result.benchmark_type == BenchmarkType.ALPACA_EVAL
        assert 0.0 <= result.score <= 1.0

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        model = MockModel(responses=["Excellent response"] * 5)
        runner = AlpacaEvalRunner()
        result = runner.run(model, num_samples=5)
        
        # Score should be win rate
        assert 0.0 <= result.score <= 1.0
        assert "wins" in result.metadata
        assert "ties" in result.metadata
        assert "losses" in result.metadata
        
        total = result.metadata["wins"] + result.metadata["ties"] + result.metadata["losses"]
        assert total == 5

    def test_individual_scores_structure(self):
        """Test structure of individual scores."""
        model = MockModel()
        runner = AlpacaEvalRunner()
        result = runner.run(model, num_samples=3)
        
        for score_dict in result.individual_scores:
            assert "instruction" in score_dict
            assert "response" in score_dict
            assert "outcome" in score_dict
            assert score_dict["outcome"] in ["win", "tie", "loss"]

    def test_win_rate_metadata(self):
        """Test that metadata includes win rate details."""
        model = MockModel()
        runner = AlpacaEvalRunner()
        result = runner.run(model, num_samples=10)
        
        assert "wins" in result.metadata
        assert "ties" in result.metadata
        assert "losses" in result.metadata
        assert "win_rate" in result.metadata
        assert result.metadata["win_rate"] == result.score


class TestTruthfulQARunner:
    """Test TruthfulQA evaluation."""

    def test_initialization(self):
        """Test TruthfulQA runner initialization."""
        runner = TruthfulQARunner()
        
        assert runner.benchmark_name == "TruthfulQA"
        assert runner.benchmark_type == BenchmarkType.TRUTHFUL_QA

    def test_run_with_mock_model(self):
        """Test running TruthfulQA with mock model."""
        model = MockModel(responses=[
            "The Earth orbits around the Sun.",
            "Water boils at 100 degrees Celsius at sea level.",
            "Humans need oxygen to survive.",
        ])
        
        runner = TruthfulQARunner()
        result = runner.run(model, num_samples=3)
        
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "TruthfulQA"
        assert result.benchmark_type == BenchmarkType.TRUTHFUL_QA
        assert 0.0 <= result.score <= 1.0

    def test_truthful_and_informative_scores(self):
        """Test separate truthful and informative scoring."""
        model = MockModel()
        runner = TruthfulQARunner()
        result = runner.run(model, num_samples=5)
        
        assert "truthful_rate" in result.metadata
        assert "informative_rate" in result.metadata
        assert 0.0 <= result.metadata["truthful_rate"] <= 1.0
        assert 0.0 <= result.metadata["informative_rate"] <= 1.0

    def test_combined_score(self):
        """Test that overall score combines truthful and informative."""
        model = MockModel()
        runner = TruthfulQARunner()
        result = runner.run(model, num_samples=5)
        
        # Score should be average of truthful and informative
        expected_score = (
            result.metadata["truthful_rate"] + result.metadata["informative_rate"]
        ) / 2
        assert abs(result.score - expected_score) < 0.01

    def test_individual_scores_structure(self):
        """Test structure of individual scores."""
        model = MockModel()
        runner = TruthfulQARunner()
        result = runner.run(model, num_samples=3)
        
        for score_dict in result.individual_scores:
            assert "question" in score_dict
            assert "response" in score_dict
            assert "truthful" in score_dict
            assert "informative" in score_dict
            assert isinstance(score_dict["truthful"], bool)
            assert isinstance(score_dict["informative"], bool)

    def test_question_variety(self):
        """Test that questions come from different categories."""
        model = MockModel()
        runner = TruthfulQARunner()
        result = runner.run(model, num_samples=10)
        
        questions = [s["question"] for s in result.individual_scores]
        # Should have variety (not all same question)
        assert len(set(questions)) >= 5


class TestBenchmarkSuite:
    """Test benchmark suite for running multiple benchmarks."""

    def test_initialization(self):
        """Test suite initialization."""
        suite = BenchmarkSuite()
        
        assert hasattr(suite, 'benchmarks')

    def test_add_benchmark(self):
        """Test adding benchmarks to suite."""
        suite = BenchmarkSuite()
        runner = MTBenchRunner()
        
        suite.benchmarks.append(runner)
        assert len(suite.benchmarks) >= 1

    def test_run_all_benchmarks(self):
        """Test running all benchmarks in suite."""
        model = MockModel(responses=["Response"] * 20)
        
        suite = BenchmarkSuite()
        suite.benchmarks = [
            MTBenchRunner(),
            AlpacaEvalRunner(),
        ]
        
        results = suite.run_all(model, num_samples=5)
        
        assert isinstance(results, dict)
        assert "MT-Bench" in results
        assert "AlpacaEval" in results
        assert all(isinstance(r, BenchmarkResult) for r in results.values())

    def test_save_results(self):
        """Test saving results to JSON."""
        import tempfile
        import json
        import os
        
        model = MockModel()
        suite = BenchmarkSuite()
        suite.benchmarks = [MTBenchRunner()]
        
        results = suite.run_all(model, num_samples=2)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_path = f.name
        
        try:
            suite.save_results(results, output_path)
            
            # Verify file was created and contains valid JSON
            assert os.path.exists(output_path)
            
            with open(output_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert "MT-Bench" in loaded_data
            assert "score" in loaded_data["MT-Bench"]
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_empty_suite(self):
        """Test running empty suite."""
        model = MockModel()
        suite = BenchmarkSuite()
        suite.benchmarks = []
        
        results = suite.run_all(model, num_samples=5)
        
        assert isinstance(results, dict)
        assert len(results) == 0

    def test_single_benchmark(self):
        """Test suite with single benchmark."""
        model = MockModel()
        suite = BenchmarkSuite()
        suite.benchmarks = [TruthfulQARunner()]
        
        results = suite.run_all(model, num_samples=3)
        
        assert len(results) == 1
        assert "TruthfulQA" in results


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_result_creation(self):
        """Test creating BenchmarkResult."""
        result = BenchmarkResult(
            benchmark_name="TestBench",
            benchmark_type=BenchmarkType.CUSTOM,
            model_name="test_model",
            score=0.85,
            timestamp=datetime.now(),
            metadata={"key": "value"},
            individual_scores=[{"score": 0.8}]
        )
        
        assert result.benchmark_name == "TestBench"
        assert result.benchmark_type == BenchmarkType.CUSTOM
        assert result.model_name == "test_model"
        assert result.score == 0.85
        assert isinstance(result.timestamp, datetime)
        assert result.metadata["key"] == "value"
        assert len(result.individual_scores) == 1

    def test_result_with_empty_individual_scores(self):
        """Test result with empty individual scores."""
        result = BenchmarkResult(
            benchmark_name="Test",
            benchmark_type=BenchmarkType.CUSTOM,
            model_name="model",
            score=0.5,
            timestamp=datetime.now(),
            metadata={},
            individual_scores=[]
        )
        
        assert len(result.individual_scores) == 0


class TestBenchmarkType:
    """Test BenchmarkType enum."""

    def test_benchmark_types_exist(self):
        """Test all benchmark types exist."""
        assert BenchmarkType.MT_BENCH
        assert BenchmarkType.ALPACA_EVAL
        assert BenchmarkType.TRUTHFUL_QA
        assert BenchmarkType.HUMAN_EVAL
        assert BenchmarkType.CUSTOM

    def test_benchmark_type_values(self):
        """Test benchmark type enum values."""
        assert BenchmarkType.MT_BENCH.value == "mt_bench"
        assert BenchmarkType.ALPACA_EVAL.value == "alpaca_eval"
        assert BenchmarkType.TRUTHFUL_QA.value == "truthful_qa"
        assert BenchmarkType.HUMAN_EVAL.value == "human_eval"
        assert BenchmarkType.CUSTOM.value == "custom"


class TestBenchmarkIntegration:
    """Integration tests across multiple benchmarks."""

    def test_all_benchmarks_with_same_model(self):
        """Test all benchmarks with same model."""
        model = MockModel(responses=["Quality response"] * 30)
        
        mt_bench = MTBenchRunner()
        alpaca_eval = AlpacaEvalRunner()
        truthful_qa = TruthfulQARunner()
        
        mt_result = mt_bench.run(model, num_samples=5)
        alpaca_result = alpaca_eval.run(model, num_samples=5)
        truthful_result = truthful_qa.run(model, num_samples=5)
        
        # All should return valid results
        assert all(isinstance(r, BenchmarkResult) for r in [mt_result, alpaca_result, truthful_result])
        
        # Scores should be in valid ranges
        assert 1.0 <= mt_result.score <= 10.0
        assert 0.0 <= alpaca_result.score <= 1.0
        assert 0.0 <= truthful_result.score <= 1.0

    def test_benchmark_comparison(self):
        """Test comparing results across benchmarks."""
        model = MockModel()
        
        suite = BenchmarkSuite()
        suite.benchmarks = [
            MTBenchRunner(),
            AlpacaEvalRunner(),
            TruthfulQARunner(),
        ]
        
        results = suite.run_all(model, num_samples=3)
        
        # All benchmarks should have results
        assert len(results) == 3
        
        # Each should have timestamp
        for result in results.values():
            assert isinstance(result.timestamp, datetime)

    def test_consistent_model_name(self):
        """Test that model name is consistent across benchmarks."""
        model = MockModel()
        model.name = "TestModel"
        
        runners = [
            MTBenchRunner(),
            AlpacaEvalRunner(),
            TruthfulQARunner(),
        ]
        
        for runner in runners:
            result = runner.run(model, num_samples=2)
            # Model name should be captured
            assert result.model_name is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
