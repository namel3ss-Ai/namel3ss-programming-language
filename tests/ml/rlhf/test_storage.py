"""
Unit tests for RLHF evaluation result storage.

Tests SQLAlchemy models and EvaluationStorage API:
- EvaluationRun
- MetricResult
- BenchmarkResultDB
- SafetyResult
- EvaluationStorage operations
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

# Try to import SQLAlchemy components
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from namel3ss.ml.rlhf.evaluation.storage import (
        Base,
        EvaluationRun,
        MetricResult as MetricResultDB,
        BenchmarkResultDB,
        SafetyResult,
        EvaluationStorage,
        create_tables,
    )
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="SQLAlchemy not installed")


@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database for testing."""
    if not SQLALCHEMY_AVAILABLE:
        pytest.skip("SQLAlchemy not available")
    
    engine = create_engine("sqlite:///:memory:")
    create_tables(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()


@pytest.fixture
def storage(in_memory_db):
    """Create EvaluationStorage instance with in-memory DB."""
    return EvaluationStorage(in_memory_db)


class TestEvaluationRunModel:
    """Test EvaluationRun database model."""

    def test_create_run(self, in_memory_db):
        """Test creating evaluation run."""
        run = EvaluationRun(
            model_name="test_model",
            model_version="v1.0",
            run_name="test_run",
            config={"learning_rate": 0.001},
            notes="Test run notes"
        )
        
        in_memory_db.add(run)
        in_memory_db.commit()
        
        assert run.id is not None
        assert run.model_name == "test_model"
        assert run.model_version == "v1.0"
        assert run.run_name == "test_run"
        assert isinstance(run.timestamp, datetime)

    def test_run_relationships(self, in_memory_db):
        """Test relationships between run and results."""
        run = EvaluationRun(
            model_name="test_model",
            model_version="v1.0"
        )
        in_memory_db.add(run)
        in_memory_db.commit()
        
        # Add metric result
        metric = MetricResultDB(
            run_id=run.id,
            metric_name="accuracy",
            value=0.95
        )
        in_memory_db.add(metric)
        in_memory_db.commit()
        
        # Check relationship
        assert len(run.metric_results) == 1
        assert run.metric_results[0].metric_name == "accuracy"

    def test_cascade_delete(self, in_memory_db):
        """Test cascade delete of related results."""
        run = EvaluationRun(model_name="test_model")
        in_memory_db.add(run)
        in_memory_db.commit()
        
        # Add results
        metric = MetricResultDB(run_id=run.id, metric_name="test", value=0.5)
        in_memory_db.add(metric)
        in_memory_db.commit()
        
        run_id = run.id
        
        # Delete run
        in_memory_db.delete(run)
        in_memory_db.commit()
        
        # Results should be deleted too
        remaining = in_memory_db.query(MetricResultDB).filter_by(run_id=run_id).all()
        assert len(remaining) == 0


class TestMetricResultModel:
    """Test MetricResult database model."""

    def test_create_metric_result(self, in_memory_db):
        """Test creating metric result."""
        run = EvaluationRun(model_name="test")
        in_memory_db.add(run)
        in_memory_db.commit()
        
        metric = MetricResultDB(
            run_id=run.id,
            metric_name="rouge1",
            value=0.75,
            metadata={"samples": 100}
        )
        in_memory_db.add(metric)
        in_memory_db.commit()
        
        assert metric.id is not None
        assert metric.metric_name == "rouge1"
        assert metric.value == 0.75
        assert metric.metadata["samples"] == 100

    def test_query_by_metric_name(self, in_memory_db):
        """Test querying metrics by name."""
        run = EvaluationRun(model_name="test")
        in_memory_db.add(run)
        in_memory_db.commit()
        
        # Add multiple metrics
        for i, name in enumerate(["accuracy", "f1", "accuracy"]):
            metric = MetricResultDB(run_id=run.id, metric_name=name, value=0.5 + i * 0.1)
            in_memory_db.add(metric)
        in_memory_db.commit()
        
        # Query accuracy metrics
        accuracy_metrics = in_memory_db.query(MetricResultDB).filter_by(
            metric_name="accuracy"
        ).all()
        
        assert len(accuracy_metrics) == 2


class TestBenchmarkResultModel:
    """Test BenchmarkResultDB database model."""

    def test_create_benchmark_result(self, in_memory_db):
        """Test creating benchmark result."""
        run = EvaluationRun(model_name="test")
        in_memory_db.add(run)
        in_memory_db.commit()
        
        benchmark = BenchmarkResultDB(
            run_id=run.id,
            benchmark_name="MT-Bench",
            benchmark_type="mt_bench",
            score=7.5,
            metadata={"categories": 8}
        )
        in_memory_db.add(benchmark)
        in_memory_db.commit()
        
        assert benchmark.id is not None
        assert benchmark.benchmark_name == "MT-Bench"
        assert benchmark.benchmark_type == "mt_bench"
        assert benchmark.score == 7.5

    def test_query_by_benchmark_type(self, in_memory_db):
        """Test querying by benchmark type."""
        run = EvaluationRun(model_name="test")
        in_memory_db.add(run)
        in_memory_db.commit()
        
        # Add different benchmark types
        benchmarks = [
            ("MT-Bench", "mt_bench", 7.5),
            ("AlpacaEval", "alpaca_eval", 0.85),
            ("TruthfulQA", "truthful_qa", 0.72),
        ]
        
        for name, bench_type, score in benchmarks:
            b = BenchmarkResultDB(
                run_id=run.id,
                benchmark_name=name,
                benchmark_type=bench_type,
                score=score
            )
            in_memory_db.add(b)
        in_memory_db.commit()
        
        # Query MT-Bench results
        mt_results = in_memory_db.query(BenchmarkResultDB).filter_by(
            benchmark_type="mt_bench"
        ).all()
        
        assert len(mt_results) == 1
        assert mt_results[0].score == 7.5


class TestSafetyResultModel:
    """Test SafetyResult database model."""

    def test_create_safety_result(self, in_memory_db):
        """Test creating safety result."""
        run = EvaluationRun(model_name="test")
        in_memory_db.add(run)
        in_memory_db.commit()
        
        safety = SafetyResult(
            run_id=run.id,
            filter_name="toxicity",
            passed=True,
            score=0.15,
            severity="low",
            violations_count=0,
            metadata={"threshold": 0.7}
        )
        in_memory_db.add(safety)
        in_memory_db.commit()
        
        assert safety.id is not None
        assert safety.filter_name == "toxicity"
        assert safety.passed is True
        assert safety.score == 0.15
        assert safety.severity == "low"

    def test_query_failed_checks(self, in_memory_db):
        """Test querying failed safety checks."""
        run = EvaluationRun(model_name="test")
        in_memory_db.add(run)
        in_memory_db.commit()
        
        # Add mix of passed/failed
        results = [
            ("toxicity", True, 0.2),
            ("pii", False, 1.0),
            ("profanity", False, 0.8),
        ]
        
        for name, passed, score in results:
            s = SafetyResult(
                run_id=run.id,
                filter_name=name,
                passed=passed,
                score=score
            )
            in_memory_db.add(s)
        in_memory_db.commit()
        
        # Query failed checks
        failed = in_memory_db.query(SafetyResult).filter_by(passed=False).all()
        
        assert len(failed) == 2
        assert all(not r.passed for r in failed)


class TestEvaluationStorageAPI:
    """Test EvaluationStorage high-level API."""

    def test_create_run(self, storage):
        """Test creating evaluation run via API."""
        run_id = storage.create_run(
            model_name="llama-3-8b",
            model_version="v2.0",
            run_name="dpo_alignment",
            config={"beta": 0.1},
            notes="DPO training run"
        )
        
        assert run_id is not None
        
        # Verify in database
        run = storage.session.query(EvaluationRun).filter_by(id=run_id).first()
        assert run.model_name == "llama-3-8b"
        assert run.model_version == "v2.0"
        assert run.run_name == "dpo_alignment"

    def test_save_metric_result(self, storage):
        """Test saving metric result."""
        run_id = storage.create_run(model_name="test_model")
        
        storage.save_metric_result(
            run_id=run_id,
            metric_name="rouge1",
            value=0.75,
            metadata={"num_samples": 100}
        )
        
        # Verify saved
        metrics = storage.session.query(MetricResultDB).filter_by(run_id=run_id).all()
        assert len(metrics) == 1
        assert metrics[0].metric_name == "rouge1"
        assert metrics[0].value == 0.75

    def test_save_benchmark_result(self, storage):
        """Test saving benchmark result."""
        run_id = storage.create_run(model_name="test_model")
        
        storage.save_benchmark_result(
            run_id=run_id,
            benchmark_name="MT-Bench",
            benchmark_type="mt_bench",
            score=8.2,
            metadata={"judge": "gpt-4"}
        )
        
        # Verify saved
        benchmarks = storage.session.query(BenchmarkResultDB).filter_by(run_id=run_id).all()
        assert len(benchmarks) == 1
        assert benchmarks[0].benchmark_name == "MT-Bench"
        assert benchmarks[0].score == 8.2

    def test_save_safety_result(self, storage):
        """Test saving safety result."""
        run_id = storage.create_run(model_name="test_model")
        
        storage.save_safety_result(
            run_id=run_id,
            filter_name="toxicity",
            passed=True,
            score=0.15,
            severity="low",
            violations_count=0,
            metadata={"threshold": 0.7}
        )
        
        # Verify saved
        safety = storage.session.query(SafetyResult).filter_by(run_id=run_id).all()
        assert len(safety) == 1
        assert safety[0].filter_name == "toxicity"
        assert safety[0].passed is True

    def test_get_run_results(self, storage):
        """Test retrieving complete run results."""
        run_id = storage.create_run(model_name="test_model")
        
        # Add various results
        storage.save_metric_result(run_id, "accuracy", 0.95)
        storage.save_metric_result(run_id, "f1", 0.88)
        storage.save_benchmark_result(run_id, "MT-Bench", "mt_bench", 7.5)
        storage.save_safety_result(run_id, "toxicity", True, 0.2)
        
        results = storage.get_run_results(run_id)
        
        assert results is not None
        assert results["model_name"] == "test_model"
        assert len(results["metrics"]) == 2
        assert len(results["benchmarks"]) == 1
        assert len(results["safety"]) == 1

    def test_get_model_history(self, storage):
        """Test retrieving model evaluation history."""
        # Create multiple runs for same model
        for i in range(3):
            run_id = storage.create_run(
                model_name="llama-3",
                model_version=f"v1.{i}"
            )
            storage.save_metric_result(run_id, "accuracy", 0.8 + i * 0.05)
        
        history = storage.get_model_history("llama-3", limit=10)
        
        assert len(history) == 3
        assert all(r["model_name"] == "llama-3" for r in history)
        # Should be ordered by timestamp (newest first)
        assert history[0]["model_version"] == "v1.2"

    def test_compare_runs(self, storage):
        """Test comparing multiple runs."""
        # Create two runs
        run_id_1 = storage.create_run(model_name="model_a")
        storage.save_metric_result(run_id_1, "accuracy", 0.85)
        storage.save_benchmark_result(run_id_1, "MT-Bench", "mt_bench", 7.0)
        
        run_id_2 = storage.create_run(model_name="model_b")
        storage.save_metric_result(run_id_2, "accuracy", 0.90)
        storage.save_benchmark_result(run_id_2, "MT-Bench", "mt_bench", 8.0)
        
        comparison = storage.compare_runs([run_id_1, run_id_2])
        
        assert len(comparison) == 2
        assert comparison[0]["run_id"] == run_id_1
        assert comparison[1]["run_id"] == run_id_2
        
        # Both should have metrics and benchmarks
        for run in comparison:
            assert "metrics" in run
            assert "benchmarks" in run

    def test_get_nonexistent_run(self, storage):
        """Test getting results for nonexistent run."""
        results = storage.get_run_results(999999)
        
        assert results is None

    def test_multiple_metrics_same_run(self, storage):
        """Test saving multiple metrics for same run."""
        run_id = storage.create_run(model_name="test")
        
        metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88,
            "f1": 0.90,
        }
        
        for name, value in metrics.items():
            storage.save_metric_result(run_id, name, value)
        
        results = storage.get_run_results(run_id)
        assert len(results["metrics"]) == 4

    def test_model_history_ordering(self, storage):
        """Test that model history is ordered by timestamp."""
        model_name = "test_model"
        
        # Create runs (newest last)
        import time
        run_ids = []
        for i in range(3):
            run_id = storage.create_run(model_name=model_name, model_version=f"v{i}")
            run_ids.append(run_id)
            time.sleep(0.01)  # Ensure different timestamps
        
        history = storage.get_model_history(model_name)
        
        # Should be in reverse chronological order
        assert history[0]["model_version"] == "v2"
        assert history[-1]["model_version"] == "v0"

    def test_empty_history(self, storage):
        """Test getting history for model with no runs."""
        history = storage.get_model_history("nonexistent_model")
        
        assert isinstance(history, list)
        assert len(history) == 0


class TestDatabaseIndexes:
    """Test that database indexes work correctly."""

    def test_model_timestamp_index(self, in_memory_db):
        """Test querying with model_timestamp index."""
        # Create multiple runs
        for i in range(5):
            run = EvaluationRun(model_name=f"model_{i % 2}")
            in_memory_db.add(run)
        in_memory_db.commit()
        
        # Query should be efficient with index
        runs = in_memory_db.query(EvaluationRun).filter_by(
            model_name="model_0"
        ).order_by(EvaluationRun.timestamp.desc()).all()
        
        assert len(runs) >= 2

    def test_metric_value_index(self, in_memory_db):
        """Test querying with metric_value index."""
        run = EvaluationRun(model_name="test")
        in_memory_db.add(run)
        in_memory_db.commit()
        
        # Add metrics with different values
        for i in range(10):
            metric = MetricResultDB(
                run_id=run.id,
                metric_name="accuracy",
                value=0.5 + i * 0.05
            )
            in_memory_db.add(metric)
        in_memory_db.commit()
        
        # Query high accuracy metrics
        high_accuracy = in_memory_db.query(MetricResultDB).filter(
            MetricResultDB.metric_name == "accuracy",
            MetricResultDB.value > 0.8
        ).all()
        
        assert len(high_accuracy) >= 3


class TestStorageEdgeCases:
    """Test edge cases in storage."""

    def test_null_config(self, storage):
        """Test run with null config."""
        run_id = storage.create_run(model_name="test", config=None)
        
        run = storage.session.query(EvaluationRun).filter_by(id=run_id).first()
        assert run.config is None

    def test_empty_metadata(self, storage):
        """Test results with empty metadata."""
        run_id = storage.create_run(model_name="test")
        storage.save_metric_result(run_id, "test", 0.5, metadata={})
        
        metrics = storage.session.query(MetricResultDB).filter_by(run_id=run_id).all()
        assert metrics[0].metadata == {}

    def test_very_long_model_name(self, storage):
        """Test with very long model name."""
        long_name = "a" * 200
        run_id = storage.create_run(model_name=long_name)
        
        run = storage.session.query(EvaluationRun).filter_by(id=run_id).first()
        assert len(run.model_name) >= 100  # Should store at least 100 chars

    def test_negative_scores(self, storage):
        """Test storing negative scores (valid for some metrics)."""
        run_id = storage.create_run(model_name="test")
        storage.save_metric_result(run_id, "log_likelihood", -2.5)
        
        metrics = storage.session.query(MetricResultDB).filter_by(run_id=run_id).all()
        assert metrics[0].value == -2.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
