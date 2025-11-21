"""
Storage layer for evaluation results and benchmark scores.

Provides database models and utilities for storing:
- Evaluation metric results
- Benchmark scores
- Safety filter results
- Comparison data over time
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import json


try:
    from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text, ForeignKey, Index
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import relationship
    HAS_SQLALCHEMY = True
    Base = declarative_base()
except ImportError:
    HAS_SQLALCHEMY = False
    Base = object  # Fallback for when SQLAlchemy not installed


if HAS_SQLALCHEMY:
    class EvaluationRun(Base):
        """
        Evaluation run metadata.
        
        Tracks each evaluation session with model, timestamp,
        and configuration details.
        """
        __tablename__ = "evaluation_runs"
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        model_name = Column(String(255), nullable=False, index=True)
        model_version = Column(String(100))
        run_name = Column(String(255))
        timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
        config = Column(JSON)
        notes = Column(Text)
        
        # Relationships
        metric_results = relationship("MetricResult", back_populates="run", cascade="all, delete-orphan")
        benchmark_results = relationship("BenchmarkResultDB", back_populates="run", cascade="all, delete-orphan")
        safety_results = relationship("SafetyResult", back_populates="run", cascade="all, delete-orphan")
        
        # Indexes
        __table_args__ = (
            Index('idx_model_timestamp', 'model_name', 'timestamp'),
        )


    class MetricResult(Base):
        """
        Individual metric evaluation result.
        
        Stores scores from metrics like ROUGE, BLEU, diversity, etc.
        """
        __tablename__ = "metric_results"
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        run_id = Column(Integer, ForeignKey("evaluation_runs.id"), nullable=False, index=True)
        metric_name = Column(String(100), nullable=False, index=True)
        value = Column(Float, nullable=False)
        timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
        metadata = Column(JSON)
        
        # Relationship
        run = relationship("EvaluationRun", back_populates="metric_results")
        
        # Indexes
        __table_args__ = (
            Index('idx_metric_value', 'metric_name', 'value'),
        )


    class BenchmarkResultDB(Base):
        """
        Benchmark evaluation result.
        
        Stores results from benchmarks like MT-Bench, AlpacaEval, TruthfulQA.
        """
        __tablename__ = "benchmark_results"
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        run_id = Column(Integer, ForeignKey("evaluation_runs.id"), nullable=False, index=True)
        benchmark_name = Column(String(100), nullable=False, index=True)
        benchmark_type = Column(String(50), nullable=False)
        score = Column(Float, nullable=False)
        timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
        metadata = Column(JSON)
        
        # Relationship
        run = relationship("EvaluationRun", back_populates="benchmark_results")
        
        # Indexes
        __table_args__ = (
            Index('idx_benchmark_score', 'benchmark_name', 'score'),
        )


    class SafetyResult(Base):
        """
        Safety filter result.
        
        Stores results from toxicity, PII, profanity, and bias filters.
        """
        __tablename__ = "safety_results"
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        run_id = Column(Integer, ForeignKey("evaluation_runs.id"), nullable=False, index=True)
        filter_name = Column(String(100), nullable=False, index=True)
        passed = Column(Integer, nullable=False)  # 1 for pass, 0 for fail
        score = Column(Float, nullable=False)
        severity = Column(String(20))
        violations_count = Column(Integer, default=0)
        timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
        metadata = Column(JSON)
        
        # Relationship
        run = relationship("EvaluationRun", back_populates="safety_results")
        
        # Indexes
        __table_args__ = (
            Index('idx_filter_passed', 'filter_name', 'passed'),
            Index('idx_filter_score', 'filter_name', 'score'),
        )


@dataclass
class EvaluationStorage:
    """
    Storage manager for evaluation results.
    
    Provides high-level API for saving and retrieving
    evaluation data from database.
    """
    session: Any  # SQLAlchemy session
    
    def create_run(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Create a new evaluation run.
        
        Args:
            model_name: Name of the model being evaluated
            model_version: Model version/checkpoint
            run_name: Optional name for this run
            config: Configuration dict
            notes: Optional notes
        
        Returns:
            Run ID
        """
        if not HAS_SQLALCHEMY:
            raise RuntimeError("SQLAlchemy not installed")
        
        run = EvaluationRun(
            model_name=model_name,
            model_version=model_version,
            run_name=run_name,
            config=config,
            notes=notes
        )
        self.session.add(run)
        self.session.commit()
        return run.id
    
    def save_metric_result(
        self,
        run_id: int,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save a metric result.
        
        Args:
            run_id: Evaluation run ID
            metric_name: Name of the metric
            value: Metric value
            metadata: Additional metadata
        """
        if not HAS_SQLALCHEMY:
            raise RuntimeError("SQLAlchemy not installed")
        
        result = MetricResult(
            run_id=run_id,
            metric_name=metric_name,
            value=value,
            metadata=metadata
        )
        self.session.add(result)
        self.session.commit()
    
    def save_benchmark_result(
        self,
        run_id: int,
        benchmark_name: str,
        benchmark_type: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save a benchmark result.
        
        Args:
            run_id: Evaluation run ID
            benchmark_name: Name of the benchmark
            benchmark_type: Type of benchmark
            score: Benchmark score
            metadata: Additional metadata
        """
        if not HAS_SQLALCHEMY:
            raise RuntimeError("SQLAlchemy not installed")
        
        result = BenchmarkResultDB(
            run_id=run_id,
            benchmark_name=benchmark_name,
            benchmark_type=benchmark_type,
            score=score,
            metadata=metadata
        )
        self.session.add(result)
        self.session.commit()
    
    def save_safety_result(
        self,
        run_id: int,
        filter_name: str,
        passed: bool,
        score: float,
        severity: Optional[str] = None,
        violations_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save a safety filter result.
        
        Args:
            run_id: Evaluation run ID
            filter_name: Name of the safety filter
            passed: Whether content passed the filter
            score: Safety score
            severity: Severity level if failed
            violations_count: Number of violations detected
            metadata: Additional metadata
        """
        if not HAS_SQLALCHEMY:
            raise RuntimeError("SQLAlchemy not installed")
        
        result = SafetyResult(
            run_id=run_id,
            filter_name=filter_name,
            passed=1 if passed else 0,
            score=score,
            severity=severity,
            violations_count=violations_count,
            metadata=metadata
        )
        self.session.add(result)
        self.session.commit()
    
    def get_run_results(self, run_id: int) -> Dict[str, Any]:
        """
        Get all results for an evaluation run.
        
        Args:
            run_id: Evaluation run ID
        
        Returns:
            Dictionary with metrics, benchmarks, and safety results
        """
        if not HAS_SQLALCHEMY:
            raise RuntimeError("SQLAlchemy not installed")
        
        run = self.session.query(EvaluationRun).filter_by(id=run_id).first()
        if not run:
            return {}
        
        return {
            "run": {
                "id": run.id,
                "model_name": run.model_name,
                "model_version": run.model_version,
                "run_name": run.run_name,
                "timestamp": run.timestamp.isoformat(),
                "config": run.config,
                "notes": run.notes
            },
            "metrics": [
                {
                    "name": m.metric_name,
                    "value": m.value,
                    "metadata": m.metadata
                }
                for m in run.metric_results
            ],
            "benchmarks": [
                {
                    "name": b.benchmark_name,
                    "type": b.benchmark_type,
                    "score": b.score,
                    "metadata": b.metadata
                }
                for b in run.benchmark_results
            ],
            "safety": [
                {
                    "filter": s.filter_name,
                    "passed": bool(s.passed),
                    "score": s.score,
                    "severity": s.severity,
                    "violations_count": s.violations_count,
                    "metadata": s.metadata
                }
                for s in run.safety_results
            ]
        }
    
    def get_model_history(
        self,
        model_name: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get evaluation history for a model.
        
        Args:
            model_name: Name of the model
            limit: Max number of runs to return
        
        Returns:
            List of evaluation run summaries
        """
        if not HAS_SQLALCHEMY:
            raise RuntimeError("SQLAlchemy not installed")
        
        query = self.session.query(EvaluationRun).filter_by(model_name=model_name).order_by(EvaluationRun.timestamp.desc())
        
        if limit:
            query = query.limit(limit)
        
        runs = query.all()
        
        return [
            {
                "id": run.id,
                "model_name": run.model_name,
                "model_version": run.model_version,
                "run_name": run.run_name,
                "timestamp": run.timestamp.isoformat(),
                "metrics_count": len(run.metric_results),
                "benchmarks_count": len(run.benchmark_results),
                "safety_checks_count": len(run.safety_results)
            }
            for run in runs
        ]
    
    def compare_runs(
        self,
        run_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Compare multiple evaluation runs.
        
        Args:
            run_ids: List of run IDs to compare
        
        Returns:
            Comparison data with metrics side-by-side
        """
        if not HAS_SQLALCHEMY:
            raise RuntimeError("SQLAlchemy not installed")
        
        runs = self.session.query(EvaluationRun).filter(EvaluationRun.id.in_(run_ids)).all()
        
        comparison = {
            "runs": [],
            "metrics": {},
            "benchmarks": {},
            "safety": {}
        }
        
        for run in runs:
            comparison["runs"].append({
                "id": run.id,
                "model_name": run.model_name,
                "model_version": run.model_version,
                "timestamp": run.timestamp.isoformat()
            })
            
            # Collect metrics
            for metric in run.metric_results:
                if metric.metric_name not in comparison["metrics"]:
                    comparison["metrics"][metric.metric_name] = {}
                comparison["metrics"][metric.metric_name][run.id] = metric.value
            
            # Collect benchmarks
            for benchmark in run.benchmark_results:
                if benchmark.benchmark_name not in comparison["benchmarks"]:
                    comparison["benchmarks"][benchmark.benchmark_name] = {}
                comparison["benchmarks"][benchmark.benchmark_name][run.id] = benchmark.score
            
            # Collect safety
            for safety in run.safety_results:
                if safety.filter_name not in comparison["safety"]:
                    comparison["safety"][safety.filter_name] = {}
                comparison["safety"][safety.filter_name][run.id] = {
                    "passed": bool(safety.passed),
                    "score": safety.score
                }
        
        return comparison


def create_tables(engine):
    """
    Create all evaluation storage tables.
    
    Args:
        engine: SQLAlchemy engine
    """
    if not HAS_SQLALCHEMY:
        raise RuntimeError("SQLAlchemy not installed")
    
    Base.metadata.create_all(engine)


__all__ = [
    "EvaluationStorage",
    "create_tables",
]

if HAS_SQLALCHEMY:
    __all__.extend([
        "EvaluationRun",
        "MetricResult",
        "BenchmarkResultDB",
        "SafetyResult",
    ])
