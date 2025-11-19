"""Integration tests for training and tuning system."""

from __future__ import annotations

import textwrap
from typing import Any, Dict

import pytest

from namel3ss.lang.grammar import parse_module
from namel3ss.ast import TrainingJob, TuningJob
from namel3ss.resolver import resolve_program
from namel3ss.codegen.backend import build_backend_state


def test_parse_enhanced_training_job():
    """Test parsing training job with all new fields."""
    source = textwrap.dedent("""
        dataset "customer_churn" from csv:
            path: "data/churn.csv"
            schema:
                - tenure: int
                - monthly_spend: float
                - support_calls: int
                - churned: bool

        training "churn_predictor":
            model: churn_model
            dataset: customer_churn
            target: churned
            features:
                - tenure
                - monthly_spend
                - support_calls
            framework: sklearn
            objective: maximize_accuracy
            hyperparameters:
                n_estimators: 100
                max_depth: 10
            split:
                train: 0.7
                validation: 0.15
                test: 0.15
            compute:
                backend: local
            metrics:
                - accuracy
                - f1
    """).strip()

    module = parse_module(source)
    assert module is not None
    
    jobs = [node for node in module.body if isinstance(node, TrainingJob)]
    assert len(jobs) == 1
    
    job = jobs[0]
    assert job.name == "churn_predictor"
    assert job.model == "churn_model"
    assert job.dataset == "customer_churn"
    assert job.target == "churned"
    assert job.features == ["tenure", "monthly_spend", "support_calls"]
    assert job.framework == "sklearn"
    assert job.objective == "maximize_accuracy"
    assert job.hyperparameters["n_estimators"] == 100
    assert job.split["train"] == 0.7
    assert job.split["validation"] == 0.15
    assert job.split["test"] == 0.15
    assert job.metrics == ["accuracy", "f1"]


def test_training_job_codegen():
    """Test that training job is properly encoded in backend state."""
    source = textwrap.dedent("""
        app:
            name: test_app
        
        dataset "test_data":
            source: csv
            path: "test.csv"
            schema:
                - feature_a: float
                - feature_b: float
                - label: int
        
        model "test_model":
            type: classifier
        
        training "baseline_training":
            model: test_model
            dataset: test_data
            target: label
            features:
                - feature_a
                - feature_b
            framework: sklearn
            objective: accuracy
            hyperparameters:
                max_depth: 5
            split:
                train: 0.8
                test: 0.2
    """).strip()
    
    module = parse_module(source)
    program = resolve_program(module)
    app = program.resolved_modules["<main>"].exports.app
    
    state = build_backend_state(app)
    
    assert "baseline_training" in state.training_jobs
    job_spec = state.training_jobs["baseline_training"]
    
    assert job_spec["name"] == "baseline_training"
    assert job_spec["model"] == "test_model"
    assert job_spec["dataset"] == "test_data"
    assert job_spec["target"] == "label"
    assert job_spec["features"] == ["feature_a", "feature_b"]
    assert job_spec["framework"] == "sklearn"
    assert job_spec["objective"] == "accuracy"
    assert job_spec["hyperparameters"]["max_depth"] == 5
    assert job_spec["split"]["train"] == 0.8
    assert job_spec["split"]["test"] == 0.2


def test_tuning_job_with_search_space():
    """Test tuning job with hyperparameter search space."""
    source = textwrap.dedent("""
        dataset "test_data" from csv:
            path: "test.csv"
            schema:
                - x: float
                - y: int
        
        model "tuned_model":
            type: classifier
        
        training "base_training":
            model: tuned_model
            dataset: test_data
            target: y
            features:
                - x
            framework: sklearn
            objective: accuracy
            hyperparameters:
                n_estimators: 100
        
        tuning "hyperparameter_search":
            training_job: base_training
            strategy: random
            max_trials: 10
            objective_metric: accuracy
            search_space:
                n_estimators:
                    type: int
                    min: 50
                    max: 200
                    step: 50
                max_depth:
                    type: int
                    min: 5
                    max: 20
    """).strip()
    
    module = parse_module(source)
    
    tuning_jobs = [node for node in module.body if isinstance(node, TuningJob)]
    assert len(tuning_jobs) == 1
    
    job = tuning_jobs[0]
    assert job.name == "hyperparameter_search"
    assert job.training_job == "base_training"
    assert job.strategy == "random"
    assert job.max_trials == 10
    assert job.objective_metric == "accuracy"
    assert "n_estimators" in job.search_space
    assert job.search_space["n_estimators"].type == "int"
    assert job.search_space["n_estimators"].min == 50
    assert job.search_space["n_estimators"].max == 200


def test_tuning_job_codegen():
    """Test that tuning job is properly encoded in backend state."""
    source = textwrap.dedent("""
        app:
            name: test_app
        
        dataset "data" from csv:
            path: "data.csv"
            schema:
                - x: float
                - y: int
        
        model "model":
            type: classifier
        
        training "train":
            model: model
            dataset: data
            target: y
            features:
                - x
            objective: accuracy
        
        tuning "tune":
            training_job: train
            strategy: grid
            max_trials: 5
            search_space:
                param_a:
                    type: float
                    min: 0.1
                    max: 1.0
    """).strip()
    
    module = parse_module(source)
    program = resolve_program(module)
    app = program.resolved_modules["<main>"].exports.app
    
    state = build_backend_state(app)
    
    assert "tune" in state.tuning_jobs
    job_spec = state.tuning_jobs["tune"]
    
    assert job_spec["name"] == "tune"
    assert job_spec["training_job"] == "train"
    assert job_spec["strategy"] == "grid"
    assert job_spec["max_trials"] == 5
    assert "param_a" in job_spec["search_space"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
