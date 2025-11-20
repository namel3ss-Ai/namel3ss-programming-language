"""
Training and tuning job definitions for offline ML model development.

This module contains AST nodes for defining:
- Training jobs: Full model training specifications
- Tuning jobs: Hyperparameter optimization
- Compute specifications: Resource allocation
- Hyperparameter specifications: Search spaces
- Early stopping: Training convergence criteria
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .base import Expression


# Type alias for hyperparameter values (can be expressions or literals)
HyperparameterValue = Union[Expression, Any]


@dataclass
class TrainingComputeSpec:
    """
    Compute resource specification for training jobs.
    
    Defines where and how training should be executed:
    - Local development
    - Cloud platforms (AWS, GCP, Azure)
    - Distributed training clusters
    
    Example:
        compute: {
            backend: "aws_sagemaker",
            resources: {
                instance_type: "ml.p3.2xlarge",
                instance_count: 1
            },
            queue: "high_priority"
        }
    """
    backend: str = "local"  # local, aws_sagemaker, gcp_vertex, azure_ml
    resources: Dict[str, Any] = field(default_factory=dict)
    queue: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EarlyStoppingSpec:
    """
    Early stopping criteria for training convergence.
    
    Automatically stops training when a metric stops improving,
    preventing overfitting and saving compute resources.
    
    Example:
        early_stopping: {
            metric: "val_loss",
            patience: 3,
            min_delta: 0.001,
            mode: "min"
        }
    """
    metric: str
    patience: int
    min_delta: float = 0.0
    mode: str = "min"  # min, max
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingJob:
    """
    Complete specification for a model training job.
    
    Defines all aspects of training a machine learning model:
    - Data: dataset, features, target
    - Model: architecture, framework
    - Hyperparameters: learning rate, batch size, etc.
    - Compute: where and how to train
    - Evaluation: metrics, validation strategy
    
    Example DSL:
        training customer_churn_model {
            model: "churn_predictor"
            dataset: "customer_data"
            objective: "classification"
            target: "churned"
            features: ["tenure", "monthly_charges", "contract_type"]
            
            framework: "xgboost"
            hyperparameters: {
                learning_rate: 0.1,
                max_depth: 6,
                n_estimators: 100
            }
            
            compute: {
                backend: "aws_sagemaker",
                resources: {
                    instance_type: "ml.m5.xlarge"
                }
            }
            
            split: {
                train: 0.7,
                validation: 0.15,
                test: 0.15
            }
            
            early_stopping: {
                metric: "auc",
                patience: 5,
                mode: "max"
            }
            
            metrics: ["accuracy", "precision", "recall", "auc"]
        }
    """
    name: str
    model: str
    dataset: str
    objective: str  # classification, regression, clustering, etc.
    target: Optional[str] = None
    features: List[str] = field(default_factory=list)
    framework: Optional[str] = None  # xgboost, sklearn, tensorflow, pytorch
    hyperparameters: Dict[str, HyperparameterValue] = field(default_factory=dict)
    compute: TrainingComputeSpec = field(default_factory=TrainingComputeSpec)
    split: Dict[str, float] = field(default_factory=dict)
    validation_split: Optional[float] = None
    early_stopping: Optional[EarlyStoppingSpec] = None
    output_registry: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HyperparamSpec:
    """
    Hyperparameter search space specification for tuning.
    
    Defines the range or set of values to explore during
    hyperparameter optimization.
    
    Supported types:
    - Continuous: min/max range with optional log scaling
    - Discrete: specific set of values
    - Integer: min/max with optional step size
    
    Example:
        learning_rate: {
            type: "float",
            min: 0.001,
            max: 0.1,
            log: true
        }
        
        n_estimators: {
            type: "int",
            min: 50,
            max: 200,
            step: 10
        }
        
        optimizer: {
            type: "categorical",
            values: ["adam", "sgd", "rmsprop"]
        }
    """
    type: str  # float, int, categorical
    min: Optional[float] = None
    max: Optional[float] = None
    values: Optional[List[Any]] = None  # For categorical
    log: bool = False  # Use log scale for continuous parameters
    step: Optional[float] = None  # For discrete integer ranges
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TuningJob:
    """
    Hyperparameter optimization job specification.
    
    Automates the search for optimal hyperparameters by:
    - Defining search spaces for each parameter
    - Selecting a search strategy (grid, random, bayesian)
    - Running multiple trials in parallel
    - Optimizing for a target metric
    
    Example DSL:
        tuning optimize_churn_model {
            training_job: "customer_churn_model"
            
            search_space: {
                learning_rate: {
                    type: "float",
                    min: 0.001,
                    max: 0.3,
                    log: true
                },
                max_depth: {
                    type: "int",
                    min: 3,
                    max: 10
                },
                n_estimators: {
                    type: "categorical",
                    values: [50, 100, 200, 500]
                }
            }
            
            strategy: "bayesian"
            max_trials: 50
            parallel_trials: 5
            
            objective_metric: "auc"
            early_stopping: {
                metric: "auc",
                patience: 10,
                mode: "max"
            }
        }
    """
    name: str
    training_job: str  # Reference to the training job to tune
    search_space: Dict[str, HyperparamSpec] = field(default_factory=dict)
    strategy: str = "grid"  # grid, random, bayesian, hyperband
    max_trials: int = 1
    parallel_trials: int = 1
    early_stopping: Optional[EarlyStoppingSpec] = None
    objective_metric: str = "loss"
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "TrainingComputeSpec",
    "EarlyStoppingSpec",
    "TrainingJob",
    "HyperparamSpec",
    "TuningJob",
    "HyperparameterValue",
]
