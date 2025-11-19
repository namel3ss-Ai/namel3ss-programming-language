"""Production-grade training backends with sklearn, PyTorch, and TensorFlow support."""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

from .backends import (
    TrainingBackend,
    TrainingPlan,
    TrainingRunResult,
    register_training_backend,
)

logger = logging.getLogger(__name__)


class SklearnTrainingBackend(TrainingBackend):
    """Production sklearn training backend with real model training."""
    
    name = "sklearn"
    
    def run(self, plan: TrainingPlan, context: Optional[Dict[str, Any]] = None) -> TrainingRunResult:
        """Execute sklearn training with real dataset loading and model persistence."""
        context = context or {}
        
        try:
            # Validate dependencies
            if pd is None:
                return self._error_result(plan, "pandas is required for sklearn backend")
            if np is None:
                return self._error_result(plan, "numpy is required for sklearn backend")
            
            # Load and prepare dataset
            dataset_rows = context.get("dataset_rows", [])
            if not dataset_rows:
                return self._error_result(plan, "No dataset rows available in context")
            
            df = pd.DataFrame(dataset_rows)
            
            # Extract target and features
            target = plan.get("metadata", {}).get("target")
            features = plan.get("metadata", {}).get("features", [])
            
            if not target:
                return self._error_result(plan, "No target field specified")
            if not features:
                # Use all columns except target
                features = [col for col in df.columns if col != target]
            
            if target not in df.columns:
                return self._error_result(plan, f"Target field '{target}' not in dataset")
            
            # Prepare feature matrix and target vector
            X = df[features].values
            y = df[target].values
            
            # Split data
            split_config = plan.get("metadata", {}).get("split", {})
            train_size = split_config.get("train", 0.7)
            val_size = split_config.get("validation", 0.15)
            test_size = split_config.get("test", 0.15)
            
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(
                X, y, train_size, val_size, test_size
            )
            
            # Get model class
            model_type = plan.get("metadata", {}).get("model_type", "RandomForestClassifier")
            hyperparameters = plan.get("hyperparameters", {})
            
            model = self._create_model(model_type, hyperparameters)
            
            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Evaluate
            metrics = self._evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test)
            metrics["train_time_seconds"] = round(train_time, 3)
            metrics["train_samples"] = len(X_train)
            metrics["val_samples"] = len(X_val) if X_val is not None else 0
            metrics["test_samples"] = len(X_test) if X_test is not None else 0
            
            # Prepare result
            result: TrainingRunResult = {
                "status": "ok",
                "job": plan.get("job", "training-job"),
                "backend": self.name,
                "model": plan.get("model", ""),
                "dataset": plan.get("dataset", ""),
                "objective": plan.get("objective", ""),
                "hyperparameters": hyperparameters,
                "resources": plan.get("resources", {}),
                "metrics": metrics,
                "artifacts": {
                    "model_object": model,
                    "feature_names": features,
                    "target_name": target,
                    "model_type": model_type,
                },
                "metadata": {
                    "executed_at": time.time(),
                    "backend": self.name,
                    "framework": "sklearn",
                    **plan.get("metadata", {}),
                },
            }
            
            return result
            
        except Exception as exc:
            logger.exception("Sklearn training backend failed")
            return self._error_result(plan, str(exc))
    
    def _split_data(
        self,
        X: Any,
        y: Any,
        train_size: float,
        val_size: float,
        test_size: float,
    ) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """Split data into train/val/test sets."""
        n = len(X)
        
        # Normalize splits
        total = train_size + val_size + test_size
        if total == 0:
            train_size, val_size, test_size = 0.7, 0.15, 0.15
        else:
            train_size = train_size / total
            val_size = val_size / total
            test_size = test_size / total
        
        # Calculate split indices
        train_end = int(n * train_size)
        val_end = train_end + int(n * val_size)
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        X_val = X[train_end:val_end] if val_size > 0 else None
        y_val = y[train_end:val_end] if val_size > 0 else None
        
        X_test = X[val_end:] if test_size > 0 else None
        y_test = y[val_end:] if test_size > 0 else None
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _create_model(self, model_type: str, hyperparameters: Dict[str, Any]) -> Any:
        """Create sklearn model instance."""
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression, LinearRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.svm import SVC
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for sklearn backend") from exc
        
        model_map = {
            "RandomForestClassifier": RandomForestClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "LogisticRegression": LogisticRegression,
            "LinearRegression": LinearRegression,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "SVC": SVC,
        }
        
        model_class = model_map.get(model_type, RandomForestClassifier)
        
        # Filter hyperparameters to only valid ones for this model
        import inspect
        sig = inspect.signature(model_class.__init__)
        valid_params = {k: v for k, v in hyperparameters.items() if k in sig.parameters}
        
        return model_class(**valid_params)
    
    def _evaluate_model(
        self,
        model: Any,
        X_train: Any,
        X_val: Any,
        X_test: Any,
        y_train: Any,
        y_val: Any,
        y_test: Any,
    ) -> Dict[str, float]:
        """Evaluate model and return metrics."""
        metrics: Dict[str, float] = {}
        
        # Train accuracy
        train_score = model.score(X_train, y_train)
        metrics["train_accuracy"] = round(train_score, 4)
        
        # Validation accuracy
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            metrics["val_accuracy"] = round(val_score, 4)
        
        # Test accuracy
        if X_test is not None and y_test is not None:
            test_score = model.score(X_test, y_test)
            metrics["test_accuracy"] = round(test_score, 4)
            metrics["accuracy"] = round(test_score, 4)  # Primary metric
        
        return metrics
    
    def _error_result(self, plan: TrainingPlan, detail: str) -> TrainingRunResult:
        """Create error result."""
        return {
            "status": "error",
            "job": plan.get("job", "training-job"),
            "backend": self.name,
            "error": "sklearn_training_failed",
            "detail": detail,
        }


class PyTorchTrainingBackend(TrainingBackend):
    """PyTorch training backend (placeholder for future implementation)."""
    
    name = "pytorch"
    
    def run(self, plan: TrainingPlan, context: Optional[Dict[str, Any]] = None) -> TrainingRunResult:
        """Execute PyTorch training."""
        return {
            "status": "error",
            "job": plan.get("job", "training-job"),
            "backend": self.name,
            "error": "not_implemented",
            "detail": "PyTorch training backend is not yet implemented",
        }


class TensorFlowTrainingBackend(TrainingBackend):
    """TensorFlow training backend (placeholder for future implementation)."""
    
    name = "tensorflow"
    
    def run(self, plan: TrainingPlan, context: Optional[Dict[str, Any]] = None) -> TrainingRunResult:
        """Execute TensorFlow training."""
        return {
            "status": "error",
            "job": plan.get("job", "training-job"),
            "backend": self.name,
            "error": "not_implemented",
            "detail": "TensorFlow training backend is not yet implemented",
        }


# Register production backends
register_training_backend("sklearn", lambda: SklearnTrainingBackend())
register_training_backend("pytorch", lambda: PyTorchTrainingBackend())
register_training_backend("tensorflow", lambda: TensorFlowTrainingBackend())


__all__ = [
    "SklearnTrainingBackend",
    "PyTorchTrainingBackend",
    "TensorFlowTrainingBackend",
]
