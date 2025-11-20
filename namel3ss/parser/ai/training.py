"""Training and tuning job parsing for ML model workflows."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from namel3ss.ast import (
    EarlyStoppingSpec,
    HyperparamSpec,
    TrainingComputeSpec,
    TrainingJob,
    TuningJob,
)

if TYPE_CHECKING:
    from ..base import ParserBase

_TRAINING_HEADER = re.compile(r'^training\s+"([^"]+)"\s*:?', re.IGNORECASE)
_TUNING_HEADER = re.compile(r'^tuning\s+"([^"]+)"\s*:?', re.IGNORECASE)


class TrainingParserMixin:
    """Mixin for parsing training and tuning job definitions."""
    
    def _parse_training_job(self: 'ParserBase', line: str, line_no: int, base_indent: int) -> TrainingJob:
        """
        Parse training job definition for model training.
        
        Training jobs specify model training configurations including datasets,
        objectives, hyperparameters, compute resources, and validation strategies.
        
        Syntax:
            training "Name":
                model: "ModelName"
                dataset: "TrainingData"
                objective: classification|regression|clustering
                target: "outcome_field"
                features: ["feature1", "feature2"]
                hyperparameters:
                    learning_rate: 0.001
                    batch_size: 32
                compute:
                    backend: local|kubernetes|aws
                    resources:
                        gpu: 1
                split:
                    train: 0.8
                    test: 0.2
                early_stopping:
                    metric: val_loss
                    patience: 5
        """
        match = _TRAINING_HEADER.match(line.strip())
        if not match:
            raise self._error(
                'Expected: training "Name":',
                line_no,
                line,
                hint='Training jobs require a name, e.g., training "TrainClassifier":'
            )
        name = match.group(1)
        model_name: Optional[str] = None
        dataset_name: Optional[str] = None
        objective: Optional[str] = None
        target: Optional[str] = None
        features: List[str] = []
        framework: Optional[str] = None
        hyperparameters: Dict[str, Any] = {}
        metrics: List[str] = []
        metadata: Dict[str, Any] = {}
        compute_spec = TrainingComputeSpec()
        split: Dict[str, float] = {}
        validation_split: Optional[float] = None
        early_stopping: Optional[EarlyStoppingSpec] = None
        output_registry: Optional[str] = None
        description: Optional[str] = None

        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            lowered = stripped.lower()
            if lowered.startswith('hyperparameters:'):
                self._advance()
                block = self._parse_kv_block(indent)
                hyperparameters = self._transform_config(block)
                continue
            if lowered.startswith('compute:'):
                self._advance()
                compute_spec = self._parse_training_compute_block(indent)
                continue
            if lowered.startswith('split:'):
                self._advance()
                block = self._parse_kv_block(indent)
                split = {k: float(v) if isinstance(v, (int, float)) else 0.0 for k, v in block.items()}
                continue
            if lowered.startswith('early_stopping:'):
                self._advance()
                block = self._parse_kv_block(indent)
                early_stopping = self._build_early_stopping_spec(block)
                continue
            if lowered.startswith('features:'):
                self._advance()
                features.extend(self._parse_string_list(indent))
                continue
            if lowered.startswith('metrics:'):
                self._advance()
                metrics.extend(self._parse_string_list(indent))
                continue
            if lowered.startswith('metadata:'):
                self._advance()
                block = self._parse_kv_block(indent)
                metadata.update(self._transform_config(block))
                continue
            assign = re.match(r'([\w\.\- ]+)\s*:\s*(.*)$', stripped)
            if not assign:
                raise self._error("Invalid entry inside training block", self.pos + 1, nxt)
            key = assign.group(1).strip().lower()
            remainder = assign.group(2)
            self._advance()
            if remainder:
                value = self._coerce_scalar(remainder)
            else:
                value = self._parse_kv_block(indent)
            if key == 'model':
                model_name = self._strip_quotes(self._stringify_value(value))
            elif key == 'dataset':
                dataset_name = self._strip_quotes(self._stringify_value(value))
            elif key == 'objective':
                objective = self._strip_quotes(self._stringify_value(value))
            elif key == 'target':
                target = self._strip_quotes(self._stringify_value(value))
            elif key == 'framework':
                framework = self._strip_quotes(self._stringify_value(value))
            elif key == 'validation_split':
                validation_split = float(value) if isinstance(value, (int, float)) else None
            elif key in {'output_registry', 'registry', 'output'}:
                output_registry = self._strip_quotes(self._stringify_value(value))
            elif key == 'description':
                description = self._stringify_value(value)
            else:
                metadata[key] = self._transform_config(value)

        if not model_name:
            raise self._error(
                "Training job must define 'model:'",
                line_no,
                line,
                hint='Add model: "ModelName" to specify which model to train'
            )
        if not dataset_name:
            raise self._error(
                "Training job must define 'dataset:'",
                line_no,
                line,
                hint='Add dataset: "DatasetName" to specify training data'
            )
        if not objective:
            raise self._error(
                "Training job must define 'objective:'",
                line_no,
                line,
                hint='Add objective: classification|regression|clustering'
            )

        return TrainingJob(
            name=name,
            model=model_name,
            dataset=dataset_name,
            objective=objective,
            target=target,
            features=features,
            framework=framework,
            hyperparameters=hyperparameters,
            compute=compute_spec,
            split=split,
            validation_split=validation_split,
            early_stopping=early_stopping,
            output_registry=output_registry,
            metrics=metrics,
            description=description,
            metadata=metadata,
        )

    def _parse_training_compute_block(self: 'ParserBase', parent_indent: int) -> TrainingComputeSpec:
        """
        Parse compute resource specification for training jobs.
        
        Defines backend infrastructure and resource allocations for training,
        including GPU, memory, and execution queue specifications.
        
        Syntax:
            compute:
                backend: local|kubernetes|aws|azure
                queue: "gpu-high-priority"
                resources:
                    gpu: 2
                    memory: 16GB
                    cpu: 8
        """
        config = self._parse_kv_block(parent_indent)
        backend_raw = config.pop('backend', config.pop('target', 'local'))
        queue_raw = config.pop('queue', None)
        resources_raw = config.pop('resources', {})
        metadata_raw = config.pop('metadata', {})
        backend = self._strip_quotes(self._stringify_value(backend_raw)) or 'local'
        queue = self._strip_quotes(self._stringify_value(queue_raw)) if queue_raw is not None else None
        resources = self._coerce_options_dict(resources_raw)
        metadata = self._coerce_options_dict(metadata_raw)
        if config:
            metadata.update({key: self._transform_config(val) for key, val in config.items()})
        return TrainingComputeSpec(backend=backend, resources=resources, queue=queue, metadata=metadata)

    def _parse_tuning_job(self: 'ParserBase', line: str, line_no: int, base_indent: int) -> TuningJob:
        """
        Parse hyperparameter tuning job definition.
        
        Tuning jobs automate hyperparameter optimization across search spaces
        using strategies like grid search, random search, or Bayesian optimization.
        
        Syntax:
            tuning "Name":
                training_job: "JobName"
                strategy: grid|random|bayesian
                max_trials: 100
                parallel_trials: 4
                objective_metric: accuracy
                search_space:
                    learning_rate:
                        type: float
                        min: 0.0001
                        max: 0.1
                        log: true
                    batch_size:
                        type: categorical
                        values: [16, 32, 64, 128]
                early_stopping:
                    metric: val_loss
                    patience: 3
        """
        match = _TUNING_HEADER.match(line.strip())
        if not match:
            raise self._error(
                'Expected: tuning "Name":',
                line_no,
                line,
                hint='Tuning jobs require a name, e.g., tuning "OptimizeModel":'
            )
        name = match.group(1)
        training_job_name: Optional[str] = None
        strategy = "grid"
        max_trials = 1
        parallel_trials = 1
        objective_metric = "loss"
        search_space_specs: Dict[str, HyperparamSpec] = {}
        early_stopping: Optional[EarlyStoppingSpec] = None
        metadata: Dict[str, Any] = {}

        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            lowered = stripped.lower()
            if lowered.startswith('search_space:'):
                self._advance()
                block = self._parse_kv_block(indent)
                search_space_specs = self._build_hyperparam_specs(block)
                continue
            if lowered.startswith('early_stopping:'):
                self._advance()
                block = self._parse_kv_block(indent)
                early_stopping = self._build_early_stopping_spec(block)
                continue
            if lowered.startswith('metadata:'):
                self._advance()
                block = self._parse_kv_block(indent)
                metadata.update(self._transform_config(block))
                continue
            assign = re.match(r'([\w\.\- ]+)\s*:\s*(.*)$', stripped)
            if not assign:
                raise self._error("Invalid entry inside tuning block", self.pos + 1, nxt)
            key = assign.group(1).strip().lower()
            remainder = assign.group(2)
            self._advance()
            value = self._coerce_scalar(remainder) if remainder else None
            if key == 'training_job':
                training_job_name = self._strip_quotes(self._stringify_value(value))
            elif key == 'strategy':
                strategy = self._strip_quotes(self._stringify_value(value)) or strategy
            elif key == 'max_trials':
                max_trials = self._coerce_int(value) or max_trials
            elif key == 'parallel_trials':
                parallel_trials = self._coerce_int(value) or parallel_trials
            elif key in {'objective_metric', 'metric'}:
                objective_metric = self._strip_quotes(self._stringify_value(value)) or objective_metric
            else:
                metadata[key] = self._transform_config(value)

        if not training_job_name:
            raise self._error(
                "Tuning job must reference 'training_job:'",
                line_no,
                line,
                hint='Add training_job: "JobName" to specify which training job to optimize'
            )
        if not search_space_specs:
            raise self._error(
                "Tuning job must define a non-empty 'search_space:' block",
                line_no,
                line,
                hint='Add search_space: block with hyperparameter specifications'
            )

        return TuningJob(
            name=name,
            training_job=training_job_name,
            search_space=search_space_specs,
            strategy=strategy,
            max_trials=max_trials,
            parallel_trials=parallel_trials,
            early_stopping=early_stopping,
            objective_metric=objective_metric,
            metadata=metadata,
        )

    def _build_hyperparam_specs(self: 'ParserBase', block: Dict[str, Any]) -> Dict[str, HyperparamSpec]:
        """
        Build hyperparameter specifications for tuning jobs.
        
        Converts configuration blocks into typed hyperparameter specs
        supporting categorical, continuous, and discrete search spaces.
        
        Spec Types:
            categorical: Discrete choices from list of values
            float: Continuous range with optional log scale
            int: Discrete integer range with optional step
        """
        specs: Dict[str, HyperparamSpec] = {}
        for name, entry in (block or {}).items():
            if isinstance(entry, dict):
                spec_data = dict(entry)
            else:
                spec_data = {"values": entry}
            param_type = str(spec_data.pop('type', spec_data.pop('kind', 'categorical')) or 'categorical')
            min_value = self._to_float(spec_data.pop('min', spec_data.pop('low', None)))
            max_value = self._to_float(spec_data.pop('max', spec_data.pop('high', None)))
            step_value = self._to_float(spec_data.pop('step', None))
            values_entry = spec_data.pop('values', spec_data.pop('choices', None))
            if values_entry is None and isinstance(entry, list):
                values_entry = entry
            if values_entry is not None and not isinstance(values_entry, list):
                values_entry = [values_entry]
            log_value = spec_data.pop('log', spec_data.pop('log_scale', False))
            metadata = {key: self._transform_config(val) for key, val in spec_data.items()}
            specs[name] = HyperparamSpec(
                type=param_type,
                min=min_value,
                max=max_value,
                values=values_entry,
                log=bool(log_value),
                step=step_value,
                metadata=metadata,
            )
        return specs

    def _build_early_stopping_spec(self: 'ParserBase', block: Dict[str, Any]) -> EarlyStoppingSpec:
        """
        Build early stopping specification for training optimization.
        
        Defines criteria for terminating training early when metrics
        stop improving, preventing overfitting and saving resources.
        
        Configuration:
            metric: Metric to monitor (e.g., 'val_loss')
            patience: Epochs to wait without improvement
            min_delta: Minimum change to count as improvement
            mode: 'min' (lower is better) or 'max' (higher is better)
        """
        metric_name = self._strip_quotes(self._stringify_value(block.get('metric')))
        patience_value = self._coerce_int(block.get('patience')) or 0
        min_delta_value = self._to_float(block.get('min_delta')) or 0.0
        mode_value = self._strip_quotes(self._stringify_value(block.get('mode'))) or 'min'
        metadata = {key: self._transform_config(val) for key, val in block.items() if key not in {'metric', 'patience', 'min_delta', 'mode'}}
        return EarlyStoppingSpec(metric=metric_name, patience=patience_value, min_delta=min_delta_value, mode=mode_value, metadata=metadata)

    # Import _to_float from utils
    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """Convert value to float, returning None if invalid."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
