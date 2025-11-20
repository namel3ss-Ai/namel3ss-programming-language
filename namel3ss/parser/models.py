from __future__ import annotations

import shlex
from typing import Any, Dict, List, Optional

from namel3ss.ast import (
    Model,
    ModelDatasetReference,
    ModelDeploymentTarget,
    ModelEvaluationMetric,
    ModelFeatureSpec,
    ModelHyperParameter,
    ModelMonitoringSpec,
    ModelServingSpec,
)
# KeywordRegistry import removed - class does not exist

from .base import ParserBase


class ModelParserMixin(ParserBase):
    """
    Parse ML model declarations with centralized validation.
    
    Handles parsing of machine learning model specifications including:
    - Training configuration (datasets, features, hyperparameters)
    - Model framework and engine selection
    - Feature engineering specifications
    - Monitoring and alerting setup
    - Serving endpoints and deployment targets
    - Registry metadata and versioning
    
    Syntax:
        model "MyModel" using xgboost:
            from dataset train_data
            target: churn
            features: age, income, tenure
            hyperparameters:
                learning_rate: 0.1
                max_depth: 6
            monitoring:
                schedule: daily
                metrics:
                    accuracy: { threshold: 0.8 }
            serving:
                endpoints: [api.example.com]
    
    Uses centralized indentation validation for consistent error messages.
    """

    def _parse_model(self, line: str, line_no: int, base_indent: int) -> Model:
        """
        Parse model definition with training and serving configuration.
        
        Validates model name, type, engine, and processes training specs,
        monitoring, and deployment configuration.
        """
        raw = line.strip()
        if raw.endswith(':'):
            raw = raw[:-1]
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            raise self._error(
                f"Unable to parse model declaration: {exc}",
                line_no,
                line,
                hint='Check for unmatched quotes or special characters'
            )
        if len(parts) < 2 or parts[0] != 'model':
            raise self._error(
                "Expected: model \"Name\" using TYPE",
                line_no,
                line,
                hint='Model declarations must have a name and optionally a type'
            )

        name = parts[1]
        model_type = 'custom'
        engine: Optional[str] = None
        idx = 2
        if idx < len(parts) and parts[idx].lower() == 'using':
            idx += 1
            if idx >= len(parts):
                raise self._error(
                    "Model type required after 'using'",
                    line_no,
                    line,
                    hint='Specify model type like xgboost, sklearn, tensorflow'
                )
            model_type = parts[idx]
            idx += 1
        if idx < len(parts) and parts[idx].lower() == 'engine':
            idx += 1
            if idx >= len(parts):
                raise self._error(
                    "Engine name required after 'engine'",
                    line_no,
                    line,
                    hint='Specify execution engine for the model'
                )
            engine = parts[idx]
            idx += 1

        model = Model(name=name, model_type=model_type, engine=engine)
        
        # Validate indented block
        indent_info = self._expect_indent_greater_than(
            base_indent,
            context=f'model "{name}"',
            line_no=line_no
        )
        if not indent_info:
            # Model without configuration is allowed, return early
            return model

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
            if lowered.startswith('from '):
                tokens = shlex.split(stripped)
                if len(tokens) < 3:
                    raise self._error("Expected: from dataset|table SOURCE", self.pos + 1, nxt)
                source_kind = tokens[1].lower()
                source_ref = tokens[2]
                model.training.source_type = source_kind
                model.training.source = source_ref
                self._advance()
            elif lowered.startswith('target:'):
                model.training.target = stripped[len('target:'):].strip()
                self._advance()
            elif lowered.startswith('features:'):
                feats = stripped[len('features:'):].strip()
                model.training.features = [f.strip() for f in feats.split(',') if f.strip()]
                self._advance()
            elif lowered.startswith('framework:'):
                model.training.framework = stripped[len('framework:'):].strip() or None
                self._advance()
            elif lowered.startswith('objective:'):
                model.training.objective = stripped[len('objective:'):].strip() or None
                self._advance()
            elif lowered.startswith('loss:'):
                model.training.loss = stripped[len('loss:'):].strip() or None
                self._advance()
            elif lowered.startswith('optimizer:'):
                model.training.optimizer = stripped[len('optimizer:'):].strip() or None
                self._advance()
            elif lowered.startswith('batch size:') or lowered.startswith('batch_size:'):
                value = stripped.split(':', 1)[1].strip()
                model.training.batch_size = self._coerce_int(value)
                self._advance()
            elif lowered.startswith('epochs:'):
                value = stripped[len('epochs:'):].strip()
                model.training.epochs = self._coerce_int(value)
                self._advance()
            elif lowered.startswith('learning rate:') or lowered.startswith('learning_rate:'):
                value = stripped.split(':', 1)[1].strip()
                try:
                    model.training.learning_rate = float(value)
                except (ValueError, TypeError):
                    model.training.learning_rate = None
                self._advance()
            elif lowered.startswith('schedule:'):
                model.training.schedule = stripped[len('schedule:'):].strip()
                self._advance()
            elif lowered.startswith('callbacks:'):
                value = stripped[len('callbacks:'):].strip()
                model.training.callbacks = self._ensure_string_list(value)
                self._advance()
            elif lowered.startswith('split:'):
                block_indent = indent
                self._advance()
                split_config = self._parse_kv_block(block_indent)
                model.training.split.update(split_config)
            elif lowered.startswith('datasets:'):
                block_indent = indent
                self._advance()
                dataset_config = self._parse_kv_block(block_indent)
                refs = self._parse_model_dataset_block(dataset_config)
                model.training.datasets.extend(refs)
            elif lowered.startswith('transform '):
                header_line = self._advance()
                if header_line is None:
                    break
                transform = self._parse_dataset_transform_block(header_line, indent)
                model.training.transforms.append(transform)
            elif lowered.startswith('hyperparameters:'):
                block_indent = indent
                self._advance()
                params = self._parse_model_hyperparameters(block_indent)
                model.training.hyperparameters.extend(params)
            elif lowered.startswith('training metadata:'):
                block_indent = indent
                self._advance()
                metadata = self._parse_kv_block(block_indent)
                model.training.metadata.update(metadata)
            elif lowered.startswith('options:'):
                block_indent = indent
                self._advance()
                options = self._parse_kv_block(block_indent)
                model.options.update(options)
            elif lowered.startswith('registry:'):
                block_indent = indent
                self._advance()
                registry_data = self._parse_kv_block(block_indent)
                if 'version' in registry_data:
                    model.registry.version = str(registry_data.pop('version'))
                if 'accuracy' in registry_data:
                    try:
                        model.registry.accuracy = float(registry_data.pop('accuracy'))
                    except (ValueError, TypeError):
                        model.registry.accuracy = None
                registry_name = registry_data.pop('registry_name', registry_data.pop('name', None))
                if registry_name is not None:
                    model.registry.registry_name = str(registry_name)
                owner = registry_data.pop('owner', None)
                if owner is not None:
                    model.registry.owner = str(owner)
                stage = registry_data.pop('stage', registry_data.pop('phase', None))
                if stage is not None:
                    model.registry.stage = str(stage)
                updated_at = registry_data.pop('last_updated', registry_data.pop('updated_at', None))
                if updated_at is not None:
                    model.registry.last_updated = str(updated_at)
                tags_value = registry_data.pop('tags', None)
                if tags_value is not None:
                    model.registry.tags = self._ensure_string_list(tags_value)
                checks_value = registry_data.pop('checks', None)
                if isinstance(checks_value, dict):
                    model.registry.checks.update(checks_value)
                metadata_value = registry_data.pop('metadata', None)
                if isinstance(metadata_value, dict):
                    model.registry.metadata.update(metadata_value)
                if registry_data:
                    model.registry.metadata.update(registry_data)
            elif lowered.startswith('monitoring:'):
                block_indent = indent
                self._advance()
                monitoring = self._parse_model_monitoring(block_indent)
                model.monitoring = monitoring
            elif lowered.startswith('serving:'):
                block_indent = indent
                self._advance()
                serving = self._parse_model_serving(block_indent)
                model.serving = serving
            elif lowered.startswith('deployments:'):
                block_indent = indent
                self._advance()
                deployments = self._parse_model_deployments(block_indent)
                model.deployments.extend(deployments)
            elif lowered.startswith('description:'):
                model.description = stripped[len('description:'):].strip() or None
                self._advance()
            elif lowered.startswith('task:'):
                model.task = stripped[len('task:'):].strip() or None
                self._advance()
            elif lowered.startswith('feature '):
                header_line = self._advance()
                if header_line is None:
                    break
                feature_spec = self._parse_model_feature_spec(header_line, indent)
                model.features_spec.append(feature_spec)
            elif lowered.startswith('tags:'):
                tag_text = stripped[len('tags:'):].strip()
                model.tags = self._ensure_string_list(tag_text)
                self._advance()
            elif lowered.startswith('metadata:'):
                block_indent = indent
                self._advance()
                metadata = self._parse_kv_block(block_indent)
                model.metadata.update(metadata)
            else:
                raise self._error(
                    "Unknown directive inside model block",
                    self.pos + 1,
                    nxt,
                    hint='Valid model directives: from, target, features, framework, hyperparameters, monitoring, serving, etc.'
                )

        return model

    def _parse_model_hyperparameters(self, parent_indent: int) -> List[ModelHyperParameter]:
        mapping = self._parse_kv_block(parent_indent)
        params: List[ModelHyperParameter] = []
        for name, raw in mapping.items():
            value: Any = None
            tunable = False
            search_space: Optional[Dict[str, Any]] = None
            if isinstance(raw, dict):
                config = dict(raw)
                value = config.pop('value', config.pop('default', None))
                tunable = self._to_bool(config.pop('tunable', config.pop('search', False)))
                space_raw = config.pop('search_space', config.pop('space', None))
                if isinstance(space_raw, dict):
                    search_space = dict(space_raw)
                elif space_raw is not None:
                    search_space = {'value': space_raw}
                if config:
                    if search_space is None:
                        search_space = {}
                    search_space.update(config)
            else:
                value = raw
            params.append(
                ModelHyperParameter(
                    name=name,
                    value=value,
                    tunable=tunable,
                    search_space=search_space,
                )
            )
        return params

    def _parse_model_dataset_block(self, mapping: Dict[str, Any]) -> List[ModelDatasetReference]:
        refs: List[ModelDatasetReference] = []
        for role, raw in mapping.items():
            name = ''
            filters: List[Any] = []
            options: Dict[str, Any] = {}
            if isinstance(raw, dict):
                config = dict(raw)
                name_value = config.pop('name', config.pop('dataset', config.pop('source', None)))
                if name_value is None:
                    name_value = role
                name = str(name_value)
                filter_raw = config.pop('filter', config.pop('where', None))
                if isinstance(filter_raw, list):
                    filters = [self._coerce_expression(item) for item in filter_raw]
                elif filter_raw is not None:
                    filters = [self._coerce_expression(filter_raw)]
                options = self._coerce_options_dict(config.pop('options', {}))
                if config:
                    options.update(config)
            else:
                name = str(raw)
            refs.append(
                ModelDatasetReference(
                    role=role.lower(),
                    name=name,
                    filters=filters,
                    options=options,
                )
            )
        return refs

    def _parse_model_feature_spec(self, header_line: str, header_indent: int) -> ModelFeatureSpec:
        stripped = header_line.strip()
        has_block = stripped.endswith(':')
        header = stripped[:-1] if has_block else stripped
        try:
            tokens = shlex.split(header)
        except ValueError as exc:
            raise self._error(f"Unable to parse model feature declaration: {exc}", self.pos, header_line)
        if len(tokens) < 2 or tokens[0].lower() != 'feature':
            raise self._error(
                "Expected: feature \"Name\"",
                self.pos,
                header_line,
                hint='Feature specifications must have a name in quotes'
            )

        name = self._strip_quotes(tokens[1])
        config: Dict[str, Any] = {}
        if has_block:
            config = self._parse_kv_block(header_indent)

        role_raw = config.pop('role', 'feature')
        role = str(role_raw) if role_raw is not None else 'feature'
        source_raw = config.pop('source', config.pop('from', None))
        source = str(source_raw) if source_raw is not None else None
        dtype_raw = config.pop('dtype', None)
        dtype = str(dtype_raw) if dtype_raw is not None else None
        expr_raw = config.pop('expression', config.pop('expr', None))
        expression = self._coerce_expression(expr_raw) if expr_raw is not None else None
        required_raw = config.pop('required', True)
        required = self._to_bool(required_raw, True)
        description_raw = config.pop('description', config.pop('desc', None))
        description = str(description_raw) if description_raw is not None else None
        stats_raw = config.pop('stats', {})
        stats = self._coerce_options_dict(stats_raw)
        options = self._coerce_options_dict(config.pop('options', {}))
        if config:
            options.update(config)

        return ModelFeatureSpec(
            name=name,
            dtype=dtype,
            role=role,
            source=source,
            expression=expression,
            required=required,
            description=description,
            stats=stats,
            options=options,
        )

    def _parse_model_monitoring(self, parent_indent: int) -> ModelMonitoringSpec:
        config = self._parse_kv_block(parent_indent)
        schedule_raw = config.pop('schedule', config.pop('frequency', None))
        schedule = str(schedule_raw) if schedule_raw is not None else None
        metrics_raw = config.pop('metrics', {})
        metrics = self._parse_model_monitoring_metrics(metrics_raw)
        alerts_raw = config.pop('alerts', {})
        alerts = self._coerce_options_dict(alerts_raw)
        drift_raw = config.pop('drift_thresholds', config.pop('drift', {}))
        drift = self._coerce_options_dict(drift_raw)
        if config:
            extras_bucket = alerts.setdefault('extras', {})
            if not isinstance(extras_bucket, dict):
                extras_bucket = {}
                alerts['extras'] = extras_bucket
            extras_bucket.update(config)
        return ModelMonitoringSpec(
            schedule=schedule,
            metrics=metrics,
            alerts=alerts,
            drift_thresholds=drift,
        )

    def _parse_model_monitoring_metrics(self, raw: Any) -> List[ModelEvaluationMetric]:
        metrics: List[ModelEvaluationMetric] = []
        if not isinstance(raw, dict):
            return metrics
        for name, entry in raw.items():
            config = dict(entry) if isinstance(entry, dict) else {'value': entry}
            value_raw = config.pop('value', config.pop('score', None))
            value = self._coerce_optional_float(value_raw)
            threshold_raw = config.pop('threshold', config.pop('limit', None))
            threshold = self._coerce_optional_float(threshold_raw)
            goal_raw = config.pop('goal', None)
            goal = str(goal_raw) if goal_raw is not None else None
            higher_raw = config.pop('higher_is_better', config.pop('maximize', True))
            higher = self._to_bool(higher_raw, True)
            dataset_raw = config.pop('dataset', config.pop('source', None))
            dataset = str(dataset_raw) if dataset_raw is not None else None
            tags_raw = config.pop('tags', None)
            tags = self._ensure_string_list(tags_raw) if tags_raw is not None else []
            extras = self._coerce_options_dict(config.pop('extras', {}))
            if config:
                extras.update(config)
            metrics.append(
                ModelEvaluationMetric(
                    name=name,
                    value=value,
                    threshold=threshold,
                    goal=goal,
                    higher_is_better=higher,
                    dataset=dataset,
                    tags=tags,
                    extras=extras,
                )
            )
        return metrics

    def _parse_model_serving(self, parent_indent: int) -> ModelServingSpec:
        config = self._parse_kv_block(parent_indent)
        endpoints_raw = config.pop('endpoints', config.pop('urls', []))
        if isinstance(endpoints_raw, list):
            endpoints = [str(item) for item in endpoints_raw]
        elif isinstance(endpoints_raw, str):
            endpoints = [part.strip() for part in endpoints_raw.split(',') if part.strip()]
        else:
            endpoints = []
        realtime = self._coerce_options_dict(config.pop('realtime', config.pop('online', {})))
        batch = self._coerce_options_dict(config.pop('batch', config.pop('offline', {})))
        resources = self._coerce_options_dict(config.pop('resources', {}))
        options = self._coerce_options_dict(config.pop('options', {}))
        if config:
            options.update(config)
        return ModelServingSpec(
            endpoints=endpoints,
            realtime=realtime,
            batch=batch,
            resources=resources,
            options=options,
        )

    def _parse_model_deployments(self, parent_indent: int) -> List[ModelDeploymentTarget]:
        mapping = self._parse_kv_block(parent_indent)
        deployments: List[ModelDeploymentTarget] = []
        for name, raw in mapping.items():
            if isinstance(raw, dict):
                config = dict(raw)
            else:
                config = {'environment': raw}
            environment_raw = config.pop('environment', config.pop('env', None))
            strategy_raw = config.pop('strategy', config.pop('mode', None))
            options = self._coerce_options_dict(config.pop('options', {}))
            if config:
                options.update(config)
            deployments.append(
                ModelDeploymentTarget(
                    name=name,
                    environment=str(environment_raw) if environment_raw is not None else None,
                    strategy=str(strategy_raw) if strategy_raw is not None else None,
                    options=options,
                )
            )
        return deployments

    def _coerce_optional_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except (ValueError, TypeError):
            return None
