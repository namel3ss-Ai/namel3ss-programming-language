"""Helpers to translate the Namel3ss AST into backend-friendly state."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set

from namel3ss.plugins.utils import normalize_plugin_category

from ...ast import (
	Action,
	ActionOperation,
	ActionOperationType,
	AskConnectorOperation,
	AggregateOp,
	App,
	AttributeRef,
	BinaryOp,
	CallExpression,
	CallPythonOperation,
	Chain,
	ChainStep,
	Connector,
	ComputedColumnOp,
	ContextValue,
	Dataset,
	DatasetFeature,
	DatasetOp,
	DatasetProfile,
	DatasetQualityCheck,
	DatasetSchemaField,
	DatasetTarget,
	DatasetTransformStep,
	Frame,
	FrameExpression,
	FrameAccessPolicy,
	FrameColumn,
	FrameColumnConstraint,
	FrameConstraint,
	FrameIndex,
	FrameRelationship,
	FrameSourceDef,
	Experiment,
	ExperimentMetric,
	ExperimentVariant,
	Expression,
	ElifBlock,
	FilterOp,
	ForLoop,
	GroupByOp,
	IfBlock,
	Insight,
	InsightAssignment,
	InsightAudience,
	InsightDatasetRef,
	InsightDeliveryChannel,
	InsightEmit,
	InsightLogicStep,
	InsightMetric,
	InsightNarrative,
	InsightSelect,
	InsightThreshold,
	JoinOp,
	LayoutMeta,
	LayoutSpec,
	Literal,
	Model,
	ModelDatasetReference,
	ModelDeploymentTarget,
	ModelEvaluationMetric,
	ModelFeatureSpec,
	ModelHyperParameter,
	ModelMonitoringSpec,
	ModelTrainingSpec,
	ModelRegistryInfo,
	ModelServingSpec,
	Evaluator,
	Metric,
	Guardrail,
	NameRef,
	OrderByOp,
	Page,
	PageStatement,
	PredictStatement,
	RunChainOperation,
	RunPromptOperation,
	ShowChart,
	ShowForm,
	ShowTable,
	ShowText,
	Prompt,
	PromptField,
	AIModel,
	WorkflowIfBlock,
	WorkflowForBlock,
	WorkflowWhileBlock,
	WorkflowNode,
	WhileLoop,
	BreakStatement,
	ContinueStatement,
	ToastOperation,
	GoToPageOperation,
	Template,
	Memory,
	TrainingJob,
	TuningJob,
	TrainingComputeSpec,
	HyperparamSpec,
	EarlyStoppingSpec,
	UnaryOp,
	UpdateOperation,
	VariableAssignment,
	WindowFrame,
	WindowOp,
	CrudResource,
)
from ...effects import EffectAnalyzer
from ...frames import FrameExpressionAnalyzer, FrameTypeError


@dataclass
class PageComponent:
	"""Serializable representation of a page component."""

	type: str
	payload: Dict[str, Any] = field(default_factory=dict)
	index: Optional[int] = None


def _component_to_serializable(component: PageComponent) -> Dict[str, Any]:
	data = {"type": component.type}
	if component.index is not None:
		data["__component_index"] = component.index
	data.update(component.payload)
	return data


@dataclass
class PageSpec:
	"""Encoded data for a page required by the backend generator."""

	name: str
	route: str
	slug: str
	index: int
	api_path: str
	reactive: bool = False
	refresh_policy: Optional[Dict[str, Any]] = None
	components: List[PageComponent] = field(default_factory=list)
	layout: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendState:
	"""Container for backend-facing data derived from the AST."""

	app: Dict[str, Any]
	datasets: Dict[str, Dict[str, Any]]
	frames: Dict[str, Dict[str, Any]]
	connectors: Dict[str, Dict[str, Any]]
	ai_connectors: Dict[str, Dict[str, Any]]
	ai_models: Dict[str, Dict[str, Any]]
	memories: Dict[str, Dict[str, Any]]
	prompts: Dict[str, Dict[str, Any]]
	insights: Dict[str, Dict[str, Any]]
	models: Dict[str, Dict[str, Any]]
	templates: Dict[str, Dict[str, Any]]
	chains: Dict[str, Dict[str, Any]]
	experiments: Dict[str, Dict[str, Any]]
	training_jobs: Dict[str, Dict[str, Any]]
	tuning_jobs: Dict[str, Dict[str, Any]]
	crud_resources: Dict[str, Dict[str, Any]]
	evaluators: Dict[str, Dict[str, Any]]
	metrics: Dict[str, Dict[str, Any]]
	guardrails: Dict[str, Dict[str, Any]]
	pages: List[PageSpec]
	env_keys: List[str]


_TEMPLATE_PATTERN = re.compile(r"\{([^{}]+)\}")

_FRAME_ANALYZER: Optional[FrameExpressionAnalyzer] = None


def build_backend_state(app: App) -> BackendState:
	"""Build the serialisable backend state for the provided :class:`App`."""

	env_keys: Set[str] = set()
	global _FRAME_ANALYZER
	_FRAME_ANALYZER = FrameExpressionAnalyzer(app.frames) if app.frames else None
	try:
		EffectAnalyzer(app).analyze()

		datasets: Dict[str, Dict[str, Any]] = {}
		connectors: Dict[str, Dict[str, Any]] = {}
		for dataset in app.datasets:
			encoded = _encode_dataset(dataset, env_keys)
			datasets[dataset.name] = encoded
			if encoded.get("connector"):
				connectors[dataset.name] = encoded["connector"]

		frames: Dict[str, Dict[str, Any]] = {}
		for frame in app.frames:
			frames[frame.name] = _encode_frame(frame, env_keys)

		ai_connectors: Dict[str, Dict[str, Any]] = {}
		for connector in app.connectors:
			aiconfig = _encode_ai_connector(connector, env_keys)
			aiconfig.pop("name", None)
			ai_connectors[connector.name] = aiconfig

		ai_models: Dict[str, Dict[str, Any]] = {}
		for model in app.ai_models:
			ai_models[model.name] = _encode_ai_model(model, env_keys)

		memories: Dict[str, Dict[str, Any]] = {}
		for memory in app.memories:
			memories[memory.name] = _encode_memory(memory, env_keys)
		memory_names = set(memories.keys())

		prompt_lookup: Dict[str, Prompt] = {}
		prompts: Dict[str, Dict[str, Any]] = {}
		for prompt in app.prompts:
			model_name = prompt.model
			if model_name not in ai_models:
				raise ValueError(f"Prompt '{prompt.name}' references undefined model '{model_name}'")
			prompts[prompt.name] = _encode_prompt(prompt, env_keys)
			prompt_lookup[prompt.name] = prompt

		insights: Dict[str, Dict[str, Any]] = {}
		for insight in app.insights:
			insights[insight.name] = _encode_insight(insight, env_keys)

		models: Dict[str, Dict[str, Any]] = {}
		for model in app.models:
			models[model.name] = _encode_model(model, env_keys)

		templates: Dict[str, Dict[str, Any]] = {}
		for template in app.templates:
			templates[template.name] = _encode_template(template, env_keys)

		chains: Dict[str, Dict[str, Any]] = {}
		for chain in app.chains:
			chains[chain.name] = _encode_chain(chain, env_keys, memory_names)

		experiments: Dict[str, Dict[str, Any]] = {}
		for experiment in app.experiments:
			experiments[experiment.name] = _encode_experiment(experiment, env_keys)

		training_jobs: Dict[str, Dict[str, Any]] = {}
		for job in app.training_jobs:
			training_jobs[job.name] = _encode_training_job(job, env_keys)

		tuning_jobs: Dict[str, Dict[str, Any]] = {}
		for job in app.tuning_jobs:
			tuning_jobs[job.name] = _encode_tuning_job(job, env_keys)

		crud_resources: Dict[str, Dict[str, Any]] = {}
		for resource in app.crud_resources:
			crud_resources[resource.name] = _encode_crud_resource(resource, env_keys)

		evaluators: Dict[str, Dict[str, Any]] = {}
		for evaluator in app.evaluators:
			evaluators[evaluator.name] = _encode_evaluator(evaluator, env_keys)

		metrics: Dict[str, Dict[str, Any]] = {}
		for metric in app.metrics:
			metrics[metric.name] = _encode_metric(metric)

		guardrails: Dict[str, Dict[str, Any]] = {}
		for guardrail in app.guardrails:
			guardrails[guardrail.name] = _encode_guardrail(guardrail)

		pages: List[PageSpec] = []
		for index, page in enumerate(app.pages):
			pages.append(_encode_page(index, page, env_keys, prompt_lookup))

		app_payload: Dict[str, Any] = {
			"name": app.name,
			"database": app.database,
			"theme": dict(app.theme.values),
			"variables": [_encode_variable(var, env_keys) for var in app.variables],
		}

		sorted_env_keys = sorted(env_keys)

		return BackendState(
			app=app_payload,
			datasets=datasets,
			frames=frames,
			connectors=connectors,
			ai_connectors=ai_connectors,
			memories=memories,
			ai_models=ai_models,
			prompts=prompts,
			insights=insights,
			models=models,
			templates=templates,
			chains=chains,
			experiments=experiments,
			training_jobs=training_jobs,
			tuning_jobs=tuning_jobs,
			crud_resources=crud_resources,
			evaluators=evaluators,
			metrics=metrics,
			guardrails=guardrails,
			pages=pages,
			env_keys=sorted_env_keys,
		)
	finally:
		_FRAME_ANALYZER = None


def _encode_dataset(dataset: Dataset, env_keys: Set[str]) -> Dict[str, Any]:
	operations = [_encode_dataset_op(operation, env_keys) for operation in dataset.operations]

	connector = None
	if dataset.connector:
		connector = {
			"type": dataset.connector.connector_type,
			"name": dataset.connector.connector_name,
			"options": _encode_value(dataset.connector.options, env_keys),
		}

	payload: Dict[str, Any] = {
		"name": dataset.name,
		"source_type": dataset.source_type,
		"source": dataset.source,
		"operations": [op for op in operations if op],
		"transforms": [_encode_dataset_transform(step, env_keys) for step in dataset.transforms],
		"schema": [_encode_dataset_schema(field, env_keys) for field in dataset.schema],
		"features": [_encode_dataset_feature(feature, env_keys) for feature in dataset.features],
		"targets": [_encode_dataset_target(target, env_keys) for target in dataset.targets],
		"quality_checks": [_encode_dataset_quality_check(check, env_keys) for check in dataset.quality_checks],
		"profile": _encode_dataset_profile(dataset.profile, env_keys),
		"connector": connector,
		"reactive": dataset.reactive,
		"refresh_policy": _encode_value(dataset.refresh_policy, env_keys),
		"cache_policy": _encode_value(dataset.cache_policy, env_keys),
		"pagination": _encode_value(dataset.pagination, env_keys),
		"streaming": _encode_value(dataset.streaming, env_keys),
		"metadata": _encode_value(dataset.metadata, env_keys),
		"lineage": _encode_value(dataset.lineage, env_keys),
		"tags": list(dataset.tags or []),
		"sample_rows": [{"id": idx + 1, "value": (idx + 1) * 10} for idx in range(3)],
	}
	return payload


def _encode_dataset_op(operation: DatasetOp, env_keys: Set[str]) -> Dict[str, Any]:
	if isinstance(operation, FilterOp):
		return {
			"type": "filter",
			"condition": _expression_to_source(operation.condition),
			"condition_expr": _expression_to_runtime(operation.condition),
		}
	if isinstance(operation, GroupByOp):
		return {"type": "group_by", "columns": list(operation.columns)}
	if isinstance(operation, AggregateOp):
		return {
			"type": "aggregate",
			"function": operation.function,
			"expression": operation.expression,
		}
	if isinstance(operation, OrderByOp):
		return {"type": "order_by", "columns": list(operation.columns)}
	if isinstance(operation, ComputedColumnOp):
		return {
			"type": "computed_column",
			"name": operation.name,
			"expression": _expression_to_source(operation.expression),
			"expression_expr": _expression_to_runtime(operation.expression),
		}
	if isinstance(operation, WindowOp):
		return {
			"type": "window",
			"name": operation.name,
			"function": operation.function,
			"target": operation.target,
			"partition_by": list(operation.partition_by or []),
			"order_by": list(operation.order_by or []),
			"frame": _encode_window_frame(operation.frame),
		}
	if isinstance(operation, JoinOp):
		return {
			"type": "join",
			"target_type": operation.target_type,
			"target_name": operation.target_name,
			"join_type": operation.join_type,
			"condition": _expression_to_source(operation.condition),
			"condition_expr": _expression_to_runtime(operation.condition),
		}
	return {"type": type(operation).__name__}


def _encode_window_frame(frame: WindowFrame) -> Dict[str, Any]:
	return {
		"mode": frame.mode,
		"interval_value": frame.interval_value,
		"interval_unit": frame.interval_unit,
	}


def _encode_dataset_transform(step: DatasetTransformStep, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"name": step.name,
		"type": step.transform_type,
		"inputs": list(step.inputs or []),
		"output": step.output,
		"expression": _expression_to_source(step.expression),
		"expression_expr": _expression_to_runtime(step.expression),
		"options": _encode_value(step.options, env_keys),
	}


def _encode_dataset_schema(field: DatasetSchemaField, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"name": field.name,
		"dtype": field.dtype,
		"nullable": field.nullable,
		"description": field.description,
		"tags": list(field.tags or []),
		"constraints": _encode_value(field.constraints, env_keys),
		"stats": _encode_value(field.stats, env_keys),
	}


def _encode_dataset_feature(feature: DatasetFeature, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"name": feature.name,
		"role": feature.role,
		"source": feature.source,
		"dtype": feature.dtype,
		"expression": _expression_to_source(feature.expression),
		"expression_expr": _expression_to_runtime(feature.expression),
		"description": feature.description,
		"options": _encode_value(feature.options, env_keys),
	}


def _encode_dataset_target(target: DatasetTarget, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"name": target.name,
		"kind": target.kind,
		"expression": _expression_to_source(target.expression),
		"expression_expr": _expression_to_runtime(target.expression),
		"positive_class": target.positive_class,
		"horizon": target.horizon,
		"options": _encode_value(target.options, env_keys),
	}


def _encode_dataset_quality_check(check: DatasetQualityCheck, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"name": check.name,
		"condition": _expression_to_source(check.condition),
		"condition_expr": _expression_to_runtime(check.condition),
		"metric": check.metric,
		"threshold": check.threshold,
		"severity": check.severity,
		"message": check.message,
		"extras": _encode_value(check.extras, env_keys),
	}



def _encode_dataset_profile(profile: Optional[DatasetProfile], env_keys: Set[str]) -> Optional[Dict[str, Any]]:
	if profile is None:
		return None
	return {
		"row_count": profile.row_count,
		"column_count": profile.column_count,
		"freshness": profile.freshness,
		"updated_at": profile.updated_at,
		"stats": _encode_value(profile.stats, env_keys),
	}


def _encode_frame(frame: Frame, env_keys: Set[str]) -> Dict[str, Any]:
	metadata_value = _encode_value(frame.metadata, env_keys)
	if not isinstance(metadata_value, dict):
		metadata_value = {"value": metadata_value} if metadata_value is not None else {}
	options_value = _encode_value(frame.options, env_keys)
	if not isinstance(options_value, dict):
		options_value = {"value": options_value} if options_value is not None else {}
	examples: List[Dict[str, Any]] = []
	for example in frame.examples:
		encoded = _encode_value(example, env_keys)
		if isinstance(encoded, dict):
			examples.append(encoded)
		else:
			examples.append({"value": encoded})
	return {
		"name": frame.name,
		"source_type": str(frame.source_type or "dataset").lower(),
		"source": frame.source,
		"description": frame.description,
		"columns": [_encode_frame_column(column, env_keys) for column in frame.columns],
		"indexes": [_encode_frame_index(index, env_keys) for index in frame.indexes],
		"relationships": [_encode_frame_relationship(relationship, env_keys) for relationship in frame.relationships],
		"constraints": [_encode_frame_constraint(constraint, env_keys) for constraint in frame.constraints],
		"access": _encode_frame_access(frame.access, env_keys),
		"tags": list(frame.tags or []),
		"metadata": metadata_value,
		"examples": examples,
		"options": options_value,
		"key": list(frame.key or []),
		"splits": dict(frame.splits or {}),
		"source_config": _encode_frame_source(frame.source_config),
	}


def _encode_frame_column(column: FrameColumn, env_keys: Set[str]) -> Dict[str, Any]:
	metadata_value = _encode_value(column.metadata, env_keys)
	if not isinstance(metadata_value, dict):
		metadata_value = {"value": metadata_value} if metadata_value is not None else {}
	return {
		"name": column.name,
		"dtype": column.dtype,
		"nullable": column.nullable,
		"description": column.description,
		"role": column.role,
		"default": _encode_value(column.default, env_keys),
		"expression": _expression_to_source(column.expression),
		"expression_expr": _expression_to_runtime(column.expression),
		"source": column.source,
		"tags": list(column.tags or []),
		"metadata": metadata_value,
		"validations": [_encode_frame_column_validation(validation, env_keys) for validation in column.validations],
	}


def _encode_frame_column_validation(validation: FrameColumnConstraint, env_keys: Set[str]) -> Dict[str, Any]:
	config_value = _encode_value(validation.config, env_keys)
	if not isinstance(config_value, dict):
		config_value = {"value": config_value} if config_value is not None else {}
	return {
		"name": validation.name,
		"expression": _expression_to_source(validation.expression),
		"expression_expr": _expression_to_runtime(validation.expression),
		"message": validation.message,
		"severity": validation.severity,
		"config": config_value,
	}


def _encode_frame_index(index: FrameIndex, env_keys: Set[str]) -> Dict[str, Any]:
	options_value = _encode_value(index.options, env_keys)
	if not isinstance(options_value, dict):
		options_value = {"value": options_value} if options_value is not None else {}
	return {
		"name": index.name,
		"columns": list(index.columns or []),
		"unique": index.unique,
		"method": index.method,
		"options": options_value,
	}


def _encode_frame_relationship(relationship: FrameRelationship, env_keys: Set[str]) -> Dict[str, Any]:
	metadata_value = _encode_value(relationship.metadata, env_keys)
	if not isinstance(metadata_value, dict):
		metadata_value = {"value": metadata_value} if metadata_value is not None else {}
	return {
		"name": relationship.name,
		"target_frame": relationship.target_frame,
		"target_dataset": relationship.target_dataset,
		"local_key": relationship.local_key,
		"remote_key": relationship.remote_key,
		"cardinality": relationship.cardinality,
		"join_type": relationship.join_type,
		"description": relationship.description,
		"metadata": metadata_value,
	}


def _encode_frame_constraint(constraint: FrameConstraint, env_keys: Set[str]) -> Dict[str, Any]:
	metadata_value = _encode_value(constraint.metadata, env_keys)
	if not isinstance(metadata_value, dict):
		metadata_value = {"value": metadata_value} if metadata_value is not None else {}
	return {
		"name": constraint.name,
		"expression": _expression_to_source(constraint.expression),
		"expression_expr": _expression_to_runtime(constraint.expression),
		"message": constraint.message,
		"severity": constraint.severity,
		"metadata": metadata_value,
	}


def _encode_frame_access(policy: Optional[FrameAccessPolicy], env_keys: Set[str]) -> Optional[Dict[str, Any]]:
	if policy is None:
		return None
	metadata_value = _encode_value(policy.metadata, env_keys)
	if not isinstance(metadata_value, dict):
		metadata_value = {"value": metadata_value} if metadata_value is not None else {}
	return {
		"public": policy.public,
		"roles": list(policy.roles or []),
		"allow_anonymous": policy.allow_anonymous,
		"rate_limit_per_minute": policy.rate_limit_per_minute,
		"cache_seconds": policy.cache_seconds,
		"metadata": metadata_value,
	}


def _encode_frame_source(source: Optional[FrameSourceDef]) -> Optional[Dict[str, Any]]:
	if source is None:
		return None
	payload: Dict[str, Any] = {"kind": source.kind}
	if source.connection:
		payload["connection"] = source.connection
	if source.table:
		payload["table"] = source.table
	if source.path:
		payload["path"] = source.path
	if source.format:
		payload["format"] = source.format
	return payload


def _encode_statement(
	statement: PageStatement,
	env_keys: Set[str],
	prompt_lookup: Dict[str, Prompt],
) -> Optional[PageComponent]:
	if isinstance(statement, ShowText):
		_collect_template_markers(statement.text, env_keys)
		payload = {
			"text": statement.text,
			"styles": dict(statement.styles),
		}
		return PageComponent(type="text", payload=payload)

	if isinstance(statement, ShowTable):
		payload = {
			"title": statement.title,
			"source_type": statement.source_type,
			"source": statement.source,
			"columns": list(statement.columns or []),
			"filter": statement.filter_by,
			"sort": statement.sort_by,
			"style": dict(statement.style or {}),
			"layout": _encode_layout_meta(statement.layout),
			"insight": statement.insight,
			"dynamic_columns": _encode_value(statement.dynamic_columns, env_keys),
		}
		return PageComponent(type="table", payload=payload)

	if isinstance(statement, ShowChart):
		payload = {
			"heading": statement.heading,
			"title": statement.title,
			"source_type": statement.source_type,
			"source": statement.source,
			"chart_type": statement.chart_type,
			"x": statement.x,
			"y": statement.y,
			"color": statement.color,
			"layout": _encode_layout_meta(statement.layout),
			"insight": statement.insight,
			"encodings": _encode_value(statement.encodings, env_keys),
			"style": dict(statement.style or {}),
			"legend": _encode_value(statement.legend, env_keys),
		}
		return PageComponent(type="chart", payload=payload)

	if isinstance(statement, ShowForm):
		payload = {
			"title": statement.title,
			"fields": [
				{"name": field.name, "field_type": field.field_type}
				for field in statement.fields
			],
			"layout": _encode_layout_spec(statement.layout),
			"operations": [_encode_action_operation(op, env_keys, prompt_lookup) for op in statement.on_submit_ops],
			"styles": dict(statement.styles),
		}
		return PageComponent(type="form", payload=payload)

	if isinstance(statement, Action):
		payload = {
			"name": statement.name,
			"trigger": statement.trigger,
			"operations": [_encode_action_operation(op, env_keys, prompt_lookup) for op in statement.operations],
		}
		return PageComponent(type="action", payload=payload)

	if isinstance(statement, VariableAssignment):
		payload = {
			"name": statement.name,
			"value": _encode_value(statement.value, env_keys),
			"value_source": _expression_to_source(statement.value),
			"value_expr": _expression_to_runtime(statement.value),
		}
		return PageComponent(type="variable", payload=payload)

	if isinstance(statement, IfBlock):
		body_payload: List[Dict[str, Any]] = []
		for stmt in statement.body:
			encoded = _encode_statement_dict(stmt, env_keys, prompt_lookup)
			if encoded:
				body_payload.append(encoded)
		elif_payload: List[Dict[str, Any]] = []
		for branch in statement.elifs:
			branch_body: List[Dict[str, Any]] = []
			for stmt in branch.body:
				encoded = _encode_statement_dict(stmt, env_keys, prompt_lookup)
				if encoded:
					branch_body.append(encoded)
			elif_payload.append({
				"condition": _expression_to_runtime(branch.condition),
				"body": branch_body,
			})
		else_payload: List[Dict[str, Any]] = []
		for stmt in statement.else_body or []:
			encoded = _encode_statement_dict(stmt, env_keys, prompt_lookup)
			if encoded:
				else_payload.append(encoded)
		payload = {
			"condition": _expression_to_runtime(statement.condition),
			"body": body_payload,
			"elifs": elif_payload,
			"else_body": else_payload,
		}
		return PageComponent(type="if", payload=payload)

	if isinstance(statement, ForLoop):
		loop_body: List[Dict[str, Any]] = []
		for stmt in statement.body:
			encoded = _encode_statement_dict(stmt, env_keys, prompt_lookup)
			if encoded:
				loop_body.append(encoded)
		payload = {
			"loop_var": statement.loop_var,
			"source_kind": statement.source_kind,
			"source_name": statement.source_name,
			"body": loop_body,
		}
		return PageComponent(type="for_loop", payload=payload)

	if isinstance(statement, WhileLoop):
		loop_body: List[Dict[str, Any]] = []
		for stmt in statement.body:
			encoded = _encode_statement_dict(stmt, env_keys, prompt_lookup)
			if encoded:
				loop_body.append(encoded)
		payload = {
			"condition": _expression_to_runtime(statement.condition),
			"body": loop_body,
		}
		return PageComponent(type="while_loop", payload=payload)

	if isinstance(statement, BreakStatement):
		return PageComponent(type="break", payload={})

	if isinstance(statement, ContinueStatement):
		return PageComponent(type="continue", payload={})

	if isinstance(statement, PredictStatement):
		payload = {
			"model_name": statement.model_name,
			"input_kind": statement.input_kind,
			"input_ref": statement.input_ref,
			"assign": _encode_value(statement.assign, env_keys),
			"parameters": _encode_value(statement.parameters, env_keys),
		}
		return PageComponent(type="predict", payload=payload)

	return None


def _encode_statement_dict(
	statement: PageStatement,
	env_keys: Set[str],
	prompt_lookup: Dict[str, Prompt],
) -> Optional[Dict[str, Any]]:
	component = _encode_statement(statement, env_keys, prompt_lookup)
	if component is None:
		return None
	return _component_to_serializable(component)


def _encode_page(index: int, page: Page, env_keys: Set[str], prompt_lookup: Dict[str, Prompt]) -> PageSpec:
	slug = _slugify_page_name(page.name)
	api_path = _page_api_path(page.route)
	components: List[PageComponent] = []
	for statement in page.statements:
		component = _encode_statement(statement, env_keys, prompt_lookup)
		if component is not None:
			component.index = len(components)
			components.append(component)
	layout = dict(page.layout) if page.layout else {}
	refresh_policy = None
	if page.refresh_policy is not None:
		refresh_policy = {
			"interval_seconds": page.refresh_policy.interval_seconds,
			"mode": page.refresh_policy.mode,
		}
	return PageSpec(
		name=page.name,
		route=page.route,
		slug=slug,
		index=index,
		api_path=api_path,
		reactive=page.reactive,
		refresh_policy=refresh_policy,
		components=components,
		layout=layout,
	)


def _encode_layout_meta(meta: Optional[LayoutMeta]) -> Optional[Dict[str, Any]]:
	if meta is None:
		return None
	return {
		"width": meta.width,
		"height": meta.height,
		"variant": meta.variant,
		"align": meta.align,
		"emphasis": meta.emphasis,
		"extras": dict(meta.extras),
	}


def _encode_layout_spec(spec: LayoutSpec) -> Dict[str, Any]:
	return {
		"width": spec.width,
		"height": spec.height,
		"variant": spec.variant,
		"order": spec.order,
		"area": spec.area,
		"breakpoint": spec.breakpoint,
		"props": dict(spec.props),
	}


def _encode_insight(insight: Insight, env_keys: Set[str]) -> Dict[str, Any]:
	logic = [_encode_insight_step(step, env_keys) for step in insight.logic]
	metrics = [_encode_insight_metric(metric, env_keys) for metric in insight.metrics]
	thresholds = [_encode_insight_threshold(threshold, env_keys) for threshold in insight.thresholds]
	narratives = [_encode_insight_narrative(narrative, env_keys) for narrative in insight.narratives]
	expose_sources = {key: _expression_to_source(value) for key, value in insight.expose_as.items()}
	expose_exprs = {key: _expression_to_runtime(value) for key, value in insight.expose_as.items()}
	parameter_sources = {key: _expression_to_source(value) for key, value in insight.parameters.items()}
	parameter_exprs = {key: _expression_to_runtime(value) for key, value in insight.parameters.items()}
	return {
		"name": insight.name,
		"source_dataset": insight.source_dataset,
		"logic": logic,
		"metrics": metrics,
		"thresholds": thresholds,
		"narratives": narratives,
		"expose_as": expose_sources,
		"expose_expr": expose_exprs,
		"datasets": [_encode_insight_dataset_ref(ref, env_keys) for ref in insight.datasets],
		"parameters": parameter_sources,
		"parameters_expr": parameter_exprs,
		"audiences": [_encode_insight_audience(audience, env_keys) for audience in insight.audiences],
		"channels": [_encode_insight_channel(channel, env_keys) for channel in insight.channels],
		"tags": list(insight.tags or []),
		"metadata": _encode_value(insight.metadata, env_keys),
	}


def _encode_insight_step(step: InsightLogicStep, env_keys: Set[str]) -> Dict[str, Any]:
	if isinstance(step, InsightAssignment):
		return {
			"type": "assign",
			"name": step.name,
			"expression": _expression_to_source(step.expression),
			"expression_expr": _expression_to_runtime(step.expression),
		}
	if isinstance(step, InsightSelect):
		return {
			"type": "select",
			"dataset": step.dataset,
			"condition": _expression_to_source(step.condition),
			"condition_expr": _expression_to_runtime(step.condition),
			"limit": step.limit,
			"order_by": list(step.order_by or []),
		}
	if isinstance(step, InsightEmit):
		_collect_template_markers(step.content, env_keys)
		return {
			"type": "emit",
			"kind": step.kind,
			"content": step.content,
			"props": _encode_value(step.props, env_keys),
		}
	return {"type": type(step).__name__}


def _encode_insight_metric(metric: InsightMetric, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"name": metric.name,
		"value": _expression_to_source(metric.value),
		"value_expr": _expression_to_runtime(metric.value),
		"label": metric.label,
		"format": metric.format,
		"unit": metric.unit,
		"baseline": _expression_to_source(metric.baseline),
		"baseline_expr": _expression_to_runtime(metric.baseline),
		"target": _expression_to_source(metric.target),
		"target_expr": _expression_to_runtime(metric.target),
		"window": metric.window,
		"extras": _encode_value(metric.extras, env_keys),
	}


def _encode_insight_threshold(threshold: InsightThreshold, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"name": threshold.name,
		"metric": threshold.metric,
		"operator": threshold.operator,
		"value": _expression_to_source(threshold.value),
		"value_expr": _expression_to_runtime(threshold.value),
		"level": threshold.level,
		"message": threshold.message,
		"window": threshold.window,
		"extras": _encode_value(threshold.extras, env_keys),
	}


def _encode_insight_narrative(narrative: InsightNarrative, env_keys: Set[str]) -> Dict[str, Any]:
	_collect_template_markers(narrative.template, env_keys)
	return {
		"name": narrative.name,
		"template": narrative.template,
		"variant": narrative.variant,
		"style": _encode_value(narrative.style, env_keys),
		"extras": _encode_value(narrative.extras, env_keys),
	}


def _encode_insight_dataset_ref(ref: InsightDatasetRef, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"name": ref.name,
		"role": ref.role,
		"transforms": [_encode_dataset_transform(step, env_keys) for step in ref.transforms],
		"options": _encode_value(ref.options, env_keys),
	}


def _encode_insight_audience(audience: InsightAudience, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"name": audience.name,
		"persona": audience.persona,
		"needs": _encode_value(audience.needs, env_keys),
		"channels": list(audience.channels or []),
	}


def _encode_insight_channel(channel: InsightDeliveryChannel, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"kind": channel.kind,
		"target": channel.target,
		"schedule": channel.schedule,
		"options": _encode_value(channel.options, env_keys),
	}


def _encode_model(model: Model, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"name": model.name,
		"type": model.model_type,
		"engine": model.engine,
		"task": model.task,
		"description": model.description,
		"options": _encode_value(model.options, env_keys),
		"training": _encode_model_training(model.training, env_keys),
		"features_spec": [_encode_model_feature_spec(feature, env_keys) for feature in model.features_spec],
		"registry": _encode_model_registry(model.registry, env_keys),
		"monitoring": _encode_model_monitoring(model.monitoring, env_keys),
		"serving": _encode_model_serving_spec(model.serving, env_keys),
		"deployments": [_encode_model_deployment(deployment, env_keys) for deployment in model.deployments],
		"tags": list(model.tags or []),
		"metadata": _encode_value(model.metadata, env_keys),
	}


def _encode_model_training(training: ModelTrainingSpec, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"source_type": training.source_type,
		"source": training.source,
		"target": training.target,
		"features": list(training.features or []),
		"split": _encode_value(training.split, env_keys),
		"schedule": training.schedule,
		"framework": training.framework,
		"objective": training.objective,
		"loss": training.loss,
		"optimizer": training.optimizer,
		"batch_size": training.batch_size,
		"epochs": training.epochs,
		"learning_rate": training.learning_rate,
		"hyperparameters": [_encode_model_hyperparameter(param, env_keys) for param in training.hyperparameters],
		"datasets": [_encode_model_dataset_reference(reference, env_keys) for reference in training.datasets],
		"transforms": [_encode_dataset_transform(step, env_keys) for step in training.transforms],
		"callbacks": list(training.callbacks or []),
		"metadata": _encode_value(training.metadata, env_keys),
	}


def _encode_model_hyperparameter(parameter: ModelHyperParameter, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"name": parameter.name,
		"value": _encode_value(parameter.value, env_keys),
		"tunable": parameter.tunable,
		"search_space": _encode_value(parameter.search_space, env_keys),
	}


def _encode_model_dataset_reference(reference: ModelDatasetReference, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"role": reference.role,
		"name": reference.name,
		"filters": [_expression_to_source(expr) for expr in reference.filters],
		"options": _encode_value(reference.options, env_keys),
	}


def _encode_model_feature_spec(feature: ModelFeatureSpec, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"name": feature.name,
		"dtype": feature.dtype,
		"role": feature.role,
		"source": feature.source,
		"expression": _expression_to_source(feature.expression),
		"required": feature.required,
		"description": feature.description,
		"stats": _encode_value(feature.stats, env_keys),
		"options": _encode_value(feature.options, env_keys),
	}


def _encode_model_registry(registry: ModelRegistryInfo, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"version": registry.version,
		"accuracy": registry.accuracy,
		"metrics": _encode_value(registry.metrics, env_keys),
		"metadata": _encode_value(registry.metadata, env_keys),
		"registry_name": registry.registry_name,
		"owner": registry.owner,
		"stage": registry.stage,
		"last_updated": registry.last_updated,
		"tags": list(registry.tags or []),
		"checks": _encode_value(registry.checks, env_keys),
	}


def _encode_model_monitoring(monitoring: Optional[ModelMonitoringSpec], env_keys: Set[str]) -> Optional[Dict[str, Any]]:
	if monitoring is None:
		return None
	return {
		"schedule": monitoring.schedule,
		"metrics": [_encode_model_monitoring_metric(metric, env_keys) for metric in monitoring.metrics],
		"alerts": _encode_value(monitoring.alerts, env_keys),
		"drift_thresholds": _encode_value(monitoring.drift_thresholds, env_keys),
	}


def _encode_model_monitoring_metric(metric: ModelEvaluationMetric, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"name": metric.name,
		"value": _encode_value(metric.value, env_keys),
		"threshold": _encode_value(metric.threshold, env_keys),
		"goal": metric.goal,
		"higher_is_better": metric.higher_is_better,
		"dataset": metric.dataset,
		"tags": list(metric.tags or []),
		"extras": _encode_value(metric.extras, env_keys),
	}


def _encode_model_serving_spec(serving: Optional[ModelServingSpec], env_keys: Set[str]) -> Optional[Dict[str, Any]]:
	if serving is None:
		return None
	return {
		"endpoints": list(serving.endpoints or []),
		"realtime": _encode_value(serving.realtime, env_keys),
		"batch": _encode_value(serving.batch, env_keys),
		"resources": _encode_value(serving.resources, env_keys),
		"options": _encode_value(serving.options, env_keys),
	}


def _encode_model_deployment(deployment: ModelDeploymentTarget, env_keys: Set[str]) -> Dict[str, Any]:
	return {
		"name": deployment.name,
		"environment": deployment.environment,
		"strategy": deployment.strategy,
		"options": _encode_value(deployment.options, env_keys),
	}


def _encode_training_job(job: TrainingJob, env_keys: Set[str]) -> Dict[str, Any]:
	compute_spec = job.compute or TrainingComputeSpec()
	return {
		"name": job.name,
		"model": job.model,
		"dataset": job.dataset,
		"objective": job.objective,
		"hyperparameters": {
			key: _encode_value(value, env_keys) for key, value in (job.hyperparameters or {}).items()
		},
		"compute": _encode_training_compute_spec(compute_spec, env_keys),
		"output_registry": job.output_registry,
		"metrics": list(job.metrics or []),
		"description": job.description,
		"metadata": _encode_metadata_dict(job.metadata, env_keys),
	}


def _encode_training_compute_spec(compute: TrainingComputeSpec, env_keys: Set[str]) -> Dict[str, Any]:
	resources = {
		key: _encode_value(value, env_keys) for key, value in (compute.resources or {}).items()
	}
	return {
		"backend": compute.backend or "local",
		"resources": resources,
		"queue": compute.queue,
		"metadata": _encode_metadata_dict(compute.metadata, env_keys),
	}


def _encode_tuning_job(job: TuningJob, env_keys: Set[str]) -> Dict[str, Any]:
	search_space = {
		name: _encode_hyperparam_spec(spec, env_keys)
		for name, spec in (job.search_space or {}).items()
	}
	return {
		"name": job.name,
		"training_job": job.training_job,
		"strategy": job.strategy,
		"max_trials": job.max_trials,
		"parallel_trials": job.parallel_trials,
		"objective_metric": job.objective_metric,
		"search_space": search_space,
		"early_stopping": _encode_early_stopping_spec(job.early_stopping, env_keys),
		"metadata": _encode_metadata_dict(job.metadata, env_keys),
	}


def _encode_hyperparam_spec(spec: HyperparamSpec, env_keys: Set[str]) -> Dict[str, Any]:
	values = [_encode_value(value, env_keys) for value in (spec.values or [])]
	return {
		"type": spec.type,
		"min": spec.min,
		"max": spec.max,
		"values": values,
		"log": bool(spec.log),
		"step": spec.step,
		"metadata": _encode_metadata_dict(spec.metadata, env_keys),
	}


def _encode_early_stopping_spec(spec: Optional[EarlyStoppingSpec], env_keys: Set[str]) -> Optional[Dict[str, Any]]:
	if spec is None:
		return None
	return {
		"metric": spec.metric,
		"patience": spec.patience,
		"min_delta": spec.min_delta,
		"mode": spec.mode,
		"metadata": _encode_metadata_dict(spec.metadata, env_keys),
	}


def _encode_metadata_dict(value: Optional[Dict[str, Any]], env_keys: Set[str]) -> Dict[str, Any]:
	encoded = _encode_value(value or {}, env_keys)
	if isinstance(encoded, dict):
		return dict(encoded)
	if encoded is None:
		return {}
	return {"value": encoded}

def _encode_ai_connector(connector: Connector, env_keys: Set[str]) -> Dict[str, Any]:
	config_encoded = _encode_value(connector.config, env_keys)
	if isinstance(config_encoded, dict):
		config_payload = config_encoded
	else:
		config_payload = connector.config
	encoded: Dict[str, Any] = {
		"name": connector.name,
		"type": connector.connector_type,
		"category": normalize_plugin_category(connector.connector_type),
		"config": config_payload,
	}
	if connector.provider:
		encoded["provider"] = connector.provider
	if connector.description:
		encoded["description"] = connector.description
	return encoded


def _encode_template(template: Template, env_keys: Set[str]) -> Dict[str, Any]:
	metadata_encoded = _encode_value(template.metadata, env_keys)
	if isinstance(metadata_encoded, dict):
		metadata_payload = metadata_encoded
	else:
		metadata_payload = template.metadata
	return {
		"name": template.name,
		"prompt": template.prompt,
		"metadata": metadata_payload,
	}


def _encode_memory(memory: Memory, env_keys: Set[str]) -> Dict[str, Any]:
	config_encoded = _encode_value(memory.config, env_keys)
	if not isinstance(config_encoded, dict):
		config_encoded = {"value": config_encoded} if config_encoded is not None else {}
	metadata_encoded = _encode_value(memory.metadata, env_keys)
	if not isinstance(metadata_encoded, dict):
		metadata_encoded = {"value": metadata_encoded} if metadata_encoded is not None else {}
	payload: Dict[str, Any] = {
		"name": memory.name,
		"scope": memory.scope,
		"kind": memory.kind,
		"config": config_encoded,
		"metadata": metadata_encoded,
	}
	if memory.max_items is not None:
		payload["max_items"] = int(memory.max_items)
	return payload


def _encode_ai_model(model: AIModel, env_keys: Set[str]) -> Dict[str, Any]:
	config_payload = _encode_value(model.config, env_keys)
	metadata_value = _encode_value(model.metadata, env_keys)
	if not isinstance(metadata_value, dict):
		metadata_value = {"value": metadata_value} if metadata_value is not None else {}
	return {
		"name": model.name,
		"provider": model.provider,
		"model": model.model_name,
		"config": config_payload if isinstance(config_payload, dict) else model.config,
		"description": model.description,
		"metadata": metadata_value,
	}


def _encode_prompt(prompt: Prompt, env_keys: Set[str]) -> Dict[str, Any]:
	_collect_template_markers(prompt.template, env_keys)
	parameters_value = _encode_value(prompt.parameters, env_keys)
	if not isinstance(parameters_value, dict):
		parameters_value = {"value": parameters_value} if parameters_value is not None else {}
	metadata_value = _encode_value(prompt.metadata, env_keys)
	if not isinstance(metadata_value, dict):
		metadata_value = {"value": metadata_value} if metadata_value is not None else {}
	return {
		"name": prompt.name,
		"model": prompt.model,
		"template": prompt.template,
		"input": [_encode_prompt_field(field, env_keys) for field in prompt.input_fields],
		"output": [_encode_prompt_field(field, env_keys) for field in prompt.output_fields],
		"parameters": parameters_value,
		"metadata": metadata_value,
		"description": prompt.description,
	}


def _encode_prompt_field(field: PromptField, env_keys: Set[str]) -> Dict[str, Any]:
	metadata_value = _encode_value(field.metadata, env_keys)
	if not isinstance(metadata_value, dict):
		metadata_value = {"value": metadata_value} if metadata_value is not None else {}
	payload: Dict[str, Any] = {
		"name": field.name,
		"type": field.field_type,
		"required": field.required,
		"enum": list(field.enum or []),
		"description": field.description,
		"metadata": metadata_value,
	}
	if field.default is not None:
		payload["default"] = _encode_value(field.default, env_keys)
	return payload


def _encode_chain(chain: Chain, env_keys: Set[str], memory_names: Set[str]) -> Dict[str, Any]:
	encoded_steps = [
		_encode_workflow_node(node, env_keys, memory_names, chain.name) for node in chain.steps
	]
	metadata_encoded = _encode_value(chain.metadata, env_keys)
	if not isinstance(metadata_encoded, dict):
		metadata_encoded = {"value": metadata_encoded}
	return {
		"name": chain.name,
		"input_key": chain.input_key,
		"steps": encoded_steps,
		"metadata": metadata_encoded,
	}


def _encode_workflow_node(
	node: WorkflowNode,
	env_keys: Set[str],
	memory_names: Set[str],
	chain_name: str,
) -> Dict[str, Any]:
	if isinstance(node, ChainStep):
		return _encode_chain_step(node, env_keys, memory_names, chain_name)
	if isinstance(node, WorkflowIfBlock):
		payload: Dict[str, Any] = {
			"type": "if",
			"condition": _expression_to_runtime(node.condition),
			"condition_source": _expression_to_source(node.condition),
			"then": [_encode_workflow_node(child, env_keys, memory_names, chain_name) for child in node.then_steps],
			"elif": [
				{
					"condition": _expression_to_runtime(branch_condition),
					"condition_source": _expression_to_source(branch_condition),
					"steps": [_encode_workflow_node(child, env_keys, memory_names, chain_name) for child in branch_steps],
				}
				for branch_condition, branch_steps in node.elif_steps
			],
			"else": [_encode_workflow_node(child, env_keys, memory_names, chain_name) for child in node.else_steps],
		}
		return payload
	if isinstance(node, WorkflowForBlock):
		payload: Dict[str, Any] = {
			"type": "for",
			"loop_var": node.loop_var,
			"source_kind": node.source_kind,
			"body": [_encode_workflow_node(child, env_keys, memory_names, chain_name) for child in node.body],
		}
		if node.source_name:
			payload["source_name"] = node.source_name
		if node.source_expression is not None:
			payload["source_expression"] = _expression_to_runtime(node.source_expression)
			payload["source_expression_source"] = _expression_to_source(node.source_expression)
		if node.max_iterations:
			payload["max_iterations"] = int(node.max_iterations)
		return payload
	if isinstance(node, WorkflowWhileBlock):
		payload = {
			"type": "while",
			"condition": _expression_to_runtime(node.condition),
			"condition_source": _expression_to_source(node.condition),
			"body": [_encode_workflow_node(child, env_keys, memory_names, chain_name) for child in node.body],
		}
		if node.max_iterations:
			payload["max_iterations"] = int(node.max_iterations)
		return payload
	raise TypeError(f"Unsupported workflow node '{type(node).__name__}' in chain '{chain_name}'")


def _encode_chain_step(
	step: ChainStep,
	env_keys: Set[str],
	memory_names: Set[str],
	chain_name: str,
) -> Dict[str, Any]:
	options_encoded = _encode_value(step.options, env_keys)
	if not isinstance(options_encoded, dict):
		options_encoded = {"value": options_encoded}
	_validate_chain_memory_options(step.options, memory_names, chain_name, step.kind, step.target)
	payload: Dict[str, Any] = {
		"type": "step",
		"kind": step.kind,
		"target": step.target,
		"options": options_encoded,
		"stop_on_error": bool(step.stop_on_error),
	}
	if step.name:
		payload["name"] = step.name
	if step.evaluation:
		evaluation_payload: Dict[str, Any] = {
			"evaluators": list(step.evaluation.evaluators),
		}
		if step.evaluation.guardrail:
			evaluation_payload["guardrail"] = step.evaluation.guardrail
		payload["evaluation"] = evaluation_payload
	return payload


def _encode_evaluator(evaluator: Evaluator, env_keys: Set[str]) -> Dict[str, Any]:
	config_encoded = _encode_value(evaluator.config, env_keys)
	config_payload = config_encoded if isinstance(config_encoded, dict) else {}
	return {
		"name": evaluator.name,
		"kind": evaluator.kind,
		"provider": evaluator.provider,
		"config": config_payload,
	}


def _encode_metric(metric: Metric) -> Dict[str, Any]:
	return {
		"name": metric.name,
		"evaluator": metric.evaluator,
		"aggregation": metric.aggregation,
		"params": dict(metric.params or {}),
	}


def _encode_guardrail(guardrail: Guardrail) -> Dict[str, Any]:
	return {
		"name": guardrail.name,
		"evaluators": list(guardrail.evaluators),
		"action": guardrail.action,
		"message": guardrail.message,
	}


def _validate_chain_memory_options(
	options: Dict[str, Any],
	memory_names: Set[str],
	chain_name: str,
	step_kind: str,
	target: str,
) -> None:
	if not options:
		return
	for key in ("read_memory", "write_memory"):
		if key not in options:
			continue
		names = options[key]
		if isinstance(names, str):
			collection = [names]
		elif isinstance(names, list):
			collection = names
		else:
			continue
		missing = [name for name in collection if name not in memory_names]
		if missing:
			missing_str = ", ".join(sorted(missing))
			raise ValueError(
				f"Chain '{chain_name}' step '{step_kind}:{target}' references undefined memory: {missing_str}"
			)


def _encode_experiment(experiment: Experiment, env_keys: Set[str]) -> Dict[str, Any]:
	variants_payload: List[Dict[str, Any]] = []
	for variant in experiment.variants:
		config_encoded = _encode_value(variant.config, env_keys)
		if not isinstance(config_encoded, dict):
			config_encoded = {"value": config_encoded}
		variants_payload.append(
			{
				"name": variant.name,
				"slug": _slugify_identifier(variant.name),
				"target_type": variant.target_type,
				"target_name": variant.target_name,
				"config": config_encoded,
			}
		)
	metrics_payload: List[Dict[str, Any]] = []
	for metric in experiment.metrics:
		metadata_encoded = _encode_value(metric.metadata, env_keys)
		if not isinstance(metadata_encoded, dict):
			metadata_encoded = {"value": metadata_encoded}
		metrics_payload.append(
			{
				"name": metric.name,
				"slug": _slugify_identifier(metric.name),
				"source_kind": metric.source_kind,
				"source_name": metric.source_name,
				"goal": metric.goal,
				"metadata": metadata_encoded,
			}
		)
	metadata_value = _encode_value(experiment.metadata, env_keys)
	if not isinstance(metadata_value, dict):
		metadata_value = {"value": metadata_value} if metadata_value is not None else {}
	data_config: Optional[Dict[str, Any]] = None
	if isinstance(metadata_value, dict):
		data_block = metadata_value.pop("data", metadata_value.pop("dataset", None))
		if data_block is not None:
			try:
				data_config = _encode_experiment_data_config(data_block)
			except ValueError as exc:
				raise ValueError(f"Experiment '{experiment.name}' data configuration is invalid: {exc}") from exc
	return {
		"name": experiment.name,
		"slug": _slugify_identifier(experiment.name),
		"description": experiment.description,
		"variants": variants_payload,
		"metrics": metrics_payload,
		"metadata": metadata_value,
		"data_config": data_config,
	}


def _encode_experiment_data_config(raw: Any) -> Dict[str, Any]:
	if raw is None:
		return {}
	if isinstance(raw, str):
		return {"frame": raw}
	if not isinstance(raw, dict):
		raise ValueError("data config must be provided as a mapping or string frame name")
	config: Dict[str, Any] = {}
	frame_name = raw.get("frame") or raw.get("name")
	if frame_name:
		config["frame"] = str(frame_name)
	pipeline_value = raw.get("pipeline") or raw.get("frame_pipeline")
	if pipeline_value is not None:
		config["pipeline"] = pipeline_value
	features = _coerce_column_name_list(raw.get("features") or raw.get("feature_columns"))
	if features:
		config["features"] = features
	target_value = raw.get("target") or raw.get("label")
	if isinstance(target_value, dict):
		target_value = target_value.get("name")
	if target_value:
		config["target"] = str(target_value)
	time_value = raw.get("time") or raw.get("time_column") or raw.get("timestamp")
	if isinstance(time_value, dict):
		time_value = time_value.get("name")
	if time_value:
		config["time_column"] = str(time_value)
	group_columns = _coerce_column_name_list(
		raw.get("groups") or raw.get("group_columns") or raw.get("ids") or raw.get("identifiers")
	)
	if group_columns:
		config["group_columns"] = group_columns
	weight_value = raw.get("weight") or raw.get("weight_column") or raw.get("sample_weight")
	if isinstance(weight_value, dict):
		weight_value = weight_value.get("name")
	if weight_value:
		config["weight_column"] = str(weight_value)
	split_override = raw.get("splits") or raw.get("split")
	if isinstance(split_override, dict):
		normalized_splits = _normalize_split_mapping(split_override)
		if normalized_splits:
			config["splits"] = normalized_splits
	if not config.get("frame") and not config.get("pipeline"):
		raise ValueError("data config requires at least a 'frame' or 'pipeline' entry")
	return config


def _coerce_column_name_list(value: Any) -> List[str]:
	if not value:
		return []
	if isinstance(value, str):
		parts = [segment.strip() for segment in value.split(",") if segment.strip()]
		return parts or [value.strip()]
	if isinstance(value, dict):
		if "name" in value:
			return [str(value["name"])]
		value = value.get("columns") or value.get("names")
	if isinstance(value, (list, tuple, set)):
		names: List[str] = []
		for item in value:
			if isinstance(item, str):
				names.append(item)
			elif isinstance(item, dict) and item.get("name"):
				names.append(str(item["name"]))
		return names
	return []


def _normalize_split_mapping(raw: Dict[str, Any]) -> Dict[str, float]:
	splits: Dict[str, float] = {}
	for key, value in raw.items():
		name = str(key).strip()
		if not name:
			continue
		try:
			number = float(value)
		except (TypeError, ValueError):
			continue
		if number <= 0:
			continue
		splits[name] = number
	return splits


def _encode_crud_resource(resource: CrudResource, env_keys: Set[str]) -> Dict[str, Any]:
	select_fields = list(dict.fromkeys(resource.select_fields or []))
	mutable_fields = list(dict.fromkeys(resource.mutable_fields or []))
	primary_key = resource.primary_key or "id"
	if not mutable_fields:
		mutable_fields = [field for field in select_fields if field and field != primary_key]
	allowed: List[str] = []
	seen: Set[str] = set()
	for operation in resource.allowed_operations:
		candidate = str(operation or "").lower()
		if candidate in {"list", "retrieve", "create", "update", "delete"} and candidate not in seen:
			seen.add(candidate)
			allowed.append(candidate)
	if str(resource.source_type or "table").lower() != "table":
		allowed = [op for op in allowed if op in {"list", "retrieve"}]
	read_only = resource.read_only or not any(op in {"create", "update", "delete"} for op in allowed)
	default_limit = int(resource.default_limit or 100)
	max_limit = int(resource.max_limit or max(default_limit, 100))
	if default_limit <= 0:
		default_limit = 100
	if max_limit < default_limit:
		max_limit = default_limit
	label = resource.label or resource.name.replace("-", " ").title()
	return {
		"slug": resource.name,
		"label": label,
		"source_type": str(resource.source_type or "table").lower(),
		"source_name": resource.source_name,
		"primary_key": primary_key,
		"select_fields": select_fields,
		"mutable_fields": mutable_fields,
		"allowed_operations": allowed,
		"tenant_column": resource.tenant_column,
		"default_limit": default_limit,
		"max_limit": max_limit,
		"read_only": read_only,
	}


def _encode_variable(variable: VariableAssignment, env_keys: Set[str]) -> Dict[str, Any]:
	if isinstance(variable.value, FrameExpression):
		pipeline_payload = _encode_frame_expression_value(variable.value, env_keys)
		return {
			"name": variable.name,
			"value": {"__frame_pipeline__": pipeline_payload},
			"value_source": pipeline_payload.get("root"),
			"value_expr": None,
			"frame_pipeline": pipeline_payload,
		}
	return {
		"name": variable.name,
		"value": _encode_value(variable.value, env_keys),
		"value_source": _expression_to_source(variable.value),
		"value_expr": _expression_to_runtime(variable.value),
	}


def _encode_frame_expression_value(expression: FrameExpression, env_keys: Set[str]) -> Dict[str, Any]:
	if _FRAME_ANALYZER is None:
		raise FrameTypeError("Frame expressions require frame analyzer context during encoding")
	plan = _FRAME_ANALYZER.analyze(expression)
	return plan.to_payload(_expression_to_runtime, _expression_to_source)


def _encode_action_operation(
	operation: ActionOperationType,
	env_keys: Set[str],
	prompt_lookup: Dict[str, Prompt],
) -> Dict[str, Any]:
	if isinstance(operation, UpdateOperation):
		return {
			"type": "update",
			"table": operation.table,
			"set_expression": operation.set_expression,
			"where_expression": operation.where_expression,
		}
	if isinstance(operation, ToastOperation):
		return {"type": "toast", "message": operation.message}
	if isinstance(operation, GoToPageOperation):
		return {
			"type": "navigate",
			"page_name": operation.page_name,
		}
	if isinstance(operation, CallPythonOperation):
		return {
			"type": "python_call",
			"module": operation.module,
			"method": operation.method,
			"arguments": {
				key: _encode_value(value, env_keys) for key, value in operation.arguments.items()
			},
		}
	if isinstance(operation, AskConnectorOperation):
		return {
			"type": "connector_call",
			"name": operation.connector_name,
			"arguments": {
				key: _encode_value(value, env_keys) for key, value in operation.arguments.items()
			},
		}
	if isinstance(operation, RunChainOperation):
		return {
			"type": "chain_run",
			"name": operation.chain_name,
			"inputs": {
				key: _encode_value(value, env_keys) for key, value in operation.inputs.items()
			},
		}
	if isinstance(operation, RunPromptOperation):
		_validate_prompt_arguments(prompt_lookup, operation.prompt_name, operation.arguments)
		return {
			"type": "prompt_call",
			"prompt": operation.prompt_name,
			"arguments": {
				key: _encode_value(value, env_keys) for key, value in operation.arguments.items()
			},
		}
	if isinstance(operation, ActionOperation):
		return {"type": type(operation).__name__}
	return {"type": "operation"}


def _validate_prompt_arguments(
	prompt_lookup: Dict[str, Prompt],
	prompt_name: str,
	arguments: Dict[str, Expression],
) -> None:
	prompt = prompt_lookup.get(prompt_name)
	if prompt is None:
		raise ValueError(f"Prompt '{prompt_name}' is not defined")
	valid_names = {field.name for field in prompt.input_fields}
	required_names = {field.name for field in prompt.input_fields if field.required}
	provided = set(arguments.keys())
	missing = sorted(required_names - provided)
	if missing:
		raise ValueError(
			f"Prompt '{prompt_name}' is missing required inputs: {', '.join(missing)}"
		)
	extra = sorted(provided - valid_names)
	if extra:
		raise ValueError(
			f"Prompt '{prompt_name}' does not accept inputs: {', '.join(extra)}"
		)


def _encode_value(value: Any, env_keys: Set[str]) -> Any:
	if isinstance(value, ContextValue):
		marker = {
			"__context__": {
				"scope": value.scope,
				"path": list(value.path),
			}
		}
		if value.default is not None:
			marker["__context__"]["default"] = value.default
		if value.scope == "env" and value.path:
			env_keys.add(value.path[0])
		return marker
	if isinstance(value, Literal):
		return value.value
	if isinstance(value, FrameExpression):
		pipeline_payload = _encode_frame_expression_value(value, env_keys)
		return {"__frame_pipeline__": pipeline_payload}
	if isinstance(value, (NameRef, AttributeRef, BinaryOp, UnaryOp, CallExpression)):
		return _expression_to_source(value)
	if isinstance(value, list):
		return [_encode_value(item, env_keys) for item in value]
	if isinstance(value, dict):
		return {key: _encode_value(val, env_keys) for key, val in value.items()}
	if hasattr(value, "__dict__"):
		return _encode_value(value.__dict__, env_keys)
	return value


def _expression_to_source(expression: Optional[Expression]) -> Optional[str]:
	if expression is None:
		return None
	if isinstance(expression, Literal):
		return repr(expression.value)
	if isinstance(expression, NameRef):
		return expression.name
	if isinstance(expression, AttributeRef):
		return f"{expression.base}.{expression.attr}"
	if isinstance(expression, BinaryOp):
		left = _expression_to_source(expression.left) or ""
		right = _expression_to_source(expression.right) or ""
		return f"{left} {expression.op} {right}".strip()
	if isinstance(expression, UnaryOp):
		operand = _expression_to_source(expression.operand) or ""
		return f"{expression.op}{operand}".strip()
	if isinstance(expression, CallExpression):
		func = _expression_to_source(expression.function) or "call"
		args = ", ".join(
			arg for arg in [_expression_to_source(arg) for arg in expression.arguments] if arg is not None
		)
		return f"{func}({args})"
	if isinstance(expression, ContextValue):
		return None
	return str(expression)


def _expression_to_runtime(expression: Optional[Expression]) -> Optional[Dict[str, Any]]:
	if expression is None:
		return None
	if isinstance(expression, Literal):
		return {"type": "literal", "value": expression.value}
	if isinstance(expression, NameRef):
		return {"type": "name", "name": expression.name}
	if isinstance(expression, AttributeRef):
		path: List[str] = []
		if expression.base:
			path.extend(segment for segment in expression.base.split(".") if segment)
		if expression.attr:
			path.append(expression.attr)
		return {"type": "attribute", "path": path}
	if isinstance(expression, ContextValue):
		return {
			"type": "context",
			"scope": expression.scope,
			"path": list(expression.path),
			"default": expression.default,
		}
	if isinstance(expression, BinaryOp):
		return {
			"type": "binary",
			"op": expression.op,
			"left": _expression_to_runtime(expression.left),
			"right": _expression_to_runtime(expression.right),
		}
	if isinstance(expression, UnaryOp):
		return {
			"type": "unary",
			"op": expression.op,
			"operand": _expression_to_runtime(expression.operand),
		}
	if isinstance(expression, CallExpression):
		return {
			"type": "call",
			"function": _expression_to_runtime(expression.function),
			"arguments": [
				_expression_to_runtime(arg) for arg in expression.arguments
			],
		}
	return {"type": "literal", "value": _expression_to_source(expression)}


def _slugify_route(route: str) -> str:
	slug = route.strip("/") or "root"
	slug = slug.replace("/", "_")
	slug = re.sub(r"[^a-zA-Z0-9_]+", "_", slug)
	return slug.lower() or "page"


def _slugify_page_name(name: str) -> str:
	slug = name.strip()
	slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug)
	slug = slug.strip("_")
	return slug.lower() or "page"


def _slugify_identifier(value: str) -> str:
	slug = value.strip()
	slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug)
	slug = slug.strip("_")
	return slug.lower() or "item"


def _page_api_path(route: str) -> str:
	cleaned = route.strip("/")
	if not cleaned:
		cleaned = "root"
	return f"/api/pages/{cleaned}"


def _collect_template_markers(value: Any, env_keys: Set[str]) -> None:
	if isinstance(value, str):
		for match in _TEMPLATE_PATTERN.finditer(value):
			token = match.group(1).strip()
			if not token:
				continue
			if token.startswith("$"):
				env_keys.add(token[1:])
				continue
			if ":" in token:
				scope, _, path = token.partition(":")
				parts = [segment for segment in path.split(".") if segment]
				if scope == "env" and parts:
					env_keys.add(parts[0])
	elif isinstance(value, dict):
		for item in value.values():
			_collect_template_markers(item, env_keys)
	elif isinstance(value, list):
		for item in value:
			_collect_template_markers(item, env_keys)
