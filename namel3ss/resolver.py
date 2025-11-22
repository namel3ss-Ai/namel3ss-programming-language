"""Module/import resolution for Namel3ss programs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from namel3ss.ast import App, Module, Program, ChainStep, WorkflowIfBlock, WorkflowForBlock, WorkflowWhileBlock
from namel3ss.lang import LANGUAGE_VERSION, SUPPORTED_LANGUAGE_VERSIONS
from namel3ss.types import check_app
from namel3ss.ast.modules import Import
from namel3ss.errors import N3ResolutionError
from namel3ss.ast.expressions import FunctionDef, RuleDef


EXPORT_ATTRS: Tuple[str, ...] = (
    "variables",
    "datasets",
    "frames",
    "pages",
    "insights",
    "models",
    "connectors",
    "ai_models",
    "prompts",
    "memories",
    "templates",
    "chains",
    "experiments",
    "crud_resources",
    "evaluators",
    "metrics",
    "guardrails",
    "eval_suites",
    "training_jobs",
    "tuning_jobs",
    "indices",
    "rag_pipelines",
    "policies",
    "functions",
    "rules",
)

EXPORT_LABELS: Dict[str, str] = {
    "variables": "variable",
    "datasets": "dataset",
    "frames": "frame",
    "pages": "page",
    "insights": "insight",
    "models": "model",
    "connectors": "connector",
    "ai_models": "AI model",
    "prompts": "prompt",
    "memories": "memory",
    "templates": "template",
    "chains": "chain",
    "experiments": "experiment",
    "crud_resources": "crud resource",
    "evaluators": "evaluator",
    "metrics": "metric",
    "guardrails": "guardrail",
    "eval_suites": "eval_suite",
    "training_jobs": "training job",
    "tuning_jobs": "tuning job",
    "indices": "index",
    "rag_pipelines": "rag_pipeline",
    "policies": "policy",
    "functions": "function",
    "rules": "rule",
}


class ModuleResolutionError(N3ResolutionError):
    """Raised when module/import resolution fails."""


@dataclass
class ExportedSymbol:
    kind: str
    name: str
    value: Any


@dataclass
class ModuleExports:
    app: Optional[App]
    has_explicit_app: bool
    exports_by_kind: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    symbol_index: Dict[str, List[ExportedSymbol]] = field(default_factory=dict)

    def resolve_symbol(self, name: str) -> ExportedSymbol:
        matches = self.symbol_index.get(name, [])
        if not matches:
            raise ModuleResolutionError(f"Symbol '{name}' is not exported by the target module")
        if len(matches) > 1:
            kinds = ", ".join(sorted(symbol.kind for symbol in matches))
            raise ModuleResolutionError(f"Symbol '{name}' is ambiguous across kinds: {kinds}")
        return matches[0]


@dataclass
class ResolvedImport:
    statement: Import
    target_module: str
    alias: Optional[str]
    imported_symbols: Dict[str, ExportedSymbol]


@dataclass
class ResolvedModule:
    module: Module
    exports: ModuleExports
    imports: List[ResolvedImport]


@dataclass
class ResolvedProgram:
    modules: Dict[str, ResolvedModule]
    root: ResolvedModule
    app: App
    language_version: str


def resolve_program(program: Program, *, entry_path: Optional[str | Path] = None) -> ResolvedProgram:
    module_by_name: Dict[str, Module] = {}
    for module in program.modules:
        if not module.name:
            raise ModuleResolutionError("Module is missing a name; ensure module declarations or file naming are set")
        if module.name in module_by_name:
            raise ModuleResolutionError(f"Duplicate module name detected: {module.name!r}")
        module_by_name[module.name] = module

    resolved_modules: Dict[str, ResolvedModule] = {}
    for name, module in module_by_name.items():
        exports = _build_module_exports(module)
        resolved_modules[name] = ResolvedModule(module=module, exports=exports, imports=[])

    root_candidates = [resolved for resolved in resolved_modules.values() if resolved.module.has_explicit_app]
    if not root_candidates:
        raise ModuleResolutionError("Program does not define an app entrypoint; add an 'app \"Name\"' declaration")
    if len(root_candidates) > 1:
        names = ", ".join(sorted(candidate.module.name or candidate.module.path for candidate in root_candidates))
        raise ModuleResolutionError(f"Multiple root apps detected: {names}. Exactly one module may declare an app.")
    root = root_candidates[0]
    if entry_path:
        entry_resolved = str(Path(entry_path).resolve())
        if Path(root.module.path).resolve() != Path(entry_resolved):
            raise ModuleResolutionError(
                f"Entry file {entry_resolved} does not contain the root app (found in {root.module.path})"
            )

    language_version = _resolve_language_versions(resolved_modules)
    for resolved in resolved_modules.values():
        resolved.imports = _resolve_imports(resolved, resolved_modules)

    _validate_prompts(resolved_modules)
    _validate_chains(resolved_modules)
    _validate_evaluations(resolved_modules)
    _validate_eval_suites(resolved_modules)
    _validate_training_artifacts(resolved_modules)
    _validate_symbolic_expressions(resolved_modules)
    merged_app = _merge_apps(resolved_modules, root)
    check_app(merged_app, path=root.module.path)
    return ResolvedProgram(modules=resolved_modules, root=root, app=merged_app, language_version=language_version)


def _build_module_exports(module: Module) -> ModuleExports:
    """
    Build exports for a module.
    
    This function extracts all declarations from the module's App and builds
    an export registry for cross-module references.
    
    For backward compatibility, it also collects any declarations that are
    directly in module.body (not yet attached to the App).
    """
    app: Optional[App] = None
    if module.body and isinstance(module.body[0], App):
        app = module.body[0]

    exports_by_kind: Dict[str, Dict[str, Any]] = {kind: {} for kind in EXPORT_ATTRS}
    symbol_index: Dict[str, List[ExportedSymbol]] = {}
    
    if app is not None:
        # Collect from App collections (primary mechanism)
        for kind in EXPORT_ATTRS:
            collection = getattr(app, kind, [])
            for item in collection:
                name = getattr(item, "name", None)
                if not name:
                    continue
                registry = exports_by_kind[kind]
                if name in registry:
                    label = EXPORT_LABELS.get(kind, kind)
                    raise ModuleResolutionError(
                        f"Duplicate {label} '{name}' defined in module {module.name or module.path}"
                    )
                registry[name] = item
                symbol_index.setdefault(name, []).append(ExportedSymbol(kind=kind, name=name, value=item))
        
        # BACKWARD COMPATIBILITY: Collect any loose declarations from module.body
        # that weren't properly attached to the App (legacy parser behavior)
        _collect_loose_declarations_into_app(module, app, exports_by_kind, symbol_index)

    return ModuleExports(
        app=app,
        has_explicit_app=module.has_explicit_app,
        exports_by_kind=exports_by_kind,
        symbol_index=symbol_index,
    )


def _collect_loose_declarations_into_app(
    module: Module,
    app: App,
    exports_by_kind: Dict[str, Dict[str, Any]],
    symbol_index: Dict[str, List[ExportedSymbol]],
) -> None:
    """
    Collect any declarations from module.body that aren't in the App.
    
    This is a backward compatibility mechanism for code that might bypass
    the normal App attachment process. It also serves as a safety net.
    """
    from namel3ss.ast import (
        Dataset, Frame, Page, Prompt, Chain, Memory, Template,
        Connector, Model, AgentDefinition, GraphDefinition, 
        IndexDefinition, RagPipelineDefinition,
        LLMDefinition, ToolDefinition, TrainingJob, TuningJob,
        FunctionDef, RuleDef, KnowledgeModule, LogicQuery,
    )
    
    # Import PolicyDefinition conditionally - may not be in __init__
    try:
        from namel3ss.ast import PolicyDefinition
    except ImportError:
        from namel3ss.ast.policy import PolicyDefinition
    
    # Map AST types to collection names
    type_to_collection = {
        Dataset: 'datasets',
        Frame: 'frames',
        Page: 'pages',
        Prompt: 'prompts',
        Chain: 'chains',
        Memory: 'memories',
        Template: 'templates',
        Connector: 'connectors',
        Model: 'models',
        AgentDefinition: 'agents',
        GraphDefinition: 'graphs',
        IndexDefinition: 'indices',
        RagPipelineDefinition: 'rag_pipelines',
        PolicyDefinition: 'policies',
        LLMDefinition: 'llms',
        ToolDefinition: 'tools',
        TrainingJob: 'training_jobs',
        TuningJob: 'tuning_jobs',
        FunctionDef: 'functions',
        RuleDef: 'rules',
        KnowledgeModule: 'knowledge_modules',
        LogicQuery: 'queries',
    }
    
    # Iterate through module.body looking for declarations
    for item in module.body:
        # Skip the App itself
        if isinstance(item, App):
            continue
        
        # Check if this is a known declaration type
        item_type = type(item)
        collection_name = type_to_collection.get(item_type)
        
        if collection_name:
            name = getattr(item, "name", None)
            if not name:
                continue
            
            # Check if already in exports (attached to App)
            registry = exports_by_kind.get(collection_name, {})
            if name in registry:
                # Already exported from App, skip
                continue
            
            # Not in App yet - attach it and add to exports
            collection = getattr(app, collection_name, None)
            if collection is not None:
                collection.append(item)
                registry[name] = item
                symbol_index.setdefault(name, []).append(
                    ExportedSymbol(kind=collection_name, name=name, value=item)
                )


def _resolve_imports(
    resolved_module: ResolvedModule,
    modules: Dict[str, ResolvedModule],
) -> List[ResolvedImport]:
    imports: List[ResolvedImport] = []
    for entry in resolved_module.module.imports:
        target = modules.get(entry.module)
        if target is None:
            raise ModuleResolutionError(
                f"Module '{resolved_module.module.name}' imports unknown module '{entry.module}'"
            )
        imported: Dict[str, ExportedSymbol] = {}
        if entry.names:
            for imported_name in entry.names:
                symbol = target.exports.resolve_symbol(imported_name.name)
                alias = imported_name.alias or imported_name.name
                if alias in imported:
                    raise ModuleResolutionError(
                        f"Duplicate imported name '{alias}' in module '{resolved_module.module.name}'"
                    )
                imported[alias] = symbol
            alias_value: Optional[str] = None
        else:
            alias_value = entry.alias or entry.module
        imports.append(
            ResolvedImport(
                statement=entry,
                target_module=target.module.name or target.module.path,
                alias=alias_value,
                imported_symbols=imported,
            )
        )
    return imports


def _merge_apps(modules: Dict[str, ResolvedModule], root: ResolvedModule) -> App:
    root_app = root.exports.app
    if root_app is None:
        raise ModuleResolutionError("Root module does not contain an app definition")
    merged = deepcopy(root_app)
    owner_map: Dict[str, Dict[str, str]] = {
        kind: {name: root.module.name or root.module.path for name in items}
        for kind, items in root.exports.exports_by_kind.items()
    }
    for name, resolved in modules.items():
        if resolved is root:
            continue
        app = resolved.exports.app
        if app is None:
            continue
        for kind in EXPORT_ATTRS:
            items = resolved.exports.exports_by_kind.get(kind, {})
            if not items:
                continue
            registry = owner_map.setdefault(kind, {})
            target_collection: List[Any] = getattr(merged, kind)
            for symbol_name, value in items.items():
                if symbol_name in registry:
                    label = EXPORT_LABELS.get(kind, kind)
                    owner = registry[symbol_name]
                    raise ModuleResolutionError(
                        f"Duplicate {label} '{symbol_name}' exported by modules '{owner}' and '{name}'"
                    )
                registry[symbol_name] = name
                target_collection.append(deepcopy(value))
    return merged


def _collect_global_symbols(resolved_modules: Dict[str, ResolvedModule], kind: str) -> Dict[str, str]:
    owners: Dict[str, str] = {}
    for module_name, resolved in resolved_modules.items():
        items = resolved.exports.exports_by_kind.get(kind, {})
        for symbol_name in items:
            if symbol_name in owners:
                label = EXPORT_LABELS.get(kind, kind)
                raise ModuleResolutionError(
                    f"Duplicate {label} '{symbol_name}' exported by modules '{owners[symbol_name]}' and '{module_name}'"
                )
            owners[symbol_name] = module_name
    return owners


def _validate_evaluations(resolved_modules: Dict[str, ResolvedModule]) -> None:
    evaluator_owners = _collect_global_symbols(resolved_modules, "evaluators")
    guardrail_owners = _collect_global_symbols(resolved_modules, "guardrails")
    for module_name, resolved in resolved_modules.items():
        metrics = resolved.exports.exports_by_kind.get("metrics", {})
        for metric in metrics.values():
            if metric.evaluator not in evaluator_owners:
                raise ModuleResolutionError(
                    f"Metric '{metric.name}' in module '{module_name}' references unknown evaluator '{metric.evaluator}'"
                )
        guardrails = resolved.exports.exports_by_kind.get("guardrails", {})
        for guardrail in guardrails.values():
            for evaluator_name in guardrail.evaluators:
                if evaluator_name not in evaluator_owners:
                    raise ModuleResolutionError(
                        f"Guardrail '{guardrail.name}' in module '{module_name}' references unknown evaluator '{evaluator_name}'"
                    )
        chains = resolved.exports.exports_by_kind.get("chains", {})
        for chain in chains.values():
            for step in _iter_chain_steps(chain.steps):
                evaluation = getattr(step, "evaluation", None)
                if evaluation is None:
                    continue
                for evaluator_name in evaluation.evaluators:
                    if evaluator_name not in evaluator_owners:
                        raise ModuleResolutionError(
                            f"Chain '{chain.name}' step '{step.name or step.target}' references unknown evaluator '{evaluator_name}'"
                        )
                if evaluation.guardrail and evaluation.guardrail not in guardrail_owners:
                    raise ModuleResolutionError(
                        f"Chain '{chain.name}' step '{step.name or step.target}' references unknown guardrail '{evaluation.guardrail}'"
                    )


def _validate_eval_suites(resolved_modules: Dict[str, ResolvedModule]) -> None:
    """Validate eval_suite references to datasets, chains, and LLMs."""
    dataset_owners = _collect_global_symbols(resolved_modules, "datasets")
    frame_owners = _collect_global_symbols(resolved_modules, "frames")
    chain_owners = _collect_global_symbols(resolved_modules, "chains")
    ai_model_owners = _collect_global_symbols(resolved_modules, "ai_models")
    llm_owners = _collect_global_symbols(resolved_modules, "llms")
    
    def _has_dataset(name: str) -> bool:
        return name in dataset_owners or name in frame_owners
    
    def _has_llm(name: str) -> bool:
        return name in ai_model_owners or name in llm_owners
    
    # Known metric types - basic validation
    known_metric_types = {
        "builtin_latency",
        "builtin_cost",
        "ragas_relevance",
        "ragas_context_precision",
        "ragas_context_recall",
        "ragas_faithfulness",
        "ragas_answer_similarity",
        "ragas_answer_correctness",
    }
    
    for module_name, resolved in resolved_modules.items():
        eval_suites = resolved.exports.exports_by_kind.get("eval_suites", {})
        for suite in eval_suites.values():
            # Validate dataset reference
            if not _has_dataset(suite.dataset_name):
                raise ModuleResolutionError(
                    f"eval_suite '{suite.name}' in module '{module_name}' references unknown dataset or frame '{suite.dataset_name}'"
                )
            
            # Validate target_chain reference
            if suite.target_chain_name not in chain_owners:
                raise ModuleResolutionError(
                    f"eval_suite '{suite.name}' in module '{module_name}' references unknown chain '{suite.target_chain_name}'"
                )
            
            # Validate judge_llm reference if specified
            if suite.judge_llm_name and not _has_llm(suite.judge_llm_name):
                raise ModuleResolutionError(
                    f"eval_suite '{suite.name}' in module '{module_name}' references unknown LLM '{suite.judge_llm_name}'"
                )
            
            # Validate rubric consistency
            if suite.rubric and not suite.judge_llm_name:
                raise ModuleResolutionError(
                    f"eval_suite '{suite.name}' in module '{module_name}' defines rubric but no judge_llm"
                )
            
            # Validate metrics
            for metric_spec in suite.metrics:
                if not metric_spec.name:
                    raise ModuleResolutionError(
                        f"eval_suite '{suite.name}' in module '{module_name}' has metric with empty name"
                    )
                if not metric_spec.type:
                    raise ModuleResolutionError(
                        f"eval_suite '{suite.name}' metric '{metric_spec.name}' in module '{module_name}' has no type"
                    )
                # Warn about unknown metric types (but don't fail - allows custom metrics)
                if not (metric_spec.type in known_metric_types or 
                       metric_spec.type.startswith("custom_") or
                       metric_spec.type.startswith("ragas_") or
                       metric_spec.type.startswith("builtin_")):
                    # Log warning but don't fail - extensibility
                    pass


def _validate_training_artifacts(resolved_modules: Dict[str, ResolvedModule]) -> None:
    model_owners = _collect_global_symbols(resolved_modules, "models")
    ai_model_owners = _collect_global_symbols(resolved_modules, "ai_models")
    dataset_owners = _collect_global_symbols(resolved_modules, "datasets")
    frame_owners = _collect_global_symbols(resolved_modules, "frames")
    metric_owners = _collect_global_symbols(resolved_modules, "metrics")
    training_owners = _collect_global_symbols(resolved_modules, "training_jobs")
    tuning_owners = _collect_global_symbols(resolved_modules, "tuning_jobs")

    def _has_model(name: str) -> bool:
        return name in model_owners or name in ai_model_owners

    def _has_dataset(name: str) -> bool:
        return name in dataset_owners or name in frame_owners
    
    def _get_dataset_fields(dataset_name: str) -> Set[str]:
        """Extract field names from a dataset or frame schema."""
        fields: Set[str] = set()
        
        # Check in datasets
        for resolved in resolved_modules.values():
            datasets = resolved.exports.exports_by_kind.get("datasets", {})
            if dataset_name in datasets:
                dataset = datasets[dataset_name]
                # Extract from schema fields
                if hasattr(dataset, 'schema') and dataset.schema:
                    fields.update(field.name for field in dataset.schema if hasattr(field, 'name'))
                # Extract from features
                if hasattr(dataset, 'features') and dataset.features:
                    fields.update(feature.name for feature in dataset.features if hasattr(feature, 'name'))
                return fields
        
        # Check in frames
        for resolved in resolved_modules.values():
            frames = resolved.exports.exports_by_kind.get("frames", {})
            if dataset_name in frames:
                frame = frames[dataset_name]
                # Extract from columns
                if hasattr(frame, 'columns') and frame.columns:
                    fields.update(col.name for col in frame.columns if hasattr(col, 'name'))
                return fields
        
        return fields

    for module_name, resolved in resolved_modules.items():
        training_jobs = resolved.exports.exports_by_kind.get("training_jobs", {})
        for job in training_jobs.values():
            if job.model and not _has_model(job.model):
                raise ModuleResolutionError(
                    f"Training job '{job.name}' in module '{module_name}' references unknown model '{job.model}'"
                )
            if job.dataset and not _has_dataset(job.dataset):
                raise ModuleResolutionError(
                    f"Training job '{job.name}' in module '{module_name}' references unknown dataset or frame '{job.dataset}'"
                )
            
            # Validate target and features against dataset schema
            if job.dataset and (job.target or job.features):
                dataset_fields = _get_dataset_fields(job.dataset)
                if dataset_fields:  # Only validate if schema is available
                    if job.target and job.target not in dataset_fields:
                        raise ModuleResolutionError(
                            f"Training job '{job.name}' in module '{module_name}' specifies target field '{job.target}' "
                            f"which is not in dataset '{job.dataset}' schema. Available fields: {sorted(dataset_fields)}"
                        )
                    for feature in job.features or []:
                        if feature not in dataset_fields:
                            raise ModuleResolutionError(
                                f"Training job '{job.name}' in module '{module_name}' specifies feature '{feature}' "
                                f"which is not in dataset '{job.dataset}' schema. Available fields: {sorted(dataset_fields)}"
                            )
            
            for metric_name in job.metrics or []:
                if metric_name not in metric_owners:
                    # Allow built-in metric names without validation
                    if not metric_name.lower() in {'accuracy', 'precision', 'recall', 'f1', 'auc', 'loss', 'mse', 'mae', 'rmse', 'r2'}:
                        raise ModuleResolutionError(
                            f"Training job '{job.name}' in module '{module_name}' references unknown metric '{metric_name}'"
                        )

        tuning_jobs = resolved.exports.exports_by_kind.get("tuning_jobs", {})
        for job in tuning_jobs.values():
            if job.training_job and job.training_job not in training_owners:
                raise ModuleResolutionError(
                    f"Tuning job '{job.name}' in module '{module_name}' references unknown training job '{job.training_job}'"
                )
            if job.objective_metric and job.objective_metric not in metric_owners:
                raise ModuleResolutionError(
                    f"Tuning job '{job.name}' in module '{module_name}' references unknown metric '{job.objective_metric}'"
                )

        experiments = resolved.exports.exports_by_kind.get("experiments", {})
        for experiment in experiments.values():
            for training_name in experiment.training_jobs:
                if training_name not in training_owners:
                    raise ModuleResolutionError(
                        f"Experiment '{experiment.name}' in module '{module_name}' references unknown training job '{training_name}'"
                    )
            for tuning_name in experiment.tuning_jobs:
                if tuning_name not in tuning_owners:
                    raise ModuleResolutionError(
                        f"Experiment '{experiment.name}' in module '{module_name}' references unknown tuning job '{tuning_name}'"
                    )
            for dataset_name in experiment.eval_datasets:
                if not _has_dataset(dataset_name):
                    raise ModuleResolutionError(
                        f"Experiment '{experiment.name}' in module '{module_name}' references unknown evaluation dataset '{dataset_name}'"
                    )
            for metric_name in experiment.eval_metrics:
                if metric_name not in metric_owners:
                    raise ModuleResolutionError(
                        f"Experiment '{experiment.name}' in module '{module_name}' references unknown evaluation metric '{metric_name}'"
                    )
            comparison = getattr(experiment, "comparison", None)
            if comparison:
                if comparison.baseline_model and not _has_model(comparison.baseline_model):
                    raise ModuleResolutionError(
                        f"Experiment '{experiment.name}' baseline model '{comparison.baseline_model}' is not defined"
                    )
                if comparison.best_of and comparison.best_of not in tuning_owners and comparison.best_of not in training_owners:
                    raise ModuleResolutionError(
                        f"Experiment '{experiment.name}' comparison references unknown job '{comparison.best_of}'"
                    )
                for challenger in comparison.challengers:
                    if challenger not in tuning_owners and challenger not in training_owners:
                        raise ModuleResolutionError(
                            f"Experiment '{experiment.name}' comparison challenger '{challenger}' is not defined"
                        )


def _validate_prompts(resolved_modules: Dict[str, ResolvedModule]) -> None:
    """
    Validate structured prompts:
    1. Check that args and output_schema are well-formed
    2. Validate template placeholders match defined args
    3. Check that model references exist
    """
    import re
    from namel3ss.ast import Prompt, OutputSchema
    
    ai_model_owners = _collect_global_symbols(resolved_modules, "ai_models")
    
    for module_name, resolved in resolved_modules.items():
        prompts = resolved.exports.exports_by_kind.get("prompts", {})
        
        for prompt in prompts.values():
            if not isinstance(prompt, Prompt):
                continue
            
            # Validate model reference
            if prompt.model and prompt.model not in ai_model_owners:
                raise ModuleResolutionError(
                    f"Prompt '{prompt.name}' in module '{module_name}' references unknown model '{prompt.model}'"
                )
            
            # Validate args and output_schema if present (structured prompts)
            if prompt.args or prompt.output_schema:
                _validate_structured_prompt(prompt, module_name)


def _validate_chains(resolved_modules: Dict[str, ResolvedModule]) -> None:
    """
    Validate chains:
    1. Check that referenced policies exist
    2. Validate step references
    """
    from namel3ss.ast import Chain
    
    policy_owners = _collect_global_symbols(resolved_modules, "policies")
    
    for module_name, resolved in resolved_modules.items():
        chains = resolved.exports.exports_by_kind.get("chains", {})
        
        for chain in chains.values():
            if not isinstance(chain, Chain):
                continue
            
            # Validate policy reference if specified
            if chain.policy_name and chain.policy_name not in policy_owners:
                raise ModuleResolutionError(
                    f"Chain '{chain.name}' in module '{module_name}' references unknown policy '{chain.policy_name}'"
                )


def _validate_structured_prompt(prompt, module_name: str) -> None:
    """Validate a structured prompt's args, output_schema, and template."""
    import re
    from namel3ss.ast import OutputFieldType
    
    # Validate arg names are unique
    if prompt.args:
        arg_names = [arg.name for arg in prompt.args]
        duplicates = [name for name in arg_names if arg_names.count(name) > 1]
        if duplicates:
            raise ModuleResolutionError(
                f"Prompt '{prompt.name}' in module '{module_name}' has duplicate argument names: {', '.join(set(duplicates))}"
            )
        
        # Validate arg types
        valid_arg_types = {'string', 'int', 'float', 'bool', 'list', 'object'}
        for arg in prompt.args:
            base_type = arg.arg_type.split('[')[0]  # Handle list[T]
            if base_type not in valid_arg_types:
                raise ModuleResolutionError(
                    f"Prompt '{prompt.name}' in module '{module_name}' has invalid arg type '{arg.arg_type}' for argument '{arg.name}'"
                )
    
    # Validate output_schema
    if prompt.output_schema:
        _validate_output_schema(prompt.output_schema, prompt.name, module_name)
    
    # Validate template placeholders match args
    if prompt.template and prompt.args:
        _validate_template_placeholders(prompt.template, prompt.args, prompt.name, module_name)


def _validate_output_schema(schema, prompt_name: str, module_name: str) -> None:
    """Validate output schema field types and enums."""
    from namel3ss.ast import OutputFieldType
    
    if not schema.fields:
        raise ModuleResolutionError(
            f"Prompt '{prompt_name}' in module '{module_name}' has empty output_schema"
        )
    
    # Validate field names are unique
    field_names = [f.name for f in schema.fields]
    duplicates = [name for name in field_names if field_names.count(name) > 1]
    if duplicates:
        raise ModuleResolutionError(
            f"Prompt '{prompt_name}' in module '{module_name}' has duplicate output field names: {', '.join(set(duplicates))}"
        )
    
    # Validate each field type
    for field in schema.fields:
        _validate_output_field_type(field.field_type, field.name, prompt_name, module_name)


def _validate_output_field_type(field_type, field_name: str, prompt_name: str, module_name: str) -> None:
    """Recursively validate an output field type."""
    from namel3ss.ast import OutputFieldType
    
    # Validate base type
    valid_base_types = {'string', 'int', 'float', 'bool', 'list', 'object', 'enum'}
    if field_type.base_type not in valid_base_types:
        raise ModuleResolutionError(
            f"Prompt '{prompt_name}' in module '{module_name}' field '{field_name}' has invalid type '{field_type.base_type}'"
        )
    
    # Validate enum has values
    if field_type.base_type == 'enum':
        if not field_type.enum_values:
            raise ModuleResolutionError(
                f"Prompt '{prompt_name}' in module '{module_name}' field '{field_name}' has enum type with no values"
            )
        # Check all enum values are strings
        for val in field_type.enum_values:
            if not isinstance(val, str):
                raise ModuleResolutionError(
                    f"Prompt '{prompt_name}' in module '{module_name}' field '{field_name}' enum values must be strings"
                )
        # Check for duplicates
        if len(field_type.enum_values) != len(set(field_type.enum_values)):
            raise ModuleResolutionError(
                f"Prompt '{prompt_name}' in module '{module_name}' field '{field_name}' has duplicate enum values"
            )
    
    # Validate list has element type
    if field_type.base_type == 'list':
        if not field_type.element_type:
            raise ModuleResolutionError(
                f"Prompt '{prompt_name}' in module '{module_name}' field '{field_name}' has list type without element type"
            )
        _validate_output_field_type(field_type.element_type, f"{field_name}[]", prompt_name, module_name)
    
    # Validate nested object fields
    if field_type.base_type == 'object' and field_type.nested_fields:
        for nested_field in field_type.nested_fields:
            _validate_output_field_type(nested_field.field_type, f"{field_name}.{nested_field.name}", prompt_name, module_name)


def _validate_template_placeholders(template: str, args: list, prompt_name: str, module_name: str) -> None:
    """Validate that template placeholders reference defined args."""
    import re
    
    # Extract placeholders: {arg_name}
    placeholder_pattern = r'\{(\w+)\}'
    placeholders = set(re.findall(placeholder_pattern, template))
    
    # Get defined arg names
    arg_names = {arg.name for arg in args}
    
    # Check for undefined placeholders
    undefined = placeholders - arg_names
    if undefined:
        raise ModuleResolutionError(
            f"Prompt '{prompt_name}' in module '{module_name}' template references undefined arguments: {', '.join(sorted(undefined))}"
        )
    
    # Optionally warn about unused args (not an error, just informational)
    # unused = arg_names - placeholders
    # if unused:
    #     # Could log a warning here


def _validate_symbolic_expressions(resolved_modules: Dict[str, ResolvedModule]) -> None:
    """Validate symbolic expressions (functions and rules) across all modules."""
    try:
        from namel3ss.resolver_symbolic import validate_symbolic_expressions
    except ImportError:
        # If resolver_symbolic isn't available, skip validation
        return
    
    # Collect all functions and rules from all modules
    all_functions: List[FunctionDef] = []
    all_rules: List[RuleDef] = []
    
    for resolved in resolved_modules.values():
        if not resolved.exports.app:
            continue
        
        app = resolved.exports.app
        
        # Collect functions and rules from App fields
        all_functions.extend(app.functions)
        all_rules.extend(app.rules)
        
        # Note: app.body was from legacy parser and is not used in the new parser
        # All declarations are now in App collection fields
    
    # Validate collected expressions
    if all_functions or all_rules:
        validate_symbolic_expressions(all_functions, all_rules)


def _iter_chain_steps(nodes: List[Any]) -> List[ChainStep]:
    steps: List[ChainStep] = []
    for node in nodes or []:
        if isinstance(node, ChainStep):
            steps.append(node)
            continue
        if isinstance(node, WorkflowIfBlock):
            steps.extend(_iter_chain_steps(node.then_steps))
            for _, branch_steps in node.elif_steps:
                steps.extend(_iter_chain_steps(branch_steps))
            steps.extend(_iter_chain_steps(node.else_steps))
        elif isinstance(node, WorkflowForBlock):
            steps.extend(_iter_chain_steps(node.body))
        elif isinstance(node, WorkflowWhileBlock):
            steps.extend(_iter_chain_steps(node.body))
    return steps


def _resolve_language_versions(resolved_modules: Dict[str, ResolvedModule]) -> str:
    versions: Set[str] = set()
    for resolved in resolved_modules.values():
        declared = resolved.module.language_version or LANGUAGE_VERSION
        if declared not in SUPPORTED_LANGUAGE_VERSIONS:
            raise ModuleResolutionError(
                f"Module '{resolved.module.name}' targets unsupported language version '{declared}'."
            )
        resolved.module.language_version = declared
        versions.add(declared)
    if len(versions) > 1:
        values = ", ".join(sorted(versions))
        raise ModuleResolutionError(
            f"Modules declare conflicting language versions: {values}. Use a single language spec version across the project."
        )
    return versions.pop() if versions else LANGUAGE_VERSION


__all__ = [
    "ModuleResolutionError",
    "ModuleExports",
    "ResolvedImport",
    "ResolvedModule",
    "ResolvedProgram",
    "resolve_program",
]
