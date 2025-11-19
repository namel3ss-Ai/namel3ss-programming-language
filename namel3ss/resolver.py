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
    "training_jobs",
    "tuning_jobs",
    "indices",
    "rag_pipelines",
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
    "training_jobs": "training job",
    "tuning_jobs": "tuning job",
    "indices": "index",
    "rag_pipelines": "rag_pipeline",
}
    "chains": "chain",
    "experiments": "experiment",
    "crud_resources": "CRUD resource",
    "evaluators": "evaluator",
    "metrics": "metric",
    "guardrails": "guardrail",
    "training_jobs": "training job",
    "tuning_jobs": "tuning job",
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

    _validate_evaluations(resolved_modules)
    _validate_training_artifacts(resolved_modules)
    merged_app = _merge_apps(resolved_modules, root)
    check_app(merged_app, path=root.module.path)
    return ResolvedProgram(modules=resolved_modules, root=root, app=merged_app, language_version=language_version)


def _build_module_exports(module: Module) -> ModuleExports:
    app: Optional[App] = None
    if module.body and isinstance(module.body[0], App):
        app = module.body[0]

    exports_by_kind: Dict[str, Dict[str, Any]] = {kind: {} for kind in EXPORT_ATTRS}
    symbol_index: Dict[str, List[ExportedSymbol]] = {}
    if app is not None:
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

    return ModuleExports(
        app=app,
        has_explicit_app=module.has_explicit_app,
        exports_by_kind=exports_by_kind,
        symbol_index=symbol_index,
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
            for metric_name in job.metrics or []:
                if metric_name not in metric_owners:
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
