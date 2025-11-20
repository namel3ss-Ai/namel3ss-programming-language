"""
Utility functions for CLI operations.

This module provides common utility functions used across CLI commands,
including string manipulation, resource finding, and output formatting.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..ast import App, Chain, Experiment
from ..ml import get_default_model_registry


# Default model registry for fallback lookups
DEFAULT_MODEL_REGISTRY = get_default_model_registry()


def pluralize(word: str, count: int) -> str:
    """
    Simple English pluralization.
    
    Args:
        word: Word to pluralize
        count: Count determining singular vs plural
    
    Returns:
        Singular word if count==1, else word with 's' suffix
    
    Examples:
        >>> pluralize("dataset", 1)
        'dataset'
        >>> pluralize("dataset", 5)
        'datasets'
        >>> pluralize("chain", 0)
        'chains'
    """
    return word if count == 1 else f"{word}s"


def slugify_model_name(value: str) -> str:
    """
    Convert model name to filesystem-safe slug.
    
    Replaces non-alphanumeric characters with underscores and
    converts to lowercase for consistent naming.
    
    Args:
        value: Model name to slugify
    
    Returns:
        Slugified model name safe for filesystem use
    
    Examples:
        >>> slugify_model_name("GPT-4 Turbo")
        'gpt_4_turbo'
        >>> slugify_model_name("My-Model-v2.1")
        'my_model_v2_1'
        >>> slugify_model_name("___special___")
        'special'
    """
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return normalized.lower() or "model"


def find_first_n3_file() -> Optional[Path]:
    """
    Find first .n3 file in current working directory.
    
    Useful for commands that can infer source file when only
    one .n3 file is present.
    
    Returns:
        Path to first .n3 file (sorted alphabetically) or None
    
    Examples:
        >>> # In directory with app.n3
        >>> find_first_n3_file()  # doctest: +SKIP
        PosixPath('app.n3')
    """
    candidates = sorted(Path.cwd().glob('*.n3'))
    return candidates[0] if candidates else None


def get_program_root(source_path: Path) -> Path:
    """
    Determine program root directory from source path.
    
    For file paths, returns parent directory.
    For directory paths, returns the directory itself.
    
    Args:
        source_path: Source file or directory path
    
    Returns:
        Root directory for the program
    
    Examples:
        >>> get_program_root(Path("/workspace/app.n3"))
        PosixPath('/workspace')
        >>> get_program_root(Path("/workspace/"))
        PosixPath('/workspace')
    """
    return source_path.parent if source_path.is_file() else source_path


def find_chain(app: App, name: str) -> Optional[Chain]:
    """
    Find chain by name in app.
    
    Args:
        app: App to search
        name: Chain name to find
    
    Returns:
        Matching Chain or None if not found
    
    Examples:
        >>> app = App(chains=[Chain(name="process", ...)])
        >>> chain = find_chain(app, "process")
        >>> chain.name
        'process'
        >>> find_chain(app, "nonexistent")
        None
    """
    for chain in app.chains:
        if chain.name == name:
            return chain
    return None


def find_experiment(app: App, name: str) -> Optional[Experiment]:
    """
    Find experiment by name in app.
    
    Args:
        app: App to search
        name: Experiment name to find
    
    Returns:
        Matching Experiment or None if not found
    
    Examples:
        >>> app = App(experiments=[Experiment(name="test_variant", ...)])
        >>> exp = find_experiment(app, "test_variant")
        >>> exp.name
        'test_variant'
        >>> find_experiment(app, "nonexistent")
        None
    """
    for experiment in app.experiments:
        if experiment.name == name:
            return experiment
    return None


def resolve_model_spec(app: App, model_name: str) -> Dict[str, Any]:
    """
    Resolve model specification from app or registry.
    
    Searches app models first, then falls back to default model registry.
    Returns spec with type, framework, version, metrics, and metadata.
    
    Args:
        app: App containing model definitions
        model_name: Name of model to resolve
    
    Returns:
        Model specification dictionary with standardized fields
    
    Examples:
        >>> app = App(models=[...])
        >>> spec = resolve_model_spec(app, "gpt-4")
        >>> spec['type']
        'llm'
        >>> spec['framework']
        'openai'
    """
    # Search app models first
    for model in app.models:
        if model.name == model_name:
            registry_info = model.registry
            return {
                "type": model.model_type,
                "framework": model.engine or model.model_type,
                "version": registry_info.version or "v1",
                "metrics": dict(registry_info.metrics or {}),
                "metadata": dict(registry_info.metadata or {}),
            }
    
    # Fallback to default registry
    fallback = DEFAULT_MODEL_REGISTRY.get(model_name)
    if fallback:
        return dict(fallback)
    
    # Unknown model - return generic spec
    return {
        "type": "custom",
        "framework": "unknown",
        "version": "v1",
        "metrics": {},
        "metadata": {},
    }


def generate_backend_summary(app: App) -> List[str]:
    """
    Generate summary lines describing backend resources.
    
    Creates human-readable list of backend components including
    datasets, connectors, insights, models, templates, chains, and experiments.
    
    Args:
        app: App to summarize
    
    Returns:
        List of summary lines with counts and proper pluralization
    
    Examples:
        >>> app = App(datasets=[...], chains=[...], ...)
        >>> summary = generate_backend_summary(app)
        >>> summary[0]
        '✓ 3 datasets available'
        >>> summary[-1]
        '✓ 2 experiments queued'
    """
    # Collect unique connectors
    connectors = {
        str(ds.connector.connector_name or ds.source or ds.name)
        for ds in app.datasets
        if ds.connector is not None
    }
    connectors.update(conn.name for conn in app.connectors)
    
    # Count resources
    dataset_count = len(app.datasets)
    insight_count = len(app.insights)
    model_count = len(app.models)
    template_count = len(app.templates)
    chain_count = len(app.chains)
    experiment_count = len(app.experiments)
    
    return [
        f"✓ {dataset_count} {pluralize('dataset', dataset_count)} available",
        f"✓ {len(connectors)} {pluralize('connector', len(connectors))} registered",
        f"✓ {insight_count} {pluralize('insight', insight_count)} routed",
        f"✓ {model_count} {pluralize('model', model_count)} declared",
        f"✓ {template_count} {pluralize('template', template_count)} cached",
        f"✓ {chain_count} {pluralize('chain', chain_count)} composed",
        f"✓ {experiment_count} {pluralize('experiment', experiment_count)} queued",
    ]
