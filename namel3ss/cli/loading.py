"""
Program and module loading for CLI operations.

This module handles loading N3 programs, resolving dependencies,
loading runtime modules, and managing the runtime module cache.
"""

import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ..ast import App
from ..codegen import generate_backend
from ..config import WorkspaceDefaults, extract_connector_config
from ..loader import load_program
from ..parser import N3SyntaxError
from ..resolver import ModuleResolutionError, resolve_program
from ..types import N3TypeError
from .errors import CLIError, CLIFileNotFoundError, CLIValidationError, format_cli_error
from .utils import get_program_root


# Global cache for runtime modules to avoid regeneration
_RUNTIME_CACHE: Dict[str, Any] = {}


def load_n3_app(source_path: Path) -> App:
    """
    Load and resolve N3 application from source file.
    
    Handles full pipeline: loading, parsing, dependency resolution,
    and type checking. Provides user-friendly error messages for
    common failure modes.
    
    Args:
        source_path: Path to .n3 source file
    
    Returns:
        Resolved App AST
    
    Raises:
        CLIFileNotFoundError: If source file doesn't exist
        CLIError: For parse, resolution, or type errors
    
    Examples:
        >>> app = load_n3_app(Path("app.n3"))  # doctest: +SKIP
        >>> print(f"Loaded {app.name}")
    """
    # Validate file exists
    if not source_path.exists():
        raise CLIFileNotFoundError(
            f"Source file not found: {source_path}",
            hint="Check the file path and try again"
        )
    
    # Load and resolve program
    try:
        project_root = get_program_root(source_path)
        program = load_program(project_root)
        resolved = resolve_program(program, entry_path=source_path)
        return resolved.app
        
    except FileNotFoundError as exc:
        raise CLIFileNotFoundError(
            f"File not found: {exc}",
            hint="Ensure all imported files exist"
        ) from exc
        
    except ModuleResolutionError as exc:
        # Module resolution errors have custom formatting
        print(format_cli_error(exc), file=sys.stderr)
        sys.exit(1)
        
    except N3SyntaxError as exc:
        # Syntax errors have custom formatting
        print(format_cli_error(exc), file=sys.stderr)
        sys.exit(1)
        
    except N3TypeError as exc:
        # Type errors have custom formatting
        print(format_cli_error(exc), file=sys.stderr)
        sys.exit(1)


def clear_generated_module_cache() -> None:
    """
    Clear Python import cache for generated modules.
    
    Removes 'generated' module and all submodules from sys.modules
    to force reimport on next load_runtime_module call.
    
    Used when regenerating backend to ensure fresh imports.
    
    Examples:
        >>> clear_generated_module_cache()
        >>> # Next import will load fresh generated code
    """
    for name in list(sys.modules):
        if name == "generated" or name.startswith("generated."):
            sys.modules.pop(name, None)


def load_runtime_module(app: App, cache_key: str) -> Any:
    """
    Load runtime module for app, generating if needed.
    
    Manages runtime module cache to avoid regenerating backend
    for repeated operations. Generates ephemeral backend in
    temporary directory if not cached.
    
    Args:
        app: App to load runtime for
        cache_key: Unique key for caching (e.g., app name or source path)
    
    Returns:
        Imported runtime module with app runtime functions
    
    Raises:
        CLIError: If backend generation or import fails
    
    Examples:
        >>> app = App(...)
        >>> runtime = load_runtime_module(app, "my_app")
        >>> # Use runtime.execute_chain, etc.
    """
    # Check cache first
    runtime = _RUNTIME_CACHE.get(cache_key)
    if runtime is not None:
        return runtime
    
    # Generate backend in temp directory
    backend_dir = tempfile.mkdtemp(prefix="namel3ss_cli_backend_")
    
    try:
        # Use default connector config for ephemeral runtime
        connector_cfg = extract_connector_config(None, WorkspaceDefaults())
        
        generate_backend(
            app,
            backend_dir,
            embed_insights=False,
            enable_realtime=False,
            connector_config=connector_cfg
        )
        
        # Clear cached modules to force fresh import
        clear_generated_module_cache()
        
        # Add backend to sys.path if not present
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        
        # Import runtime module
        runtime = importlib.import_module("generated.runtime")
        
        # Cache for future use
        _RUNTIME_CACHE[cache_key] = runtime
        
        return runtime
        
    except Exception as exc:
        raise CLIError(
            f"Failed to load runtime module: {exc}",
            code="CLI_RUNTIME_LOAD_ERROR",
            hint="Check that backend generation succeeded",
            context={'backend_dir': backend_dir, 'cache_key': cache_key}
        ) from exc


def load_json_argument(
    inline_value: Optional[str],
    file_path: Optional[str],
    label: str,
) -> Dict[str, Any]:
    """
    Load JSON object from inline string or file.
    
    Supports two input modes:
    1. Inline JSON string (e.g., --inputs '{"key": "value"}')
    2. JSON file path (e.g., --inputs-file inputs.json)
    
    Args:
        inline_value: Inline JSON string
        file_path: Path to JSON file
        label: Argument name for error messages (e.g., 'inputs', 'config')
    
    Returns:
        Parsed JSON dictionary
    
    Raises:
        CLIValidationError: If both provided, file not found, or invalid JSON
    
    Examples:
        >>> load_json_argument('{"x": 1}', None, "inputs")
        {'x': 1}
        >>> load_json_argument(None, "config.json", "config")  # doctest: +SKIP
        {'key': 'value', ...}
    """
    # Validate mutual exclusivity
    if inline_value and file_path:
        raise CLIValidationError(
            f"Specify either --{label} or --{label}-file, not both",
            hint=f"Use --{label} for inline JSON or --{label}-file for file path"
        )
    
    # Get JSON data
    data: Optional[str] = inline_value
    if file_path:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise CLIFileNotFoundError(
                f"{label.capitalize()} file not found: {file_path}",
                hint="Check the file path and try again"
            )
        try:
            data = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CLIValidationError(
                f"Unable to read {label} file '{file_path}': {exc}",
                hint="Check file permissions and format"
            ) from exc
    
    # Handle empty input
    if not data:
        return {}
    
    # Parse JSON
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise CLIValidationError(
            f"Invalid JSON for {label}: {exc}",
            hint="Check JSON syntax (quotes, commas, brackets)",
            context={'data_preview': data[:100] if len(data) > 100 else data}
        ) from exc
    
    # Validate type
    if not isinstance(parsed, dict):
        raise CLIValidationError(
            f"{label.capitalize()} must be a JSON object (dictionary)",
            hint="Use {...} for JSON objects, not arrays or primitives"
        )
    
    return parsed


def prepare_backend(
    source_path: Path,
    backend_dir: str,
    *,
    embed_insights: bool = False,
    enable_realtime: bool = False,
    connector_config: Optional[Dict[str, Any]] = None,
) -> App:
    """
    Parse N3 file and generate backend scaffold.
    
    Complete pipeline for backend generation:
    1. Load and resolve N3 app
    2. Generate FastAPI backend code
    3. Return resolved app for further processing
    
    Args:
        source_path: Path to .n3 source file
        backend_dir: Output directory for backend
        embed_insights: Whether to embed insight routes in API
        enable_realtime: Whether to enable real-time features
        connector_config: Optional connector configuration override
    
    Returns:
        Resolved App AST
    
    Raises:
        CLIError: If loading or generation fails
    
    Examples:
        >>> app = prepare_backend(
        ...     Path("app.n3"),
        ...     "./backend",
        ...     embed_insights=True
        ... )  # doctest: +SKIP
    """
    # Load app
    app = load_n3_app(source_path)
    
    # Use default connector config if not provided
    if connector_config is None:
        connector_config = extract_connector_config(None, WorkspaceDefaults())
    
    # Generate backend
    try:
        generate_backend(
            app,
            backend_dir,
            embed_insights=embed_insights,
            enable_realtime=enable_realtime,
            connector_config=connector_config
        )
    except Exception as exc:
        raise CLIError(
            f"Backend generation failed: {exc}",
            code="CLI_BACKEND_GENERATION_ERROR",
            hint="Check N3 source for errors and try again",
            context={'source': str(source_path), 'backend_dir': backend_dir}
        ) from exc
    
    return app
