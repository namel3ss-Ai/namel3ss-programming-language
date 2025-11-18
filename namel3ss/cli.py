"""
Command line interface for Namel3ss.

This module exposes a CLI with subcommands for building and running
Namel3ss applications. It provides:

- build: Generate static frontend and/or backend scaffold
- run: Start a development server with hot reload

The CLI is the primary entrypoint for working with .n3 files.
"""

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import __version__
from .lang import LANGUAGE_VERSION as LANGUAGE_SPEC_VERSION
from .ast import App, Chain, Experiment
from .parser import N3SyntaxError
from .loader import load_program
from .codegen import generate_backend, generate_site
from .ml import get_default_model_registry
from .utils import iter_dependency_reports
from .config import (
    AppConfig,
    WorkspaceConfig,
    apply_cli_overrides,
    env_strings_from_config,
    default_app_name,
    load_workspace_config,
    merge_env,
    resolve_apps_from_args,
)
from .devserver import DevAppSession, summarize_sessions
from .plugins import PluginManager
from .resolver import ModuleResolutionError, resolve_program

ENV_ALIAS_MAP = {
    "production": ".env.prod",
    "prod": ".env.prod",
    "development": ".env.dev",
    "dev": ".env.dev",
    "local": ".env.local",
    "locally": ".env.local",
    "test": ".env.test",
}

DEFAULT_MODEL_REGISTRY = get_default_model_registry()

_RUN_ENV_PREPOSITIONS = {"using", "in", "on", "with"}

_CLI_TRACE_LIMIT = 4000


@dataclass
class CLIContext:
    """Shared context resolved from workspace configuration."""

    workspace_root: Path
    config: WorkspaceConfig
    plugin_manager: PluginManager


def _cli_context(args: argparse.Namespace) -> CLIContext:
    ctx = getattr(args, "cli_context", None)
    if ctx is None:
        raise RuntimeError("CLIContext was not initialised before command execution.")
    return ctx


def _match_app_config(config: WorkspaceConfig, source_path: Path) -> Optional[AppConfig]:
    resolved = source_path.resolve()
    for entry in config.apps.values():
        if entry.file.resolve() == resolved:
            return entry
    return None


def _runtime_env(config: WorkspaceConfig, app_cfg: AppConfig, cli_env: Sequence[str]) -> List[str]:
    base = env_strings_from_config(config.defaults.env)
    base = merge_env(base, env_strings_from_config(app_cfg.env))
    return merge_env(base, cli_env)


def _create_ephemeral_app_config(
    workspace: WorkspaceConfig,
    source_path: Path,
    args: argparse.Namespace,
) -> AppConfig:
    defaults = workspace.defaults
    name = default_app_name(source_path)
    backend_override = _as_path_string(getattr(args, "backend_out", None))
    backend = Path(backend_override or defaults.backend_out)
    if not backend.is_absolute():
        backend = (source_path.parent / backend).resolve()
    frontend_override = _as_path_string(getattr(args, "frontend_out", None))
    frontend = Path(frontend_override or defaults.frontend_out)
    if not frontend.is_absolute():
        frontend = (source_path.parent / frontend).resolve()
    raw_target = _as_string(getattr(args, "target", None))
    target = raw_target if raw_target in {"static", "react-vite"} else defaults.target
    port_value = getattr(args, "port", None)
    port = int(port_value) if isinstance(port_value, int) else defaults.port
    frontend_port_value = getattr(args, "frontend_port", None) if hasattr(args, "frontend_port") else None
    if isinstance(frontend_port_value, int):
        frontend_port = frontend_port_value
    else:
        frontend_port = defaults.frontend_port
    realtime_flag = _bool_from_flag(getattr(args, "realtime", None))
    enable_realtime = True if realtime_flag else bool(defaults.enable_realtime)
    env = env_strings_from_config(defaults.env)
    return AppConfig(
        name=name,
        file=source_path,
        backend_out=backend,
        frontend_out=frontend,
        port=port,
        frontend_port=frontend_port,
        target=target,
        enable_realtime=enable_realtime,
        env=env,
    )


def _effective_realtime(app_cfg: AppConfig, workspace: WorkspaceConfig) -> bool:
    if app_cfg.enable_realtime is None:
        return bool(workspace.defaults.enable_realtime)
    return bool(app_cfg.enable_realtime)


@dataclass
class BuildInvocation:
    app: App
    source: Path
    backend_dir: Path
    frontend_dir: Path
    target: str
    enable_realtime: bool


@dataclass
class RunInvocation:
    apps: Sequence[AppConfig]
    dev_mode: bool
    enable_realtime: bool


def _format_error_detail(exc: BaseException) -> str:
    message = f"{exc.__class__.__name__}: {exc}"
    return message if len(message) <= 280 else f"{message[:277]}..."


def _traceback_excerpt() -> str:
    trace = traceback.format_exc().strip()
    return trace if len(trace) <= _CLI_TRACE_LIMIT else f"{trace[:_CLI_TRACE_LIMIT - 3]}..."


def _find_first_n3_file() -> Optional[Path]:
    candidates = sorted(Path.cwd().glob('*.n3'))
    return candidates[0] if candidates else None


def _resolve_env_reference(token: str) -> Optional[str]:
    normalized = token.strip().strip('"\'')
    if not normalized:
        return None
    lower = normalized.lower()
    if lower in ENV_ALIAS_MAP:
        return ENV_ALIAS_MAP[lower]
    if normalized.startswith('.env'):
        return normalized
    return None


def _normalize_run_command_args(argv: List[str]) -> List[str]:
    if not argv or argv[0] != 'run':
        return argv

    new_args: List[str] = ['run']
    env_specs: List[str] = []
    tokens = argv[1:]
    idx = 0
    file_seen = False
    default_file_requested = False

    while idx < len(tokens):
        token = tokens[idx]
        lower = token.lower()

        if lower in _RUN_ENV_PREPOSITIONS and idx + 1 < len(tokens):
            candidate = tokens[idx + 1]
            resolved = _resolve_env_reference(candidate)
            if resolved is not None:
                env_specs.append(resolved)
                if not file_seen:
                    default_file_requested = True
                idx += 2
                continue

        resolved_single = _resolve_env_reference(token)
        if resolved_single is not None:
            env_specs.append(resolved_single)
            if not file_seen:
                default_file_requested = True
            idx += 1
            continue

        if token.endswith('.n3') and not file_seen:
            new_args.append(token)
            file_seen = True
            idx += 1
            continue

        new_args.append(token)
        idx += 1

    if not file_seen and default_file_requested:
        default_file = _find_first_n3_file()
        if default_file is not None:
            new_args.append(str(default_file))

    for env_entry in env_specs:
        new_args.extend(['--env', env_entry])

    return new_args


def _apply_env_overrides(overrides: Optional[List[str]]) -> None:
    """Set environment variables provided as KEY=VALUE strings."""

    if not overrides:
        return

    for entry in overrides:
        if not entry:
            continue

        if '=' in entry:
            key, value = entry.split('=', 1)
            os.environ[key] = value
            continue

        _load_env_file(entry)


def _as_path_string(value: Any) -> Optional[str]:
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    return None


def _as_string(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    return None


def _bool_from_flag(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _load_env_file(path_str: str) -> None:
    env_path = Path(path_str)
    if not env_path.exists():
        print(f"Error: environment file not found: {env_path}", file=sys.stderr)
        sys.exit(1)

    try:
        for raw_line in env_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[len('export '):].strip()
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()
    except Exception as exc:  # pragma: no cover - IO error handling
        print(f"Error reading environment file {env_path}: {exc}", file=sys.stderr)
        sys.exit(1)


def _plural(word: str, count: int) -> str:
    return word if count == 1 else f"{word}s"


def _program_root_for(source_path: Path) -> Path:
    return source_path.parent if source_path.is_file() else source_path


def _resolve_program(source_path: Path):
    if not source_path.exists():
        raise FileNotFoundError(f"File not found: {source_path}")
    project_root = _program_root_for(source_path)
    program = load_program(project_root)
    return resolve_program(program, entry_path=source_path)


def _load_cli_app(source_path: Path) -> App:
    try:
        resolved = _resolve_program(source_path)
    except (FileNotFoundError, ModuleResolutionError, N3SyntaxError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    return resolved.app


def _slugify_model_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return normalized.lower() or "model"


def _resolve_model_spec(app: App, model_name: str) -> Dict[str, Any]:
    for model in app.models:
        if model.name == model_name:
            registry_info = model.registry
            spec = {
                "type": model.model_type,
                "framework": model.engine or model.model_type,
                "version": registry_info.version or "v1",
                "metrics": dict(registry_info.metrics or {}),
                "metadata": dict(registry_info.metadata or {}),
            }
            return spec

    fallback = DEFAULT_MODEL_REGISTRY.get(model_name)
    if fallback:
        return dict(fallback)

    return {
        "type": "custom",
        "framework": "unknown",
        "version": "v1",
        "metrics": {},
        "metadata": {},
    }


def _backend_summary_lines(app: App) -> List[str]:
    connectors = {
        str(ds.connector.connector_name or ds.source or ds.name)
        for ds in app.datasets
        if ds.connector is not None
    }
    connectors.update(conn.name for conn in app.connectors)
    dataset_count = len(app.datasets)
    insight_count = len(app.insights)
    model_count = len(app.models)
    template_count = len(app.templates)
    chain_count = len(app.chains)
    experiment_count = len(app.experiments)

    return [
        f"✓ {dataset_count} {_plural('dataset', dataset_count)} available",
        f"✓ {len(connectors)} {_plural('connector', len(connectors))} registered",
        f"✓ {insight_count} {_plural('insight', insight_count)} routed",
        f"✓ {model_count} {_plural('model', model_count)} declared",
        f"✓ {template_count} {_plural('template', template_count)} cached",
        f"✓ {chain_count} {_plural('chain', chain_count)} composed",
        f"✓ {experiment_count} {_plural('experiment', experiment_count)} queued",
    ]


def _find_chain(app: App, name: str) -> Optional[Chain]:
    for chain in app.chains:
        if chain.name == name:
            return chain
    return None


def _find_experiment(app: App, name: str) -> Optional[Experiment]:
    for experiment in app.experiments:
        if experiment.name == name:
            return experiment
    return None

_RUNTIME_CACHE: Dict[str, Any] = {}


def _clear_generated_module_cache() -> None:
    for name in list(sys.modules):
        if name == "generated" or name.startswith("generated."):
            sys.modules.pop(name, None)


def _load_runtime_module(app: App, cache_key: str) -> Any:
    runtime = _RUNTIME_CACHE.get(cache_key)
    if runtime is not None:
        return runtime
    backend_dir = tempfile.mkdtemp(prefix="namel3ss_cli_backend_")
    generate_backend(app, backend_dir, embed_insights=False, enable_realtime=False)
    _clear_generated_module_cache()
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    runtime = importlib.import_module("generated.runtime")
    _RUNTIME_CACHE[cache_key] = runtime
    return runtime


def _print_prediction_response_text(response: Dict[str, Any]) -> None:
    """Pretty-print chain prediction output with status-aware messaging."""

    status = str(response.get("status") or response.get("output", {}).get("status") or "").lower()
    framework = response.get("framework") or "n/a"
    version = response.get("version") or "n/a"
    print(f"Model: {response.get('model', 'n/a')} (framework: {framework}, version: {version})")

    if status == "error":
        error_code = response.get("error") or response.get("output", {}).get("error") or "unknown_error"
        detail = response.get("detail") or response.get("output", {}).get("detail")
        print(f"[error] {error_code}")
        if detail:
            print(f"Detail: {detail}")
        return

    if status == "partial":
        print("[warning] Partial result reported by downstream components.")

    inputs = response.get("inputs") or response.get("input")
    if inputs:
        print("Inputs:")
        print(json.dumps(inputs, indent=2))

    if "result" in response:
        print("Result:")
        print(json.dumps(response["result"], indent=2) if isinstance(response["result"], dict) else response["result"])
    elif "output" in response:
        print("Output:")
        print(json.dumps(response["output"], indent=2))

    if response.get("notes"):
        print("Notes:")
        print(json.dumps(response["notes"], indent=2))

    metadata = response.get("metadata", {})
    if metadata:
        print("Metadata:")
        print(json.dumps(metadata, indent=2))


def _print_experiment_result_text(result: Dict[str, Any]) -> None:
    """Pretty-print experiment evaluation output with status awareness."""

    status = str(result.get("status") or "").lower()
    print(f"Experiment: {result.get('experiment', 'n/a')}")
    print(f"Status: {status or 'unknown'}")

    if status == "error":
        error_code = result.get("error", "experiment_error")
        detail = result.get("detail")
        print(f"[error] {error_code}")
        if detail:
            print(f"Detail: {detail}")
        return

    if status == "partial":
        print("[warning] Partial metrics returned; inspect notes for details.")

    print(f"Winner: {result.get('winner') or 'n/a'}")

    variants = result.get("variants") or []
    if variants:
        print("Variants:")
        for entry in variants:
            target = f"{entry.get('target_type')}:{entry.get('target_name') or 'default'}"
            score = entry.get("score")
            score_display = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
            print(f"  - {entry.get('name')} ({target}) score={score_display}")

    metrics = result.get("metrics") or []
    if metrics:
        print("Metrics:")
        for metric in metrics:
            name = metric.get("name", "metric")
            value = metric.get("value")
            value_display = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
            print(f"  - {name}: {value_display}")

    if result.get("inputs"):
        print("Inputs:")
        print(json.dumps(result["inputs"], indent=2))

    if result.get("notes"):
        print("Notes:")
        print(json.dumps(result["notes"], indent=2))


def prepare_backend(
    source_path: Path,
    backend_dir: str,
    *,
    embed_insights: bool = False,
    enable_realtime: bool = False,
) -> App:
    """Parse N3 file and generate backend scaffold.
    
    Parameters
    ----------
    source_path : Path
        Path to the .n3 source file
    backend_dir : str
        Directory where backend scaffold will be generated
    embed_insights : bool, optional
        Whether to embed evaluated insight payloads into dataset endpoints
        generated for tables and charts. Default is ``False``.
        
    Returns
    -------
    App
        The parsed application AST
        
    Raises
    ------
    N3SyntaxError
        If the source file contains syntax errors
    FileNotFoundError
        If the source file does not exist
    """
    resolved = _resolve_program(source_path)
    app = resolved.app
    
    # Generate backend scaffold
    generate_backend(
        app,
        backend_dir,
        embed_insights=embed_insights,
        enable_realtime=enable_realtime,
    )
    
    return app


def check_uvicorn_available() -> bool:
    """Check if uvicorn is installed and importable.
    
    Returns
    -------
    bool
        True if uvicorn is available, False otherwise
    """
    try:
        import uvicorn  # type: ignore
        return True
    except ImportError:
        return False


def run_dev_server(
    source_path: Path,
    backend_dir: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = True,
    *,
    embed_insights: bool = False,
    enable_realtime: bool = False,
    frontend_out: Optional[str] = None,
    frontend_target: str = "static",
    env: Optional[Sequence[str]] = None,
) -> None:
    """Run a development server for a Namel3ss app.
    
    This function:
    1. Parses the .n3 file
    2. Generates a backend scaffold
    3. Starts a uvicorn dev server with hot reload
    
    Parameters
    ----------
    source_path : Path
        Path to the .n3 source file
    backend_dir : Optional[str]
        Directory for backend scaffold. If None, uses a temp directory
    host : str
        Host to bind the server to
    port : int
        Port to bind the server to
    reload : bool
        Whether to enable hot reload
    embed_insights : bool, optional
        Forwarded to backend generation to control insight embedding.
    """
    # Check uvicorn availability first
    if not check_uvicorn_available():
        print(
            "Error: uvicorn is not installed.\n"
            "Please install it with: pip install uvicorn[standard]",
            file=sys.stderr
        )
        sys.exit(1)
    
    # Import uvicorn only after checking it's available
    import uvicorn  # type: ignore
    
    # Apply environment overrides before generation
    if env:
        _apply_env_overrides(list(env))

    # Use temp directory if none specified
    if backend_dir is None:
        backend_dir = os.path.join(tempfile.gettempdir(), '.namel3ss_dev_backend')
    backend_dir_path = Path(backend_dir).resolve()
    frontend_dir_path = Path(frontend_out or (backend_dir_path.parent / 'build')).resolve()
    
    try:
        # Prepare backend
        app = prepare_backend(
            source_path,
            str(backend_dir_path),
            embed_insights=embed_insights,
            enable_realtime=enable_realtime,
        )

        generate_site(
            app,
            str(frontend_dir_path),
            enable_realtime=enable_realtime,
            target=frontend_target,
        )
        
        print(f"✓ Parsed: {app.name}")
        print(f"✓ Backend generated in: {backend_dir_path}")
        print(f"✓ Frontend generated in: {frontend_dir_path}")
        for line in _backend_summary_lines(app):
            print(line)
        if enable_realtime:
            print("✓ Realtime streaming enabled (ws://{host}:{port}/ws/pages/<slug>)".format(host=host, port=port))
        print(f"\nNamel3ss dev server running at http://{host}:{port}")
        print("Press CTRL+C to stop\n")
        
        # Change to backend directory so imports work
        original_dir = os.getcwd()
        os.chdir(backend_dir_path)

        try:
            # Start uvicorn dev server
            uvicorn.run(
                "main:app",
                host=host,
                port=port,
                reload=reload,
                log_level="info"
            )
        finally:
            # Restore original directory
            os.chdir(original_dir)
            
    except (N3SyntaxError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _invoke_tool(command: Optional[str], label: str) -> None:
    if not command:
        print(f"No {label} command configured. Update [tools.{label}] in namel3ss.toml to enable it.")
        return
    print(f"→ Running {label}: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def cmd_build(args: argparse.Namespace) -> None:
    """Handle the 'build' subcommand."""

    ctx = _cli_context(args)
    source_path = Path(args.file).resolve()

    if not source_path.exists():
        print(f"Error: file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    workspace = ctx.config
    workspace.ensure_apps([])
    app_entry = _match_app_config(workspace, source_path)

    defaults = workspace.defaults
    backend_override = _as_path_string(getattr(args, "backend_out", None))
    backend_source = backend_override or (app_entry.backend_out if app_entry else defaults.backend_out)
    backend_dir = Path(backend_source)
    if not backend_dir.is_absolute():
        backend_dir = (source_path.parent / backend_dir).resolve()

    frontend_override = _as_path_string(getattr(args, "out", None))
    frontend_source = frontend_override or (app_entry.frontend_out if app_entry else defaults.frontend_out)
    frontend_dir = Path(frontend_source)
    if not frontend_dir.is_absolute():
        frontend_dir = (source_path.parent / frontend_dir).resolve()

    explicit_target = _as_string(getattr(args, "target", None))
    target_default = app_entry.target if app_entry else defaults.target
    target = explicit_target or target_default

    explicit_rt = _bool_from_flag(getattr(args, "realtime", None))
    if explicit_rt:
        realtime_flag = True
    else:
        rt_default = app_entry.enable_realtime if app_entry and app_entry.enable_realtime is not None else defaults.enable_realtime
        realtime_flag = bool(rt_default)

    env_entries: List[str] = env_strings_from_config(workspace.defaults.env)
    if app_entry:
        env_entries = merge_env(env_entries, env_strings_from_config(app_entry.env))
    env_entries = merge_env(env_entries, getattr(args, "env", []))
    _apply_env_overrides(env_entries)

    app = _load_cli_app(source_path)

    if args.print_ast:
        def default(o):
            if hasattr(o, "__dict__"):
                return o.__dict__
            return str(o)

        print(json.dumps(app, default=default, indent=2))
        return

    embed_flag = getattr(args, "embed_insights", False)

    if not args.backend_only:
        generate_site(app, str(frontend_dir), enable_realtime=realtime_flag, target=target)
        if target == "static":
            print(f"✓ Static site generated in {frontend_dir}")
        else:
            print(f"✓ Frontend [{target}] generated in {frontend_dir}")

    if args.build_backend or args.backend_only:
        generate_backend(
            app,
            str(backend_dir),
            embed_insights=embed_flag,
            enable_realtime=realtime_flag,
        )
        print(f"✓ Backend scaffold generated in {backend_dir}")
        for line in _backend_summary_lines(app):
            print(line)

    ctx.plugin_manager.emit_build(
        BuildInvocation(
            app=app,
            source=source_path,
            backend_dir=backend_dir,
            frontend_dir=frontend_dir,
            target=target,
            enable_realtime=realtime_flag,
        ),
        args=args,
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the 'run' subcommand."""
    ctx = _cli_context(args)
    namespace = vars(args)
    target = namespace.get("target")
    force_dev = namespace.get("dev", False)

    chain_mode = False
    if not force_dev and target:
        if target.endswith(".n3"):
            chain_mode = False
        else:
            potential_path = Path(target)
            chain_mode = not potential_path.exists()
    elif not target:
        chain_mode = False

    workspace = ctx.config
    workspace_root = ctx.workspace_root

    if chain_mode:
        chain_name = target or ""
        source_arg = namespace.get("file")
        if source_arg is None:
            default_file = _find_first_n3_file()
            if default_file is None:
                print("Error: no .n3 file found to resolve chain execution.", file=sys.stderr)
                sys.exit(1)
            source_arg = str(default_file)
        source_path = Path(source_arg)
        app_cfg = _match_app_config(workspace, source_path)
        if app_cfg is None:
            app_cfg = _create_ephemeral_app_config(workspace, source_path, args)
        _apply_env_overrides(_runtime_env(workspace, app_cfg, namespace.get("env", [])))
        app = _load_cli_app(source_path)
        chain = _find_chain(app, chain_name)
        if chain is None:
            available = ", ".join(sorted(c.name for c in app.chains)) or "none"
            print(
                f"Error: chain '{chain_name}' not found. Available chains: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        payload: Optional[Dict[str, Any]] = None
        cache_key = str(source_path.resolve())
        try:
            runtime = _load_runtime_module(app, cache_key)
            runtime_result = runtime.run_chain(chain_name, payload or {})
            if not isinstance(runtime_result, dict):
                raise TypeError("run_chain returned non-dict result")
            result = dict(runtime_result)
            result.setdefault("status", "ok")
            result.setdefault("inputs", payload or {})
        except Exception as exc:
            result = {
                "status": "error",
                "error": "chain_execution_failed",
                "detail": _format_error_detail(exc),
                "traceback": _traceback_excerpt(),
                "model": chain_name,
                "inputs": payload or {},
            }
        result.setdefault("model", chain_name)
        result.setdefault("inputs", payload or {})
        if namespace.get("json", False):
            print(json.dumps(result, indent=2))
        else:
            _print_prediction_response_text(result)
        return

    selected_names = getattr(args, "apps", None)
    source_path_arg: Optional[str] = None
    if target and (target.endswith(".n3") or Path(target).exists()):
        source_path_arg = target
    if source_path_arg is None and namespace.get("file"):
        source_path_arg = namespace["file"]

    apps: List[AppConfig]
    if selected_names:
        apps = resolve_apps_from_args(workspace, selected_names, workspace_root)
    elif source_path_arg:
        source_path = Path(source_path_arg)
        matched = _match_app_config(workspace, source_path)
        if matched is None:
            apps = [_create_ephemeral_app_config(workspace, source_path, args)]
        else:
            apps = [matched]
    elif namespace.get("workspace") or workspace.apps:
        apps = resolve_apps_from_args(workspace, None, workspace_root)
    else:
        if source_path_arg is None:
            default_file = _find_first_n3_file()
            if default_file is None:
                print("Error: no .n3 file found in the current directory.", file=sys.stderr)
                sys.exit(1)
            source_path_arg = str(default_file)
        source_path = Path(source_path_arg)
        matched = _match_app_config(workspace, source_path)
        if matched is None:
            apps = [_create_ephemeral_app_config(workspace, source_path, args)]
        else:
            apps = [matched]

    if not apps:
        print("Error: no applications resolved to run.", file=sys.stderr)
        sys.exit(1)

    realtime_toggle = _bool_from_flag(getattr(args, "realtime", None))
    enable_override = True if realtime_toggle else None
    apps = apply_cli_overrides(
        apps,
        backend_out=namespace.get("backend_out"),
        frontend_out=getattr(args, "frontend_out", None),
        port=namespace.get("port"),
        frontend_port=getattr(args, "frontend_port", None),
        target=None,
        enable_realtime=enable_override,
    )

    env_union: List[str] = []
    effective_envs: Dict[str, List[str]] = {}
    for app_cfg in apps:
        env_values = _runtime_env(workspace, app_cfg, namespace.get("env", []))
        effective_envs[app_cfg.name] = env_values
        env_union = merge_env(env_union, env_values)

    if env_union:
        _apply_env_overrides(env_union)

    dev_mode = namespace.get("dev", False) or len(apps) > 1
    realtime_enabled = any(_effective_realtime(app, workspace) for app in apps)

    if dev_mode:
        sessions: List[DevAppSession] = []
        try:
            for app_cfg in apps:
                enable_rt = _effective_realtime(app_cfg, workspace)
                session = DevAppSession(
                    name=app_cfg.name,
                    source=app_cfg.file,
                    backend_out=app_cfg.backend_out,
                    frontend_out=app_cfg.frontend_out,
                    host=namespace.get("host", workspace.defaults.host),
                    port=app_cfg.port,
                    frontend_target=app_cfg.target,
                    enable_realtime=enable_rt,
                    env=effective_envs.get(app_cfg.name, []),
                )
                session.start()
                sessions.append(session)
            summarize_sessions(sessions)
            ctx.plugin_manager.emit_run(
                RunInvocation(apps=apps, dev_mode=True, enable_realtime=realtime_enabled),
                args=args,
            )
            try:
                while any(session.is_running() for session in sessions):
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
        finally:
            for session in sessions:
                session.stop()
        return

    app_cfg = apps[0]
    enable_rt = _effective_realtime(app_cfg, workspace)
    run_dev_server(
        app_cfg.file,
        backend_dir=str(app_cfg.backend_out),
        host=namespace.get("host", workspace.defaults.host),
        port=app_cfg.port,
        reload=not namespace.get("no_reload", False),
        embed_insights=namespace.get("embed_insights", False),
        enable_realtime=enable_rt,
        frontend_out=str(app_cfg.frontend_out),
        frontend_target=app_cfg.target,
        env=effective_envs.get(app_cfg.name, []),
    )
    ctx.plugin_manager.emit_run(
        RunInvocation(apps=[app_cfg], dev_mode=False, enable_realtime=realtime_enabled),
        args=args,
    )


def cmd_test(args: argparse.Namespace) -> None:
    ctx = _cli_context(args)
    override = getattr(args, "command", None)
    command = override or ctx.config.tools.test
    ctx.plugin_manager.emit_workspace(task="test", command=command, args=args)
    _invoke_tool(command, "test")


def cmd_lint(args: argparse.Namespace) -> None:
    ctx = _cli_context(args)
    override = getattr(args, "command", None)
    command = override or ctx.config.tools.lint
    ctx.plugin_manager.emit_workspace(task="lint", command=command, args=args)
    _invoke_tool(command, "lint")


def cmd_typecheck(args: argparse.Namespace) -> None:
    ctx = _cli_context(args)
    override = getattr(args, "command", None)
    command = override or ctx.config.tools.typecheck
    ctx.plugin_manager.emit_workspace(task="typecheck", command=command, args=args)
    _invoke_tool(command, "typecheck")


def cmd_eval(args: argparse.Namespace) -> None:
    """Handle the 'eval' subcommand for experiments."""

    experiment_name = args.experiment
    source_arg = args.file
    if source_arg is None:
        default_file = _find_first_n3_file()
        if default_file is None:
            print("Error: no .n3 file found to evaluate experiments.", file=sys.stderr)
            sys.exit(1)
        source_arg = str(default_file)

    app = _load_cli_app(Path(source_arg))
    experiment = _find_experiment(app, experiment_name)
    if experiment is None:
        available = ", ".join(sorted(exp.name for exp in app.experiments)) or "none"
        print(
            f"Error: experiment '{experiment_name}' not found. Available experiments: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    payload: Optional[Dict[str, Any]] = None
    cache_key = str(Path(source_arg).resolve())
    try:
        runtime = _load_runtime_module(app, cache_key)
        runtime_result = runtime.evaluate_experiment(experiment_name, payload or {})
        if not isinstance(runtime_result, dict):
            raise TypeError("evaluate_experiment returned non-dict result")
        result = dict(runtime_result)
        result.setdefault("status", "ok")
        result.setdefault("inputs", payload or {})
    except Exception as exc:
        result = {
            "status": "error",
            "error": "experiment_execution_failed",
            "detail": _format_error_detail(exc),
            "traceback": _traceback_excerpt(),
            "experiment": experiment_name,
        }
    result.setdefault("experiment", experiment_name)
    result.setdefault("inputs", payload or {})

    if getattr(args, 'format', 'json') == 'text':
        _print_experiment_result_text(result)
        return

    print(json.dumps(result, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    """Handle the 'train' subcommand using user-provided hooks."""

    source_path = Path(args.file)
    app = _load_cli_app(source_path)
    model_name = args.model
    spec = _resolve_model_spec(app, model_name)
    framework = spec.get("framework", "unknown")
    metadata = spec.get("metadata", {})
    trainer_hook: Optional[str] = metadata.get("trainer")

    if not trainer_hook:
        print(json.dumps({
            "status": "error",
            "error": "trainer_not_configured",
            "detail": f"No trainer hook configured for model '{model_name}'.",
        }))
        return

    module_path, _, attr = trainer_hook.partition(":")
    if not module_path or not attr:
        print(json.dumps({
            "status": "error",
            "error": "trainer_invalid_hook",
            "detail": f"Trainer hook '{trainer_hook}' is not importable.",
        }))
        return

    try:
        module = importlib.import_module(module_path)
        trainer = getattr(module, attr)
        if not callable(trainer):
            raise TypeError(f"Trainer '{trainer_hook}' is not callable")
        trainer(model_name, spec, args)
    except Exception as exc:
        print(json.dumps({
            "status": "error",
            "error": "trainer_failed",
            "detail": str(exc),
        }))
        return

    print(f"Training hook completed for '{model_name}'")
    print(json.dumps({
        "status": "ok",
        "model": model_name,
        "detail": "Training hook completed.",
    }))


def cmd_deploy(args: argparse.Namespace) -> None:
    """Handle the 'deploy' subcommand invoking optional user hooks."""

    source_path = Path(args.file)
    app = _load_cli_app(source_path)
    model_name = args.model
    spec = _resolve_model_spec(app, model_name)
    slug = _slugify_model_name(model_name)
    version = spec.get("version", "v1")
    metadata = spec.get("metadata", {})
    deployer_hook: Optional[str] = metadata.get("deployer")
    if not deployer_hook:
        print(json.dumps({
            "status": "error",
            "error": "deployer_not_configured",
            "detail": f"No deployer hook configured for model '{model_name}'.",
        }))
        return

    module_path, _, attr = deployer_hook.partition(":")
    if not module_path or not attr:
        print(json.dumps({
            "status": "error",
            "error": "deployer_invalid_hook",
            "detail": f"Deployer hook '{deployer_hook}' is not importable.",
        }))
        return

    try:
        module = importlib.import_module(module_path)
        deployer = getattr(module, attr)
        if not callable(deployer):
            raise TypeError(f"Deployer '{deployer_hook}' is not callable")
        output = deployer(model_name, spec, args)
    except Exception as exc:
        print(json.dumps({
            "status": "error",
            "error": "deployer_failed",
            "detail": str(exc),
        }))
        return

    if isinstance(output, dict):
        result = {"status": output.get("status", "ok"), **{k: v for k, v in output.items() if k != "status"}}
    elif isinstance(output, str) and output:
        result = {"status": "ok", "endpoint": output}
    else:
        result = {"status": "ok", "detail": "Deployment hook executed."}
    result.setdefault("model", model_name)
    result.setdefault("version", version)
    result.setdefault("endpoint", result.get("endpoint"))
    endpoint = result.get("endpoint")
    if endpoint:
        print(f"Model '{model_name}' deployed at {endpoint}")
    print(json.dumps(result))


def cmd_doctor(args: argparse.Namespace) -> None:
    """Report the availability of core and optional dependencies."""

    reports = iter_dependency_reports()
    core_missing = False

    for report in reports:
        prefix = "Optional" if report.optional else "Core"
        status = "available" if not report.missing else "missing"
        symbol = "✓" if status == "available" else "✗"
        print(f"{symbol} {prefix} {report.title}: {status}")
        if report.missing:
            missing_modules = ", ".join(report.missing)
            print(f"    Missing modules: {missing_modules}")
            if report.advice:
                print(f"    → {report.advice}")
            if not report.optional:
                core_missing = True

    if core_missing:
        raise SystemExit(1)


def main(argv: Optional[list] = None) -> None:
    """Main CLI entrypoint with subcommand support."""
    if argv is None:
        argv = sys.argv[1:]

    if (
        len(argv) > 0
        and not argv[0].startswith('-')
        and argv[0] not in ['build', 'run', 'help']
        and (argv[0].endswith('.n3') or Path(argv[0]).exists())
    ):
        print("Note: Using legacy invocation. Consider using 'namel3ss build' instead.", file=sys.stderr)
        argv = ['build'] + argv

    if argv and argv[0] == 'run':
        argv = _normalize_run_command_args(argv)

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config')
    pre_parser.add_argument('--workspace')
    pre_args, _ = pre_parser.parse_known_args(argv)
    workspace_root = Path(pre_args.workspace).resolve() if pre_args.workspace else Path.cwd()
    config_path = Path(pre_args.config).resolve() if pre_args.config else None
    config = load_workspace_config(workspace_root, config_path)
    plugin_manager = PluginManager(config.plugins)
    plugin_manager.load()

    parser = argparse.ArgumentParser(
        description="Namel3ss (N3) language compiler – build full‑stack apps in plain English",
        prog="namel3ss"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} (language {LANGUAGE_SPEC_VERSION})"
    )

    parser.add_argument(
        '--config',
        default=str(config_path) if config_path else None,
        help='Path to a namel3ss.toml configuration file'
    )
    parser.add_argument(
        '--workspace',
        default=str(workspace_root),
        help='Workspace root directory (defaults to current working directory)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    plugin_manager.register_commands(subparsers)
    
    # Build subcommand
    build_parser = subparsers.add_parser(
        'build',
        help='Generate static site and/or backend scaffold'
    )
    build_parser.add_argument('file', help='Path to the .n3 source file')
    build_parser.add_argument(
        '--out', '-o', default='build', help='Output directory for static files'
    )
    build_parser.add_argument(
        '--print-ast', action='store_true', help='Print the parsed AST and exit'
    )
    build_parser.add_argument(
        '--build-backend', action='store_true', help='Also generate FastAPI backend scaffold'
    )
    build_parser.add_argument(
        '--realtime', action='store_true', help='Enable realtime websocket scaffolding'
    )
    build_parser.add_argument(
        '--target',
        choices=['static', 'react-vite'],
        default='static',
        help='Frontend target to generate (default: static)'
    )
    build_parser.add_argument(
        '--backend-only', action='store_true', help='Only generate backend, skip static site'
    )
    build_parser.add_argument(
        '--backend-out', default='backend_build', help='Output directory for backend scaffold'
    )
    build_parser.add_argument(
        '--embed-insights', action='store_true', help='Embed insight results directly in dataset responses'
    )
    build_parser.add_argument(
        '--env',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='Set environment variable for backend generation (may be provided multiple times)'
    )
    build_parser.set_defaults(func=cmd_build)
    
    # Run subcommand
    run_parser = subparsers.add_parser(
        'run',
        help='Execute an AI chain or launch the development server'
    )
    run_parser.add_argument(
        'target',
        nargs='?',
        help='Chain name to execute or path to the .n3 source file'
    )
    run_parser.add_argument(
        '-f', '--file',
        dest='file',
        help='Explicit .n3 source file to use when running a chain'
    )
    run_parser.add_argument(
        '--dev',
        action='store_true',
        help='Force dev server mode even if the target looks like a chain'
    )
    run_parser.add_argument(
        '--backend-out',
        default=None,
        help='Output directory for backend scaffold (default: temp directory)'
    )
    run_parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind server to (default: 127.0.0.1)'
    )
    run_parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind server to (default: 8000)'
    )
    run_parser.add_argument(
        '--no-reload',
        action='store_true',
        help='Disable hot reload'
    )
    run_parser.add_argument(
        '--embed-insights',
        action='store_true',
        help='Embed insight results directly in dataset responses'
    )
    run_parser.add_argument(
        '--realtime',
        action='store_true',
        help='Enable realtime websocket scaffolding'
    )
    run_parser.add_argument(
        '--apps',
        nargs='+',
        help='Names of apps defined in the workspace configuration to run'
    )
    run_parser.add_argument(
        '--frontend-out',
        default=None,
        help='Override frontend output directory for generated assets'
    )
    run_parser.add_argument(
        '--frontend-port',
        type=int,
        default=None,
        help='Base frontend port when running multiple apps concurrently'
    )
    run_parser.add_argument(
        '--env',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='Set environment variable before launching the dev server (may be provided multiple times)'
    )
    run_parser.add_argument(
        '--json',
        action='store_true',
        help='Emit structured JSON when executing a chain'
    )
    run_parser.set_defaults(func=cmd_run)

    eval_parser = subparsers.add_parser(
            'eval',
        help='Evaluate experiment variants'
    )
    eval_parser.add_argument('experiment', help='Name of the experiment to evaluate')
    eval_parser.add_argument(
        '-f', '--file',
        dest='file',
        help='Path to the .n3 source file (defaults to first .n3 in current directory)'
    )
    eval_parser.add_argument(
        '--format',
        choices=['json', 'text'],
        default='json',
        help='Output format for experiment results'
    )
    eval_parser.set_defaults(func=cmd_eval)

    train_parser = subparsers.add_parser(
        'train',
        help='Run model training hooks'
    )
    train_parser.add_argument('file', help='Path to the .n3 source file')
    train_parser.add_argument(
        '--model',
        required=True,
        help='Name of the model to train (must exist in the DSL or model registry)'
    )
    train_parser.set_defaults(func=cmd_train)

    deploy_parser = subparsers.add_parser(
        'deploy',
        help='Deploy a model prediction endpoint'
    )
    deploy_parser.add_argument('file', help='Path to the .n3 source file')
    deploy_parser.add_argument(
        '--model',
        required=True,
        help='Name of the model to deploy (must exist in the DSL or model registry)'
    )
    deploy_parser.set_defaults(func=cmd_deploy)

    doctor_parser = subparsers.add_parser(
        'doctor',
        help='Diagnose installed dependencies and optional extras'
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    test_parser = subparsers.add_parser(
        'test',
        help='Execute the configured test command for the workspace'
    )
    test_parser.add_argument(
        '--command',
        help='Override the configured test command'
    )
    test_parser.set_defaults(func=cmd_test)

    lint_parser = subparsers.add_parser(
        'lint',
        help='Run the configured lint command'
    )
    lint_parser.add_argument(
        '--command',
        help='Override the configured lint command'
    )
    lint_parser.set_defaults(func=cmd_lint)

    typecheck_parser = subparsers.add_parser(
        'typecheck',
        help='Run the configured type checking command'
    )
    typecheck_parser.add_argument(
        '--command',
        help='Override the configured typecheck command'
    )
    typecheck_parser.set_defaults(func=cmd_typecheck)
    
    args = parser.parse_args(argv)

    runtime_workspace = Path(args.workspace).resolve() if getattr(args, 'workspace', None) else workspace_root
    runtime_config_path = Path(args.config).resolve() if getattr(args, 'config', None) else config_path
    if runtime_workspace != workspace_root or runtime_config_path != config_path:
        config = load_workspace_config(runtime_workspace, runtime_config_path)
        plugin_manager = PluginManager(config.plugins)
        plugin_manager.load()
    
    # If no command specified, print help
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    args.cli_context = CLIContext(
        workspace_root=runtime_workspace,
        config=config,
        plugin_manager=plugin_manager,
    )
    
    # Execute the command
    args.func(args)


if __name__ == '__main__':  # pragma: no cover
    main()
