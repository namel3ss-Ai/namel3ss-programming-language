"""
Run command implementation.

This module handles the 'run' subcommand which starts development servers
for Namel3ss applications, supporting both single-app and multi-app modes.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from namel3ss.codegen.backend.core import generate_backend
from namel3ss.codegen.frontend import generate_site
from namel3ss.config import AppConfig, extract_connector_config, apply_cli_overrides, resolve_apps_from_args, merge_env
from namel3ss.devserver import DevAppSession, summarize_sessions
from namel3ss.errors import N3SyntaxError, N3TypeError

from ..context import (
    create_ephemeral_app_config,
    get_cli_context,
    get_effective_realtime,
    get_runtime_env,
    match_app_config,
)
from ..errors import CLIDependencyError, CLIRuntimeError, handle_cli_exception
from ..loading import load_json_argument, load_n3_app, load_runtime_module, prepare_backend
from ..output import print_error, print_prediction_response, print_success
from ..utils import find_chain, find_first_n3_file, generate_backend_summary
from ..validation import apply_env_overrides, normalize_run_command_args, validate_bool, validate_int


_CLI_TRACE_LIMIT = 2000


class RunInvocation:
    """
    Dataclass holding the results of a run invocation for plugin hooks.
    
    Attributes:
        apps: List of applications being run
        dev_mode: Whether running in development mode (multi-app or explicit --dev)
        enable_realtime: Whether realtime streaming is enabled
    """
    
    def __init__(
        self,
        apps: Sequence[AppConfig],
        dev_mode: bool,
        enable_realtime: bool,
    ):
        self.apps = apps
        self.dev_mode = dev_mode
        self.enable_realtime = enable_realtime


def check_uvicorn_available() -> bool:
    """
    Check if uvicorn is installed and importable.
    
    Returns:
        True if uvicorn is available, False otherwise
    
    Examples:
        >>> check_uvicorn_available()  # doctest: +SKIP
        True
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
    connector_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run a development server for a Namel3ss app.
    
    This function:
    1. Parses the .n3 file
    2. Generates backend and frontend scaffolds
    3. Starts a uvicorn dev server with hot reload
    
    Args:
        source_path: Path to the .n3 source file
        backend_dir: Directory for backend scaffold (None uses temp directory)
        host: Host to bind the server to
        port: Port to bind the server to
        reload: Whether to enable hot reload
        embed_insights: Forward to backend generation to control insight embedding
        enable_realtime: Enable WebSocket realtime streaming
        frontend_out: Frontend output directory
        frontend_target: Frontend target type (static, react, etc.)
        env: Environment variable overrides
        connector_config: Connector retry/concurrency settings
    
    Raises:
        SystemExit: If uvicorn is not installed or generation fails
    
    Examples:
        >>> run_dev_server(  # doctest: +SKIP
        ...     Path("app.n3"),
        ...     host="127.0.0.1",
        ...     port=8000
        ... )
        ✓ Parsed: MyApp
        ✓ Backend generated in: backend/
        ✓ Frontend generated in: build/
        Namel3ss dev server running at http://127.0.0.1:8000
    """
    # Check uvicorn availability first
    if not check_uvicorn_available():
        raise CLIDependencyError(
            "uvicorn is not installed",
            hint="Install with: pip install uvicorn[standard]",
        )
    
    # Import uvicorn only after checking it's available
    import uvicorn  # type: ignore
    
    # Apply environment overrides before generation
    if env:
        apply_env_overrides(list(env))
    
    # Use temp directory if none specified
    if backend_dir is None:
        backend_dir = os.path.join(tempfile.gettempdir(), '.namel3ss_dev_backend')
    backend_dir_path = Path(backend_dir).resolve()
    frontend_dir_path = Path(
        frontend_out or (backend_dir_path.parent / 'build')
    ).resolve()
    
    try:
        # Prepare backend
        app = prepare_backend(
            source_path,
            str(backend_dir_path),
            embed_insights=embed_insights,
            enable_realtime=enable_realtime,
            connector_config=connector_config,
        )
        
        # Generate frontend
        generate_site(
            app,
            str(frontend_dir_path),
            enable_realtime=enable_realtime,
            target=frontend_target,
        )
        
        # Print success messages
        print_success(f"Parsed: {app.name}")
        print_success(f"Backend generated in: {backend_dir_path}")
        print_success(f"Frontend generated in: {frontend_dir_path}")
        
        summary = generate_backend_summary(app)
        print(summary)
        
        if enable_realtime:
            print_success(
                f"Realtime streaming enabled (ws://{host}:{port}/ws/pages/<slug>)"
            )
        
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
    
    except (N3SyntaxError, N3TypeError, RuntimeError) as exc:
        raise CLIRuntimeError(
            f"Failed to start dev server: {exc}",
            hint="Check your N3 source file for errors",
        ) from exc
    except FileNotFoundError as exc:
        raise CLIRuntimeError(
            f"File not found: {exc}",
            hint="Ensure the source file exists",
        ) from exc


def _format_error_detail(exc: BaseException) -> str:
    """
    Format exception as a concise error detail string.
    
    Args:
        exc: Exception to format
    
    Returns:
        Formatted error detail, truncated to 280 characters
    
    Examples:
        >>> _format_error_detail(ValueError("test error"))
        'ValueError: test error'
    """
    message = f"{exc.__class__.__name__}: {exc}"
    return message if len(message) <= 280 else f"{message[:277]}..."


def _traceback_excerpt() -> str:
    """
    Get truncated traceback for error reporting.
    
    Returns:
        Formatted traceback, truncated to _CLI_TRACE_LIMIT characters
    
    Examples:
        >>> try:
        ...     raise ValueError("test")
        ... except:
        ...     excerpt = _traceback_excerpt()  # doctest: +SKIP
    """
    trace = traceback.format_exc().strip()
    return trace if len(trace) <= _CLI_TRACE_LIMIT else f"{trace[:_CLI_TRACE_LIMIT - 3]}..."


def cmd_run(args: argparse.Namespace) -> None:
    """
    Handle the 'run' subcommand to start development servers or run chains.
    
    This command supports multiple modes:
    1. Chain execution mode: Run a specific chain and print results
    2. Single-app dev server: Start one development server
    3. Multi-app dev server: Start multiple development servers simultaneously
    
    The command automatically detects which mode based on arguments:
    - If target is a chain name (not .n3 file), runs chain mode
    - If --dev or multiple apps specified, runs multi-app dev server
    - Otherwise runs single-app dev server
    
    Args:
        args: Parsed command-line arguments containing:
            - file: Path to .n3 source file (optional)
            - target: Chain name or .n3 file to run (optional)
            - apps: List of app names from config (optional)
            - workspace: Run all workspace apps (optional)
            - dev: Force dev server mode (optional)
            - host: Server host (default: 127.0.0.1)
            - port: Server port (default: 8000)
            - frontend_port: Frontend dev server port (optional)
            - backend_out: Backend output directory (optional)
            - frontend_out: Frontend output directory (optional)
            - realtime: Enable realtime streaming (optional)
            - env: Environment variable overrides (optional)
            - json: Output JSON format (chain mode only)
            - embed_insights: Embed insight payloads (optional)
            - no_reload: Disable hot reload (optional)
    
    Raises:
        SystemExit: On any error during execution
    
    Examples:
        Run chain:
        >>> args = argparse.Namespace(target='my_chain', ...)
        >>> cmd_run(args)  # doctest: +SKIP
        Model: gpt-4 (framework: openai, version: n/a)
        Result: ...
        
        Start dev server:
        >>> args = argparse.Namespace(file='app.n3', ...)
        >>> cmd_run(args)  # doctest: +SKIP
        ✓ Parsed: MyApp
        Namel3ss dev server running at http://127.0.0.1:8000
    """
    try:
        ctx = get_cli_context(args)
        namespace = vars(args)
        target = namespace.get("target")
        force_dev = namespace.get("dev", False)
        
        # Determine if this is chain execution mode
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
        
        # Chain execution mode
        if chain_mode:
            chain_name = target or ""
            source_arg = namespace.get("file")
            if source_arg is None:
                default_file = find_first_n3_file()
                if default_file is None:
                    raise CLIRuntimeError(
                        "No .n3 file found to resolve chain execution",
                        hint="Specify a .n3 file with --file or create one in the current directory",
                    )
                source_arg = str(default_file)
            
            source_path = Path(source_arg)
            app_cfg = match_app_config(workspace, source_path)
            if app_cfg is None:
                app_cfg = create_ephemeral_app_config(workspace, source_path, args)
            
            apply_env_overrides(get_runtime_env(workspace, app_cfg, namespace.get("env", [])))
            app = load_n3_app(source_path)
            
            chain = find_chain(app, chain_name)
            if chain is None:
                available = ", ".join(sorted(c.name for c in app.chains)) or "none"
                raise CLIRuntimeError(
                    f"Chain '{chain_name}' not found",
                    hint=f"Available chains: {available}",
                )
            
            payload: Optional[Dict[str, Any]] = None
            cache_key = str(source_path.resolve())
            
            try:
                runtime = load_runtime_module(app, cache_key)
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
                print_prediction_response(result)
            return
        
        # Dev server mode - resolve apps
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
            matched = match_app_config(workspace, source_path)
            if matched is None:
                apps = [create_ephemeral_app_config(workspace, source_path, args)]
            else:
                apps = [matched]
        elif namespace.get("workspace") or workspace.apps:
            apps = resolve_apps_from_args(workspace, None, workspace_root)
        else:
            if source_path_arg is None:
                default_file = find_first_n3_file()
                if default_file is None:
                    raise CLIRuntimeError(
                        "No .n3 file found in the current directory",
                        hint="Create a .n3 file or specify one with --file",
                    )
                source_path_arg = str(default_file)
            source_path = Path(source_path_arg)
            matched = match_app_config(workspace, source_path)
            if matched is None:
                apps = [create_ephemeral_app_config(workspace, source_path, args)]
            else:
                apps = [matched]
        
        if not apps:
            raise CLIRuntimeError(
                "No applications resolved to run",
                hint="Specify a .n3 file or configure apps in namel3ss.toml",
            )
        
        # Apply CLI overrides
        realtime_toggle = validate_bool(getattr(args, "realtime", None), allow_none=True)
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
        
        # Merge environment variables
        env_union: List[str] = []
        effective_envs: Dict[str, List[str]] = {}
        for app_cfg in apps:
            env_values = get_runtime_env(workspace, app_cfg, namespace.get("env", []))
            effective_envs[app_cfg.name] = env_values
            env_union = merge_env(env_union, env_values)
        
        if env_union:
            apply_env_overrides(env_union)
        
        dev_mode = namespace.get("dev", False) or len(apps) > 1
        realtime_enabled = any(get_effective_realtime(app, workspace) for app in apps)
        
        # Multi-app dev server mode
        if dev_mode:
            sessions: List[DevAppSession] = []
            try:
                for app_cfg in apps:
                    enable_rt = get_effective_realtime(app_cfg, workspace)
                    connector_cfg = extract_connector_config(app_cfg, workspace.defaults)
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
                        connector_config=connector_cfg,
                    )
                    session.start()
                    sessions.append(session)
                
                summarize_sessions(sessions)
                ctx.plugin_manager.emit_run(
                    RunInvocation(
                        apps=apps,
                        dev_mode=True,
                        enable_realtime=realtime_enabled
                    ),
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
        
        # Single-app dev server mode
        app_cfg = apps[0]
        enable_rt = get_effective_realtime(app_cfg, workspace)
        connector_cfg = extract_connector_config(app_cfg, workspace.defaults)
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
            connector_config=connector_cfg,
        )
        ctx.plugin_manager.emit_run(
            RunInvocation(
                apps=[app_cfg],
                dev_mode=False,
                enable_realtime=realtime_enabled
            ),
            args=args,
        )
    
    except Exception as exc:
        handle_cli_exception(exc)
