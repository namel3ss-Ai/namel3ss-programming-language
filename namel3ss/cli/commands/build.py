"""
Build command implementation.

This module handles the 'build' subcommand which generates frontend and/or
backend scaffolds from N3 source files.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from namel3ss.codegen.backend.core import generate_backend
from namel3ss.codegen.frontend import generate_site
from namel3ss.config import extract_connector_config, env_strings_from_config, merge_env

from ..context import get_cli_context, match_app_config
from ..errors import CLIFileNotFoundError, handle_cli_exception
from ..loading import load_n3_app
from ..output import print_success
from ..utils import generate_backend_summary
from ..validation import (
    apply_env_overrides,
    validate_bool,
    validate_path,
    validate_string,
    validate_target_type,
)


class BuildInvocation:
    """
    Dataclass holding the results of a build invocation for plugin hooks.
    
    Attributes:
        app: The parsed application AST
        source: Path to the source .n3 file
        backend_dir: Directory where backend was generated
        frontend_dir: Directory where frontend was generated
        target: Frontend target type (static, react, etc.)
        enable_realtime: Whether realtime streaming is enabled
    """
    
    def __init__(
        self,
        app,
        source: Path,
        backend_dir: Path,
        frontend_dir: Path,
        target: str,
        enable_realtime: bool,
    ):
        self.app = app
        self.source = source
        self.backend_dir = backend_dir
        self.frontend_dir = frontend_dir
        self.target = target
        self.enable_realtime = enable_realtime


def cmd_build(args: argparse.Namespace) -> None:
    """
    Handle the 'build' subcommand to generate frontend and/or backend scaffolds.
    
    This command:
    1. Loads and validates the N3 source file
    2. Resolves configuration from workspace settings and CLI args
    3. Generates frontend site (static HTML, React, etc.)
    4. Optionally generates backend scaffold (FastAPI)
    5. Emits build event to plugin system
    
    Args:
        args: Parsed command-line arguments containing:
            - file: Path to .n3 source file
            - out: Frontend output directory (optional)
            - backend_out: Backend output directory (optional)
            - target: Frontend target type (optional)
            - realtime: Enable realtime streaming (optional)
            - env: Environment variable overrides (optional)
            - print_ast: Print AST and exit (optional)
            - embed_insights: Embed insight payloads in endpoints (optional)
            - build_backend: Force backend generation (optional)
            - backend_only: Generate only backend, skip frontend (optional)
    
    Raises:
        SystemExit: On any error during build process
    
    Examples:
        Build with defaults:
        >>> args = argparse.Namespace(file='app.n3', ...)
        >>> cmd_build(args)  # doctest: +SKIP
        ✓ Static site generated in build/
        ✓ Backend scaffold generated in backend/
    """
    try:
        # Get CLI context
        ctx = get_cli_context(args)
        source_path = validate_path(Path(args.file).resolve(), must_exist=True)
        
        # Resolve configuration
        workspace = ctx.config
        workspace.ensure_apps([])
        app_entry = match_app_config(workspace, source_path)
        
        defaults = workspace.defaults
        
        # Resolve backend directory
        backend_override = validate_path(
            getattr(args, "backend_out", None),
            allow_none=True
        )
        backend_source = backend_override or (
            app_entry.backend_out if app_entry else defaults.backend_out
        )
        backend_dir = Path(backend_source)
        if not backend_dir.is_absolute():
            backend_dir = (source_path.parent / backend_dir).resolve()
        
        # Resolve frontend directory
        frontend_override = validate_path(
            getattr(args, "out", None),
            allow_none=True
        )
        frontend_source = frontend_override or (
            app_entry.frontend_out if app_entry else defaults.frontend_out
        )
        frontend_dir = Path(frontend_source)
        if not frontend_dir.is_absolute():
            frontend_dir = (source_path.parent / frontend_dir).resolve()
        
        # Resolve frontend target
        explicit_target = validate_string(
            getattr(args, "target", None),
            allow_none=True
        )
        target_default = app_entry.target if app_entry else defaults.target
        target = validate_target_type(explicit_target or target_default)
        
        # Resolve realtime flag
        explicit_rt = validate_bool(
            getattr(args, "realtime", None),
            allow_none=True
        )
        if explicit_rt is not None:
            realtime_flag = explicit_rt
        else:
            rt_default = (
                app_entry.enable_realtime
                if app_entry and app_entry.enable_realtime is not None
                else defaults.enable_realtime
            )
            realtime_flag = bool(rt_default)
        
        # Merge environment variables
        env_entries = env_strings_from_config(workspace.defaults.env)
        if app_entry:
            env_entries = merge_env(
                env_entries,
                env_strings_from_config(app_entry.env)
            )
        env_entries = merge_env(env_entries, getattr(args, "env", []))
        apply_env_overrides(env_entries)
        
        # Load N3 application
        app = load_n3_app(source_path)
        
        # Print AST if requested
        if getattr(args, "print_ast", False):
            def default(o):
                if hasattr(o, "__dict__"):
                    return o.__dict__
                return str(o)
            
            print(json.dumps(app, default=default, indent=2))
            return
        
        # Get flags
        embed_flag = getattr(args, "embed_insights", False)
        backend_only = getattr(args, "backend_only", False)
        build_backend = getattr(args, "build_backend", False)
        export_schemas = getattr(args, "export_schemas", False)
        schema_version = getattr(args, "schema_version", "1.0.0")
        
        # Generate frontend
        if not backend_only:
            generate_site(
                app,
                str(frontend_dir),
                enable_realtime=realtime_flag,
                target=target,
            )
            if target == "static":
                print_success(f"Static site generated in {frontend_dir}")
            else:
                print_success(f"Frontend [{target}] generated in {frontend_dir}")
        
        # Generate backend
        if build_backend or backend_only:
            connector_cfg = extract_connector_config(
                app_entry, defaults
            ) if app_entry else extract_connector_config(None, defaults)
            
            generate_backend(
                app,
                str(backend_dir),
                embed_insights=embed_flag,
                enable_realtime=realtime_flag,
                connector_config=connector_cfg,
                export_schemas=export_schemas,
                schema_version=schema_version,
            )
            print_success(f"Backend scaffold generated in {backend_dir}")
            
            # Print backend summary
            summary = generate_backend_summary(app)
            print(summary)
        
        # Emit build event to plugins
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
    
    except Exception as exc:
        handle_cli_exception(exc)
