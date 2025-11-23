"""
CLI context and environment management.

This module provides the CLIContext dataclass and functions for managing
workspace configuration, environment variables, and app configuration resolution.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from ..config import (
    AppConfig,
    WorkspaceConfig,
    default_app_name,
    env_strings_from_config,
    merge_env,
)
from ..plugins import PluginManager
from .errors import CLIConfigError
from .validation import validate_bool, validate_int, validate_path, validate_string, validate_target_type


@dataclass
class CLIContext:
    """
    Shared context resolved from workspace configuration.
    
    Contains workspace-level configuration and resources that are shared
    across all CLI commands within a single invocation.
    
    Attributes:
        workspace_root: Root directory of the workspace
        config: Parsed workspace configuration
        plugin_manager: Plugin manager instance for extensibility
    """
    
    workspace_root: Path
    config: WorkspaceConfig
    plugin_manager: PluginManager


def get_cli_context(args: argparse.Namespace) -> CLIContext:
    """
    Retrieve CLIContext from parsed arguments.
    
    The context should be initialized and attached to args during
    argument parsing, before command execution.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        CLI context for the current invocation
    
    Raises:
        CLIConfigError: If context was not initialized
    
    Examples:
        >>> # In command handler
        >>> def cmd_build(args):
        ...     ctx = get_cli_context(args)
        ...     # Use ctx.config, ctx.workspace_root, etc.
    """
    ctx = getattr(args, "cli_context", None)
    if ctx is None:
        raise CLIConfigError(
            "CLI context was not initialized before command execution",
            hint="This is an internal error - please report it",
            code="CLI_CONTEXT_NOT_INITIALIZED"
        )
    return ctx


def match_app_config(config: WorkspaceConfig, source_path: Path) -> Optional[AppConfig]:
    """
    Find app configuration matching the source file path.
    
    Searches workspace configuration for an app entry with matching
    source file path (resolved to absolute path).
    
    Args:
        config: Workspace configuration to search
        source_path: Source file path to match
    
    Returns:
        Matching AppConfig or None if not found
    
    Examples:
        >>> workspace = WorkspaceConfig(...)
        >>> source = Path("app.ai")
        >>> app_cfg = match_app_config(workspace, source)
        >>> if app_cfg:
        ...     print(f"Found app: {app_cfg.name}")
    """
    resolved = source_path.resolve()
    for entry in config.apps.values():
        if entry.file.resolve() == resolved:
            return entry
    return None


def get_runtime_env(
    config: WorkspaceConfig,
    app_cfg: AppConfig,
    cli_env: Sequence[str]
) -> List[str]:
    """
    Build runtime environment variables for app execution.
    
    Merges environment variables from multiple sources in priority order:
    1. Workspace defaults
    2. App-specific configuration
    3. CLI overrides (highest priority)
    
    Args:
        config: Workspace configuration
        app_cfg: App configuration
        cli_env: Environment overrides from CLI arguments
    
    Returns:
        List of KEY=VALUE environment variable strings
    
    Examples:
        >>> workspace = WorkspaceConfig(...)
        >>> app = AppConfig(...)
        >>> cli_overrides = ["API_KEY=secret"]
        >>> env = get_runtime_env(workspace, app, cli_overrides)
        >>> "API_KEY=secret" in env
        True
    """
    # Start with workspace defaults
    base = env_strings_from_config(config.defaults.env)
    
    # Merge app-specific environment
    base = merge_env(base, env_strings_from_config(app_cfg.env))
    
    # Apply CLI overrides (highest priority)
    return merge_env(base, cli_env)


def create_ephemeral_app_config(
    workspace: WorkspaceConfig,
    source_path: Path,
    args: argparse.Namespace,
) -> AppConfig:
    """
    Create temporary app configuration from CLI arguments.
    
    Used when running apps that are not defined in workspace configuration.
    Constructs AppConfig by combining workspace defaults with CLI overrides.
    
    Args:
        workspace: Workspace configuration providing defaults
        source_path: Source .ai file path
        args: Parsed CLI arguments with optional overrides
    
    Returns:
        Ephemeral AppConfig for the execution
    
    Raises:
        CLIConfigError: If configuration values are invalid
    
    Examples:
        >>> workspace = WorkspaceConfig(...)
        >>> source = Path("adhoc.ai")
        >>> args = argparse.Namespace(port=8080, backend_out="./out")
        >>> app_cfg = create_ephemeral_app_config(workspace, source, args)
        >>> app_cfg.port
        8080
    """
    defaults = workspace.defaults
    name = default_app_name(source_path)
    
    # Backend output directory
    backend_override = validate_path(
        getattr(args, "backend_out", None),
        allow_none=True
    )
    backend = Path(backend_override or defaults.backend_out)
    if not backend.is_absolute():
        backend = (source_path.parent / backend).resolve()
    
    # Frontend output directory
    frontend_override = validate_path(
        getattr(args, "frontend_out", None),
        allow_none=True
    )
    frontend = Path(frontend_override or defaults.frontend_out)
    if not frontend.is_absolute():
        frontend = (source_path.parent / frontend).resolve()
    
    # Target type
    raw_target = validate_string(getattr(args, "target", None), allow_none=True)
    if raw_target in {"static", "react-vite"}:
        target = raw_target
    else:
        target = defaults.target
    
    # Port numbers
    port_value = getattr(args, "port", None)
    port = validate_int(port_value, allow_none=True) or defaults.port
    
    frontend_port_value = getattr(args, "frontend_port", None) if hasattr(args, "frontend_port") else None
    frontend_port = validate_int(frontend_port_value, allow_none=True) or defaults.frontend_port
    
    # Realtime flag
    realtime_flag = validate_bool(getattr(args, "realtime", None), allow_none=True)
    enable_realtime = realtime_flag if realtime_flag is not None else bool(defaults.enable_realtime)
    
    # Environment
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


def get_effective_realtime(app_cfg: AppConfig, workspace: WorkspaceConfig) -> bool:
    """
    Determine effective realtime setting for app.
    
    Falls back to workspace default if app doesn't specify realtime preference.
    
    Args:
        app_cfg: App configuration
        workspace: Workspace configuration providing defaults
    
    Returns:
        Whether realtime features should be enabled
    
    Examples:
        >>> app = AppConfig(name="app", enable_realtime=True, ...)
        >>> workspace = WorkspaceConfig(...)
        >>> get_effective_realtime(app, workspace)
        True
    """
    if app_cfg.enable_realtime is None:
        return bool(workspace.defaults.enable_realtime)
    return bool(app_cfg.enable_realtime)
