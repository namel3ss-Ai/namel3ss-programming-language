"""
Command line interface for Namel3ss.

DEPRECATED: This module is a backward compatibility wrapper.
Import from namel3ss.cli package instead.

This module re-exports all CLI functionality from the refactored cli package
to maintain backward compatibility with existing code and tests.
"""

# Re-export main entry point
from .cli import main

# Re-export command functions
from .cli.commands import (
    cmd_build,
    cmd_deploy,
    cmd_doctor,
    cmd_eval,
    cmd_eval_suite,
    cmd_lint,
    cmd_lsp,
    cmd_run,
    cmd_test,
    cmd_train,
    cmd_typecheck,
)

# Re-export context and dataclasses
from .cli.context import CLIContext

# Re-export loading functions
from .cli.loading import (
    load_n3_app as _load_cli_app,
    load_runtime_module as _load_runtime_module,
    prepare_backend,
    clear_generated_module_cache as _clear_generated_module_cache,
    load_json_argument as _load_json_argument,
)

# Re-export validation functions
from .cli.validation import (
    apply_env_overrides as _apply_env_overrides,
    normalize_run_command_args as _normalize_run_command_args,
)

# Re-export run functions
from .cli.commands.run import (
    check_uvicorn_available,
    run_dev_server,
    RunInvocation,
    _format_error_detail,
    _traceback_excerpt,
)

# Re-export build dataclass
from .cli.commands.build import BuildInvocation

# Re-export utility functions
from .cli.utils import (
    find_first_n3_file as _find_first_n3_file,
    find_chain as _find_chain,
    find_experiment as _find_experiment,
    generate_backend_summary as _backend_summary_lines,
    get_program_root as _program_root_for,
    pluralize as _plural,
    resolve_model_spec as _resolve_model_spec,
    slugify_model_name as _slugify_model_name,
)

# Re-export context functions
from .cli.context import (
    get_cli_context as _cli_context,
    match_app_config as _match_app_config,
    get_runtime_env as _runtime_env,
    create_ephemeral_app_config as _create_ephemeral_app_config,
    get_effective_realtime as _effective_realtime,
)

# Re-export error formatting
from .cli.errors import format_cli_error as _format_cli_error

# Re-export constants that were in the original module
from .cli.validation import ENV_ALIAS_MAP

# For compatibility with code that may check these
_CLI_TRACE_LIMIT = 2000
_RUN_ENV_PREPOSITIONS = {"using", "in", "on", "with"}

__all__ = [
    # Main entry point
    "main",
    
    # Command functions
    "cmd_build",
    "cmd_deploy",
    "cmd_doctor",
    "cmd_eval",
    "cmd_eval_suite",
    "cmd_lint",
    "cmd_lsp",
    "cmd_run",
    "cmd_test",
    "cmd_train",
    "cmd_typecheck",
    
    # Context and dataclasses
    "CLIContext",
    "BuildInvocation",
    "RunInvocation",
    
    # Public functions
    "check_uvicorn_available",
    "run_dev_server",
    "prepare_backend",
    
    # Internal functions (with underscore prefix for backward compatibility)
    "_load_cli_app",
    "_load_runtime_module",
    "_clear_generated_module_cache",
    "_load_json_argument",
    "_apply_env_overrides",
    "_normalize_run_command_args",
    "_format_error_detail",
    "_traceback_excerpt",
    "_find_first_n3_file",
    "_find_chain",
    "_find_experiment",
    "_backend_summary_lines",
    "_program_root_for",
    "_plural",
    "_resolve_model_spec",
    "_slugify_model_name",
    "_cli_context",
    "_match_app_config",
    "_runtime_env",
    "_create_ephemeral_app_config",
    "_effective_realtime",
    "_format_cli_error",
    
    # Constants
    "ENV_ALIAS_MAP",
    "_CLI_TRACE_LIMIT",
    "_RUN_ENV_PREPOSITIONS",
]


if __name__ == '__main__':  # pragma: no cover
    main()
