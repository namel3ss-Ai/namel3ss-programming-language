"""
Namel3ss CLI entry point.

This module provides the main CLI interface for the Namel3ss language,
dispatching commands to focused command modules while maintaining backward
compatibility with the original monolithic CLI.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from namel3ss import __version__
from namel3ss.lang import LANGUAGE_VERSION as LANGUAGE_SPEC_VERSION
from namel3ss.config import load_workspace_config
from namel3ss.plugins import PluginManager

# Import core commands that work with minimal dependencies
from .commands import (
    cmd_build,
    cmd_run,
    cmd_doctor,
    _lazy_import_deploy,
    _lazy_import_debug,
    _lazy_import_local_deploy,
    _lazy_import_ai_commands,
    _lazy_import_dev_tools,
    _lazy_import_enhanced_cli,
)

# Import lazy loaders for optional commands
from .lazy_imports import (
    lazy_cmd_eval,
    lazy_cmd_eval_suite,
    lazy_cmd_train,
    lazy_cmd_deploy,
    lazy_cmd_test,
    lazy_cmd_lint,
    lazy_cmd_typecheck,
    lazy_cmd_format,
    lazy_cmd_lsp,
)

# Set up add_command functions through lazy imports
try:
    # Local deployment commands
    _, add_local_deploy_command = _lazy_import_local_deploy()
except:
    def add_local_deploy_command(parser):
        local_parser = parser.add_parser('local', help='Local model deployment (requires local-deploy extra)')
        local_parser.set_defaults(func=lambda args: (print("Error: Local deployment requires the 'local-deploy' extra."), print("Install with: pip install namel3ss[local-deploy]"), sys.exit(1))[2])

try:
    # Debug commands
    _, add_debug_command = _lazy_import_debug()
except:
    def add_debug_command(parser):
        debug_parser = parser.add_parser('debug', help='Debug features (requires cli extra)')
        debug_parser.set_defaults(func=lambda args: (print("Error: Debug features require the 'cli' extra."), print("Install with: pip install namel3ss[cli]"), sys.exit(1))[2])

try:
    # Enhanced CLI commands
    (_, add_stdlib_command, _, add_packages_command, 
     _, add_modules_command, add_security_command,
     _, add_conformance_command) = _lazy_import_enhanced_cli()
except:
    def add_stdlib_command(parser):
        stdlib_parser = parser.add_parser('stdlib', help='Standard library features (requires cli extra)')
        stdlib_parser.set_defaults(func=lambda args: (print("Error: Standard library features require the 'cli' extra."), print("Install with: pip install namel3ss[cli]"), sys.exit(1))[2])
    
    def add_packages_command(parser):
        packages_parser = parser.add_parser('packages', help='Package management (requires cli extra)')
        packages_parser.set_defaults(func=lambda args: (print("Error: Package management requires the 'cli' extra."), print("Install with: pip install namel3ss[cli]"), sys.exit(1))[2])
    
    def add_modules_command(parser):
        modules_parser = parser.add_parser('modules', help='Module introspection (requires cli extra)')
        modules_parser.set_defaults(func=lambda args: (print("Error: Module introspection requires the 'cli' extra."), print("Install with: pip install namel3ss[cli]"), sys.exit(1))[2])
    
    def add_security_command(parser):
        security_parser = parser.add_parser('security', help='Security validation (requires cli extra)')
        security_parser.set_defaults(func=lambda args: (print("Error: Security validation requires the 'cli' extra."), print("Install with: pip install namel3ss[cli]"), sys.exit(1))[2])
    
    def add_conformance_command(parser):
        conformance_parser = parser.add_parser('conformance', help='Language conformance testing (requires cli extra)')
        conformance_parser.set_defaults(func=lambda args: (print("Error: Language conformance testing requires the 'cli' extra."), print("Install with: pip install namel3ss[cli]"), sys.exit(1))[2])

# Try to import optional SDK sync (may not be available)
try:
    from namel3ss.sdk_sync.cli import add_sdk_sync_command
except ImportError:
    def add_sdk_sync_command(parser):
        def sdk_sync_cmd(args):
            print("Error: SDK sync requires additional dependencies.")
            print("Install with: pip install 'namel3ss[dev]'")
            import sys
            sys.exit(1)
        sdk_parser = parser.add_parser('sdk-sync', help="Generate typed SDK from N3 schemas (requires dev extra)")
        sdk_parser.set_defaults(func=sdk_sync_cmd)

from .context import CLIContext
from .validation import normalize_run_command_args


def _configure_runtime_logging(args) -> None:
    """Configure logging level for runtime log statements."""
    import logging
    import os
    
    # Determine log level from CLI arg, environment, or default
    log_level = (
        getattr(args, 'log_level', None) or 
        os.getenv('NAMEL3SS_LOG_LEVEL', 'info')
    ).lower()
    
    # Map string to logging level
    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'error': logging.ERROR,
    }
    
    numeric_level = level_map.get(log_level, logging.INFO)
    
    # Configure the namel3ss.runtime logger used by log statements
    runtime_logger = logging.getLogger('namel3ss.runtime')
    runtime_logger.setLevel(numeric_level)
    
    # Add console handler if not already present
    if not runtime_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        runtime_logger.addHandler(handler)
        # Prevent propagation to root logger to avoid duplicate messages
        runtime_logger.propagate = False


def main(argv: Optional[list] = None) -> None:
    """
    Main CLI entrypoint with subcommand support.
    
    Provides comprehensive command-line interface for the Namel3ss language,
    supporting building, running, evaluation, training, deployment, and
    development tooling commands.
    
    Args:
        argv: Command-line arguments (None uses sys.argv[1:])
    
    Examples:
        Build a project:
        >>> main(['build', 'app.ai'])  # doctest: +SKIP
        
        Run development server:
        >>> main(['run', 'app.ai'])  # doctest: +SKIP
        
        Run experiment:
        >>> main(['eval', 'my_experiment', '--file', 'app.ai'])  # doctest: +SKIP
    """
    if argv is None:
        argv = sys.argv[1:]
    
    # Valid command names for legacy detection
    valid_commands = {
        'build', 'run', 'eval', 'eval-suite', 'train', 'deploy', 'doctor', 
        'test', 'lint', 'typecheck', 'format', 'lsp', 'sdk-sync', 'stdlib', 
        'packages', 'pkg', 'modules', 'mod', 'build-index', 'help'
    }
    
    # Legacy invocation support: convert bare .ai/.ai file to 'build' command
    if (
        len(argv) > 0
        and not argv[0].startswith('-')
        and argv[0] not in valid_commands
        and (argv[0].endswith('.ai') or argv[0].endswith('.ai') or Path(argv[0]).exists())
    ):
        print(
            "Note: Using legacy invocation. Consider using 'namel3ss build' instead.",
            file=sys.stderr
        )
        argv = ['build'] + argv
    
    # Normalize run command arguments (natural language syntax)
    if argv and argv[0] == 'run':
        argv = normalize_run_command_args(argv)
    
    # Pre-parse to get workspace and config for plugin loading
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config')
    pre_parser.add_argument('--workspace')
    pre_args, _ = pre_parser.parse_known_args(argv)
    
    workspace_root = (
        Path(pre_args.workspace).resolve()
        if pre_args.workspace
        else Path.cwd()
    )
    config_path = (
        Path(pre_args.config).resolve()
        if pre_args.config
        else None
    )
    config = load_workspace_config(workspace_root, config_path)
    plugin_manager = PluginManager(config.plugins)
    plugin_manager.load()
    
    # Main parser
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
        '--verbose',
        action='store_true',
        help='Print full tracebacks and detailed CLI errors (or set NAMEL3SS_VERBOSE=1)'
    )
    parser.add_argument(
        '--workspace',
        default=str(workspace_root),
        help='Workspace root directory (defaults to current working directory)'
    )
    parser.add_argument(
        '--log-level',
        choices=['debug', 'info', 'warn', 'error'],
        default=None,
        help='Set logging level for runtime log statements (or set NAMEL3SS_LOG_LEVEL)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    plugin_manager.register_commands(subparsers)
    
    # Build subcommand
    build_parser = subparsers.add_parser(
        'build',
        help='Generate static site and/or backend scaffold'
    )
    build_parser.add_argument('file', help='Path to the .ai source file (.ai also supported)')
    build_parser.add_argument(
        '--out', '-o', default='build', help='Output directory for static files'
    )
    build_parser.add_argument(
        '--print-ast', action='store_true', help='Print the parsed AST and exit'
    )
    build_parser.add_argument(
        '--build-backend', action='store_true',
        help='Also generate FastAPI backend scaffold'
    )
    build_parser.add_argument(
        '--realtime', action='store_true',
        help='Enable realtime websocket scaffolding'
    )
    build_parser.add_argument(
        '--target',
        choices=['static', 'react-vite'],
        default='static',
        help='Frontend target to generate (default: static)'
    )
    build_parser.add_argument(
        '--backend-only', action='store_true',
        help='Only generate backend, skip static site'
    )
    build_parser.add_argument(
        '--backend-out', default='backend_build',
        help='Output directory for backend scaffold'
    )
    build_parser.add_argument(
        '--embed-insights', action='store_true',
        help='Embed insight results directly in dataset responses'
    )
    build_parser.add_argument(
        '--export-schemas', action='store_true',
        help='Export schemas for SDK generation (enables /api/_meta endpoints)'
    )
    build_parser.add_argument(
        '--schema-version',
        default='1.0.0',
        help='Version for exported schemas (default: 1.0.0)'
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
        help='Chain name to execute or path to the .ai source file (.ai also supported)'
    )
    run_parser.add_argument(
        '-f', '--file',
        dest='file',
        help='Explicit .ai source file to use when running a chain (.ai also supported)'
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
        help='Enable realtime websocket streaming'
    )
    run_parser.add_argument(
        '--frontend-out',
        default=None,
        help='Output directory for frontend scaffold'
    )
    run_parser.add_argument(
        '--frontend-port',
        type=int,
        default=None,
        help='Port for frontend dev server (if separate from backend)'
    )
    run_parser.add_argument(
        '--json',
        action='store_true',
        help='Output chain results in JSON format'
    )
    run_parser.add_argument(
        '--env',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='Set environment variable (may be provided multiple times)'
    )
    run_parser.add_argument(
        '--apps',
        nargs='*',
        help='Run specific apps from workspace configuration'
    )
    run_parser.add_argument(
        '--workspace-flag',
        dest='workspace',
        action='store_true',
        help='Run all workspace apps'
    )
    run_parser.set_defaults(func=cmd_run)
    
    # Eval subcommand
    eval_parser = subparsers.add_parser(
        'eval',
        help='Run an experiment evaluation'
    )
    eval_parser.add_argument('experiment', help='Name of the experiment to evaluate')
    eval_parser.add_argument(
        '-f', '--file',
        dest='file',
        help='Path to the .ai source file (.ai also supported)'
    )
    eval_parser.add_argument(
        '--format',
        choices=['json', 'text'],
        default='json',
        help='Output format (default: json)'
    )
    eval_parser.set_defaults(func=lazy_cmd_eval)
    
    # Eval-suite subcommand
    eval_suite_parser = subparsers.add_parser(
        'eval-suite',
        help='Run a comprehensive evaluation suite'
    )
    eval_suite_parser.add_argument('suite', help='Name of the evaluation suite to run')
    eval_suite_parser.add_argument(
        '-f', '--file',
        dest='file',
        help='Path to the .ai source file (.ai also supported)'
    )
    eval_suite_parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of examples to evaluate'
    )
    eval_suite_parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for evaluation (default: 1)'
    )
    eval_suite_parser.add_argument(
        '--output',
        help='Output file path for results (default: stdout)'
    )
    eval_suite_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Include per-example metrics in output'
    )
    eval_suite_parser.set_defaults(func=lazy_cmd_eval_suite)
    
    # Train subcommand
    train_parser = subparsers.add_parser(
        'train',
        help='Train or fine-tune models via declared training jobs'
    )
    train_parser.add_argument('file', help='Path to the .ai source file (.ai also supported)')
    train_parser.add_argument(
        '--job',
        help='Name of the training job to execute'
    )
    train_parser.add_argument(
        '--list',
        action='store_true',
        help='List available training jobs and exit'
    )
    train_parser.add_argument(
        '--backends',
        action='store_true',
        help='List registered training backends and exit'
    )
    train_parser.add_argument(
        '--plan',
        action='store_true',
        help='Resolve the training plan instead of executing it'
    )
    train_parser.add_argument(
        '--history',
        action='store_true',
        help='Show recent execution history for the selected job'
    )
    train_parser.add_argument(
        '--history-limit',
        type=int,
        default=5,
        help='Number of history entries to display with --history (default: 5)'
    )
    train_parser.add_argument(
        '--payload',
        help='Inline JSON payload to provide as training inputs'
    )
    train_parser.add_argument(
        '--payload-file',
        help='Path to a JSON file containing the training payload'
    )
    train_parser.add_argument(
        '--overrides',
        help='Inline JSON overrides for hyperparameters or resource hints'
    )
    train_parser.add_argument(
        '--overrides-file',
        help='Path to a JSON file containing override values'
    )
    train_parser.add_argument(
        '--json',
        action='store_true',
        help='Emit compact JSON output (default pretty prints)'
    )
    train_parser.set_defaults(func=lazy_cmd_train)
    
    # Deploy subcommand with subcommands for different deployment types
    deploy_parser = subparsers.add_parser(
        'deploy',
        help='Deploy models to various environments'
    )
    deploy_subparsers = deploy_parser.add_subparsers(
        dest='deploy_command',
        help='Deployment targets'
    )
    
    # Original deploy command (for backward compatibility)
    cloud_deploy_parser = deploy_subparsers.add_parser(
        'cloud',
        help='Deploy a model to cloud endpoints (original deploy command)'
    )
    cloud_deploy_parser.add_argument('file', help='Path to the .ai source file (.ai also supported)')
    cloud_deploy_parser.add_argument(
        '--model',
        required=True,
        help='Name of the model to deploy (must exist in the DSL or model registry)'
    )
    cloud_deploy_parser.set_defaults(func=lazy_cmd_deploy)
    
    # Add local deployment subcommands
    add_local_deploy_command(deploy_subparsers)
    
    # Doctor subcommand
    doctor_parser = subparsers.add_parser(
        'doctor',
        help='Diagnose installed dependencies and optional extras'
    )
    doctor_parser.set_defaults(func=cmd_doctor)
    
    # Test subcommand
    test_parser = subparsers.add_parser(
        'test',
        help='Run namel3ss application tests with deterministic mocks'
    )
    test_parser.add_argument(
        'files',
        nargs='*',
        help='Test files or directories to run (default: discover tests in tests/ and current directory)'
    )
    test_parser.add_argument(
        '--pattern',
        default='*.test.yaml',
        help='Pattern for test file discovery (default: *.test.yaml)'
    )
    test_parser.add_argument(
        '--external',
        action='store_true',
        help='Use external test runner instead of native namel3ss testing'
    )
    test_parser.add_argument(
        '--command',
        help='Override the configured test command (only used with --external)'
    )
    test_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output showing test execution details'
    )
    test_parser.add_argument(
        '--fail-fast', '-f',
        action='store_true',
        dest='fail_fast',
        help='Stop on first test failure'
    )
    test_parser.set_defaults(func=lazy_cmd_test)
    
    # Lint subcommand
    lint_parser = subparsers.add_parser(
        'lint',
        help='Run semantic analysis and linting on Namel3ss files'
    )
    lint_parser.add_argument(
        'files',
        nargs='*',
        default=['.'],
        help='Files or directories to lint (default: current directory)'
    )
    lint_parser.add_argument(
        '--command',
        help='Override with external lint command (e.g., ruff, flake8)'
    )
    lint_parser.add_argument(
        '--external',
        action='store_true',
        help='Force use of external linter instead of native semantic analysis'
    )
    lint_parser.set_defaults(func=lazy_cmd_lint)
    
    # Typecheck subcommand
    typecheck_parser = subparsers.add_parser(
        'typecheck',
        help='Run the configured type checking command'
    )
    typecheck_parser.add_argument(
        '--command',
        help='Override the configured typecheck command'
    )
    typecheck_parser.set_defaults(func=lazy_cmd_typecheck)
    
    # Format subcommand
    format_parser = subparsers.add_parser(
        'format',
        help='Format Namel3ss source files using AST-based formatter'
    )
    format_parser.add_argument(
        'files',
        nargs='*',
        default=['.'],
        help='Files or directories to format (default: current directory)'
    )
    format_parser.add_argument(
        '--check',
        action='store_true',
        help='Check if files need formatting without making changes'
    )
    format_parser.add_argument(
        '--diff',
        action='store_true',
        help='Show diff of formatting changes'
    )
    format_parser.add_argument(
        '--write',
        action='store_true',
        default=True,
        help='Write formatting changes to files (default: true)'
    )
    format_parser.set_defaults(func=lazy_cmd_format)
    
    # LSP subcommand
    lsp_parser = subparsers.add_parser(
        'lsp',
        help='Start the Namel3ss language server for editor integrations'
    )
    lsp_parser.set_defaults(func=lazy_cmd_lsp)
    
    # SDK-Sync subcommand
    add_sdk_sync_command(subparsers)
    add_debug_command(subparsers)
    add_stdlib_command(subparsers)
    add_packages_command(subparsers)
    add_modules_command(subparsers)
    add_security_command(subparsers)
    add_conformance_command(subparsers)
    
    # RAG index building command
    build_index_parser = subparsers.add_parser(
        'build-index',
        help='Build a RAG vector index from a dataset'
    )
    build_index_parser.add_argument(
        'source',
        help='Path to the .ai source file (.ai also supported)'
    )
    build_index_parser.add_argument(
        'index',
        help='Name of the index to build'
    )
    build_index_parser.add_argument(
        '--dataset',
        help='Override the source dataset name'
    )
    build_index_parser.add_argument(
        '-n', '--max-documents',
        type=int,
        metavar='N',
        help='Maximum number of documents to index'
    )
    build_index_parser.add_argument(
        '--filter',
        action='append',
        metavar='KEY=VALUE',
        help='Metadata filter (can be specified multiple times, '
             'e.g., --filter tag=support --filter lang=en)'
    )
    build_index_parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume indexing from previous checkpoint'
    )
    build_index_parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild the index from scratch (delete previous state)'
    )
    build_index_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    build_index_parser.set_defaults(
        func=lambda args: __import__(
            'namel3ss.cli_rag',
            fromlist=['cmd_build_index']
        ).cmd_build_index(args)
    )
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    # Reload config if workspace or config path changed during parsing
    runtime_workspace = (
        Path(args.workspace).resolve()
        if getattr(args, 'workspace', None)
        else workspace_root
    )
    runtime_config_path = (
        Path(args.config).resolve()
        if getattr(args, 'config', None)
        else config_path
    )
    if runtime_workspace != workspace_root or runtime_config_path != config_path:
        config = load_workspace_config(runtime_workspace, runtime_config_path)
        plugin_manager = PluginManager(config.plugins)
        plugin_manager.load()
    
    # If no command specified, print help
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    # Attach CLI context to args
    args.cli_context = CLIContext(
        workspace_root=runtime_workspace,
        config=config,
        plugin_manager=plugin_manager,
    )
    args.verbose = getattr(args, "verbose", False)
    
    # Configure logging level
    _configure_runtime_logging(args)
    
    # Execute command
    args.func(args)


if __name__ == '__main__':  # pragma: no cover
    main()
