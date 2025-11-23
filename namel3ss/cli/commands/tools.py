"""
Development tools commands.

This module handles tool commands (test, lint, typecheck, lsp) that
invoke configured development tools or start language servers.
"""

import argparse
import os
import subprocess
import sys

from ..context import get_cli_context
from ..errors import CLIRuntimeError, handle_cli_exception


def _invoke_tool(command: str | None, label: str) -> None:
    """
    Invoke a development tool command.
    
    Args:
        command: Shell command to execute (None if not configured)
        label: Tool label for user messages
    
    Raises:
        SystemExit: If command returns non-zero exit code
    
    Examples:
        >>> _invoke_tool("pytest tests/", "test")  # doctest: +SKIP
        → Running test: pytest tests/
        ======================== test session starts =========================
        ...
    """
    if not command:
        print(
            f"No {label} command configured. "
            f"Update [tools.{label}] in namel3ss.toml to enable it."
        )
        return
    
    print(f"→ Running {label}: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def cmd_test(args: argparse.Namespace) -> None:
    """
    Handle the 'test' subcommand to run test suite.
    
    Invokes the test command configured in workspace configuration
    (tools.test in namel3ss.toml) or uses the command override provided
    via CLI argument.
    
    Args:
        args: Parsed command-line arguments containing:
            - command: Optional command override
    
    Raises:
        SystemExit: If tests fail
    
    Examples:
        >>> args = argparse.Namespace(command=None)
        >>> cmd_test(args)  # doctest: +SKIP
        → Running test: pytest tests/
    """
    try:
        ctx = get_cli_context(args)
        override = getattr(args, "command", None)
        command = override or ctx.config.tools.test
        ctx.plugin_manager.emit_workspace(task="test", command=command, args=args)
        _invoke_tool(command, "test")
    except Exception as exc:
        handle_cli_exception(exc, verbose=getattr(args, "verbose", False))


def cmd_lint(args: argparse.Namespace) -> None:
    """
    Handle the 'lint' subcommand to run linter.
    
    Invokes the lint command configured in workspace configuration
    (tools.lint in namel3ss.toml) or uses the command override provided
    via CLI argument.
    
    Args:
        args: Parsed command-line arguments containing:
            - command: Optional command override
    
    Raises:
        SystemExit: If linting fails
    
    Examples:
        >>> args = argparse.Namespace(command=None)
        >>> cmd_lint(args)  # doctest: +SKIP
        → Running lint: ruff check .
    """
    try:
        ctx = get_cli_context(args)
        override = getattr(args, "command", None)
        command = override or ctx.config.tools.lint
        ctx.plugin_manager.emit_workspace(task="lint", command=command, args=args)
        _invoke_tool(command, "lint")
    except Exception as exc:
        handle_cli_exception(exc, verbose=getattr(args, "verbose", False))


def cmd_typecheck(args: argparse.Namespace) -> None:
    """
    Handle the 'typecheck' subcommand to run type checker.
    
    Invokes the typecheck command configured in workspace configuration
    (tools.typecheck in namel3ss.toml) or uses the command override provided
    via CLI argument.
    
    Args:
        args: Parsed command-line arguments containing:
            - command: Optional command override
    
    Raises:
        SystemExit: If type checking fails
    
    Examples:
        >>> args = argparse.Namespace(command=None)
        >>> cmd_typecheck(args)  # doctest: +SKIP
        → Running typecheck: mypy namel3ss/
    """
    try:
        ctx = get_cli_context(args)
        override = getattr(args, "command", None)
        command = override or ctx.config.tools.typecheck
        ctx.plugin_manager.emit_workspace(task="typecheck", command=command, args=args)
        _invoke_tool(command, "typecheck")
    except Exception as exc:
        handle_cli_exception(exc, verbose=getattr(args, "verbose", False))


def cmd_lsp(args: argparse.Namespace) -> None:
    """
    Handle the 'lsp' subcommand to launch the Namel3ss language server.
    
    Starts the Namel3ss language server over stdio for editor integration.
    The server provides features like autocomplete, diagnostics, and
    go-to-definition for N3 source files.
    
    Args:
        args: Parsed command-line arguments (no specific args required)
    
    Raises:
        SystemExit: If language server fails to start or pygls is not installed
    
    Examples:
        >>> args = argparse.Namespace()
        >>> cmd_lsp(args)  # doctest: +SKIP
        Starting Namel3ss language server (pid=12345)
        [LSP server running...]
    """
    try:
        get_cli_context(args)
        
        try:
            from namel3ss.lsp.server import create_server
        except ImportError as exc:
            raise CLIRuntimeError(
                "pygls is not installed",
                hint="Install with: pip install namel3ss[dev]",
            ) from exc
        
        server = create_server()
        pid = os.getpid()
        print(f"Starting Namel3ss language server (pid={pid})", file=sys.stderr)
        
        try:
            server.start_io()
        except KeyboardInterrupt:
            print("Language server interrupted by user.", file=sys.stderr)
        except Exception as exc:
            raise CLIRuntimeError(
                f"Language server stopped unexpectedly: {exc}",
            ) from exc
    
    except Exception as exc:
        handle_cli_exception(exc, verbose=getattr(args, "verbose", False))
