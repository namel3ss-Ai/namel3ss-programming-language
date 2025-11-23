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
    
    Uses the native Namel3ss semantic linter for deep analysis,
    with fallback to configured external lint command if needed.
    
    Args:
        args: Parsed command-line arguments containing:
            - command: Optional command override
            - files: Optional list of files to lint (default: current directory)
            - external: Force use of external linter
    
    Raises:
        SystemExit: If linting fails
    
    Examples:
        >>> args = argparse.Namespace(command=None, files=[], external=False)
        >>> cmd_lint(args)  # doctest: +SKIP
        → Running native semantic linter
    """
    try:
        ctx = get_cli_context(args)
        override = getattr(args, "command", None)
        use_external = getattr(args, "external", False)
        files = getattr(args, "files", ["."])
        
        # Use native semantic linter by default
        if not override and not use_external:
            try:
                from namel3ss.linter import SemanticLinter, get_default_rules
                import pathlib
                
                linter = SemanticLinter(get_default_rules())
                
                # Collect files to lint
                files_to_lint = []
                for file_arg in files:
                    path = pathlib.Path(file_arg)
                    if path.is_file() and path.suffix in ['.ai', '.n3']:
                        files_to_lint.append(path)
                    elif path.is_dir():
                        files_to_lint.extend(path.rglob('*.ai'))
                        files_to_lint.extend(path.rglob('*.n3'))
                
                if not files_to_lint:
                    print("No .ai or .n3 files found to lint")
                    return
                
                total_findings = 0
                error_count = 0
                
                print(f"→ Running native semantic linter on {len(files_to_lint)} file(s)")
                
                for file_path in files_to_lint:
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        result = linter.lint_document(content, str(file_path))
                        
                        if result.errors:
                            print(f"\\n{file_path}: ERRORS")
                            for error in result.errors:
                                print(f"  {error}")
                            error_count += 1
                            continue
                        
                        if result.warnings:
                            for warning in result.warnings:
                                print(f"Warning: {warning}")
                        
                        if result.findings:
                            print(f"\\n{file_path}:")
                            for finding in result.findings:
                                icon = "❌" if finding.severity.value == "error" else "⚠️" if finding.severity.value == "warning" else "💡"
                                location = f":{finding.line}" if finding.line else ""
                                print(f"  {icon} {finding.severity.value.upper()}{location}: {finding.message}")
                                if finding.suggestion:
                                    print(f"     💡 {finding.suggestion}")
                                if finding.code_context:
                                    print(f"     📄 {finding.code_context}")
                            
                            total_findings += len(result.findings)
                            error_count += result.error_count()
                    
                    except Exception as exc:
                        print(f"Error linting {file_path}: {exc}")
                        error_count += 1
                
                # Summary
                if total_findings == 0:
                    print("\\n✅ No issues found")
                else:
                    print(f"\\n📊 Found {total_findings} issue(s)")
                    if error_count > 0:
                        raise SystemExit(1)
                
                ctx.plugin_manager.emit_workspace(task="lint", command="native", args=args)
                return
                
            except ImportError:
                print("Native linter not available, falling back to external command")
        
        # Fall back to external linter
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


def cmd_format(args: argparse.Namespace) -> None:
    """
    Handle the 'format' subcommand to format Namel3ss source files.
    
    Formats N3 source files using the AST-based formatter for consistent
    code style. Supports both single file and directory formatting.
    
    Args:
        args: Parsed command-line arguments containing:
            - files: List of files or directories to format
            - check: If True, only check if formatting is needed
            - diff: If True, show diff of changes
            - write: If True, write changes to files (default)
    
    Raises:
        SystemExit: If formatting encounters errors or check mode finds changes
    
    Examples:
        >>> args = argparse.Namespace(files=['app.ai'], check=False, diff=False, write=True)
        >>> cmd_format(args)  # doctest: +SKIP
        Formatted 1 file successfully
    """
    try:
        from namel3ss.formatting import ASTFormatter, DefaultFormattingRules
        import pathlib
        
        ctx = get_cli_context(args)
        formatter = ASTFormatter(DefaultFormattingRules.standard())
        
        files_to_format = []
        for file_arg in args.files:
            path = pathlib.Path(file_arg)
            if path.is_file() and path.suffix in ['.ai', '.n3']:
                files_to_format.append(path)
            elif path.is_dir():
                files_to_format.extend(path.rglob('*.ai'))
                files_to_format.extend(path.rglob('*.n3'))
            else:
                print(f"Warning: Skipping {file_arg} (not a .ai/.n3 file or directory)")
        
        if not files_to_format:
            print("No files to format")
            return
        
        formatted_count = 0
        error_count = 0
        
        for file_path in files_to_format:
            try:
                content = file_path.read_text(encoding='utf-8')
                result = formatter.format_document(content, str(file_path))
                
                if result.errors:
                    print(f"Error formatting {file_path}:")
                    for error in result.errors:
                        print(f"  {error}")
                    error_count += 1
                    continue
                
                if result.warnings:
                    for warning in result.warnings:
                        print(f"Warning in {file_path}: {warning}")
                
                if result.is_changed:
                    if args.check:
                        print(f"Would reformat {file_path}")
                        formatted_count += 1
                    elif args.diff:
                        # Show diff (simplified)
                        print(f"--- {file_path}")
                        print(f"+++ {file_path}")
                        print("@@ Changes @@")
                        original_lines = content.splitlines()
                        formatted_lines = result.formatted_text.splitlines()
                        # Simple diff display
                        if len(original_lines) != len(formatted_lines):
                            print(f" Lines: {len(original_lines)} → {len(formatted_lines)}")
                        print(f" Content changed: {result.is_changed}")
                    elif args.write:
                        file_path.write_text(result.formatted_text, encoding='utf-8')
                        print(f"Formatted {file_path}")
                        formatted_count += 1
                
            except Exception as exc:
                print(f"Error processing {file_path}: {exc}")
                error_count += 1
        
        # Summary
        if args.check:
            if formatted_count > 0:
                print(f"{formatted_count} file(s) would be reformatted")
                raise SystemExit(1)  # Indicate changes needed
            else:
                print("All files are already formatted")
        else:
            if formatted_count > 0:
                print(f"Formatted {formatted_count} file(s) successfully")
            if error_count > 0:
                print(f"Encountered {error_count} error(s)")
                raise SystemExit(1)
    
    except ImportError as exc:
        raise CLIRuntimeError(
            "AST formatter not available",
            hint="Ensure namel3ss.formatting module is properly installed",
        ) from exc
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
