"""
Production-grade error handling for Namel3ss CLI.

This module provides a comprehensive exception hierarchy for CLI operations,
with rich context, error codes, and user-friendly formatting.
"""

import os
import sys
import traceback
from typing import Any, Dict, Optional


# Maximum length for traceback output in CLI
_CLI_TRACE_LIMIT = 4000


class CLIError(Exception):
    """
    Base exception for all CLI operations.
    
    Provides structured error information including error codes,
    contextual hints, and metadata for programmatic handling.
    
    Attributes:
        message: Human-readable error description
        code: Machine-readable error code for programmatic handling
        hint: Optional suggestion for resolving the error
        context: Additional metadata about the error
    """
    
    def __init__(
        self,
        message: str,
        *,
        code: str,
        hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize CLI error with rich context.
        
        Args:
            message: Human-readable error description
            code: Error code (e.g., 'CLI001', 'BUILD_FAILED')
            hint: Optional suggestion for fixing the error
            context: Additional metadata (file paths, line numbers, etc.)
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.hint = hint
        self.context = context or {}
    
    def __str__(self) -> str:
        """Format error for display."""
        return self.message


class CLIConfigError(CLIError):
    """
    Configuration file or workspace setup errors.
    
    Raised when:
    - Workspace configuration file is invalid or missing
    - App configuration has validation errors
    - Environment variables are misconfigured
    """
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('code', 'CLI_CONFIG_ERROR')
        super().__init__(message, **kwargs)


class CLIValidationError(CLIError):
    """
    Invalid command arguments or options.
    
    Raised when:
    - Required arguments are missing
    - Argument values are invalid or out of range
    - Incompatible options are used together
    """
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('code', 'CLI_VALIDATION_ERROR')
        super().__init__(message, **kwargs)


class CLIRuntimeError(CLIError):
    """
    Errors during command execution.
    
    Base class for runtime failures that occur during command execution,
    as opposed to configuration or validation errors.
    """
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('code', 'CLI_RUNTIME_ERROR')
        super().__init__(message, **kwargs)


class CLIBuildError(CLIRuntimeError):
    """
    Build command failures.
    
    Raised when:
    - Frontend or backend generation fails
    - Output directories cannot be created
    - Generated code has syntax errors
    """
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('code', 'CLI_BUILD_ERROR')
        super().__init__(message, **kwargs)


class CLIServerError(CLIRuntimeError):
    """
    Development server failures.
    
    Raised when:
    - Server cannot bind to port
    - uvicorn is not available
    - Server crashes during startup
    """
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('code', 'CLI_SERVER_ERROR')
        super().__init__(message, **kwargs)


class CLIFileNotFoundError(CLIError):
    """
    Required file or directory not found.
    
    Raised when:
    - Source .n3 file doesn't exist
    - Configuration file is missing
    - Required directory is not present
    """
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('code', 'CLI_FILE_NOT_FOUND')
        super().__init__(message, **kwargs)


class CLIDependencyError(CLIError):
    """
    Missing or incompatible dependencies.
    
    Raised when:
    - Required Python packages are not installed
    - Package versions are incompatible
    - System dependencies are missing
    """
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('code', 'CLI_DEPENDENCY_ERROR')
        super().__init__(message, **kwargs)


def format_cli_error(
    exc: BaseException,
    *,
    verbose: bool = False,
    include_traceback: bool = False
) -> str:
    """
    Format exception for CLI display with context and hints.
    
    Provides user-friendly error messages with:
    - Clear error type and message
    - Actionable hints when available
    - Optional traceback for debugging
    - Consistent formatting across all error types
    
    Args:
        exc: Exception to format
        verbose: Include additional context and metadata
        include_traceback: Include full Python traceback
    
    Returns:
        Formatted error message suitable for CLI output
    
    Examples:
        >>> try:
        ...     raise CLIValidationError("Invalid port", hint="Use port 1024-65535")
        ... except Exception as e:
        ...     print(format_cli_error(e))
        Error: Invalid port
        Hint: Use port 1024-65535
    """
    # Check if exception has custom formatting method
    formatter = getattr(exc, "format", None)
    if callable(formatter):
        try:
            return formatter()
        except Exception:
            pass
    
    # Build error message
    lines = []
    
    # Error type and message
    if isinstance(exc, CLIError):
        lines.append(f"Error [{exc.code}]: {exc.message}")
        
        # Add hint if available
        if exc.hint:
            lines.append(f"Hint: {exc.hint}")
        
        # Add context in verbose mode
        if verbose and exc.context:
            lines.append("\nContext:")
            for key, value in exc.context.items():
                lines.append(f"  {key}: {value}")
    else:
        # Generic exception formatting
        error_type = exc.__class__.__name__
        lines.append(f"Error: {error_type}: {exc}")
    
    # Add traceback if requested
    if include_traceback:
        lines.append("\nTraceback:")
        lines.append(format_traceback_excerpt())
    
    return "\n".join(lines)


def format_error_detail(exc: BaseException) -> str:
    """
    Format error as a single-line detail string.
    
    Useful for logging or compact error display. Truncates long messages
    to prevent overwhelming output.
    
    Args:
        exc: Exception to format
    
    Returns:
        Single-line error detail, truncated if too long
    
    Examples:
        >>> exc = ValueError("Invalid configuration value")
        >>> format_error_detail(exc)
        'ValueError: Invalid configuration value'
    """
    message = f"{exc.__class__.__name__}: {exc}"
    return message if len(message) <= 280 else f"{message[:277]}..."


def format_traceback_excerpt() -> str:
    """
    Format current exception traceback with size limit.
    
    Captures the current exception's traceback and formats it for display,
    truncating if it exceeds the CLI trace limit to prevent overwhelming
    terminal output.
    
    Returns:
        Formatted traceback string, truncated if necessary
    
    Note:
        Should only be called within an exception handler context.
    """
    trace = traceback.format_exc().strip()
    if len(trace) <= _CLI_TRACE_LIMIT:
        return trace
    return f"{trace[:_CLI_TRACE_LIMIT - 3]}..."


def wrap_exception(
    exc: BaseException,
    *,
    message: str,
    error_class: type = CLIRuntimeError,
    **kwargs
) -> CLIError:
    """
    Wrap a generic exception as a CLI-specific error.
    
    Preserves the original exception as context while providing
    CLI-appropriate error messaging and structure.
    
    Args:
        exc: Original exception to wrap
        message: CLI-friendly error message
        error_class: CLIError subclass to use for wrapping
        **kwargs: Additional arguments for error class constructor
    
    Returns:
        CLI-specific error with original exception in context
    
    Examples:
        >>> try:
        ...     open('/nonexistent')
        ... except FileNotFoundError as e:
        ...     cli_err = wrap_exception(
        ...         e,
        ...         message="Could not open configuration file",
        ...         error_class=CLIConfigError,
        ...         hint="Check that the file exists"
        ...     )
        ...     raise cli_err
    """
    context = kwargs.get('context', {})
    context['original_exception'] = str(exc)
    context['original_type'] = exc.__class__.__name__
    kwargs['context'] = context
    
    return error_class(message, **kwargs)


def _env_flag(name: str) -> bool:
    val = os.getenv(name)
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


def cli_verbose_enabled(verbose_flag: bool = False) -> bool:
    """
    Determine whether verbose error output is enabled.
    
    Respects an explicit flag and the NAMEL3SS_VERBOSE/NAMEL3SS_DEBUG
    environment variables.
    """
    return verbose_flag or _env_flag("NAMEL3SS_VERBOSE") or _env_flag("NAMEL3SS_DEBUG")


def cli_reraise_enabled() -> bool:
    """
    Determine whether exceptions should be re-raised instead of exiting.
    
    Controlled by NAMEL3SS_RERAISE or NAMEL3SS_DEBUG environment variables.
    """
    return _env_flag("NAMEL3SS_RERAISE") or _env_flag("NAMEL3SS_DEBUG")


def handle_cli_exception(
    exc: BaseException,
    *,
    verbose: bool = False,
    exit_code: int = 1
) -> None:
    """
    Handle exception at CLI top-level with proper formatting and exit.
    
    Central error handler for CLI commands. Formats the error appropriately
    and exits with the specified code. Should be called from command handlers.
    
    Args:
        exc: Exception to handle
        verbose: Enable verbose error output
        exit_code: Exit code to use (default: 1)
    
    Note:
        This function calls sys.exit() and does not return.
    """
    verbose_effective = cli_verbose_enabled(verbose)
    if cli_reraise_enabled():
        if verbose_effective:
            raise
        raise exc

    # Format and print error
    error_message = format_cli_error(
        exc,
        verbose=verbose_effective,
        include_traceback=verbose_effective
    )
    print(error_message, file=sys.stderr)
    
    # Exit with appropriate code
    sys.exit(exit_code)
