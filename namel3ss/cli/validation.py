"""
Centralized validation for CLI arguments and configuration.

This module provides reusable validation functions for CLI operations,
eliminating duplicate validation logic and ensuring consistent behavior.
"""

import os
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence

from .errors import CLIValidationError


# Environment alias mapping for common names
ENV_ALIAS_MAP = {
    "production": ".env.prod",
    "prod": ".env.prod",
    "development": ".env.dev",
    "dev": ".env.dev",
    "local": ".env.local",
    "locally": ".env.local",
    "test": ".env.test",
}

# Prepositions that may appear in 'run' command for environment specification
_RUN_ENV_PREPOSITIONS = {"using", "in", "on", "with"}


def validate_path(value: Any, *, allow_none: bool = False, must_exist: bool = False) -> Optional[Path]:
    """
    Validate and convert value to Path.
    
    Args:
        value: Value to validate (string, PathLike, or None)
        allow_none: Whether None is acceptable
        must_exist: Whether the path must exist on the filesystem
    
    Returns:
        Path object or None if allow_none=True and value is None
    
    Raises:
        CLIValidationError: If value is not a valid path type or doesn't exist when must_exist=True
    
    Examples:
        >>> validate_path("/tmp/file.txt")
        PosixPath('/tmp/file.txt')
        >>> validate_path(None, allow_none=True)
        None
        >>> validate_path("/nonexistent", must_exist=True)  # doctest: +SKIP
        CLIValidationError: Path does not exist: /nonexistent
    """
    if value is None:
        if allow_none:
            return None
        raise CLIValidationError(
            "Path value cannot be None",
            hint="Provide a valid file or directory path"
        )
    
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        
        # Check existence if required
        if must_exist and not path.exists():
            raise CLIValidationError(
                f"Path does not exist: {path}",
                hint="Ensure the file or directory exists before running this command"
            )
        
        return path
    
    raise CLIValidationError(
        f"Expected path-like value, got {type(value).__name__}",
        hint="Provide a string or Path object"
    )


def validate_string(value: Any, *, allow_none: bool = False) -> Optional[str]:
    """
    Validate and convert value to string.
    
    Args:
        value: Value to validate
        allow_none: Whether None is acceptable
    
    Returns:
        String value or None if allow_none=True and value is None
    
    Raises:
        CLIValidationError: If value is not a string
    
    Examples:
        >>> validate_string("hello")
        'hello'
        >>> validate_string(123)
        Traceback (most recent call last):
        ...
        CLIValidationError: Expected string value, got int
    """
    if value is None:
        if allow_none:
            return None
        raise CLIValidationError(
            "String value cannot be None",
            hint="Provide a valid string value"
        )
    
    if isinstance(value, str):
        return value
    
    raise CLIValidationError(
        f"Expected string value, got {type(value).__name__}",
        hint="Provide a string value"
    )


def validate_bool(value: Any, *, allow_none: bool = False) -> Optional[bool]:
    """
    Validate and convert value to boolean.
    
    Args:
        value: Value to validate
        allow_none: Whether None is acceptable
    
    Returns:
        Boolean value or None if allow_none=True and value is None
    
    Raises:
        CLIValidationError: If value is not a boolean
    
    Examples:
        >>> validate_bool(True)
        True
        >>> validate_bool("true")
        Traceback (most recent call last):
        ...
        CLIValidationError: Expected boolean value, got str
    """
    if value is None:
        if allow_none:
            return None
        raise CLIValidationError(
            "Boolean value cannot be None",
            hint="Provide a valid boolean (True/False)"
        )
    
    if isinstance(value, bool):
        return value
    
    raise CLIValidationError(
        f"Expected boolean value, got {type(value).__name__}",
        hint="Provide a boolean value (True/False)"
    )


def validate_int(
    value: Any,
    *,
    allow_none: bool = False,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None
) -> Optional[int]:
    """
    Validate and convert value to integer with optional range checking.
    
    Args:
        value: Value to validate
        allow_none: Whether None is acceptable
        min_value: Minimum acceptable value (inclusive)
        max_value: Maximum acceptable value (inclusive)
    
    Returns:
        Integer value or None if allow_none=True and value is None
    
    Raises:
        CLIValidationError: If value is not an integer or out of range
    
    Examples:
        >>> validate_int(42)
        42
        >>> validate_int(5000, min_value=1024, max_value=65535)
        5000
        >>> validate_int(100, min_value=1024)
        Traceback (most recent call last):
        ...
        CLIValidationError: Value 100 is below minimum 1024
    """
    if value is None:
        if allow_none:
            return None
        raise CLIValidationError(
            "Integer value cannot be None",
            hint="Provide a valid integer"
        )
    
    if not isinstance(value, int) or isinstance(value, bool):
        raise CLIValidationError(
            f"Expected integer value, got {type(value).__name__}",
            hint="Provide an integer value"
        )
    
    if min_value is not None and value < min_value:
        raise CLIValidationError(
            f"Value {value} is below minimum {min_value}",
            hint=f"Use a value >= {min_value}"
        )
    
    if max_value is not None and value > max_value:
        raise CLIValidationError(
            f"Value {value} exceeds maximum {max_value}",
            hint=f"Use a value <= {max_value}"
        )
    
    return value


def validate_port(value: Any, *, allow_none: bool = False) -> Optional[int]:
    """
    Validate port number (1-65535).
    
    Args:
        value: Port value to validate
        allow_none: Whether None is acceptable
    
    Returns:
        Valid port number or None if allow_none=True and value is None
    
    Raises:
        CLIValidationError: If port is invalid
    
    Examples:
        >>> validate_port(8000)
        8000
        >>> validate_port(0)
        Traceback (most recent call last):
        ...
        CLIValidationError: Value 0 is below minimum 1
    """
    return validate_int(value, allow_none=allow_none, min_value=1, max_value=65535)


def validate_env_reference(token: str) -> Optional[str]:
    """
    Validate and resolve environment reference.
    
    Resolves environment aliases (prod, dev, test) to actual .env filenames.
    
    Args:
        token: Environment reference token
    
    Returns:
        Resolved environment filename or None if not a valid reference
    
    Examples:
        >>> validate_env_reference("prod")
        '.env.prod'
        >>> validate_env_reference("development")
        '.env.dev'
        >>> validate_env_reference(".env.custom")
        '.env.custom'
        >>> validate_env_reference("random")
        None
    """
    normalized = token.strip().strip('"\'')
    if not normalized:
        return None
    
    lower = normalized.lower()
    if lower in ENV_ALIAS_MAP:
        return ENV_ALIAS_MAP[lower]
    
    if normalized.startswith('.env'):
        return normalized
    
    return None


def normalize_run_command_args(argv: List[str]) -> List[str]:
    """
    Normalize 'run' command arguments for natural language syntax.
    
    Handles syntax like:
    - n3 run app.ai using prod
    - n3 run app.ai in development
    - n3 run app.ai on local
    
    Converts prepositions to standard --env flag format.
    
    Args:
        argv: Command-line arguments
    
    Returns:
        Normalized arguments with --env flags
    
    Examples:
        >>> normalize_run_command_args(['run', 'app.ai', 'using', 'prod'])
        ['run', 'app.ai', '--env', 'prod']
        >>> normalize_run_command_args(['run', 'app.ai', 'in', 'dev'])
        ['run', 'app.ai', '--env', 'dev']
    """
    if not argv or argv[0] != 'run':
        return argv
    
    normalized = []
    skip_next = False
    
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        
        if arg.lower() in _RUN_ENV_PREPOSITIONS:
            # Check if next token is environment reference
            if i + 1 < len(argv):
                next_token = argv[i + 1]
                env_ref = validate_env_reference(next_token)
                if env_ref:
                    normalized.extend(['--env', env_ref])
                    skip_next = True
                    continue
        
        normalized.append(arg)
    
    return normalized


def apply_env_overrides(overrides: Optional[List[str]]) -> None:
    """
    Apply environment variable overrides from KEY=VALUE strings.
    
    Handles both direct KEY=VALUE assignments and .env file references.
    
    Args:
        overrides: List of KEY=VALUE strings or .env file paths
    
    Raises:
        CLIValidationError: If .env file is not found or invalid
    
    Examples:
        >>> apply_env_overrides(['API_KEY=secret123', 'DEBUG=true'])
        >>> os.environ['API_KEY']
        'secret123'
    """
    if not overrides:
        return
    
    for entry in overrides:
        if not entry:
            continue
        
        if '=' in entry:
            # Direct KEY=VALUE assignment
            key, value = entry.split('=', 1)
            os.environ[key] = value
            continue
        
        # Treat as .env file reference
        load_env_file(entry)


def load_env_file(path_str: str) -> None:
    """
    Load environment variables from .env file.
    
    Supports:
    - Comments (lines starting with #)
    - export statements (removed automatically)
    - KEY=VALUE format
    
    Args:
        path_str: Path to .env file
    
    Raises:
        CLIValidationError: If file not found or has syntax errors
    
    Examples:
        >>> load_env_file('.env.prod')  # doctest: +SKIP
    """
    env_path = Path(path_str)
    
    if not env_path.exists():
        raise CLIValidationError(
            f"Environment file not found: {env_path}",
            hint="Check the file path and try again",
            context={'path': str(env_path)}
        )
    
    try:
        for raw_line in env_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Handle export prefix
            if line.startswith('export '):
                line = line[len('export '):].strip()
            
            # Parse KEY=VALUE
            if '=' not in line:
                continue
            
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()
            
    except Exception as exc:
        raise CLIValidationError(
            f"Error reading environment file {env_path}: {exc}",
            hint="Check file format and permissions",
            context={'path': str(env_path), 'error': str(exc)}
        )


def validate_target_type(target: str) -> str:
    """
    Validate frontend target type.
    
    Args:
        target: Target type string
    
    Returns:
        Validated target type
    
    Raises:
        CLIValidationError: If target type is invalid
    
    Examples:
        >>> validate_target_type("static")
        'static'
        >>> validate_target_type("react-vite")
        'react-vite'
        >>> validate_target_type("invalid")
        Traceback (most recent call last):
        ...
        CLIValidationError: Invalid target type: invalid
    """
    valid_targets = {"static", "react-vite"}
    
    if target not in valid_targets:
        raise CLIValidationError(
            f"Invalid target type: {target}",
            hint=f"Supported targets: {', '.join(sorted(valid_targets))}"
        )
    
    return target


def validate_file_exists(path: Path, *, file_type: str = "file") -> Path:
    """
    Validate that file or directory exists.
    
    Args:
        path: Path to validate
        file_type: Description of file type for error messages
    
    Returns:
        Validated Path
    
    Raises:
        CLIValidationError: If file does not exist
    
    Examples:
        >>> validate_file_exists(Path("/etc/hosts"), file_type="config file")  # doctest: +SKIP
        PosixPath('/etc/hosts')
    """
    if not path.exists():
        raise CLIValidationError(
            f"{file_type.capitalize()} not found: {path}",
            hint=f"Check that the {file_type} exists and the path is correct",
            context={'path': str(path), 'type': file_type}
        )
    
    return path
