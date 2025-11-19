"""
Production-grade template engine for AI prompts using Jinja2 with security sandboxing.

This module provides a secure, extensible template engine for rendering AI prompts
with advanced features like conditionals, loops, filters, and nested object access.
All templates run in a sandboxed environment with no code execution capabilities.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from jinja2 import Environment, StrictUndefined, Undefined, TemplateSyntaxError, UndefinedError
from jinja2.sandbox import SandboxedEnvironment
from jinja2.meta import find_undeclared_variables


class TemplateError(Exception):
    """Base exception for template engine errors."""
    
    def __init__(
        self,
        message: str,
        *,
        template_name: Optional[str] = None,
        line_number: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.template_name = template_name
        self.line_number = line_number
        self.original_error = original_error


class TemplateSecurityError(TemplateError):
    """Raised when template attempts unsafe operations."""
    pass


class TemplateCompilationError(TemplateError):
    """Raised when template compilation fails."""
    pass


class TemplateRenderError(TemplateError):
    """Raised when template rendering fails."""
    pass


@dataclass
class CompiledTemplate:
    """
    Represents a compiled Jinja2 template ready for rendering.
    
    Templates are compiled once and can be rendered multiple times
    with different variable contexts.
    """
    
    name: str
    source: str
    template: Any  # jinja2.Template
    required_vars: Set[str]
    optional_vars: Set[str]
    
    def render(self, variables: Dict[str, Any]) -> str:
        """
        Render template with provided variables.
        
        Args:
            variables: Variable context for rendering
            
        Returns:
            Rendered string
            
        Raises:
            TemplateRenderError: If rendering fails
        """
        try:
            return self.template.render(**variables)
        except UndefinedError as e:
            raise TemplateRenderError(
                f"Undefined variable in template: {e}",
                template_name=self.name,
                original_error=e,
            )
        except Exception as e:
            raise TemplateRenderError(
                f"Template rendering failed: {e}",
                template_name=self.name,
                original_error=e,
            )


# Custom filters for AI prompts
def _filter_format_date(value: Any, format_str: str = "%Y-%m-%d") -> str:
    """Format datetime objects or ISO strings as dates."""
    if isinstance(value, datetime):
        return value.strftime(format_str)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt.strftime(format_str)
        except Exception:
            return value
    return str(value)


def _filter_truncate(value: str, length: int = 100, suffix: str = "...") -> str:
    """Truncate string to specified length with suffix."""
    if not isinstance(value, str):
        value = str(value)
    if len(value) <= length:
        return value
    return value[:length - len(suffix)] + suffix


def _filter_strip(value: str) -> str:
    """Remove leading/trailing whitespace."""
    return str(value).strip() if value is not None else ""


def _filter_title(value: str) -> str:
    """Convert to title case."""
    return str(value).title() if value is not None else ""


def _filter_uppercase(value: str) -> str:
    """Convert to uppercase."""
    return str(value).upper() if value is not None else ""


def _filter_lowercase(value: str) -> str:
    """Convert to lowercase."""
    return str(value).lower() if value is not None else ""


def _filter_json_encode(value: Any, indent: Optional[int] = None) -> str:
    """Encode value as JSON."""
    return json.dumps(value, indent=indent, ensure_ascii=False)


def _filter_list_join(value: List[Any], separator: str = ", ") -> str:
    """Join list items into string."""
    if not isinstance(value, (list, tuple)):
        return str(value)
    return separator.join(str(item) for item in value)


def _filter_default(value: Any, default_value: Any = "") -> Any:
    """Return default if value is None or empty string."""
    if value is None or value == "":
        return default_value
    return value


def _filter_length(value: Any) -> int:
    """Return length of collection or string."""
    try:
        return len(value)
    except TypeError:
        return 0


class PromptTemplateEngine:
    """
    Production-grade template engine for AI prompts with Jinja2 sandboxing.
    
    Features:
    - Secure sandboxed execution (no arbitrary code)
    - Variables: {{ variable }}
    - Conditionals: {% if condition %} ... {% endif %}
    - Loops: {% for item in items %} ... {% endfor %}
    - Filters: {{ value|filter_name }}
    - Nested object access: {{ user.profile.name }}
    - Custom filters for AI prompts
    - Compile-time validation
    - Context isolation
    
    Security:
    - No access to Python builtins (__import__, eval, exec, etc.)
    - No filesystem access
    - No arbitrary code execution
    - Restricted set of safe globals
    - Auto-escaping disabled (AI prompts don't need HTML escaping)
    """
    
    def __init__(
        self,
        *,
        autoescape: bool = False,
        strict_undefined: bool = True,
        custom_filters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the template engine.
        
        Args:
            autoescape: Enable auto-escaping (default False for AI prompts)
            strict_undefined: Raise errors on undefined variables (default True)
            custom_filters: Additional custom filters to register
        """
        # Create sandboxed environment
        undefined_class = StrictUndefined if strict_undefined else Undefined
        self.env = SandboxedEnvironment(
            autoescape=autoescape,
            undefined=undefined_class,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        # Register custom filters
        self._register_filters()
        
        # Register additional custom filters
        if custom_filters:
            for name, func in custom_filters.items():
                self.env.filters[name] = func
        
        # Configure security: remove dangerous globals
        self.env.globals = {
            # Only include safe utilities
            "range": range,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
        }
    
    def _register_filters(self) -> None:
        """Register built-in custom filters."""
        self.env.filters["format_date"] = _filter_format_date
        self.env.filters["truncate"] = _filter_truncate
        self.env.filters["strip"] = _filter_strip
        self.env.filters["title"] = _filter_title
        self.env.filters["uppercase"] = _filter_uppercase
        self.env.filters["lowercase"] = _filter_lowercase
        self.env.filters["json_encode"] = _filter_json_encode
        self.env.filters["list_join"] = _filter_list_join
        self.env.filters["default"] = _filter_default
        self.env.filters["length"] = _filter_length
    
    def compile(
        self,
        source: str,
        *,
        name: str = "<template>",
        validate: bool = True,
    ) -> CompiledTemplate:
        """
        Compile a template from source string.
        
        Args:
            source: Template source code
            name: Template identifier for error messages
            validate: Perform validation during compilation
            
        Returns:
            CompiledTemplate ready for rendering
            
        Raises:
            TemplateCompilationError: If compilation fails
            TemplateSecurityError: If template contains unsafe constructs
        """
        try:
            # Parse AST to extract variables
            ast = self.env.parse(source)
            
            # Extract undeclared variables (those that need to be provided)
            required_vars = find_undeclared_variables(ast)
            
            # Compile template
            template = self.env.from_string(source)
            
            if validate:
                self._validate_template_safety(source, name)
            
            return CompiledTemplate(
                name=name,
                source=source,
                template=template,
                required_vars=required_vars,
                optional_vars=set(),  # Can be extended with defaults
            )
            
        except TemplateSyntaxError as e:
            raise TemplateCompilationError(
                f"Template syntax error: {e.message}",
                template_name=name,
                line_number=e.lineno,
                original_error=e,
            )
        except Exception as e:
            raise TemplateCompilationError(
                f"Failed to compile template: {e}",
                template_name=name,
                original_error=e,
            )
    
    def _validate_template_safety(self, source: str, name: str) -> None:
        """
        Validate template doesn't contain unsafe constructs.
        
        Args:
            source: Template source
            name: Template name for errors
            
        Raises:
            TemplateSecurityError: If unsafe patterns detected
        """
        # Check for dangerous patterns
        dangerous_patterns = [
            "__import__",
            "__builtins__",
            "eval(",
            "exec(",
            "compile(",
            "open(",
            "__class__",
            "__bases__",
            "__subclasses__",
        ]
        
        for pattern in dangerous_patterns:
            if pattern in source:
                raise TemplateSecurityError(
                    f"Template contains dangerous pattern: {pattern}",
                    template_name=name,
                )
    
    def render(
        self,
        template_source: str,
        variables: Dict[str, Any],
        *,
        name: str = "<template>",
    ) -> str:
        """
        Compile and render template in one step.
        
        For one-off rendering. For repeated rendering, use compile() once
        and render() on the CompiledTemplate.
        
        Args:
            template_source: Template source code
            variables: Variable context
            name: Template identifier
            
        Returns:
            Rendered string
            
        Raises:
            TemplateCompilationError: If compilation fails
            TemplateRenderError: If rendering fails
        """
        compiled = self.compile(template_source, name=name)
        return compiled.render(variables)
    
    def validate_variables(
        self,
        compiled: CompiledTemplate,
        variables: Dict[str, Any],
    ) -> List[str]:
        """
        Validate that all required variables are provided.
        
        Args:
            compiled: Compiled template
            variables: Variables to validate
            
        Returns:
            List of missing variable names (empty if all provided)
        """
        missing = []
        for var_name in compiled.required_vars:
            if var_name not in variables:
                missing.append(var_name)
        return missing


def create_engine(
    *,
    autoescape: bool = False,
    strict_undefined: bool = True,
    custom_filters: Optional[Dict[str, Any]] = None,
) -> PromptTemplateEngine:
    """
    Factory function to create a configured template engine.
    
    Args:
        autoescape: Enable auto-escaping (default False for AI prompts)
        strict_undefined: Raise errors on undefined variables (default True)
        custom_filters: Additional custom filters to register
        
    Returns:
        Configured PromptTemplateEngine instance
    """
    return PromptTemplateEngine(
        autoescape=autoescape,
        strict_undefined=strict_undefined,
        custom_filters=custom_filters,
    )


# Global singleton instance
_default_engine: Optional[PromptTemplateEngine] = None


def get_default_engine() -> PromptTemplateEngine:
    """Get or create the default global template engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = create_engine()
    return _default_engine
