"""Template prompt implementation with variable substitution."""

import re
from typing import Any, Dict, Optional

from .base import BasePrompt, PromptError, PromptResult
from namel3ss.templates import get_default_engine, CompiledTemplate, TemplateCompilationError, TemplateRenderError


class TemplatePrompt(BasePrompt):
    """
    Prompt that uses Jinja2 template engine for secure variable substitution.
    
    Supports:
    - Variables: {{ variable }}
    - Conditionals: {% if condition %} ... {% endif %}
    - Loops: {% for item in items %} ... {% endfor %}
    - Filters: {{ value|filter_name }}
    - Nested objects: {{ user.profile.name }}
    
    Templates are compiled once for validation and reused for rendering.
    """
    
    def __init__(
        self,
        *,
        name: str,
        template: str,
        model: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        **config: Any,
    ):
        """
        Initialize template prompt.
        
        Args:
            name: Prompt identifier
            template: Jinja2 template string
            model: Target LLM model name
            args: Argument specifications with defaults
            **config: Additional configuration
        """
        super().__init__(
            name=name,
            template=template,
            model=model,
            args=args,
            **config,
        )
        
        # Get template engine
        self._engine = get_default_engine()
        
        # Compile template at initialization for early validation
        try:
            self._compiled_template = self._engine.compile(
                source=template,
                name=name,
                validate=True,
            )
        except (TemplateCompilationError, TemplateRenderError) as e:
            raise PromptError(
                f"Failed to compile template: {e}",
                prompt_name=name,
                original_error=e,
            )
    
    def render(self, **variables: Any) -> PromptResult:
        """
        Render the prompt template with provided variables.
        
        Args:
            **variables: Variable values to substitute
            
        Returns:
            PromptResult with rendered text
            
        Raises:
            PromptError: If required variables missing or rendering fails
        """
        try:
            # Apply defaults from args
            final_vars = {}
            
            # First, add defaults
            if self.args:
                for arg_name, arg_spec in self.args.items():
                    if isinstance(arg_spec, dict) and "default" in arg_spec:
                        final_vars[arg_name] = arg_spec["default"]
            
            # Then override with provided variables
            final_vars.update(variables)
            
            # Validate required variables
            missing = self._engine.validate_variables(
                self._compiled_template,
                final_vars,
            )
            
            # Check against arg specs for additional validation
            # Only mark as required if not in a conditional block
            if self.args:
                for var_name in self._compiled_template.required_vars:
                    if var_name in self.args:
                        arg_spec = self.args[var_name]
                        if isinstance(arg_spec, dict):
                            required = arg_spec.get("required", True)
                            has_default = "default" in arg_spec
                            # Only required if explicitly marked and no default
                            if required and not has_default and var_name not in final_vars:
                                if var_name not in missing:
                                    missing.append(var_name)
            
            if missing:
                raise PromptError(
                    f"Missing required variables: {', '.join(missing)}",
                    prompt_name=self.name,
                    missing_vars=missing,
                )
            
            # Render template using Jinja2 engine
            rendered = self._compiled_template.render(final_vars)
            
            return PromptResult(
                rendered=rendered,
                variables=final_vars,
                metadata={
                    "model": self.model,
                    "template": self.template,
                    "engine": "jinja2",
                },
            )
        
        except PromptError:
            # Re-raise PromptError as-is (don't wrap)
            raise
        except TemplateRenderError as e:
            raise PromptError(
                f"Template rendering failed: {e}",
                prompt_name=self.name,
                original_error=e,
            )
        except Exception as e:
            raise PromptError(
                f"Failed to render prompt: {e}",
                prompt_name=self.name,
                original_error=e,
            )
        except Exception as e:
            raise PromptError(
                f"Failed to render prompt: {e}",
                prompt_name=self.name,
                original_error=e,
            )
    
    def get_required_variables(self) -> set:
        """Return set of required variable names."""
        return self._compiled_template.required_vars

