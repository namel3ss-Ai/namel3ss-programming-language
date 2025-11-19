"""Template prompt implementation with variable substitution."""

import re
from typing import Any, Dict, Optional

from .base import BasePrompt, PromptError, PromptResult


class TemplatePrompt(BasePrompt):
    """
    Prompt that uses string.format-style template substitution.
    
    Supports {variable} placeholders and applies defaults for missing values.
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
            template: Template string with {variable} placeholders
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
        
        # Extract variable names from template
        self._template_vars = self._extract_variables(template)
    
    def _extract_variables(self, template: str) -> set:
        """Extract variable names from template string."""
        # Find all {variable} patterns
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, template)
        return set(matches)
    
    def render(self, **variables: Any) -> PromptResult:
        """
        Render the prompt template with provided variables.
        
        Args:
            **variables: Variable values to substitute
            
        Returns:
            PromptResult with rendered text
            
        Raises:
            PromptError: If required variables missing
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
            
            # Validate we have all required variables
            missing = []
            for var_name in self._template_vars:
                if var_name not in final_vars:
                    # Check if it's required
                    if self.args and var_name in self.args:
                        arg_spec = self.args[var_name]
                        if isinstance(arg_spec, dict):
                            required = arg_spec.get("required", True)
                            if required:
                                missing.append(var_name)
                        else:
                            missing.append(var_name)
                    else:
                        # No arg spec, assume required
                        missing.append(var_name)
            
            if missing:
                raise PromptError(
                    f"Missing required variables: {', '.join(missing)}",
                    prompt_name=self.name,
                    missing_vars=missing,
                )
            
            # Render template
            rendered = self.template.format(**final_vars)
            
            return PromptResult(
                rendered=rendered,
                variables=final_vars,
                metadata={
                    "model": self.model,
                    "template": self.template,
                },
            )
        
        except KeyError as e:
            raise PromptError(
                f"Missing variable in template: {e}",
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
        if not self.args:
            return self._template_vars
        
        required = set()
        for var_name in self._template_vars:
            if var_name in self.args:
                arg_spec = self.args[var_name]
                if isinstance(arg_spec, dict):
                    if arg_spec.get("required", True) and "default" not in arg_spec:
                        required.add(var_name)
                else:
                    required.add(var_name)
            else:
                required.add(var_name)
        
        return required
