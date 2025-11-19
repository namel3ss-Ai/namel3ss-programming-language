"""Base prompt interface and common types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromptResult:
    """Result from prompt rendering."""
    
    rendered: str
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"PromptResult(rendered={self.rendered!r}, variables={list(self.variables.keys())})"


class PromptError(Exception):
    """Base exception for prompt errors."""
    
    def __init__(
        self,
        message: str,
        *,
        prompt_name: Optional[str] = None,
        missing_vars: Optional[List[str]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.prompt_name = prompt_name
        self.missing_vars = missing_vars or []
        self.original_error = original_error


class BasePrompt(ABC):
    """
    Abstract base class for all prompt template implementations.
    
    Prompts represent parameterized text templates that can be rendered with variables.
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
        Initialize the prompt.
        
        Args:
            name: Prompt identifier
            template: Template string with {variable} placeholders
            model: Target LLM model name
            args: Argument specifications (name -> type/default)
            **config: Additional configuration
        """
        self.name = name
        self.template = template
        self.model = model
        self.args = args or {}
        self.config = config
    
    @abstractmethod
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
        pass
    
    def validate_variables(self, variables: Dict[str, Any]) -> None:
        """
        Validate variables against argument specifications.
        
        Args:
            variables: Variable dictionary to validate
            
        Raises:
            PromptError: If required variables missing
        """
        if not self.args:
            return
        
        missing = []
        for arg_name, arg_spec in self.args.items():
            if isinstance(arg_spec, dict):
                required = arg_spec.get("required", True)
                has_default = "default" in arg_spec
                if required and not has_default and arg_name not in variables:
                    missing.append(arg_name)
            elif arg_name not in variables:
                # Simple case: arg_spec is just a type string
                missing.append(arg_name)
        
        if missing:
            raise PromptError(
                f"Missing required variables: {', '.join(missing)}",
                prompt_name=self.name,
                missing_vars=missing,
            )
    
    def get_model(self) -> Optional[str]:
        """Return the target model name."""
        return self.model
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model!r})"
