"""AST nodes for inline code blocks (Python, React, etc.)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from .base import Expression
from .source_location import SourceLocation

__all__ = [
    "InlineBlock",
    "InlinePythonBlock",
    "InlineReactBlock",
]


@dataclass
class InlineBlock(Expression):
    """
    Base class for inline code blocks.
    
    Inline blocks allow embedding code from other languages/runtimes
    directly in N3 syntax. Examples:
    - python { def foo(): return 42 }
    - react { <Component prop={value} /> }
    
    This is a production-grade escape hatch that:
    - Preserves source location for debugging
    - Maintains type safety (not dicts or raw strings)
    - Supports context bindings from surrounding N3 code
    - Enables controlled execution at runtime
    """
    kind: str  # Language/runtime: "python", "react", "sql", etc.
    code: str  # Raw source code as-written by user
    location: Optional[SourceLocation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extensible metadata
    
    def __post_init__(self) -> None:
        """Validate kind and code are present."""
        if not self.kind:
            raise ValueError("InlineBlock kind cannot be empty")
        if not isinstance(self.code, str):
            raise TypeError(f"InlineBlock code must be str, got {type(self.code).__name__}")


@dataclass
class InlinePythonBlock(InlineBlock):
    """
    Inline Python code block.
    
    Syntax: python { <python code> }
    
    Features:
    - Can define functions, classes, or expressions
    - Access to context bindings from N3 runtime
    - Executes server-side in controlled environment
    - Version specified for compatibility
    
    Examples:
        python {
            def process_data(items):
                return [x * 2 for x in items]
        }
        
        python {
            import numpy as np
            result = np.mean(values)
        }
    """
    code: str = ""  # Required but with default for dataclass inheritance
    kind: Literal["python"] = "python"
    bindings: Dict[str, Any] = field(default_factory=dict)  # Context variables to inject
    python_version: Optional[str] = None  # e.g., "3.11", "3.12"
    is_expression: bool = False  # True if code is single expression, False if statements
    
    def __post_init__(self) -> None:
        """Ensure kind is set to python."""
        super().__post_init__()
        object.__setattr__(self, 'kind', 'python')


@dataclass
class InlineReactBlock(InlineBlock):
    """
    Inline React/JSX component block.
    
    Syntax: react { <React JSX> }
    
    Features:
    - Can be component definition or JSX fragment
    - Compiled to React component at codegen time
    - Rendered client-side in browser
    - Props passed from N3 page context
    
    Examples:
        react {
            <div className="alert">
                <h2>{title}</h2>
                <p>{message}</p>
            </div>
        }
        
        react {
            function CustomButton({ onClick, label }) {
                return <button onClick={onClick}>{label}</button>;
            }
        }
    """
    code: str = ""  # Required but with default for dataclass inheritance
    kind: Literal["react"] = "react"
    component_name: Optional[str] = None  # Name for component if it's a definition
    props: Dict[str, Any] = field(default_factory=dict)  # Props to pass to component
    requires_imports: list[str] = field(default_factory=list)  # e.g., ["React", "useState"]
    
    def __post_init__(self) -> None:
        """Ensure kind is set to react."""
        super().__post_init__()
        object.__setattr__(self, 'kind', 'react')
