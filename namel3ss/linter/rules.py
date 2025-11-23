"""Base class and infrastructure for lint rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import LintContext, LintFinding


class LintRule(ABC):
    """Base class for semantic lint rules."""
    
    def __init__(self, rule_id: str, description: str):
        self.rule_id = rule_id
        self.description = description
    
    @abstractmethod
    def check(self, context: "LintContext") -> List["LintFinding"]:
        """
        Apply this rule to the given context.
        
        Args:
            context: Analysis context with AST, resolver, etc.
            
        Returns:
            List of lint findings
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.rule_id!r})"