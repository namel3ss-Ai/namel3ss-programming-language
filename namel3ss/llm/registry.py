"""LLM instance registry for runtime lookup."""

from typing import Dict, Optional
from .base import BaseLLM, LLMError


class LLMRegistry:
    """
    Registry for LLM instances.
    
    Allows registering LLM instances by logical name and retrieving them
    at runtime for chain execution.
    """
    
    def __init__(self):
        self._llms: Dict[str, BaseLLM] = {}
    
    def register(self, llm: BaseLLM) -> None:
        """
        Register an LLM instance.
        
        Args:
            llm: The LLM instance to register
        
        Raises:
            ValueError: If an LLM with the same name is already registered
        """
        if llm.name in self._llms:
            raise ValueError(
                f"LLM '{llm.name}' is already registered. "
                f"Use update() to replace an existing LLM."
            )
        self._llms[llm.name] = llm
    
    def update(self, llm: BaseLLM) -> None:
        """
        Register or update an LLM instance.
        
        Args:
            llm: The LLM instance to register/update
        """
        self._llms[llm.name] = llm
    
    def get(self, name: str) -> Optional[BaseLLM]:
        """
        Retrieve an LLM by name.
        
        Args:
            name: The logical name of the LLM
        
        Returns:
            The LLM instance, or None if not found
        """
        return self._llms.get(name)
    
    def get_required(self, name: str) -> BaseLLM:
        """
        Retrieve an LLM by name, raising an error if not found.
        
        Args:
            name: The logical name of the LLM
        
        Returns:
            The LLM instance
        
        Raises:
            LLMError: If the LLM is not registered
        """
        llm = self.get(name)
        if llm is None:
            raise LLMError(
                f"LLM '{name}' is not registered. "
                f"Available LLMs: {', '.join(self.list()) or 'none'}"
            )
        return llm
    
    def has(self, name: str) -> bool:
        """
        Check if an LLM is registered.
        
        Args:
            name: The logical name of the LLM
        
        Returns:
            True if registered, False otherwise
        """
        return name in self._llms
    
    def list(self) -> list[str]:
        """
        List all registered LLM names.
        
        Returns:
            List of LLM names
        """
        return list(self._llms.keys())
    
    def clear(self) -> None:
        """Clear all registered LLMs."""
        self._llms.clear()
    
    def __len__(self) -> int:
        return len(self._llms)
    
    def __contains__(self, name: str) -> bool:
        return name in self._llms


# Global registry instance
_global_registry: Optional[LLMRegistry] = None


def get_registry() -> LLMRegistry:
    """
    Get the global LLM registry instance.
    
    Returns:
        The global LLMRegistry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = LLMRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _global_registry
    _global_registry = None
