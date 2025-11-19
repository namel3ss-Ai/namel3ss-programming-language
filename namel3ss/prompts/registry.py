"""Prompt registry for managing prompt instances at runtime."""

from typing import Dict, Optional

from .base import BasePrompt


class PromptRegistry:
    """Registry for storing and retrieving prompt instances."""
    
    def __init__(self) -> None:
        self._prompts: Dict[str, BasePrompt] = {}
    
    def register(self, name: str, prompt: BasePrompt) -> None:
        """
        Register a prompt instance.
        
        Args:
            name: Prompt identifier
            prompt: Prompt instance
            
        Raises:
            ValueError: If prompt already registered
        """
        if name in self._prompts:
            raise ValueError(f"Prompt '{name}' is already registered")
        self._prompts[name] = prompt
    
    def update(self, name: str, prompt: BasePrompt) -> None:
        """
        Update or register a prompt instance.
        
        Args:
            name: Prompt identifier
            prompt: Prompt instance
        """
        self._prompts[name] = prompt
    
    def get(self, name: str) -> Optional[BasePrompt]:
        """
        Get a prompt by name.
        
        Args:
            name: Prompt identifier
            
        Returns:
            Prompt instance or None if not found
        """
        return self._prompts.get(name)
    
    def get_required(self, name: str) -> BasePrompt:
        """
        Get a prompt by name, raising error if not found.
        
        Args:
            name: Prompt identifier
            
        Returns:
            Prompt instance
            
        Raises:
            KeyError: If prompt not found
        """
        prompt = self._prompts.get(name)
        if prompt is None:
            raise KeyError(f"Prompt '{name}' not found in registry")
        return prompt
    
    def has(self, name: str) -> bool:
        """
        Check if a prompt is registered.
        
        Args:
            name: Prompt identifier
            
        Returns:
            True if prompt exists
        """
        return name in self._prompts
    
    def list_prompts(self) -> Dict[str, Optional[str]]:
        """
        List all registered prompts with their models.
        
        Returns:
            Dict mapping prompt names to model names
        """
        return {name: prompt.get_model() for name, prompt in self._prompts.items()}
    
    def clear(self) -> None:
        """Remove all prompts from registry."""
        self._prompts.clear()
    
    def __contains__(self, name: str) -> bool:
        """Support 'name in registry' syntax."""
        return name in self._prompts
    
    def __repr__(self) -> str:
        return f"PromptRegistry(prompts={list(self._prompts.keys())})"


# Global registry instance
_GLOBAL_REGISTRY: Optional[PromptRegistry] = None


def get_registry() -> PromptRegistry:
    """
    Get the global prompt registry instance.
    
    Returns:
        Singleton PromptRegistry
    """
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = PromptRegistry()
    return _GLOBAL_REGISTRY


def reset_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = None
