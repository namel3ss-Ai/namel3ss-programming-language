"""Standard library registry for the Namel3ss language."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from .memory import (
    MemoryPolicy, 
    list_memory_policies,
    get_policy_description as get_memory_description,
    validate_memory_config,
)

from .llm import (
    LLMConfigField,
    list_llm_config_fields,
    get_field_description as get_llm_description,
    validate_llm_config,
)

from .tools import (
    ToolCategory,
    list_tool_categories,
    get_category_description as get_tool_description,
    validate_tool_config,
)


logger = logging.getLogger(__name__)


class StandardLibraryRegistry:
    """
    Central registry for all standard library components.
    
    Provides unified access to:
    - Memory policies and their specifications
    - LLM configuration fields and validation
    - Tool categories and interfaces
    - Cross-component validation and suggestions
    """
    
    def __init__(self):
        self._memory_policies = list_memory_policies()
        self._llm_fields = list_llm_config_fields() 
        self._tool_categories = list_tool_categories()
        
        logger.info(
            f"Initialized standard library registry with "
            f"{len(self._memory_policies)} memory policies, "
            f"{len(self._llm_fields)} LLM fields, "
            f"{len(self._tool_categories)} tool categories"
        )
    
    # Memory policies
    def get_memory_policies(self) -> List[str]:
        """Get list of available memory policy names."""
        return self._memory_policies.copy()
    
    def get_memory_policy_info(self, policy: str) -> Dict[str, str]:
        """Get information about a memory policy."""
        return {
            'name': policy,
            'description': get_memory_description(policy)
        }
    
    def validate_memory(self, policy: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Validate memory configuration."""
        return validate_memory_config(policy, config)
    
    # LLM configuration
    def get_llm_fields(self) -> List[str]:
        """Get list of available LLM configuration field names."""
        return self._llm_fields.copy()
    
    def get_llm_field_info(self, field: str) -> Dict[str, str]:
        """Get information about an LLM configuration field."""
        return {
            'name': field,
            'description': get_llm_description(field)
        }
    
    def validate_llm(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Validate LLM configuration."""
        return validate_llm_config(config)
    
    # Tool categories
    def get_tool_categories(self) -> List[str]:
        """Get list of available tool category names."""
        return self._tool_categories.copy()
    
    def get_tool_category_info(self, category: str) -> Dict[str, str]:
        """Get information about a tool category."""
        return {
            'name': category,
            'description': get_tool_description(category)
        }
    
    def validate_tool(self, category: str, config: Dict[str, Any]) -> Dict[str, str]:
        """Validate tool configuration."""
        return validate_tool_config(category, config)
    
    # Unified interface
    def get_all_components(self) -> Dict[str, List[str]]:
        """Get all available standard library components."""
        return {
            'memory_policies': self.get_memory_policies(),
            'llm_fields': self.get_llm_fields(),
            'tool_categories': self.get_tool_categories()
        }
    
    def search_components(self, query: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Search for standard library components by name or description.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary with matching components by type
        """
        query_lower = query.lower()
        results = {
            'memory_policies': [],
            'llm_fields': [],
            'tool_categories': []
        }
        
        # Search memory policies
        for policy in self._memory_policies:
            info = self.get_memory_policy_info(policy)
            if (query_lower in policy.lower() or 
                query_lower in info['description'].lower()):
                results['memory_policies'].append(info)
        
        # Search LLM fields
        for field in self._llm_fields:
            info = self.get_llm_field_info(field)
            if (query_lower in field.lower() or
                query_lower in info['description'].lower()):
                results['llm_fields'].append(info)
        
        # Search tool categories
        for category in self._tool_categories:
            info = self.get_tool_category_info(category) 
            if (query_lower in category.lower() or
                query_lower in info['description'].lower()):
                results['tool_categories'].append(info)
        
        return results
    
    def validate_all(
        self, 
        memory_policy: Optional[str] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Validate multiple components at once.
        
        Args:
            memory_policy: Memory policy to validate
            memory_config: Memory configuration
            llm_config: LLM configuration
            tools: List of tool configurations (each must have 'category' key)
            
        Returns:
            Dictionary of validation errors by component
        """
        all_errors = {}
        
        # Validate memory
        if memory_policy:
            memory_errors = self.validate_memory(memory_policy, memory_config)
            if memory_errors:
                all_errors['memory'] = memory_errors
        
        # Validate LLM
        if llm_config:
            llm_errors = self.validate_llm(llm_config)
            if llm_errors:
                all_errors['llm'] = llm_errors
        
        # Validate tools
        if tools:
            for i, tool_config in enumerate(tools):
                if 'category' not in tool_config:
                    all_errors[f'tool_{i}'] = {'category': 'Tool category is required'}
                    continue
                
                tool_errors = self.validate_tool(tool_config['category'], tool_config)
                if tool_errors:
                    all_errors[f'tool_{i}'] = tool_errors
        
        return all_errors


# Global registry instance
_global_registry: Optional[StandardLibraryRegistry] = None


def get_stdlib_registry() -> StandardLibraryRegistry:
    """Get the global standard library registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = StandardLibraryRegistry()
    return _global_registry


def reset_stdlib_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = None