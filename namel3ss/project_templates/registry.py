"""Template registry and base classes for N3 project templates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TemplateFile:
    """Represents a file to be generated in a project template."""
    
    path: str  # Relative path in project
    content: str  # File content (may contain template variables)
    executable: bool = False  # Whether file should be executable
    binary: bool = False  # Whether file is binary (skip template rendering)


@dataclass
class ProjectTemplate(ABC):
    """Base class for N3 project templates."""
    
    id: str  # Unique template identifier (e.g. "crud-service")
    name: str  # Display name
    description: str  # Template description
    category: str  # Category (e.g. "microservice", "dashboard", "workflow")
    tags: List[str] = field(default_factory=list)
    
    @abstractmethod
    def get_files(self, config: Dict[str, Any]) -> List[TemplateFile]:
        """
        Generate list of files for this template.
        
        Args:
            config: User-provided configuration
            
        Returns:
            List of TemplateFile objects
        """
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this template."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate user-provided configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    def get_post_generation_instructions(self, config: Dict[str, Any]) -> str:
        """
        Get instructions to display after project generation.
        
        Args:
            config: Configuration used for generation
            
        Returns:
            Human-readable instructions
        """
        return ""


class TemplateRegistry:
    """Registry for managing project templates."""
    
    def __init__(self):
        self._templates: Dict[str, ProjectTemplate] = {}
    
    def register(self, template: ProjectTemplate) -> None:
        """Register a project template."""
        if template.id in self._templates:
            raise ValueError(f"Template '{template.id}' already registered")
        self._templates[template.id] = template
    
    def get(self, template_id: str) -> Optional[ProjectTemplate]:
        """Get template by ID."""
        return self._templates.get(template_id)
    
    def list_all(self) -> List[ProjectTemplate]:
        """List all registered templates."""
        return list(self._templates.values())
    
    def list_by_category(self, category: str) -> List[ProjectTemplate]:
        """List templates in a specific category."""
        return [t for t in self._templates.values() if t.category == category]
    
    def search(self, query: str) -> List[ProjectTemplate]:
        """Search templates by name, description, or tags."""
        query_lower = query.lower()
        results = []
        
        for template in self._templates.values():
            if (
                query_lower in template.name.lower()
                or query_lower in template.description.lower()
                or any(query_lower in tag.lower() for tag in template.tags)
            ):
                results.append(template)
        
        return results
