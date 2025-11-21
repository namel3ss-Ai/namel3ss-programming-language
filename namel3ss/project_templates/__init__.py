"""
N3 Project Templates - Production-grade starter projects.

This package provides first-class, non-AI project templates that demonstrate
Namel3ss as a general-purpose orchestration and workflow platform.

Available Templates:
- crud-service: Complete CRUD microservice with REST API
- dashboard: Data visualization dashboard with metrics and charts

Each template is production-ready, fully typed, tested, and suitable for
real-world SaaS/enterprise use.
"""

from .registry import TemplateRegistry, ProjectTemplate
from .generator import TemplateGenerator
from .crud_service import CRUDServiceTemplate
from .dashboard import DashboardTemplate

__all__ = [
    "TemplateRegistry",
    "ProjectTemplate",
    "TemplateGenerator",
    "CRUDServiceTemplate",
    "DashboardTemplate",
    "get_registry",
    "list_templates",
    "generate_project",
]

# Global registry instance
_registry = None


def get_registry() -> TemplateRegistry:
    """Get the global template registry."""
    global _registry
    if _registry is None:
        _registry = TemplateRegistry()
        _registry.register(CRUDServiceTemplate())
        _registry.register(DashboardTemplate())
    return _registry


def list_templates() -> list[ProjectTemplate]:
    """List all available project templates."""
    return get_registry().list_all()


def generate_project(
    template_id: str,
    output_dir: str,
    **config
) -> None:
    """
    Generate a project from a template.
    
    Args:
        template_id: Template identifier (e.g. "crud-service")
        output_dir: Target directory for generated project
        **config: Template-specific configuration
    """
    registry = get_registry()
    generator = TemplateGenerator(registry)
    generator.generate(template_id, output_dir, config)
