"""Template generator for creating N3 projects from templates."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, BaseLoader, TemplateSyntaxError

from .registry import TemplateRegistry, TemplateFile


class TemplateGenerationError(Exception):
    """Raised when template generation fails."""
    pass


class TemplateGenerator:
    """Generates projects from templates."""
    
    def __init__(self, registry: TemplateRegistry):
        self.registry = registry
        self.jinja_env = Environment(
            loader=BaseLoader(),
            autoescape=False,  # We're generating code, not HTML
            keep_trailing_newline=True,
        )
    
    def generate(
        self,
        template_id: str,
        output_dir: str | Path,
        config: Dict[str, Any],
    ) -> None:
        """
        Generate a project from a template.
        
        Args:
            template_id: Template identifier
            output_dir: Target directory
            config: Template configuration
            
        Raises:
            TemplateGenerationError: If generation fails
        """
        # Get template
        template = self.registry.get(template_id)
        if template is None:
            raise TemplateGenerationError(f"Template not found: {template_id}")
        
        # Merge with defaults
        full_config = template.get_default_config()
        full_config.update(config)
        
        # Validate
        try:
            template.validate_config(full_config)
        except ValueError as e:
            raise TemplateGenerationError(f"Invalid configuration: {e}")
        
        # Create output directory
        output_path = Path(output_dir).resolve()
        if output_path.exists() and any(output_path.iterdir()):
            raise TemplateGenerationError(
                f"Output directory not empty: {output_path}"
            )
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate files
        try:
            files = template.get_files(full_config)
            self._generate_files(files, output_path, full_config)
        except Exception as e:
            # Clean up on failure
            if output_path.exists():
                shutil.rmtree(output_path)
            raise TemplateGenerationError(f"Generation failed: {e}")
        
        # Print post-generation instructions
        instructions = template.get_post_generation_instructions(full_config)
        if instructions:
            print("\n" + "="*70)
            print(f"✓ Project '{template.name}' created successfully!")
            print("="*70)
            print(instructions)
            print("="*70 + "\n")
    
    def _generate_files(
        self,
        files: list[TemplateFile],
        output_path: Path,
        config: Dict[str, Any],
    ) -> None:
        """Generate individual files."""
        for file_spec in files:
            file_path = output_path / file_spec.path
            
            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Render content (skip for binary files)
            if file_spec.binary:
                content = file_spec.content
            else:
                try:
                    content = self._render_template(file_spec.content, config)
                except TemplateSyntaxError as e:
                    raise TemplateGenerationError(
                        f"Template syntax error in {file_spec.path}: {e}"
                    )
            
            # Write file
            mode = "wb" if file_spec.binary else "w"
            with open(file_path, mode) as f:
                if isinstance(content, str):
                    f.write(content)
                else:
                    f.write(content)
            
            # Set executable if needed
            if file_spec.executable:
                file_path.chmod(file_path.stat().st_mode | 0o111)
    
    def _render_template(self, content: str, config: Dict[str, Any]) -> str:
        """Render a template string with Jinja2."""
        template = self.jinja_env.from_string(content)
        return template.render(**config)
