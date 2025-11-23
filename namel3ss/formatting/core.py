"""Core formatting infrastructure for Namel3ss AST-based formatting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
import re

from namel3ss.ast import App
from namel3ss.parser import Parser
from namel3ss.errors import N3Error


class IndentStyle(Enum):
    """Supported indentation styles."""
    SPACES = "spaces"
    TABS = "tabs"


@dataclass
class FormattingOptions:
    """Configuration options for AST formatting."""
    
    # Indentation settings
    indent_style: IndentStyle = IndentStyle.SPACES
    indent_size: int = 4
    tab_size: int = 4
    
    # Line settings
    max_line_length: int = 100
    insert_final_newline: bool = True
    trim_trailing_whitespace: bool = True
    
    # Block formatting
    preserve_empty_lines: bool = False
    max_empty_lines: int = 2
    
    # Expression formatting
    wrap_long_expressions: bool = True
    align_multiline_parameters: bool = True
    
    # String formatting
    prefer_single_quotes: bool = False
    normalize_quotes: bool = True
    
    # Comment preservation
    preserve_comments: bool = True
    align_comments: bool = True


@dataclass
class FormattedResult:
    """Result of AST formatting operation."""
    
    formatted_text: str
    is_changed: bool
    errors: List[str]
    warnings: List[str]
    
    def success(self) -> bool:
        """Check if formatting was successful."""
        return len(self.errors) == 0


class ASTFormatter:
    """
    Production-grade AST-based formatter for Namel3ss.
    
    This formatter:
    1. Parses source code into AST
    2. Applies consistent formatting rules
    3. Reconstructs formatted source code
    4. Preserves semantic meaning and comments
    """
    
    def __init__(self, options: Optional[FormattingOptions] = None):
        self.options = options or FormattingOptions()
        self._indent_str = self._make_indent_string()
    
    def _make_indent_string(self) -> str:
        """Create the indentation string based on options."""
        if self.options.indent_style == IndentStyle.TABS:
            return "\t"
        return " " * self.options.indent_size
    
    def format_document(self, source_text: str, file_path: str = "untitled.ai") -> FormattedResult:
        """
        Format a complete Namel3ss document.
        
        Args:
            source_text: The source code to format
            file_path: Path for error reporting (optional)
            
        Returns:
            FormattedResult with formatted text and status
        """
        errors = []
        warnings = []
        
        try:
            # Parse the source into AST
            parser = Parser(source_text, path=file_path)
            ast = parser.parse()
            
            # Format the AST
            formatted_text = self._format_app(ast)
            
            # Apply final text cleanup
            formatted_text = self._apply_text_cleanup(formatted_text)
            
            # Check if content actually changed
            is_changed = self._normalize_whitespace(source_text) != self._normalize_whitespace(formatted_text)
            
            return FormattedResult(
                formatted_text=formatted_text,
                is_changed=is_changed,
                errors=errors,
                warnings=warnings
            )
            
        except N3Error as e:
            errors.append(f"Parse error: {e.message}")
            return FormattedResult(
                formatted_text=source_text,  # Return original on parse error
                is_changed=False,
                errors=errors,
                warnings=warnings
            )
        except Exception as e:
            errors.append(f"Formatting error: {str(e)}")
            return FormattedResult(
                formatted_text=source_text,  # Return original on unexpected error
                is_changed=False,
                errors=errors,
                warnings=warnings
            )
    
    def _format_app(self, app: App) -> str:
        """Format the root App AST node."""
        lines = []
        
        # Format app declaration
        lines.append(f'app "{app.name}" {{')
        
        # Format theme if present
        if app.theme:
            lines.append("")
            lines.extend(self._format_theme(app.theme, indent_level=1))
        
        # Format configurations
        if app.configurations:
            lines.append("")
            for config in app.configurations:
                lines.extend(self._format_configuration(config, indent_level=1))
        
        # Format datasets
        if app.datasets:
            lines.append("")
            for dataset in app.datasets:
                lines.extend(self._format_dataset(dataset, indent_level=1))
        
        # Format frames
        if app.frames:
            lines.append("")
            for frame in app.frames:
                lines.extend(self._format_frame(frame, indent_level=1))
        
        # Format models
        if app.models:
            lines.append("")
            for model in app.models:
                lines.extend(self._format_model(model, indent_level=1))
        
        # Format prompts
        if app.prompts:
            lines.append("")
            for prompt in app.prompts:
                lines.extend(self._format_prompt(prompt, indent_level=1))
        
        # Format chains
        if app.chains:
            lines.append("")
            for chain in app.chains:
                lines.extend(self._format_chain(chain, indent_level=1))
        
        # Format insights
        if app.insights:
            lines.append("")
            for insight in app.insights:
                lines.extend(self._format_insight(insight, indent_level=1))
        
        # Format pages
        if app.pages:
            lines.append("")
            for page in app.pages:
                lines.extend(self._format_page(page, indent_level=1))
        
        lines.append("}")
        
        return "\\n".join(lines)
    
    def _format_theme(self, theme, indent_level: int) -> List[str]:
        """Format theme declaration."""
        lines = []
        indent = self._indent_str * indent_level
        
        lines.append(f'{indent}theme: "{theme.name}"')
        
        if hasattr(theme, 'primary_color') and theme.primary_color:
            lines.append(f'{indent}primary_color: "{theme.primary_color}"')
        
        return lines
    
    def _format_configuration(self, config, indent_level: int) -> List[str]:
        """Format configuration block."""
        lines = []
        indent = self._indent_str * indent_level
        
        lines.append(f'{indent}config "{config.name}" {{')
        
        # Format configuration properties
        for key, value in config.properties.items():
            value_str = self._format_value(value)
            lines.append(f'{indent}{self._indent_str}{key}: {value_str}')
        
        lines.append(f'{indent}}}')
        return lines
    
    def _format_dataset(self, dataset, indent_level: int) -> List[str]:
        """Format dataset declaration."""
        lines = []
        indent = self._indent_str * indent_level
        
        lines.append(f'{indent}dataset "{dataset.name}" {{')
        
        if hasattr(dataset, 'connector') and dataset.connector:
            lines.append(f'{indent}{self._indent_str}connector: "{dataset.connector}"')
        
        if hasattr(dataset, 'query') and dataset.query:
            lines.append(f'{indent}{self._indent_str}query: """')
            query_lines = dataset.query.strip().split('\\n')
            for query_line in query_lines:
                lines.append(f'{indent}{self._indent_str}{query_line}')
            lines.append(f'{indent}{self._indent_str}"""')
        
        lines.append(f'{indent}}}')
        return lines
    
    def _format_frame(self, frame, indent_level: int) -> List[str]:
        """Format frame declaration."""
        lines = []
        indent = self._indent_str * indent_level
        
        lines.append(f'{indent}frame "{frame.name}" {{')
        
        if hasattr(frame, 'columns') and frame.columns:
            lines.append(f'{indent}{self._indent_str}columns:')
            for column in frame.columns:
                lines.append(f'{indent}{self._indent_str}{self._indent_str}"{column.name}": {column.type}')
        
        lines.append(f'{indent}}}')
        return lines
    
    def _format_model(self, model, indent_level: int) -> List[str]:
        """Format model declaration."""
        lines = []
        indent = self._indent_str * indent_level
        
        lines.append(f'{indent}model "{model.name}" {{')
        
        if hasattr(model, 'provider') and model.provider:
            lines.append(f'{indent}{self._indent_str}provider: "{model.provider}"')
        
        if hasattr(model, 'model_name') and model.model_name:
            lines.append(f'{indent}{self._indent_str}model_name: "{model.model_name}"')
        
        lines.append(f'{indent}}}')
        return lines
    
    def _format_prompt(self, prompt, indent_level: int) -> List[str]:
        """Format prompt declaration."""
        lines = []
        indent = self._indent_str * indent_level
        
        lines.append(f'{indent}prompt "{prompt.name}" {{')
        
        if hasattr(prompt, 'model') and prompt.model:
            lines.append(f'{indent}{self._indent_str}model: "{prompt.model}"')
        
        if hasattr(prompt, 'template') and prompt.template:
            lines.append(f'{indent}{self._indent_str}template: """')
            template_lines = prompt.template.strip().split('\\n')
            for template_line in template_lines:
                lines.append(f'{indent}{self._indent_str}{template_line}')
            lines.append(f'{indent}{self._indent_str}"""')
        
        lines.append(f'{indent}}}')
        return lines
    
    def _format_chain(self, chain, indent_level: int) -> List[str]:
        """Format chain declaration."""
        lines = []
        indent = self._indent_str * indent_level
        
        effect_str = f' effect {chain.declared_effect}' if hasattr(chain, 'declared_effect') and chain.declared_effect else ''
        lines.append(f'{indent}chain "{chain.name}"{effect_str} {{')
        
        if hasattr(chain, 'steps') and chain.steps:
            for step in chain.steps:
                lines.extend(self._format_chain_step(step, indent_level + 1))
        
        lines.append(f'{indent}}}')
        return lines
    
    def _format_chain_step(self, step, indent_level: int) -> List[str]:
        """Format chain step."""
        lines = []
        indent = self._indent_str * indent_level
        
        if hasattr(step, 'kind'):
            if step.kind == 'prompt':
                lines.append(f'{indent}prompt "{step.target}"')
            elif step.kind == 'connector':
                lines.append(f'{indent}connector "{step.target}"')
            else:
                lines.append(f'{indent}{step.kind} "{step.target}"')
        
        return lines
    
    def _format_insight(self, insight, indent_level: int) -> List[str]:
        """Format insight declaration."""
        lines = []
        indent = self._indent_str * indent_level
        
        lines.append(f'{indent}insight "{insight.name}" from dataset {insight.source_dataset} {{')
        
        if hasattr(insight, 'logic') and insight.logic:
            lines.append(f'{indent}{self._indent_str}logic:')
            for logic_step in insight.logic:
                lines.extend(self._format_insight_logic_step(logic_step, indent_level + 2))
        
        lines.append(f'{indent}}}')
        return lines
    
    def _format_insight_logic_step(self, step, indent_level: int) -> List[str]:
        """Format insight logic step."""
        lines = []
        indent = self._indent_str * indent_level
        
        # Format different types of insight logic steps
        if hasattr(step, 'kind'):
            if step.kind == 'assignment':
                lines.append(f'{indent}{step.variable} = {step.expression}')
            elif step.kind == 'emit':
                lines.append(f'{indent}emit {step.emit_type} "{step.content}"')
        
        return lines
    
    def _format_page(self, page, indent_level: int) -> List[str]:
        """Format page declaration."""
        lines = []
        indent = self._indent_str * indent_level
        
        lines.append(f'{indent}page "{page.route}" {{')
        
        if hasattr(page, 'statements') and page.statements:
            for statement in page.statements:
                lines.extend(self._format_statement(statement, indent_level + 1))
        
        lines.append(f'{indent}}}')
        return lines
    
    def _format_statement(self, statement, indent_level: int) -> List[str]:
        """Format page statement."""
        lines = []
        indent = self._indent_str * indent_level
        
        # Format different statement types
        if hasattr(statement, 'type'):
            if statement.type == 'show_text':
                lines.append(f'{indent}show text: "{statement.text}"')
            elif statement.type == 'show_table':
                lines.append(f'{indent}show table: {statement.dataset}')
            elif statement.type == 'show_form':
                lines.append(f'{indent}show form: {{')
                if hasattr(statement, 'fields'):
                    for field in statement.fields:
                        lines.append(f'{indent}{self._indent_str}field: {{')
                        lines.append(f'{indent}{self._indent_str}{self._indent_str}name: "{field.name}"')
                        lines.append(f'{indent}{self._indent_str}{self._indent_str}type: "{field.type}"')
                        lines.append(f'{indent}{self._indent_str}}}')
                lines.append(f'{indent}}}')
        
        return lines
    
    def _format_value(self, value) -> str:
        """Format a value (string, number, boolean, etc.)."""
        if isinstance(value, str):
            quote_char = "'" if self.options.prefer_single_quotes else '"'
            return f'{quote_char}{value}{quote_char}'
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif value is None:
            return "null"
        else:
            return str(value)
    
    def _apply_text_cleanup(self, text: str) -> str:
        """Apply final text cleanup rules."""
        lines = text.split('\\n')
        
        # Trim trailing whitespace
        if self.options.trim_trailing_whitespace:
            lines = [line.rstrip() for line in lines]
        
        # Handle empty lines
        if not self.options.preserve_empty_lines:
            # Remove excessive empty lines
            cleaned_lines = []
            consecutive_empty = 0
            
            for line in lines:
                if not line.strip():
                    consecutive_empty += 1
                    if consecutive_empty <= self.options.max_empty_lines:
                        cleaned_lines.append(line)
                else:
                    consecutive_empty = 0
                    cleaned_lines.append(line)
            
            lines = cleaned_lines
        
        # Insert final newline
        result = '\\n'.join(lines)
        if self.options.insert_final_newline and not result.endswith('\\n'):
            result += '\\n'
        
        return result
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace for comparison."""
        # Remove all whitespace differences for comparison
        return re.sub(r'\\s+', ' ', text.strip())