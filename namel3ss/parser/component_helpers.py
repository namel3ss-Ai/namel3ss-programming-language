"""
Component helper transformations for unsupported patterns.

These helpers automatically transform common UI patterns that aren't directly
supported into equivalent supported components.
"""

from typing import Dict, Any, List, Optional
from ..ast.pages import (
    ShowText, ShowStatSummary, ShowDataTable, ShowCard,
    AccordionLayout, ShowDataList, DiffView, ShowDataChart
)


def transform_progress_bar(title: str, config: Dict[str, Any]) -> ShowStatSummary:
    """
    Transform progress_bar into stat_summary.
    
    Args:
        title: Progress bar title
        config: Configuration with value_binding, label, etc.
    
    Returns:
        ShowStatSummary component
    
    Example:
        show progress_bar "Job Progress":
            value_binding: "job.progress_pct"
            label: "Completion"
        
        Becomes:
        show stat_summary "Job Progress":
            stats:
                - label: "Completion"
                  value_binding: "job.progress_pct"
                  format: "percentage"
    """
    from ..ast.pages import StatConfig
    
    value_binding = config.get('value_binding', config.get('value', 'progress'))
    label = config.get('label', title)
    format_type = 'percentage' if 'pct' in value_binding or 'percent' in value_binding else 'number'
    
    stat = StatConfig(
        label=label,
        value_binding=value_binding,
        format=format_type,
        change=config.get('change'),
        trend=config.get('trend'),
        icon=config.get('icon', 'trending-up')
    )
    
    return ShowStatSummary(
        title=title,
        stats=[stat],
        layout=config.get('layout', 'horizontal')
    )


def transform_code_block(content: str, config: Dict[str, Any]) -> ShowText:
    """
    Transform code_block into show text with markdown.
    
    Args:
        content: Code content or binding
        config: Configuration with language, etc.
    
    Returns:
        ShowText component with markdown code block
    
    Example:
        show code_block:
            language: "python"
            content: "def hello(): print('hi')"
        
        Becomes markdown code block in show text.
    """
    language = config.get('language', 'text')
    line_numbers = config.get('line_numbers', False)
    
    # Check if content is a binding or literal
    if content.startswith('{{') and content.endswith('}}'):
        # Dynamic content - use template
        markdown_text = f"```{language}\n{content}\n```"
    else:
        # Static content
        markdown_text = f"```{language}\n{content}\n```"
    
    return ShowText(text=markdown_text, styles=config.get('styles', {}))


def transform_json_view(data_binding: str, config: Dict[str, Any]) -> ShowText:
    """
    Transform json_view into show text with to_json filter.
    
    Args:
        data_binding: Data to display as JSON
        config: Configuration options
    
    Returns:
        ShowText component with JSON formatting
    
    Example:
        show json_view:
            data: response
            indent: 2
        
        Becomes:
        show text "{{response | to_json}}"
    """
    indent = config.get('indent', 2)
    expanded = config.get('expanded', True)
    
    # Use to_json filter for formatting
    text = f"{{{{{data_binding} | to_json}}}}"
    
    return ShowText(
        text=text,
        styles={
            'font-family': 'monospace',
            'white-space': 'pre-wrap',
            'background': '#f5f5f5',
            'padding': '1rem',
            'border-radius': '4px',
            **config.get('styles', {})
        }
    )


def transform_tree_view(title: str, source: str, config: Dict[str, Any]) -> AccordionLayout:
    """
    Transform tree_view into accordion with nested sections.
    
    Args:
        title: Tree title
        source: Data source
        config: Configuration options
    
    Returns:
        AccordionLayout component
    
    Example:
        show tree_view "File Tree" from dataset files:
            expand_level: 1
        
        Becomes:
        accordion:
            section "{{name}}":
                show text "{{path}}"
    """
    # This is a simplified version - real implementation would need
    # to handle nested data structures dynamically
    from ..ast.pages import AccordionSection
    
    sections = []
    # In practice, this would be generated from the dataset
    # For now, return a basic accordion structure
    
    return AccordionLayout(
        sections=sections,
        allow_multiple=config.get('allow_multiple', False),
        default_expanded=config.get('default_expanded', [])
    )


# Mapping of component names to transformation functions
COMPONENT_TRANSFORMATIONS = {
    'progress_bar': transform_progress_bar,
    'progress': transform_progress_bar,
    'code_block': transform_code_block,
    'code': transform_code_block,
    'json_view': transform_json_view,
    'json': transform_json_view,
    'tree_view': transform_tree_view,
    'tree': transform_tree_view,
}


# Alternative suggestions for error messages
COMPONENT_ALTERNATIVES = {
    'progress_bar': {
        'name': 'Progress Bar',
        'primary': 'show stat_summary',
        'alternatives': [
            {
                'component': 'show stat_summary',
                'description': 'Display progress as a KPI stat with percentage formatting',
                'use_case': 'Best for: Dashboard KPIs, job completion status, metrics',
                'example': '''show stat_summary "Job Progress":
    stats:
        - label: "Completion"
          value_binding: "job.progress_pct"
          format: "percentage"
        - label: "Status"
          value_binding: "job.status"
          icon: "check-circle"'''
            },
            {
                'component': 'show data_chart',
                'description': 'Visualize progress over time with a chart',
                'use_case': 'Best for: Progress history, trends, multiple metrics',
                'example': '''show data_chart "Progress Over Time" from dataset progress_history:
    chart_type: "line"
    series:
        - data_key: "completion_pct"
          label: "Completion"
          color: "#10b981"
    x_axis:
        data_key: "timestamp"'''
            },
            {
                'component': 'show text',
                'description': 'Show progress as formatted text',
                'use_case': 'Best for: Simple inline status, text-based updates',
                'example': '''show text "Progress: {{job.progress_pct}}% Complete" style {
    color: "green"
    font-weight: "bold"
}'''
            }
        ],
        'why_not_supported': 'Progress bars require real-time updates and complex styling. The stat_summary component provides the same information in a more accessible, dashboard-friendly format.',
        'docs': ['UI_COMPONENT_REFERENCE.md', 'DATA_DISPLAY_COMPONENTS.md']
    },
    'code_block': {
        'name': 'Code Block',
        'primary': 'show text with markdown',
        'alternatives': [
            {
                'component': 'show text with markdown',
                'description': 'Use markdown code fences for syntax highlighting',
                'use_case': 'Best for: Displaying code snippets, static examples',
                'example': '''show text """
```python
def process_file(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data.upper()
```
"""'''
            },
            {
                'component': 'diff_view',
                'description': 'Compare code side-by-side with syntax highlighting',
                'use_case': 'Best for: Code reviews, version comparisons, changes',
                'example': '''diff_view:
    left_binding: "original_code"
    right_binding: "modified_code"
    content_type: "code"
    language: "python"
    mode: "split"
    show_line_numbers: true'''
            },
            {
                'component': 'show text with styling',
                'description': 'Display code in a monospace container',
                'use_case': 'Best for: Terminal output, logs, plain text code',
                'example': '''show text "{{code_output}}" style {
    font-family: "monospace"
    background: "#1e1e1e"
    color: "#d4d4d4"
    padding: "1rem"
    border-radius: "4px"
    white-space: "pre-wrap"
}'''
            }
        ],
        'why_not_supported': 'Code blocks with rich IDE features (folding, search) are complex. Markdown code blocks provide syntax highlighting, and diff_view handles comparisons.',
        'docs': ['UI_COMPONENT_REFERENCE.md', 'FEEDBACK_COMPONENTS_GUIDE.md']
    },
    'json_view': {
        'name': 'JSON Viewer',
        'primary': 'show text with to_json',
        'alternatives': [
            {
                'component': 'show text with to_json filter',
                'description': 'Format JSON data with built-in filter',
                'use_case': 'Best for: API responses, configuration display, debugging',
                'example': '''show text "{{api_response | to_json}}" style {
    font-family: "monospace"
    white-space: "pre-wrap"
    background: "#f5f5f5"
    padding: "1rem"
    border-radius: "4px"
}'''
            },
            {
                'component': 'show data_table',
                'description': 'Display JSON data as a structured table',
                'use_case': 'Best for: Tabular JSON, lists of objects, data grids',
                'example': '''# Convert JSON array to dataset first
show data_table "API Response" from dataset response_items:
    columns:
        - field: "id"
          header: "ID"
        - field: "name"
          header: "Name"
        - field: "status"
          header: "Status"
          render:
              type: "badge"'''
            },
            {
                'component': 'show card with info_grid',
                'description': 'Display JSON fields as key-value pairs',
                'use_case': 'Best for: Single objects, nested data, record details',
                'example': '''show card "Response Details":
    header:
        title: "API Response"
    sections:
        - type: "info_grid"
          items:
              - label: "Status"
                value: "{{response.status}}"
              - label: "Message"
                value: "{{response.message}}"
              - label: "Timestamp"
                value: "{{response.timestamp}}"
                format: "datetime"'''
            }
        ],
        'why_not_supported': 'Interactive JSON trees with expand/collapse require complex state management. The alternatives cover 95% of use cases with simpler implementations.',
        'docs': ['UI_COMPONENT_REFERENCE.md', 'DATA_DISPLAY_COMPONENTS.md', 'STANDARD_LIBRARY.md']
    },
    'tree_view': {
        'name': 'Tree View',
        'primary': 'accordion',
        'alternatives': [
            {
                'component': 'accordion',
                'description': 'Collapsible hierarchical sections',
                'use_case': 'Best for: File trees, nested menus, hierarchical data',
                'example': '''accordion:
    section "Documents":
        accordion:
            section "Projects":
                show data_list "Files" from dataset project_files:
                    item:
                        title: "{{filename}}"
                        subtitle: "{{path}}"
            section "Reports":
                show text "Report files here"
    section "Images":
        show text "Image files here"'''
            },
            {
                'component': 'show data_list with nesting',
                'description': 'List items with visual hierarchy',
                'use_case': 'Best for: Activity feeds, threaded comments, nested lists',
                'example': '''show data_list "File System" from dataset files:
    item:
        title: "{{name}}"
        subtitle: "{{path}}"
        metadata:
            - field: "size"
              format: "bytes"
            - field: "modified"
              format: "relative"
        badge:
            field: "type"
            style: "badge-{{type}}"'''
            },
            {
                'component': 'show card with nested sections',
                'description': 'Card sections for grouped hierarchical data',
                'use_case': 'Best for: Structured records, grouped data, categories',
                'example': '''show card "Project Structure":
    sections:
        - type: "text_section"
          title: "Source Code"
          content: "src/ directory"
        - type: "info_grid"
          items:
              - label: "Components"
                value: "12 files"
              - label: "Tests"
                value: "45 files"'''
            }
        ],
        'why_not_supported': 'Tree views with expand/collapse/drag-drop are complex widgets. Accordion provides the same hierarchical structure with simpler semantics.',
        'docs': ['UI_COMPONENT_REFERENCE.md', 'DATA_DISPLAY_COMPONENTS.md', 'LAYOUT_PRIMITIVES.md']
    }
}


def get_component_alternatives(component_name: str) -> Optional[Dict[str, Any]]:
    """Get alternatives for an unsupported component."""
    return COMPONENT_ALTERNATIVES.get(component_name)


def format_alternatives_error(component_name: str) -> str:
    """Format a comprehensive error message with alternatives."""
    info = get_component_alternatives(component_name)
    if not info:
        return f"Component '{component_name}' is not supported."
    
    # Build error message
    lines = [
        f"Component '{info['name']}' is not supported.",
        "",
        f"Why: {info['why_not_supported']}",
        "",
        f"✨ Recommended: Use {info['primary']}",
        "",
        "Alternatives:",
    ]
    
    for i, alt in enumerate(info['alternatives'], 1):
        lines.append(f"\n{i}. {alt['component']}")
        lines.append(f"   {alt['description']}")
        lines.append(f"   {alt['use_case']}")
        lines.append(f"\n   Example:")
        for line in alt['example'].split('\n'):
            lines.append(f"   {line}")
    
    lines.append("\n📚 Documentation:")
    for doc in info['docs']:
        lines.append(f"   • docs/{doc}")
    
    return '\n'.join(lines)
