"""Code generation for inline template blocks (Python, React, etc.)."""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from namel3ss.ast import App, InlinePythonBlock, InlineReactBlock


__all__ = [
    "collect_inline_blocks",
    "generate_inline_python_module",
    "generate_inline_react_components",
]


def collect_inline_blocks(app: App) -> Dict[str, List[Any]]:
    """
    Collect all inline blocks from an App AST.
    
    Recursively walks the App AST to find InlinePythonBlock and InlineReactBlock nodes.
    Returns dict mapping block type to list of blocks.
    
    Parameters
    ----------
    app : App
        The application AST
    
    Returns
    -------
    Dict[str, List[Any]]
        {"python": [InlinePythonBlock, ...], "react": [InlineReactBlock, ...]}
    """
    from namel3ss.ast import InlinePythonBlock, InlineReactBlock
    
    python_blocks: List[InlinePythonBlock] = []
    react_blocks: List[InlineReactBlock] = []
    
    def walk(node: Any) -> None:
        """Recursively walk AST nodes."""
        if isinstance(node, InlinePythonBlock):
            python_blocks.append(node)
        elif isinstance(node, InlineReactBlock):
            react_blocks.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
        elif hasattr(node, '__dict__'):
            for value in node.__dict__.values():
                walk(value)
    
    walk(app)
    
    return {
        "python": python_blocks,
        "react": react_blocks,
    }


def generate_inline_python_module(
    python_blocks: List[InlinePythonBlock],
    context_bindings: Dict[str, Any] = None,
) -> str:
    """
    Generate Python module from InlinePythonBlock nodes.
    
    Creates a Python module with:
    - Function definitions for each block
    - Imports extracted from code
    - Context bindings support
    
    Parameters
    ----------
    python_blocks : List[InlinePythonBlock]
        List of inline Python blocks to generate code for
    context_bindings : Dict[str, Any]
        Optional context variables to inject
    
    Returns
    -------
    str
        Complete Python module source code
    """
    if not python_blocks:
        return _generate_empty_inline_python_module()
    
    lines: List[str] = []
    
    # Module header
    lines.append('"""Generated inline Python blocks."""')
    lines.append('')
    lines.append('from __future__ import annotations')
    lines.append('')
    lines.append('from typing import Any, Dict, Optional')
    lines.append('')
    
    # Extract and deduplicate imports from all blocks
    imports = _extract_python_imports(python_blocks)
    if imports:
        lines.extend(imports)
        lines.append('')
    
    # Generate function for each block
    for i, block in enumerate(python_blocks):
        func_name = f"inline_python_{id(block)}"
        lines.append('')
        lines.append(f"def {func_name}(context: Optional[Dict[str, Any]] = None) -> Any:")
        lines.append('    """Generated inline Python block."""')
        
        # Add context bindings
        if block.bindings or context_bindings:
            lines.append('    # Context bindings')
            bindings = {**(context_bindings or {}), **block.bindings}
            for key in bindings:
                lines.append(f'    {key} = context.get("{key}") if context else None')
            lines.append('')
        
        # Add the inline code
        code_lines = block.code.splitlines()
        if code_lines:
            lines.append('    # Inline Python code')
            for code_line in code_lines:
                # Indent code by 4 spaces
                lines.append(f'    {code_line}' if code_line.strip() else '')
        else:
            lines.append('    pass')
        
        lines.append('')
    
    # Export all functions
    lines.append('')
    lines.append('__all__ = [')
    for block in python_blocks:
        func_name = f"inline_python_{id(block)}"
        lines.append(f'    "{func_name}",')
    lines.append(']')
    lines.append('')
    
    return '\n'.join(lines)


def _generate_empty_inline_python_module() -> str:
    """Generate empty inline Python module when no blocks exist."""
    return textwrap.dedent('''
        """Generated inline Python blocks (empty)."""
        
        # No inline Python blocks in this application
        
        __all__ = []
    ''').strip() + '\n'


def _extract_python_imports(blocks: List[InlinePythonBlock]) -> List[str]:
    """
    Extract import statements from Python code blocks.
    
    Scans code for import statements and returns deduplicated list.
    
    Parameters
    ----------
    blocks : List[InlinePythonBlock]
        Blocks to extract imports from
    
    Returns
    -------
    List[str]
        Sorted list of unique import statements
    """
    imports: Set[str] = set()
    
    for block in blocks:
        for line in block.code.splitlines():
            stripped = line.strip()
            # Simple import detection
            if stripped.startswith('import ') or stripped.startswith('from '):
                imports.add(stripped)
    
    return sorted(imports)


def generate_inline_react_components(
    react_blocks: List[InlineReactBlock],
    typescript: bool = True,
) -> Dict[str, str]:
    """
    Generate React components from InlineReactBlock nodes.
    
    Creates TypeScript/JavaScript React components.
    Returns dict mapping component names to source code.
    
    Parameters
    ----------
    react_blocks : List[InlineReactBlock]
        List of inline React blocks to generate code for
    typescript : bool
        Whether to generate TypeScript (True) or JavaScript (False)
    
    Returns
    -------
    Dict[str, str]
        Mapping from component name/ID to source code
    """
    components: Dict[str, str] = {}
    
    for block in react_blocks:
        component_id = f"InlineReact{id(block)}"
        component_name = block.component_name or component_id
        
        lines: List[str] = []
        
        # Imports
        lines.append("import React from 'react';")
        
        # Add custom imports if specified
        if block.requires_imports:
            for import_name in block.requires_imports:
                if import_name not in ['React', 'react']:
                    lines.append(f"import {{ {import_name} }} from 'react';")
        
        lines.append('')
        
        # Component definition
        if typescript:
            # Generate TypeScript interface for props
            if block.props:
                lines.append(f"interface {component_name}Props {{")
                for prop_name, prop_type in block.props.items():
                    lines.append(f"  {prop_name}?: any;")
                lines.append("}")
                lines.append('')
                
                lines.append(f"export function {component_name}(props: {component_name}Props) {{")
            else:
                lines.append(f"export function {component_name}() {{")
        else:
            lines.append(f"export function {component_name}(props) {{")
        
        # Component body
        code_lines = block.code.splitlines()
        if code_lines:
            # Check if code is already a complete function
            if 'function' in block.code or 'const' in block.code or '=>' in block.code:
                # Code contains function definition, use as-is
                for code_line in code_lines:
                    lines.append(f"  {code_line}" if code_line.strip() else '')
            else:
                # Code is JSX fragment, wrap in return
                lines.append('  return (')
                for code_line in code_lines:
                    lines.append(f"    {code_line}" if code_line.strip() else '')
                lines.append('  );')
        else:
            lines.append('  return null;')
        
        lines.append('}')
        lines.append('')
        
        # Store component
        file_ext = 'tsx' if typescript else 'jsx'
        components[f"{component_id}.{file_ext}"] = '\n'.join(lines)
    
    return components


__all__ = [
    "collect_inline_blocks",
    "generate_inline_python_module",
    "generate_inline_react_components",
]
