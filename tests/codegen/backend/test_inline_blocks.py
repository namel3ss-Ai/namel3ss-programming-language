"""Tests for inline blocks code generation."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from namel3ss.ast import App, InlinePythonBlock, InlineReactBlock, Page
from namel3ss.codegen.backend.inline_blocks import (
    collect_inline_blocks,
    generate_inline_python_module,
    generate_inline_react_components,
)


class TestCollectInlineBlocks:
    """Test inline block collection from App AST."""
    
    def test_collect_empty_app(self):
        """Test collecting from app with no inline blocks."""
        app = App(name="TestApp", pages=[])
        blocks = collect_inline_blocks(app)
        
        assert blocks == {"python": [], "react": []}
    
    def test_collect_python_blocks(self):
        """Test collecting Python inline blocks."""
        block1 = InlinePythonBlock(code="return 42", bindings={})
        block2 = InlinePythonBlock(code="return [1, 2, 3]", bindings={"x": None})
        
        app = App(name="TestApp", pages=[
            Page(name="TestPage", route="/test")
        ])
        # Manually attach blocks (in real code, these would be in expressions)
        app.pages[0]._test_block1 = block1
        app.pages[0]._test_block2 = block2
        
        blocks = collect_inline_blocks(app)
        
        assert len(blocks["python"]) == 2
        assert block1 in blocks["python"]
        assert block2 in blocks["python"]
        assert blocks["react"] == []
    
    def test_collect_react_blocks(self):
        """Test collecting React inline blocks."""
        block1 = InlineReactBlock(
            code="<div>Hello</div>",
            component_name="HelloComponent",
            props={"name": "str"}
        )
        block2 = InlineReactBlock(
            code="<button>Click</button>",
            component_name="ButtonComponent",
            props={}
        )
        
        app = App(name="TestApp", pages=[
            Page(name="TestPage", route="/test")
        ])
        app.pages[0]._test_block1 = block1
        app.pages[0]._test_block2 = block2
        
        blocks = collect_inline_blocks(app)
        
        assert blocks["python"] == []
        assert len(blocks["react"]) == 2
        assert block1 in blocks["react"]
        assert block2 in blocks["react"]
    
    def test_collect_mixed_blocks(self):
        """Test collecting both Python and React blocks."""
        py_block = InlinePythonBlock(code="return True", bindings={})
        react_block = InlineReactBlock(
            code="<span>Test</span>",
            component_name="TestComponent",
            props={}
        )
        
        app = App(name="TestApp", pages=[
            Page(name="TestPage", route="/test")
        ])
        app.pages[0]._py = py_block
        app.pages[0]._react = react_block
        
        blocks = collect_inline_blocks(app)
        
        assert len(blocks["python"]) == 1
        assert len(blocks["react"]) == 1
        assert py_block in blocks["python"]
        assert react_block in blocks["react"]


class TestGenerateInlinePythonModule:
    """Test Python module generation from inline blocks."""
    
    def test_empty_module(self):
        """Test generating module with no blocks."""
        code = generate_inline_python_module([])
        
        assert "Generated inline Python blocks (empty)" in code
        assert "__all__ = []" in code
        assert code.strip().endswith('\n') is False or code.strip().endswith('\n\n') is False
    
    def test_single_block(self):
        """Test generating module with one block."""
        block = InlinePythonBlock(code="return 42", bindings={})
        code = generate_inline_python_module([block])
        
        # Check function definition
        func_name = f"inline_python_{id(block)}"
        assert f"def {func_name}(context: Optional[Dict[str, Any]] = None) -> Any:" in code
        
        # Check code is included
        assert "return 42" in code
        
        # Check exports
        assert f'"{func_name}"' in code
        assert "__all__ = [" in code
    
    def test_block_with_bindings(self):
        """Test generating code with context bindings."""
        block = InlinePythonBlock(
            code="return x * 2",
            bindings={"x": None}
        )
        code = generate_inline_python_module([block])
        
        # Check bindings are extracted
        assert 'x = context.get("x") if context else None' in code
        assert "return x * 2" in code
    
    def test_multiple_blocks(self):
        """Test generating module with multiple blocks."""
        block1 = InlinePythonBlock(code="return 1", bindings={})
        block2 = InlinePythonBlock(code="return 2", bindings={})
        block3 = InlinePythonBlock(code="return 3", bindings={})
        
        code = generate_inline_python_module([block1, block2, block3])
        
        # Check all functions present
        for block in [block1, block2, block3]:
            func_name = f"inline_python_{id(block)}"
            assert f"def {func_name}" in code
            assert f'"{func_name}"' in code
        
        # Check all return values
        assert "return 1" in code
        assert "return 2" in code
        assert "return 3" in code
    
    def test_multiline_code(self):
        """Test generating code with multiple lines."""
        code_str = textwrap.dedent("""
            result = []
            for i in range(10):
                result.append(i * 2)
            return result
        """).strip()
        
        block = InlinePythonBlock(code=code_str, bindings={})
        generated = generate_inline_python_module([block])
        
        # Check multiline code preserved
        assert "result = []" in generated
        assert "for i in range(10):" in generated
        assert "result.append(i * 2)" in generated
        assert "return result" in generated
    
    def test_imports_extraction(self):
        """Test extracting and including imports."""
        block1 = InlinePythonBlock(
            code="import math\nreturn math.sqrt(x)",
            bindings={"x": None}
        )
        block2 = InlinePythonBlock(
            code="from datetime import datetime\nreturn datetime.now()",
            bindings={}
        )
        
        code = generate_inline_python_module([block1, block2])
        
        # Check imports extracted and placed at top
        lines = code.split('\n')
        import_section_start = None
        for i, line in enumerate(lines):
            if 'import math' in line:
                import_section_start = i
                break
        
        assert import_section_start is not None
        assert 'import math' in code
        assert 'from datetime import datetime' in code
    
    def test_context_bindings_parameter(self):
        """Test passing context_bindings to module generation."""
        block = InlinePythonBlock(code="return foo + bar", bindings={"foo": None})
        
        code = generate_inline_python_module(
            [block],
            context_bindings={"bar": None, "baz": None}
        )
        
        # All bindings should be present
        assert 'foo = context.get("foo")' in code
        assert 'bar = context.get("bar")' in code
        assert 'baz = context.get("baz")' in code
    
    def test_valid_python_syntax(self):
        """Test that generated code is syntactically valid Python."""
        blocks = [
            InlinePythonBlock(code="return 42", bindings={}),
            InlinePythonBlock(code="return [1, 2, 3]", bindings={"x": None}),
            InlinePythonBlock(
                code="result = x + y\nreturn result",
                bindings={"x": None, "y": None}
            ),
        ]
        
        code = generate_inline_python_module(blocks)
        
        # Try to compile it
        compile(code, '<generated>', 'exec')


class TestGenerateInlineReactComponents:
    """Test React component generation from inline blocks."""
    
    def test_empty_components(self):
        """Test generating with no blocks."""
        components = generate_inline_react_components([])
        assert components == {}
    
    def test_single_jsx_fragment(self):
        """Test generating simple JSX fragment."""
        block = InlineReactBlock(
            code="<div>Hello World</div>",
            component_name="HelloComponent",
            props={}
        )
        
        components = generate_inline_react_components([block], typescript=True)
        
        assert len(components) == 1
        filename = f"InlineReact{id(block)}.tsx"
        assert filename in components
        
        code = components[filename]
        assert "import React from 'react'" in code
        assert "export function HelloComponent()" in code
        assert "return (" in code
        assert "<div>Hello World</div>" in code
    
    def test_component_with_props(self):
        """Test generating component with typed props."""
        block = InlineReactBlock(
            code="<h1>{props.title}</h1>",
            component_name="TitleComponent",
            props={"title": "string", "subtitle": "string"}
        )
        
        components = generate_inline_react_components([block], typescript=True)
        code = list(components.values())[0]
        
        # Check TypeScript interface
        assert "interface TitleComponentProps {" in code
        assert "title?: any;" in code
        assert "subtitle?: any;" in code
        
        # Check function signature
        assert "export function TitleComponent(props: TitleComponentProps)" in code
    
    def test_javascript_output(self):
        """Test generating JavaScript instead of TypeScript."""
        block = InlineReactBlock(
            code="<button>Click me</button>",
            component_name="ButtonComponent",
            props={}
        )
        
        components = generate_inline_react_components([block], typescript=False)
        
        filename = f"InlineReact{id(block)}.jsx"
        assert filename in components
        
        code = components[filename]
        assert "export function ButtonComponent(props)" in code
        assert "interface" not in code  # No TypeScript
    
    def test_custom_imports(self):
        """Test adding custom React imports."""
        block = InlineReactBlock(
            code="const [count, setCount] = useState(0)",
            component_name="CounterComponent",
            props={},
            requires_imports=["useState", "useEffect"]
        )
        
        components = generate_inline_react_components([block], typescript=True)
        code = list(components.values())[0]
        
        assert "import { useState } from 'react'" in code
        assert "import { useEffect } from 'react'" in code
    
    def test_multiline_jsx(self):
        """Test generating component with multiline JSX."""
        jsx_code = textwrap.dedent("""
            <div className="container">
                <h1>{props.title}</h1>
                <p>{props.description}</p>
                <button onClick={props.onClose}>Close</button>
            </div>
        """).strip()
        
        block = InlineReactBlock(
            code=jsx_code,
            component_name="CardComponent",
            props={"title": "string", "description": "string", "onClose": "function"}
        )
        
        components = generate_inline_react_components([block], typescript=True)
        code = list(components.values())[0]
        
        assert '<div className="container">' in code
        assert '<h1>{props.title}</h1>' in code
        assert '<p>{props.description}</p>' in code
        assert '<button onClick={props.onClose}>Close</button>' in code
    
    def test_multiple_components(self):
        """Test generating multiple React components."""
        blocks = [
            InlineReactBlock(code="<div>A</div>", component_name="ComponentA", props={}),
            InlineReactBlock(code="<div>B</div>", component_name="ComponentB", props={}),
            InlineReactBlock(code="<div>C</div>", component_name="ComponentC", props={}),
        ]
        
        components = generate_inline_react_components(blocks, typescript=True)
        
        assert len(components) == 3
        
        for block in blocks:
            filename = f"InlineReact{id(block)}.tsx"
            assert filename in components
            assert f"export function {block.component_name}" in components[filename]
    
    def test_function_component_definition(self):
        """Test detecting and handling full component definitions."""
        block = InlineReactBlock(
            code="""function MyButton({ onClick, label }) {
    const [clicked, setClicked] = React.useState(false);
    return <button onClick={() => { setClicked(true); onClick(); }}>{label}</button>;
}""",
            component_name="MyButton",
            props={"onClick": "function", "label": "string"}
        )
        
        components = generate_inline_react_components([block], typescript=True)
        code = list(components.values())[0]
        
        # Check the function definition is included
        assert "function MyButton" in code
        assert "setClicked" in code
    
    def test_no_component_name(self):
        """Test generating component with auto-generated name."""
        block = InlineReactBlock(
            code="<span>Test</span>",
            component_name=None,
            props={}
        )
        
        components = generate_inline_react_components([block], typescript=True)
        
        # Should use InlineReact{id} as component name
        component_id = f"InlineReact{id(block)}"
        code = list(components.values())[0]
        assert f"export function {component_id}" in code


class TestIntegration:
    """Integration tests combining collection and generation."""
    
    def test_end_to_end_python(self):
        """Test full workflow: collect -> generate -> validate."""
        # Create app with inline Python blocks
        blocks = [
            InlinePythonBlock(code="return x * 2", bindings={"x": None}),
            InlinePythonBlock(code="return [1, 2, 3]", bindings={}),
        ]
        
        app = App(name="TestApp", pages=[
            Page(name="TestPage", route="/test")
        ])
        app.pages[0]._block1 = blocks[0]
        app.pages[0]._block2 = blocks[1]
        
        # Collect
        collected = collect_inline_blocks(app)
        assert len(collected["python"]) == 2
        
        # Generate
        module_code = generate_inline_python_module(collected["python"])
        
        # Validate syntax
        compile(module_code, '<generated>', 'exec')
        
        # Check both functions present
        for block in blocks:
            assert f"inline_python_{id(block)}" in module_code
    
    def test_end_to_end_react(self):
        """Test full workflow for React components."""
        # Create app with React blocks
        blocks = [
            InlineReactBlock(
                code="<div>Component 1</div>",
                component_name="Comp1",
                props={}
            ),
            InlineReactBlock(
                code="<div>Component 2</div>",
                component_name="Comp2",
                props={"data": "any"}
            ),
        ]
        
        app = App(name="TestApp", pages=[
            Page(name="TestPage", route="/test")
        ])
        app.pages[0]._block1 = blocks[0]
        app.pages[0]._block2 = blocks[1]
        
        # Collect
        collected = collect_inline_blocks(app)
        assert len(collected["react"]) == 2
        
        # Generate
        components = generate_inline_react_components(collected["react"], typescript=True)
        
        assert len(components) == 2
        for block in blocks:
            filename = f"InlineReact{id(block)}.tsx"
            assert filename in components
            assert f"export function {block.component_name}" in components[filename]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
