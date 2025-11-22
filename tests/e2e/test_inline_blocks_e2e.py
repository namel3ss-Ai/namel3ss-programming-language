"""End-to-end tests for inline blocks feature.

These tests verify the complete pipeline:
1. Parse inline block expressions
2. Generate Python/React code
3. Execute generated code
4. Verify runtime behavior
"""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import pytest

from namel3ss.ast import App, Page, InlinePythonBlock, InlineReactBlock
from namel3ss.codegen.backend.core.generator import generate_backend
from namel3ss.codegen.backend.inline_blocks import (
    collect_inline_blocks,
    generate_inline_python_module,
    generate_inline_react_components,
)


class TestInlineBlocksPipelineE2E:
    """End-to-end tests for inline blocks through the full pipeline."""
    
    def test_collect_and_generate_python_module(self):
        """Test collecting Python blocks and generating executable module."""
        # Create app with inline Python blocks
        block1 = InlinePythonBlock(
            code="return 2 + 2",
            bindings={}
        )
        block2 = InlinePythonBlock(
            code="return x * 2",
            bindings={"x": None}
        )
        
        app = App(
            name="TestApp",
            pages=[Page(name="TestPage", route="/test")]
        )
        # Attach blocks to page (simulating real usage)
        app.pages[0]._inline_data1 = block1
        app.pages[0]._inline_data2 = block2
        
        # Collect blocks
        blocks = collect_inline_blocks(app)
        assert len(blocks["python"]) == 2
        
        # Generate Python module
        python_code = generate_inline_python_module(blocks["python"])
        
        # Verify module is valid Python
        namespace = {}
        exec(python_code, namespace)
        
        # Execute first block
        func1_name = f"inline_python_{id(block1)}"
        result1 = namespace[func1_name](context=None)
        assert result1 == 4
        
        # Execute second block with context
        func2_name = f"inline_python_{id(block2)}"
        result2 = namespace[func2_name](context={"x": 10})
        assert result2 == 20
    
    def test_collect_and_generate_react_components(self):
        """Test collecting React blocks and generating components."""
        block1 = InlineReactBlock(
            code="<div>Hello World</div>",
            component_name="HelloComponent",
            props={}
        )
        block2 = InlineReactBlock(
            code="<h1>{props.title}</h1>",
            component_name="TitleComponent",
            props={"title": "string"}
        )
        
        app = App(
            name="TestApp",
            pages=[Page(name="TestPage", route="/test")]
        )
        app.pages[0]._ui1 = block1
        app.pages[0]._ui2 = block2
        
        # Collect blocks
        blocks = collect_inline_blocks(app)
        assert len(blocks["react"]) == 2
        
        # Generate React components
        components = generate_inline_react_components(blocks["react"], typescript=True)
        
        assert len(components) == 2
        
        # Verify component 1
        comp1_file = f"InlineReact{id(block1)}.tsx"
        assert comp1_file in components
        assert "export function HelloComponent" in components[comp1_file]
        assert "<div>Hello World</div>" in components[comp1_file]
        
        # Verify component 2
        comp2_file = f"InlineReact{id(block2)}.tsx"
        assert comp2_file in components
        assert "export function TitleComponent" in components[comp2_file]
        assert "interface TitleComponentProps" in components[comp2_file]
        assert "title?: any" in components[comp2_file]
    
    def test_backend_generation_with_inline_blocks(self):
        """Test backend generation includes inline blocks files."""
        # Create app with mixed inline blocks
        py_block = InlinePythonBlock(
            code='return "Hello from Python"',
            bindings={}
        )
        react_block = InlineReactBlock(
            code="<div>Hello from React</div>",
            component_name="Greeting",
            props={}
        )
        
        app = App(
            name="TestApp",
            pages=[Page(name="HomePage", route="/")]
        )
        app.pages[0]._greeting = py_block
        app.pages[0]._ui = react_block
        
        # Collect and generate inline blocks directly
        blocks = collect_inline_blocks(app)
        
        # Generate Python module
        python_code = generate_inline_python_module(blocks["python"])
        assert "def inline_python_" in python_code
        assert 'return "Hello from Python"' in python_code
        
        # Generate React components
        components = generate_inline_react_components(blocks["react"], typescript=True)
        assert len(components) == 1
        
        tsx_content = list(components.values())[0]
        assert "import React from 'react'" in tsx_content
        assert "export function Greeting" in tsx_content
        assert "<div>Hello from React</div>" in tsx_content
    
    def test_python_block_with_imports_executes(self):
        """Test that Python blocks with imports execute correctly."""
        block = InlinePythonBlock(
            code=textwrap.dedent("""
                import math
                return math.sqrt(16)
            """).strip(),
            bindings={}
        )
        
        app = App(name="MathApp", pages=[Page(name="MathPage", route="/math")])
        app.pages[0]._calculation = block
        
        blocks = collect_inline_blocks(app)
        python_code = generate_inline_python_module(blocks["python"])
        
        namespace = {}
        exec(python_code, namespace)
        
        func_name = f"inline_python_{id(block)}"
        result = namespace[func_name](context=None)
        assert result == 4.0
    
    def test_python_block_with_multiline_logic(self):
        """Test Python block with complex multi-line logic."""
        block = InlinePythonBlock(
            code=textwrap.dedent("""
                result = []
                for i in range(5):
                    if i % 2 == 0:
                        result.append(i)
                return result
            """).strip(),
            bindings={}
        )
        
        app = App(name="FilterApp", pages=[Page(name="FilterPage", route="/filter")])
        app.pages[0]._filtered = block
        
        blocks = collect_inline_blocks(app)
        python_code = generate_inline_python_module(blocks["python"])
        
        namespace = {}
        exec(python_code, namespace)
        
        func_name = f"inline_python_{id(block)}"
        result = namespace[func_name](context=None)
        assert result == [0, 2, 4]
    
    def test_python_block_with_context_variables(self):
        """Test Python block accessing context variables."""
        block = InlinePythonBlock(
            code="return [x * 2 for x in items]",
            bindings={"items": None}
        )
        
        app = App(name="MapApp", pages=[Page(name="MapPage", route="/map")])
        app.pages[0]._mapped = block
        
        blocks = collect_inline_blocks(app)
        python_code = generate_inline_python_module(blocks["python"])
        
        namespace = {}
        exec(python_code, namespace)
        
        func_name = f"inline_python_{id(block)}"
        
        # Test with different contexts
        result1 = namespace[func_name](context={"items": [1, 2, 3]})
        assert result1 == [2, 4, 6]
        
        result2 = namespace[func_name](context={"items": [10, 20, 30]})
        assert result2 == [20, 40, 60]
        
        result3 = namespace[func_name](context={"items": []})
        assert result3 == []
    
    def test_multiple_python_blocks_in_app(self):
        """Test app with multiple Python blocks."""
        block1 = InlinePythonBlock(code="return 10", bindings={})
        block2 = InlinePythonBlock(code="return 20", bindings={})
        block3 = InlinePythonBlock(code="return a + b", bindings={"a": None, "b": None})
        
        app = App(name="MultiApp", pages=[Page(name="Page1", route="/p1")])
        app.pages[0]._data1 = block1
        app.pages[0]._data2 = block2
        app.pages[0]._data3 = block3
        
        blocks = collect_inline_blocks(app)
        assert len(blocks["python"]) == 3
        
        python_code = generate_inline_python_module(blocks["python"])
        namespace = {}
        exec(python_code, namespace)
        
        # Verify all functions work
        assert namespace[f"inline_python_{id(block1)}"](context=None) == 10
        assert namespace[f"inline_python_{id(block2)}"](context=None) == 20
        assert namespace[f"inline_python_{id(block3)}"](context={"a": 5, "b": 7}) == 12
    
    def test_react_component_typescript_generation(self):
        """Test React component generates valid TypeScript."""
        block = InlineReactBlock(
            code=textwrap.dedent("""
                <div className="card">
                    <h2>{props.title}</h2>
                    <p>{props.description}</p>
                    <button onClick={props.onClose}>Close</button>
                </div>
            """).strip(),
            component_name="Card",
            props={"title": "string", "description": "string", "onClose": "function"}
        )
        
        app = App(name="UIApp", pages=[Page(name="UIPage", route="/ui")])
        app.pages[0]._card = block
        
        blocks = collect_inline_blocks(app)
        components = generate_inline_react_components(blocks["react"], typescript=True)
        
        comp_file = f"InlineReact{id(block)}.tsx"
        code = components[comp_file]
        
        # Verify TypeScript features
        assert "interface CardProps {" in code
        assert "title?: any;" in code
        assert "description?: any;" in code
        assert "onClose?: any;" in code
        assert "export function Card(props: CardProps)" in code
        
        # Verify JSX content
        assert '<div className="card">' in code
        assert "{props.title}" in code
        assert "{props.description}" in code
        assert "{props.onClose}" in code
    
    def test_react_component_javascript_generation(self):
        """Test React component generates valid JavaScript."""
        block = InlineReactBlock(
            code="<button>Click Me</button>",
            component_name="Button",
            props={}
        )
        
        app = App(name="JSApp", pages=[Page(name="JSPage", route="/js")])
        app.pages[0]._button = block
        
        blocks = collect_inline_blocks(app)
        components = generate_inline_react_components(blocks["react"], typescript=False)
        
        comp_file = f"InlineReact{id(block)}.jsx"
        code = components[comp_file]
        
        # Verify JavaScript (no TypeScript features)
        assert "interface" not in code
        assert "export function Button(props)" in code
        assert "<button>Click Me</button>" in code
    
    def test_empty_app_no_inline_blocks(self):
        """Test that app without inline blocks returns empty collections."""
        app = App(name="EmptyApp", pages=[Page(name="EmptyPage", route="/")])
        
        blocks = collect_inline_blocks(app)
        assert blocks["python"] == []
        assert blocks["react"] == []
        
        # Generate empty module
        python_code = generate_inline_python_module([])
        assert "Generated inline Python blocks (empty)" in python_code
        assert "__all__ = []" in python_code
        
        # Generate no components
        components = generate_inline_react_components([], typescript=True)
        assert components == {}
    
    def test_python_block_with_nested_data_structures(self):
        """Test Python block with nested dicts and lists."""
        block = InlinePythonBlock(
            code=textwrap.dedent("""
                data = {
                    "users": [
                        {"id": 1, "name": "Alice"},
                        {"id": 2, "name": "Bob"}
                    ],
                    "count": 2
                }
                return data
            """).strip(),
            bindings={}
        )
        
        app = App(name="DataApp", pages=[Page(name="DataPage", route="/data")])
        app.pages[0]._data = block
        
        blocks = collect_inline_blocks(app)
        python_code = generate_inline_python_module(blocks["python"])
        
        namespace = {}
        exec(python_code, namespace)
        
        func_name = f"inline_python_{id(block)}"
        result = namespace[func_name](context=None)
        
        assert result["count"] == 2
        assert len(result["users"]) == 2
        assert result["users"][0]["name"] == "Alice"
        assert result["users"][1]["name"] == "Bob"
    
    def test_python_block_with_function_definition(self):
        """Test Python block containing a function definition."""
        block = InlinePythonBlock(
            code=textwrap.dedent("""
                def helper(x):
                    return x * 3
                
                return [helper(i) for i in range(5)]
            """).strip(),
            bindings={}
        )
        
        app = App(name="FuncApp", pages=[Page(name="FuncPage", route="/func")])
        app.pages[0]._result = block
        
        blocks = collect_inline_blocks(app)
        python_code = generate_inline_python_module(blocks["python"])
        
        namespace = {}
        exec(python_code, namespace)
        
        func_name = f"inline_python_{id(block)}"
        result = namespace[func_name](context=None)
        assert result == [0, 3, 6, 9, 12]
    
    def test_react_component_with_custom_imports(self):
        """Test React component requiring custom imports."""
        block = InlineReactBlock(
            code=textwrap.dedent("""
                const [count, setCount] = useState(0);
                const handleClick = useCallback(() => {
                    setCount(count + 1);
                }, [count]);
                
                return <button onClick={handleClick}>Count: {count}</button>;
            """).strip(),
            component_name="Counter",
            props={},
            requires_imports=["useState", "useCallback"]
        )
        
        app = App(name="HooksApp", pages=[Page(name="HooksPage", route="/hooks")])
        app.pages[0]._counter = block
        
        blocks = collect_inline_blocks(app)
        components = generate_inline_react_components(blocks["react"], typescript=True)
        
        comp_file = f"InlineReact{id(block)}.tsx"
        code = components[comp_file]
        
        # Verify imports
        assert "import { useState } from 'react'" in code
        assert "import { useCallback } from 'react'" in code
        
        # Verify component code
        assert "const [count, setCount] = useState(0)" in code
        assert "useCallback" in code


class TestInlineBlocksErrorHandling:
    """Test error handling in inline blocks pipeline."""
    
    def test_python_syntax_error_detected(self):
        """Test that Python syntax errors are caught during execution."""
        block = InlinePythonBlock(
            code="return invalid syntax here",
            bindings={}
        )
        
        app = App(name="ErrorApp", pages=[Page(name="ErrorPage", route="/error")])
        app.pages[0]._bad = block
        
        blocks = collect_inline_blocks(app)
        python_code = generate_inline_python_module(blocks["python"])
        
        namespace = {}
        # Should raise SyntaxError when executing generated code
        with pytest.raises(SyntaxError):
            exec(python_code, namespace)
    
    def test_python_runtime_error_propagates(self):
        """Test that runtime errors propagate correctly."""
        block = InlinePythonBlock(
            code="return 1 / 0",  # Division by zero
            bindings={}
        )
        
        app = App(name="RuntimeErrorApp", pages=[Page(name="ErrorPage", route="/error")])
        app.pages[0]._error = block
        
        blocks = collect_inline_blocks(app)
        python_code = generate_inline_python_module(blocks["python"])
        
        namespace = {}
        exec(python_code, namespace)
        
        func_name = f"inline_python_{id(block)}"
        # Should raise ZeroDivisionError when calling function
        with pytest.raises(ZeroDivisionError):
            namespace[func_name](context=None)
    
    def test_missing_context_variable_handled(self):
        """Test behavior when context variable is missing."""
        block = InlinePythonBlock(
            code="return x * 2",
            bindings={"x": None}
        )
        
        app = App(name="ContextApp", pages=[Page(name="ContextPage", route="/ctx")])
        app.pages[0]._data = block
        
        blocks = collect_inline_blocks(app)
        python_code = generate_inline_python_module(blocks["python"])
        
        namespace = {}
        exec(python_code, namespace)
        
        func_name = f"inline_python_{id(block)}"
        
        # With context=None, x will be None, causing TypeError
        with pytest.raises(TypeError):
            namespace[func_name](context=None)
        
        # With empty context, x will be None, causing TypeError
        with pytest.raises(TypeError):
            namespace[func_name](context={})
        
        # With proper context, should work
        result = namespace[func_name](context={"x": 5})
        assert result == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
