"""Tests for inline block parsing (Python and React)."""

import pytest
from namel3ss.lang.parser.parse import N3Parser
from namel3ss.ast import InlinePythonBlock, InlineReactBlock
from namel3ss.lang.parser.errors import N3SyntaxError


class TestInlinePythonParsing:
    """Test parsing of python { ... } blocks."""
    
    def test_simple_python_expression(self):
        """Parse simple Python expression."""
        source = """
        app test {
            data: python { 42 }
        }
        """
        parser = N3Parser(source)
        result = parser.parse()
        
        # The inline block should be in app.data
        inline_block = result.data
        assert isinstance(inline_block, InlinePythonBlock)
        assert inline_block.kind == "python"
        assert "42" in inline_block.code
        assert inline_block.location is not None
    
    def test_python_function_definition(self):
        """Parse Python function definition."""
        source = """
        app test {
            processor: python {
                def process(items):
                    return [x * 2 for x in items]
            }
        }
        """
        parser = N3Parser(source)
        result = parser.parse()
        
        inline_block = result.processor
        assert isinstance(inline_block, InlinePythonBlock)
        assert "def process" in inline_block.code
        assert "return [x * 2 for x in items]" in inline_block.code
    
    def test_python_with_imports(self):
        """Parse Python block with imports."""
        source = """
        app test {
            calculator: python {
                import numpy as np
                result = np.mean([1, 2, 3])
            }
        }
        """
        parser = N3Parser(source)
        result = parser.parse()
        
        inline_block = result.calculator
        assert isinstance(inline_block, InlinePythonBlock)
        assert "import numpy as np" in inline_block.code
        assert "np.mean" in inline_block.code
    
    def test_python_nested_braces(self):
        """Parse Python block with nested braces (dicts, sets)."""
        source = """
        app test {
            config: python {
                settings = {
                    "nested": {"key": "value"},
                    "set": {1, 2, 3}
                }
            }
        }
        """
        parser = N3Parser(source)
        result = parser.parse()
        
        inline_block = result.config
        assert isinstance(inline_block, InlinePythonBlock)
        assert '{"key": "value"}' in inline_block.code or '"key": "value"' in inline_block.code
    
    def test_python_multiline_indentation(self):
        """Parse Python block preserving indentation."""
        source = """
        app test {
            logic: python {
                def complex_logic(data):
                    if data:
                        for item in data:
                            if item > 0:
                                yield item * 2
            }
        }
        """
        parser = N3Parser(source)
        result = parser.parse()
        
        inline_block = result.logic
        assert isinstance(inline_block, InlinePythonBlock)
        assert "def complex_logic" in inline_block.code
        # Check relative indentation is preserved
        assert "    if data:" in inline_block.code or "if data:" in inline_block.code
    
    def test_python_empty_block(self):
        """Parse empty Python block."""
        source = """
        app test {
            empty: python { }
        }
        """
        parser = N3Parser(source)
        result = parser.parse()
        
        inline_block = result.empty
        assert isinstance(inline_block, InlinePythonBlock)
        assert inline_block.code.strip() == ""
    
    def test_python_single_line(self):
        """Parse single-line Python block."""
        source = """
        app test {
            short: python { x = 42; print(x) }
        }
        """
        parser = N3Parser(source)
        result = parser.parse()
        
        inline_block = result.short
        assert isinstance(inline_block, InlinePythonBlock)
        assert "x = 42" in inline_block.code
        assert "print(x)" in inline_block.code


class TestInlineReactParsing:
    """Test parsing of react { ... } blocks."""
    
    def test_simple_react_jsx(self):
        """Parse simple JSX element."""
        source = """
        app test {
            ui: react {
                <div className="alert">Hello!</div>
            }
        }
        """
        parser = N3Parser(source)
        result = parser.parse()
        
        inline_block = result.ui
        assert isinstance(inline_block, InlineReactBlock)
        assert inline_block.kind == "react"
        assert "<div" in inline_block.code
        assert "Hello!" in inline_block.code
    
    def test_react_component_definition(self):
        """Parse React component function."""
        source = """
        app test {
            button: react {
                function CustomButton({ onClick, label }) {
                    return <button onClick={onClick}>{label}</button>;
                }
            }
        }
        """
        parser = N3Parser(source)
        result = parser.parse()
        
        inline_block = result.button
        assert isinstance(inline_block, InlineReactBlock)
        assert "function CustomButton" in inline_block.code
        assert "<button" in inline_block.code
    
    def test_react_with_props(self):
        """Parse React block with props and state."""
        source = """
        app test {
            widget: react {
                const [count, setCount] = useState(0);
                return (
                    <div>
                        <p>Count: {count}</p>
                        <button onClick={() => setCount(count + 1)}>Increment</button>
                    </div>
                );
            }
        }
        """
        parser = N3Parser(source)
        result = parser.parse()
        
        inline_block = result.widget
        assert isinstance(inline_block, InlineReactBlock)
        assert "useState" in inline_block.code
        assert "setCount" in inline_block.code
    
    def test_react_nested_braces(self):
        """Parse React block with nested braces (JSX expressions)."""
        source = """
        app test {
            card: react {
                <div className="card">
                    {items.map(item => (
                        <span key={item.id}>{item.name}</span>
                    ))}
                </div>
            }
        }
        """
        parser = N3Parser(source)
        result = parser.parse()
        
        inline_block = result.card
        assert isinstance(inline_block, InlineReactBlock)
        assert "items.map" in inline_block.code
        assert "=>" in inline_block.code or "item.name" in inline_block.code
    
    def test_react_arrow_function(self):
        """Parse React block with arrow function component."""
        source = """
        app test {
            badge: react {
                const Badge = ({ count }) => (
                    <span className="badge">{count}</span>
                );
            }
        }
        """
        parser = N3Parser(source)
        result = parser.parse()
        
        inline_block = result.badge
        assert isinstance(inline_block, InlineReactBlock)
        assert "Badge" in inline_block.code
        assert "=>" in inline_block.code or "badge" in inline_block.code


class TestInlineBlockErrors:
    """Test error handling for inline blocks."""
    
    def test_python_unclosed_brace(self):
        """Error on unclosed Python block."""
        source = """
        app test {
            bad: python {
                def foo():
                    return 42
        }
        """
        parser = N3Parser(source)
        
        with pytest.raises(N3SyntaxError, match="Unclosed inline block|expected '}'"):
            parser.parse()
    
    def test_react_unclosed_brace(self):
        """Error on unclosed React block."""
        source = """
        app test {
            bad: react {
                <div>Hello
        }
        """
        parser = N3Parser(source)
        
        with pytest.raises(N3SyntaxError, match="Unclosed inline block|expected '}'"):
            parser.parse()
    
    def test_python_missing_brace(self):
        """Error when python keyword not followed by brace."""
        source = """
        app test {
            bad: python "not a block"
        }
        """
        parser = N3Parser(source)
        
        with pytest.raises(N3SyntaxError, match="Expected"):
            parser.parse()


class TestInlineBlockLocations:
    """Test source location tracking for inline blocks."""
    
    def test_python_location(self):
        """Verify source location is captured."""
        source = """
        app test {
            data: python { 42 }
        }
        """
        parser = N3Parser(source, path="test.n3")
        result = parser.parse()
        
        inline_block = result.data
        assert inline_block.location is not None
        assert inline_block.location.file == "test.n3"
        assert inline_block.location.line > 0
    
    def test_react_location(self):
        """Verify source location is captured."""
        source = """
        app test {
            ui: react { <div>Test</div> }
        }
        """
        parser = N3Parser(source, path="test.n3")
        result = parser.parse()
        
        inline_block = result.ui
        assert inline_block.location is not None
        assert inline_block.location.file == "test.n3"
        assert inline_block.location.line > 0


class TestInlineBlockIntegration:
    """Test inline blocks in various N3 contexts."""
    
    def test_inline_in_page_data(self):
        """Use inline Python in page data preparation."""
        source = """
        app test {
            page home {
                data: python {
                    import pandas as pd
                    df = pd.read_csv("data.csv")
                    result = df.to_dict("records")
                }
                
                show table { data: result }
            }
        }
        """
        parser = N3Parser(source)
        result = parser.parse()
        
        page = result.pages[0]
        inline_block = page.data
        assert isinstance(inline_block, InlinePythonBlock)
        assert "pandas" in inline_block.code
    
    def test_inline_in_component(self):
        """Use inline React in custom component."""
        source = """
        app test {
            page dashboard {
                component: react {
                    function DashboardCard({ title, value }) {
                        return (
                            <div className="card">
                                <h3>{title}</h3>
                                <p>{value}</p>
                            </div>
                        );
                    }
                }
            }
        }
        """
        parser = N3Parser(source)
        result = parser.parse()
        
        page = result.pages[0]
        inline_block = page.component
        assert isinstance(inline_block, InlineReactBlock)
        assert "DashboardCard" in inline_block.code
    
    def test_multiple_inline_blocks(self):
        """Use multiple inline blocks in same file."""
        source = """
        app test {
            backend_logic: python {
                def calculate(x):
                    return x ** 2
            }
            
            frontend_ui: react {
                <div>Result: {result}</div>
            }
        }
        """
        parser = N3Parser(source)
        result = parser.parse()
        
        assert isinstance(result.backend_logic, InlinePythonBlock)
        assert isinstance(result.frontend_ui, InlineReactBlock)
        assert "calculate" in result.backend_logic.code
        assert "Result:" in result.frontend_ui.code
