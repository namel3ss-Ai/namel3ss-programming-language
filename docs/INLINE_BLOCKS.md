# Inline Template Blocks: Production-Grade Escape Hatches

## Overview

**Status**: ✅ Parser Complete | 🚧 Codegen In Progress | ⏳ Runtime Pending

Inline template blocks allow embedding code from other languages directly in N3 syntax, providing production-grade escape hatches for custom logic and UI components.

### Supported Inline Targets

- **`python { ... }`** - Inline Python code for server-side logic
- **`react { ... }`** - Inline React/JSX components for custom UI

---

## Quick Start

### Inline Python

```n3
app DataProcessor {
    # Inline Python for custom data processing
    transformer: python {
        import pandas as pd
        
        def process_data(raw_data):
            df = pd.DataFrame(raw_data)
            df['score'] = df['value'] * 2
            return df.to_dict('records')
    }
}
```

### Inline React

```n3
app Dashboard {
    page home {
        # Inline React for custom UI component
        custom_widget: react {
            function MetricCard({ title, value, trend }) {
                const trendColor = trend > 0 ? 'green' : 'red';
                return (
                    <div className="metric-card">
                        <h3>{title}</h3>
                        <p className="value">{value}</p>
                        <span style={{ color: trendColor }}>
                            {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}%
                        </span>
                    </div>
                );
            }
        }
    }
}
```

---

## Syntax

### Python Blocks

**Syntax**: `python { <python code> }`

**Features**:
- Full Python 3.x syntax support
- Can define functions, classes, or expressions
- Access to N3 runtime context via bindings
- Executes server-side in controlled environment

**Examples**:

```n3
# Simple expression
result: python { 42 * 2 }

# Function definition
processor: python {
    def calculate_metrics(data):
        total = sum(data)
        avg = total / len(data) if data else 0
        return {'total': total, 'average': avg}
}

# With imports
analyzer: python {
    import numpy as np
    import scipy.stats as stats
    
    def analyze(values):
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'p_value': stats.ttest_1samp(values, 0).pvalue
        }
}

# Nested braces (dicts, sets)
config: python {
    settings = {
        'model': {'type': 'gpt-4', 'temp': 0.7},
        'features': {'a', 'b', 'c'}
    }
}
```

### React Blocks

**Syntax**: `react { <jsx code> }`

**Features**:
- Full JSX syntax support
- Can be component definitions or JSX fragments
- Compiled to React components at build time
- Rendered client-side in browser
- Props passed from N3 page context

**Examples**:

```n3
# Simple JSX element
alert: react {
    <div className="alert alert-info">
        <strong>Notice:</strong> System maintenance scheduled
    </div>
}

# Component function
button: react {
    function CustomButton({ onClick, label, variant = 'primary' }) {
        return (
            <button 
                className={`btn btn-${variant}`}
                onClick={onClick}
            >
                {label}
            </button>
        );
    }
}

# With hooks
counter: react {
    const [count, setCount] = useState(0);
    
    return (
        <div className="counter">
            <p>Count: {count}</p>
            <button onClick={() => setCount(count + 1)}>
                Increment
            </button>
        </div>
    );
}

# Nested JSX expressions
list: react {
    <ul className="user-list">
        {users.map(user => (
            <li key={user.id} className={user.active ? 'active' : ''}>
                <span className="name">{user.name}</span>
                <span className="email">{user.email}</span>
            </li>
        ))}
    </ul>
}
```

---

## Technical Architecture

### AST Representation

Inline blocks are represented as typed AST nodes (not dicts or raw strings):

```python
@dataclass
class InlineBlock(Expression):
    """Base class for inline code blocks."""
    kind: str  # "python", "react", etc.
    code: str  # Raw source code
    location: Optional[SourceLocation]
    metadata: Dict[str, Any]

@dataclass
class InlinePythonBlock(InlineBlock):
    """Inline Python code block."""
    kind: Literal["python"] = "python"
    bindings: Dict[str, Any] = field(default_factory=dict)
    python_version: Optional[str] = None
    is_expression: bool = False

@dataclass
class InlineReactBlock(InlineBlock):
    """Inline React/JSX component block."""
    kind: Literal["react"] = "react"
    component_name: Optional[str] = None
    props: Dict[str, Any] = field(default_factory=dict)
    requires_imports: list[str] = field(default_factory=list)
```

### Parser Flow

1. **Lexer**: Recognizes `python` and `react` keywords → `TokenType.PYTHON`, `TokenType.REACT`
2. **Parser**: 
   - `parse_value()` detects inline keyword + `{`
   - `parse_inline_python_block()` or `parse_inline_react_block()` called
   - `_extract_inline_code_block()` extracts code:
     * Handles nested braces (dicts, JSX expressions)
     * Preserves indentation and formatting
     * Strips common leading whitespace
3. **AST**: Creates `InlinePythonBlock` or `InlineReactBlock` node with:
   - `code`: Raw source string
   - `location`: File, line, column for debugging
   - `kind`: Language identifier
   - Metadata fields (bindings, props, etc.)

### Codegen Flow (TODO)

**Python Codegen**:
- Extract `InlinePythonBlock` nodes from AST
- Generate Python module with function definitions
- Add imports, context bindings
- Emit to `generated/inline_python.py`
- Import and call from runtime

**React Codegen**:
- Extract `InlineReactBlock` nodes from AST
- Generate React component definitions
- Add required imports (React, hooks, etc.)
- Handle props mapping
- Emit to `frontend/src/components/Inline*.tsx`
- Import and render in page components

### Runtime Execution (TODO)

**Python**:
- Import generated inline module
- Execute functions with context bindings
- Return results to N3 runtime
- Sandboxed execution (trusted authoring mode)

**React**:
- Component definitions available in frontend bundle
- Rendered via React DOM
- Props passed from page context
- Standard React lifecycle

---

## Nested Braces Handling

Inline blocks correctly handle nested braces:

### Python Examples

```n3
# Dict literals
config: python {
    data = {"outer": {"inner": {"deep": 42}}}
}

# Set literals
tags: python {
    unique = {1, 2, 3, {4, 5}}  # Wait, sets can't contain sets
    unique = {1, 2, 3} | {4, 5}  # This works
}

# Dict comprehensions
mapping: python {
    result = {k: {v: v**2 for v in range(3)} for k in ['a', 'b']}
}
```

### React Examples

```n3
# JSX expression braces
component: react {
    <div>
        {items.map(item => (
            <Card key={item.id} data={{
                title: item.name,
                meta: {author: item.user, date: item.created}
            }} />
        ))}
    </div>
}

# Inline styles (objects in JSX)
styled: react {
    <div style={{
        color: 'blue',
        padding: {top: 10, bottom: 10}  # Invalid JS, but parser handles braces
    }}>
        Content
    </div>
}
```

---

## Indentation Preservation

The parser preserves Python indentation while stripping common leading whitespace:

```n3
# Input
logic: python {
    def complex_logic(data):
        if data:
            for item in data:
                if item > 0:
                    yield item * 2
}

# Extracted code (common indent removed)
def complex_logic(data):
    if data:
        for item in data:
            if item > 0:
                yield item * 2
```

---

## Error Handling

### Unclosed Braces

```n3
# ❌ ERROR: Unclosed inline block
bad: python {
    def foo():
        return 42
# Missing closing brace

# Error: Line X | [SYNTAX_ERROR] Unclosed inline block: expected '}'
```

### Missing Brace After Keyword

```n3
# ❌ ERROR: Expected opening brace
bad: python "not a block"

# Error: Line X | [SYNTAX_ERROR] Expected '{'
```

---

## Location Tracking

All inline blocks track source location for debugging:

```python
inline_block.location
# SourceLocation(
#     file='app.ai',
#     line=42,
#     column=5,
#     end_line=48,
#     end_column=1
# )
```

Used for:
- Error messages with precise line numbers
- IDE navigation (go to definition)
- Stack traces in runtime errors
- Debugging tools

---

## Security Considerations

### Trusted Authoring Mode

Inline blocks assume **trusted authoring** - code is written by developers, not end users.

**Python Security**:
- ⚠️ Full Python execution (no sandbox by default)
- ⚠️ Access to file system, network, imports
- ✅ Intended for developer-written logic
- ✅ Use external sandboxing if needed (Docker, pypy sandbox, etc.)

**React Security**:
- ✅ Standard React security model
- ✅ XSS protection via React DOM escaping
- ✅ Client-side only, no server execution
- ⚠️ Props must be sanitized if from user input

### Production Deployment

**Recommendations**:
1. **Code Review**: Inline blocks should be code-reviewed like any Python/React code
2. **Static Analysis**: Run linters (black, pylint, eslint, prettier) on extracted code
3. **Testing**: Unit test inline functions separately
4. **Monitoring**: Log inline block execution for debugging
5. **Versioning**: Track inline block changes in git

---

## Extensibility

The inline block system is designed for future expansion:

### Adding New Inline Targets

```python
# Future: SQL inline blocks
@dataclass
class InlineSQLBlock(InlineBlock):
    kind: Literal["sql"] = "sql"
    dialect: str = "postgres"
    parameters: Dict[str, Any] = field(default_factory=dict)

# Usage
query: sql {
    SELECT users.name, COUNT(orders.id) as order_count
    FROM users
    LEFT JOIN orders ON orders.user_id = users.id
    WHERE users.created_at > :start_date
    GROUP BY users.name
    ORDER BY order_count DESC;
}
```

### Supported Future Targets

- **`sql { ... }`** - Inline SQL queries
- **`graphql { ... }`** - Inline GraphQL schemas/queries
- **`typescript { ... }`** - Inline TypeScript for frontend logic
- **`css { ... }`** - Inline CSS/Sass for custom styling
- **`markdown { ... }`** - Inline Markdown for rich text content

To add a new inline target:
1. Add token type to `TokenType` enum in lexer
2. Add keyword to `KEYWORDS` dict in lexer
3. Add AST node class inheriting from `InlineBlock`
4. Add `parse_inline_X_block()` method to parser
5. Add case in `parse_value()` to detect keyword
6. Implement codegen for target language
7. Add tests

---

## Testing

### Parser Tests

```bash
# Run parser tests
pytest tests/test_inline_blocks_unit.py -v

# Tests cover:
# - Tokenization (keywords recognized)
# - Simple blocks (single expressions)
# - Complex blocks (functions, multiline)
# - Nested braces (dicts, JSX)
# - Error handling (unclosed braces)
# - Location tracking
```

### Codegen Tests (TODO)

```bash
# Test Python codegen
pytest tests/test_inline_python_codegen.py -v

# Test React codegen
pytest tests/test_inline_react_codegen.py -v
```

### End-to-End Tests (TODO)

```bash
# Test full pipeline: parse → compile → execute
pytest tests/test_inline_blocks_e2e.py -v
```

---

## Examples

### Data Processing Pipeline

```n3
app ETLPipeline {
    # Inline Python for custom transformation
    transformer: python {
        import pandas as pd
        import numpy as np
        
        def transform_sales_data(raw_data):
            df = pd.DataFrame(raw_data)
            
            # Calculate derived metrics
            df['revenue'] = df['quantity'] * df['price']
            df['profit_margin'] = (df['revenue'] - df['cost']) / df['revenue']
            
            # Aggregate by category
            summary = df.groupby('category').agg({
                'revenue': 'sum',
                'profit_margin': 'mean',
                'quantity': 'sum'
            }).reset_index()
            
            return summary.to_dict('records')
    }
    
    dataset sales {
        source: postgres table sales_raw
        transform: transformer
    }
    
    page dashboard {
        show table { data: sales }
    }
}
```

### Custom Dashboard UI

```n3
app MetricsDashboard {
    page home {
        # Inline React for custom metric card
        metric_card: react {
            function MetricCard({ metric }) {
                const [expanded, setExpanded] = useState(false);
                
                const trendIcon = metric.trend > 0 
                    ? '📈' 
                    : metric.trend < 0 
                    ? '📉' 
                    : '➡️';
                
                return (
                    <div className="metric-card">
                        <div className="header">
                            <h3>{metric.title}</h3>
                            <span className="trend">{trendIcon}</span>
                        </div>
                        
                        <div className="value">
                            {metric.value.toLocaleString()}
                        </div>
                        
                        {expanded && (
                            <div className="details">
                                <p>Previous: {metric.previous}</p>
                                <p>Change: {metric.change}%</p>
                                <p>Target: {metric.target}</p>
                            </div>
                        )}
                        
                        <button onClick={() => setExpanded(!expanded)}>
                            {expanded ? 'Show Less' : 'Show More'}
                        </button>
                    </div>
                );
            }
        }
        
        # Use custom component
        show metric_card { metric: ctx.metrics.revenue }
    }
}
```

---

## Status

### ✅ Completed (Parser Layer)

- [x] AST nodes: `InlineBlock`, `InlinePythonBlock`, `InlineReactBlock`
- [x] Lexer tokens: `PYTHON`, `REACT`
- [x] Parser methods: `parse_inline_python_block()`, `parse_inline_react_block()`
- [x] Nested braces handling
- [x] Indentation preservation
- [x] Location tracking
- [x] Error handling
- [x] Comprehensive tests (14 tests passing)

### 🚧 In Progress

- [ ] Resolver integration (preserve inline nodes)
- [ ] Python codegen (emit functions/modules)
- [ ] React codegen (emit components)

### ⏳ Pending

- [ ] Runtime execution (Python)
- [ ] Runtime rendering (React)
- [ ] Context bindings (pass N3 vars to inline code)
- [ ] End-to-end tests
- [ ] Security guidelines
- [ ] Performance benchmarks
- [ ] IDE integration (syntax highlighting, autocomplete)

---

## API Reference

### AST Nodes

#### `InlineBlock`

Base class for all inline code blocks.

**Fields**:
- `kind: str` - Language identifier ("python", "react", etc.)
- `code: str` - Raw source code as written by user
- `location: Optional[SourceLocation]` - Source file, line, column
- `metadata: Dict[str, Any]` - Extensible metadata

**Methods**:
- `__post_init__()` - Validates kind and code are present

#### `InlinePythonBlock`

Inline Python code block.

**Fields** (inherits from `InlineBlock`):
- `kind: Literal["python"]` - Always "python"
- `bindings: Dict[str, Any]` - Context variables to inject
- `python_version: Optional[str]` - Python version (e.g., "3.11")
- `is_expression: bool` - True if single expression, False if statements

#### `InlineReactBlock`

Inline React/JSX component block.

**Fields** (inherits from `InlineBlock`):
- `kind: Literal["react"]` - Always "react"
- `component_name: Optional[str]` - Component name if definition
- `props: Dict[str, Any]` - Props to pass to component
- `requires_imports: list[str]` - Required imports (e.g., ["React", "useState"])

### Parser Methods

#### `parse_inline_python_block() -> InlinePythonBlock`

Parse `python { ... }` block.

**Returns**: `InlinePythonBlock` AST node

**Raises**:
- `N3SyntaxError` if unclosed block or invalid syntax

#### `parse_inline_react_block() -> InlineReactBlock`

Parse `react { ... }` block.

**Returns**: `InlineReactBlock` AST node

**Raises**:
- `N3SyntaxError` if unclosed block or invalid syntax

#### `_extract_inline_code_block() -> str`

Extract raw code from inline block, handling nested braces.

**Returns**: Code string with common indentation removed

**Raises**:
- `N3SyntaxError` if closing brace never found

---

## Troubleshooting

### Issue: "Unclosed inline block: expected '}'"

**Cause**: Missing closing brace or unbalanced braces

**Solution**: Check that every `{` has matching `}`. Count braces in strings too.

```n3
# ❌ Bad
data: python {
    x = {"key": "value"
}

# ✅ Good
data: python {
    x = {"key": "value"}
}
```

### Issue: "Inconsistent indentation"

**Cause**: Mixing tabs and spaces, or non-4-space indents

**Solution**: Use consistent 4-space indentation

```n3
# ❌ Bad (8 spaces)
logic: python {
        def foo():
                pass
}

# ✅ Good (4 spaces)
logic: python {
    def foo():
        pass
}
```

### Issue: "Expected '{' after inline keyword"

**Cause**: Missing opening brace after `python` or `react`

**Solution**: Add opening brace immediately after keyword

```n3
# ❌ Bad
data: python "code"

# ✅ Good
data: python { "code" }
```

---

## Contributing

To extend inline block support:

1. **Add AST node** in `namel3ss/ast/inline_blocks.py`
2. **Add token type** in `namel3ss/lang/parser/grammar/lexer.py`
3. **Add parser method** in `namel3ss/lang/parser/expressions.py`
4. **Add tests** in `tests/test_inline_blocks_unit.py`
5. **Implement codegen** in `namel3ss/codegen/`
6. **Add runtime support** in generated code
7. **Update documentation**

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

---

## Related Documentation

- [Parser Architecture](PARSER_ARCHITECTURE.md)
- [AST Design](AST_DESIGN.md)
- [Frontend Escape Hatches](FRONTEND_ESCAPE_HATCHES.md)
- [Codegen Guide](CODEGEN_GUIDE.md)
- [Security Model](SECURITY.md)

---

**Version**: 1.0.0-alpha  
**Last Updated**: 2024  
**Status**: Parser Complete, Codegen In Progress
