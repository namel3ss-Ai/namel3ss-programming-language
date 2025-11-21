# Namel3ss Parser Internals

**Version:** 1.0.0  
**Date:** November 21, 2025

This document explains the internal architecture of the unified N3 parser for maintainers and contributors.

---

## Architecture Overview

The parser follows a **three-stage pipeline**:

```
Source Code → Lexer → Tokens → Parser → AST → Semantic Analyzer
```

### Components

1. **Lexer** (`namel3ss/lang/parser/grammar/lexer.py`): Tokenization with indentation tracking
2. **Parser** (`namel3ss/lang/parser/parse.py`): Recursive descent with Pratt expression parsing
3. **AST** (`namel3ss/lang/parser/ast/`): Typed syntax tree nodes
4. **Errors** (`namel3ss/lang/parser/errors.py`): Structured error reporting

---

## Lexer (Tokenization)

### Location
`namel3ss/lang/parser/grammar/lexer.py`

### Responsibility
Convert raw source text into a stream of tokens with position information.

### Key Features

#### 1. Indentation Tracking (Python-style)

The lexer emits `INDENT` and `DEDENT` tokens for block structure:

```python
app "Test" {
  field: "value"     # INDENT emitted before this line
}                    # DEDENT emitted before closing brace
```

**Implementation:**
```python
class Lexer:
    def __init__(self):
        self.indent_stack = [0]  # Track indentation levels
    
    def handle_indent(self, spaces):
        current_indent = self.indent_stack[-1]
        
        if spaces > current_indent:
            self.indent_stack.append(spaces)
            return Token(TokenType.INDENT, '', ...)
        elif spaces < current_indent:
            self.indent_stack.pop()
            return Token(TokenType.DEDENT, '', ...)
```

#### 2. Token Types

50+ token types defined in `TokenType` enum:

```python
class TokenType(Enum):
    # Keywords
    APP = "APP"
    LLM = "LLM"
    AGENT = "AGENT"
    PROMPT = "PROMPT"
    
    # Operators
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    
    # Literals
    STRING = "STRING"
    NUMBER = "NUMBER"
    TRUE = "TRUE"
    FALSE = "FALSE"
    
    # Punctuation
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    
    # Special
    NEWLINE = "NEWLINE"
    INDENT = "INDENT"
    DEDENT = "DEDENT"
    EOF = "EOF"
```

#### 3. String Literal Handling

Supports three string formats:

```python
# Single quote
text: 'Hello'

# Double quote
text: "Hello"

# Triple-quoted (multiline)
template: """
Line 1
Line 2
"""
```

**Implementation:**
```python
def tokenize_string(self, quote_char):
    """Tokenize string literal."""
    value = []
    
    # Check for triple-quoted
    if self.peek(1) == quote_char and self.peek(2) == quote_char:
        return self.tokenize_multiline_string(quote_char)
    
    # Single-line string
    while self.current() != quote_char:
        if self.current() == '\\':
            # Handle escapes: \n, \t, \", etc.
            value.append(self.handle_escape())
        else:
            value.append(self.current())
        self.advance()
```

#### 4. Numeric Literals

Handles integers and floats:

```python
# Integer
count: 42

# Float
temperature: 0.7

# Scientific notation
rate: 1e-5
```

#### 5. Special Characters

**Template variables:**
```python
template: "Hello {{ $name }}"  # $ allowed in templates
```

**Email addresses:**
```python
email: "user@example.com"  # @ allowed in strings
```

**Implementation:**
```python
def tokenize(self):
    while not self.is_at_end():
        char = self.current()
        
        # Allow $ and @ in certain contexts
        if char in ('$', '@'):
            self.advance()
            continue
```

---

## Parser (Syntax Analysis)

### Location
`namel3ss/lang/parser/parse.py`

### Responsibility
Convert token stream into Abstract Syntax Tree (AST).

### Architecture

The parser uses **mixins** for organization:

```python
class N3Parser(DeclarationParsingMixin, ExpressionParsingMixin):
    """Main parser combining declaration and expression parsing."""
```

#### Mixin Responsibilities

1. **DeclarationParsingMixin** (`declarations.py`): Top-level declarations
2. **ExpressionParsingMixin** (`expressions.py`): Expressions and operators

### Parsing Strategy

#### 1. Recursive Descent

**Definition:** Each grammar rule becomes a parsing method.

**Example:**
```python
def parse_app_declaration(self):
    """
    Grammar:
      app_decl ::= "app" STRING "connects" "to" db_ref "{" app_body "}"
    """
    self.expect(TokenType.APP)
    name = self.expect(TokenType.STRING).value
    
    # Optional database connection
    database = None
    if self.check(TokenType.CONNECTS):
        self.advance()
        self.expect(TokenType.TO)
        database = self.parse_database_ref()
    
    self.expect(TokenType.LBRACE)
    body = self.parse_app_body()
    self.expect(TokenType.RBRACE)
    
    return AppNode(name=name, database=database, **body)
```

#### 2. Pratt Expression Parsing (Precedence Climbing)

**Purpose:** Handle operator precedence correctly.

**Operator Precedence Table:**
```python
PRECEDENCE = {
    TokenType.OR: 1,           # Lowest
    TokenType.AND: 2,
    TokenType.EQ: 3,
    TokenType.LT: 3,
    TokenType.GT: 3,
    TokenType.PLUS: 4,
    TokenType.MINUS: 4,
    TokenType.STAR: 5,
    TokenType.SLASH: 5,
    TokenType.POWER: 6,        # Highest
}
```

**Implementation:**
```python
def parse_expression(self, min_precedence=0):
    """
    Precedence climbing algorithm.
    
    Example: "a + b * c"
    1. Parse "a" (primary)
    2. See "+", check precedence
    3. Parse "b * c" (higher precedence)
    4. Build BinaryOp(a, +, BinaryOp(b, *, c))
    """
    left = self.parse_primary()
    
    while self.is_binary_op() and self.precedence() >= min_precedence:
        op = self.current()
        self.advance()
        
        right_precedence = self.precedence(op) + 1
        right = self.parse_expression(right_precedence)
        
        left = BinaryOp(left=left, op=op.type, right=right)
    
    return left
```

### Token Management

#### Current Token Tracking
```python
class N3Parser:
    def __init__(self, source, path=None):
        self.tokens = Lexer(source, path).tokenize()
        self.position = 0
    
    def current(self):
        """Get current token without consuming."""
        return self.tokens[self.position]
    
    def advance(self):
        """Move to next token."""
        if not self.is_at_end():
            self.position += 1
        return self.previous()
    
    def peek(self, n=1):
        """Look ahead n tokens."""
        pos = self.position + n
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]  # EOF
```

#### Expectation Checking
```python
def expect(self, token_type):
    """
    Expect specific token type, raise error if mismatch.
    
    Example:
      self.expect(TokenType.LBRACE)
    """
    if self.check(token_type):
        return self.advance()
    
    # Generate helpful error
    raise N3SyntaxError(
        message=f"Expected {token_type.name}",
        line=self.current().line,
        column=self.current().column,
        expected=token_type.name,
        found=self.current().type.name,
        suggestion=self.get_suggestion()
    )

def check(self, token_type):
    """Check if current token matches type."""
    return self.current().type == token_type
```

---

## AST (Abstract Syntax Tree)

### Location
`namel3ss/lang/parser/ast/`

### Node Types

All AST nodes inherit from `ASTNode`:

```python
@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    line: int = 0
    column: int = 0

@dataclass
class Module(ASTNode):
    """Top-level module."""
    declarations: list[Declaration]
    imports: list[Import]

@dataclass
class AppNode(ASTNode):
    """Application declaration."""
    name: str
    database: Optional[DatabaseRef]
    description: str = ""
    version: str = "1.0.0"

@dataclass
class LLMDefinition(ASTNode):
    """LLM declaration."""
    name: str
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
```

### Why Typed AST?

1. **Type Safety:** Catch errors at compile time
2. **IDE Support:** Auto-completion and refactoring
3. **Code Generation:** Clean interface for backends
4. **Documentation:** Self-documenting structure

---

## Error Handling

### Location
`namel3ss/lang/parser/errors.py`

### Error Types

```python
class N3Error(Exception):
    """Base error class."""
    pass

class N3SyntaxError(N3Error):
    """Syntax error during parsing."""
    def __init__(self, message, line, column, expected=None, found=None):
        self.message = message
        self.line = line
        self.column = column
        self.expected = expected
        self.found = found

class N3SemanticError(N3Error):
    """Semantic error (e.g., undefined reference)."""
    pass

class N3TypeError(N3Error):
    """Type error (e.g., string + number)."""
    pass
```

### Error Reporting

#### Format
```
File: demo_app.n3
Line 6:3 | [SYNTAX_ERROR] Unexpected token
Expected: identifier
Found: filter
Suggestion: Use 'filter' as a string or identifier
```

#### Levenshtein Distance Suggestions

```python
def get_suggestion(self, found_name, valid_names):
    """
    Suggest closest valid name using edit distance.
    
    Example:
      found_name = "memroy"
      valid_names = ["memory", "agent", "prompt"]
      → Suggestion: "Did you mean 'memory'?"
    """
    distances = [
        (name, levenshtein_distance(found_name, name))
        for name in valid_names
    ]
    
    closest = min(distances, key=lambda x: x[1])
    
    if closest[1] <= 2:  # Within 2 edits
        return f"Did you mean '{closest[0]}'?"
    
    return None
```

---

## Symbol Table

### Purpose
Track declared names to detect duplicates and undefined references.

### Implementation

```python
class N3Parser:
    def __init__(self):
        self.symbols = {
            'llm': {},
            'agent': {},
            'prompt': {},
            'memory': {},
            'dataset': {},
            'page': {},
            'chain': {},
            'index': {},
            'rag_pipeline': {}
        }
    
    def declare_llm(self, name, node):
        """Register LLM declaration."""
        if name in self.symbols['llm']:
            raise N3SemanticError(
                f"Duplicate LLM declaration: '{name}'",
                line=node.line,
                previous=self.symbols['llm'][name].line
            )
        
        self.symbols['llm'][name] = node
    
    def resolve_llm(self, name):
        """Look up LLM by name."""
        if name not in self.symbols['llm']:
            suggestion = self.get_suggestion(name, self.symbols['llm'].keys())
            raise N3SemanticError(
                f"Undefined LLM: '{name}'",
                suggestion=suggestion
            )
        
        return self.symbols['llm'][name]
```

---

## Special Cases

### 1. Dataset Filter Keyword

**Problem:** `filter` is both a keyword (FILTER token) and a field name in dataset blocks.

**Solution:** Context-aware parsing in `parse_dataset_block()`:

```python
def parse_dataset_block(self):
    """
    Handle 'filter' as identifier in dataset context.
    
    Grammar:
      dataset "name" from source {
        filter: fn(x) => x.active
      }
    """
    fields = {}
    
    while not self.check(TokenType.RBRACE):
        # Special case: FILTER token as identifier
        if self.check(TokenType.FILTER):
            key = "filter"
            self.advance()
        else:
            key = self.expect(TokenType.IDENTIFIER).value
        
        self.expect(TokenType.COLON)
        value = self.parse_expression()
        
        fields[key] = value
```

### 2. Template Variables

**Problem:** `$name` and `{{ var }}` in templates need special handling.

**Solution:** Lexer allows `$` in string contexts:

```python
def tokenize_string(self):
    """Allow $ and @ in strings."""
    value = []
    
    while not self.is_string_end():
        char = self.current()
        
        if char == '$':
            # Template variable marker
            value.append(char)
        elif char == '{':
            # Check for Jinja syntax: {{ ... }}
            if self.peek() == '{':
                value.append(self.tokenize_template_expr())
        else:
            value.append(char)
```

### 3. Show Statements

**Problem:** `show text "title"` is ambiguous (text could be keyword or identifier).

**Solution:** Parser disambiguates based on context:

```python
def parse_show_statement(self):
    """
    Grammar:
      show_stmt ::= "show" (TEXT | TABLE | CHART) config_object
    """
    self.expect(TokenType.SHOW)
    
    # Check for specific keywords
    if self.check(TokenType.TEXT):
        widget_type = "text"
        self.advance()
    elif self.check(TokenType.TABLE):
        widget_type = "table"
        self.advance()
    else:
        # Generic identifier
        widget_type = self.expect(TokenType.IDENTIFIER).value
    
    config = self.parse_config_object()
    return ShowStatement(type=widget_type, config=config)
```

---

## Testing Strategy

### Unit Tests

Test each parsing method independently:

```python
def test_parse_llm():
    source = '''
    llm "gpt4" {
      provider: "openai"
      model: "gpt-4"
    }
    '''
    
    parser = N3Parser(source)
    module = parser.parse()
    
    assert len(module.declarations) == 1
    llm = module.declarations[0]
    assert llm.name == "gpt4"
    assert llm.provider == "openai"
```

### Integration Tests

Test complete N3 files:

```python
@pytest.mark.parametrize("example", [
    "demo_app.n3",
    "rag_demo.n3",
    "provider_demo.n3"
])
def test_official_examples(example):
    """Test all official examples parse successfully."""
    path = Path("examples") / example
    source = path.read_text()
    
    module = parse_module(source, str(path))
    
    assert module is not None
    assert len(module.declarations) > 0
```

### Error Testing

Verify error messages:

```python
def test_missing_brace_error():
    source = 'app "Test"'  # Missing opening brace
    
    with pytest.raises(N3SyntaxError) as exc:
        parse_module(source)
    
    assert "Expected: lbrace" in str(exc.value)
    assert exc.value.line == 1
```

---

## Performance Considerations

### 1. Single-Pass Parsing

Parser makes **one pass** over tokens (no backtracking):

```python
# Good: Single forward scan
def parse_expression():
    left = parse_primary()
    while is_binary_op():
        left = parse_binary(left)
    return left

# Bad: Multiple passes (avoid)
def parse_expression():
    if looks_like_addition():
        return parse_addition()
    elif looks_like_multiplication():
        return parse_multiplication()
```

### 2. Token Lookahead

Minimize lookahead distance:

```python
# Good: 1-token lookahead
if self.check(TokenType.LBRACE):
    return self.parse_block()

# Bad: Deep lookahead (slow)
if self.peek(10).type == TokenType.RBRACE:
    ...
```

### 3. Error Recovery

Parser **stops on first error** (fail-fast):

- Prevents cascading errors
- Gives user clear next step
- Faster than continuing with bad state

---

## Extension Points

### Adding New Declaration Types

1. **Add token type** in `lexer.py`:
```python
class TokenType(Enum):
    WIDGET = "WIDGET"  # New keyword
```

2. **Add AST node** in `ast/`:
```python
@dataclass
class WidgetDefinition(ASTNode):
    name: str
    properties: dict
```

3. **Add parsing method** in `declarations.py`:
```python
def parse_widget_declaration(self):
    self.expect(TokenType.WIDGET)
    name = self.expect(TokenType.STRING).value
    self.expect(TokenType.LBRACE)
    properties = self.parse_block_body()
    self.expect(TokenType.RBRACE)
    return WidgetDefinition(name=name, properties=properties)
```

4. **Update main parser** in `parse.py`:
```python
def parse_declaration(self):
    if self.check(TokenType.WIDGET):
        return self.parse_widget_declaration()
```

---

## Debugging Tips

### 1. Enable Token Dump

```python
parser = N3Parser(source)
for token in parser.tokens:
    print(f"{token.line}:{token.column} | {token.type.name} = {token.value}")
```

### 2. AST Visualization

```python
import json

module = parse_module(source)
print(json.dumps(module.__dict__, indent=2, default=str))
```

### 3. Parser State Inspection

```python
def parse_app(self):
    print(f"Current: {self.current()}")
    print(f"Next: {self.peek()}")
    print(f"Position: {self.position}/{len(self.tokens)}")
```

---

## Common Pitfalls

### 1. Forgetting to Advance

```python
# Wrong: Infinite loop
while self.check(TokenType.IDENTIFIER):
    name = self.current().value  # Never advances!

# Correct
while self.check(TokenType.IDENTIFIER):
    name = self.advance().value
```

### 2. Not Handling EOF

```python
# Wrong: IndexError at end of file
token = self.tokens[self.position + 1]

# Correct
def peek(self, n=1):
    pos = self.position + n
    if pos < len(self.tokens):
        return self.tokens[pos]
    return self.tokens[-1]  # EOF token
```

### 3. Incorrect Precedence

```python
# Wrong: Parses "a + b * c" as "(a + b) * c"
left = self.parse_primary()
while self.is_binary_op():
    op = self.advance()
    right = self.parse_primary()  # Should parse with higher precedence!
    left = BinaryOp(left, op, right)

# Correct: Use precedence climbing
left = self.parse_expression(min_prec)
```

---

## Architecture Decisions

### Why Recursive Descent?

**Pros:**
- Easy to understand and maintain
- One-to-one mapping with grammar
- Good error messages (know exact context)
- No parser generator dependencies

**Cons:**
- Left-recursive grammars need rewriting
- Manual precedence handling

**Conclusion:** Best fit for N3's straightforward grammar.

### Why Not PEG/Packrat?

- PEG doesn't handle operator precedence well
- Packrat's memoization adds complexity
- N3 grammar is simple enough for recursive descent

### Why Not LR/LALR?

- Requires parser generator (yacc/bison)
- Poor error messages
- Harder to customize

---

## Future Enhancements

### 1. Incremental Parsing

For IDE support, parse only changed regions:

```python
def parse_incremental(old_tree, edit):
    """Update AST for small edit without full reparse."""
    affected_range = compute_affected_range(edit)
    reparse_region(old_tree, affected_range)
```

### 2. Error Recovery

Continue parsing after errors to find multiple issues:

```python
def synchronize(self):
    """Skip tokens until next declaration."""
    while not self.is_at_end():
        if self.current().type in (TokenType.APP, TokenType.LLM):
            return
        self.advance()
```

### 3. LSP Integration

Language Server Protocol for IDE features:

```python
class N3LanguageServer:
    def on_hover(self, position):
        """Show type info on hover."""
        node = self.find_node_at(position)
        return node.type_info()
    
    def on_completion(self, position):
        """Auto-complete suggestions."""
        context = self.get_context(position)
        return context.valid_identifiers()
```

---

**End of Parser Internals Guide**
