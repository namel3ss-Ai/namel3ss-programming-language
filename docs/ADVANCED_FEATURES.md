# Advanced Language Features

**Version:** 2.0.0  
**Date:** January 2025

This document describes advanced Namel3ss features added in version 2.0:

- Static Type Checking
- Enhanced Expression Language (lambdas, subscripts, comprehensions)
- Multi-File Module System
- Editor/IDE Integration API

---

## Table of Contents

1. [Static Type Checking](#static-type-checking)
2. [Enhanced Expressions](#enhanced-expressions)
3. [Module System](#module-system)
4. [Editor API](#editor-api)
5. [Migration Guide](#migration-guide)

---

## Static Type Checking

Namel3ss 2.0 includes a comprehensive static type checker that validates your code before execution.

### Type System

The type system includes:

**Primitive Types:**
- `text`: String values
- `number`: Numeric values (int or float)
- `boolean`: True/false values
- `null`: Null value
- `any`: Dynamic type (opt-out of checking)

**Composite Types:**
- `array<T>`: Arrays with element type
- `{field: type, ...}`: Object types with structural subtyping
- `(T1, T2, ...) => R`: Function types
- `T1 | T2 | ...`: Union types
- `one_of("val1", "val2")`: Enum types

### Type Annotations

Add type annotations to improve type safety:

```n3
# Variable declarations
let name: text = "Alice"
let age: number = 30
let tags: array<text> = ["admin", "verified"]

# Function signatures
fn greet(name: text): text =>
  "Hello, " + name

fn process_items(items: array<number>): number => {
  let filtered = filter(items, fn(x: number): boolean => x > 0)
  return sum(filtered)
}

# Complex types
fn get_user(id: number): {name: text, email: text, age: number} => {
  name: "Alice",
  email: "alice@example.com",
  age: 30
}
```

### Type Inference

The type checker infers types when annotations are omitted:

```n3
# Inferred as: (number, number) => number
fn add(a, b) => a + b

# Inferred as: array<number>
let numbers = [1, 2, 3, 4, 5]

# Inferred as: text
let greeting = "Hello, " + "World"
```

### Type Errors

The static checker detects type mismatches:

```n3
# ❌ Error: Cannot assign text to number
let count: number = "hello"

# ❌ Error: Function expects (number, number), got (text, number)
add("5", 10)

# ❌ Error: Cannot call non-function value
let x = 42
x()

# ❌ Error: Array element type mismatch
let numbers: array<number> = [1, 2, "three"]
```

### Built-in Functions

The type checker knows about standard library functions:

- `str(value: any): text` - Convert to string
- `int(value: text | number): number` - Parse integer
- `float(value: text | number): number` - Parse float
- `bool(value: any): boolean` - Convert to boolean
- `len(arr: array<any>): number` - Array length
- `sum(arr: array<number>): number` - Sum numeric array
- `map(arr: array<T>, fn: (T) => R): array<R>` - Map function
- `filter(arr: array<T>, fn: (T) => boolean): array<T>` - Filter function
- `reduce(arr: array<T>, fn: (T, T) => T, init: T): T` - Reduce function

### Running Type Checking

Type checking runs automatically during compilation:

```python
from namel3ss.types import check_module_static

# Parse module
module = parse_module(source_code)

# Run static type checking
errors = check_module_static(module, path="app.ai")

if errors:
    for error in errors:
        print(f"{error.path}:{error.line}:{error.column} - {error.message}")
```

---

## Enhanced Expressions

Namel3ss 2.0 extends the expression language with functional programming features.

### Lambda Expressions

Define inline anonymous functions:

```n3
# Basic lambda
let double = fn(x) => x * 2

# With type annotations
let add: (number, number) => number = fn(a, b) => a + b

# Multiple parameters
let greet = fn(first, last) => "Hello, " + first + " " + last

# Used with higher-order functions
let numbers = [1, 2, 3, 4, 5]
let doubled = map(numbers, fn(x) => x * 2)
let positive = filter(numbers, fn(x) => x > 0)
```

**Syntax:**
```
fn(param1, param2, ...) => expression
fn(param1: type1, param2: type2): return_type => expression
```

### Subscript Operations

Access array elements and object properties:

```n3
# Array indexing
let items = [10, 20, 30, 40, 50]
let first = items[0]        # 10
let third = items[2]        # 30

# Object property access
let user = {name: "Alice", age: 30, email: "alice@example.com"}
let name = user["name"]     # "Alice"
let age = user["age"]       # 30

# Nested access
let matrix = [[1, 2], [3, 4], [5, 6]]
let value = matrix[1][0]    # 3

# Dynamic keys
let key = "email"
let email = user[key]       # "alice@example.com"
```

### Array Slicing

Extract sub-arrays:

```n3
let numbers = [0, 10, 20, 30, 40, 50]

let slice1 = numbers[1:4]   # [10, 20, 30]
let slice2 = numbers[:3]    # [0, 10, 20]
let slice3 = numbers[3:]    # [30, 40, 50]
let slice4 = numbers[:]     # Full copy

# With step (future)
let evens = numbers[::2]    # [0, 20, 40]
```

### List Comprehensions

Functional list transformations:

```n3
# Basic comprehension converts to map
let doubled = [x * 2 for x in items]
# Equivalent to: map(items, fn(x) => x * 2)

# With filter
let positive_doubled = [x * 2 for x in items if x > 0]
# Equivalent to: map(filter(items, fn(x) => x > 0), fn(x) => x * 2)

# Complex transformations
let user_names = [user["name"] for user in users if user["active"]]

# Nested (future feature)
let flattened = [item for sublist in lists for item in sublist]
```

**Current Limitations:**
- Single generator only (no nested `for`)
- Dict/set comprehensions not yet supported
- Converts to `map`/`filter` calls (functional style)

### Operator Precedence

From highest to lowest:

1. **Subscript/Call**: `arr[0]`, `func(x)`
2. **Unary**: `-x`, `not x`
3. **Multiplicative**: `*`, `/`, `%`
4. **Additive**: `+`, `-`
5. **Comparison**: `==`, `!=`, `<`, `>`, `<=`, `>=`
6. **Logical AND**: `and`
7. **Logical OR**: `or`
8. **Lambda**: `fn(x) => x`

---

## Module System

Namel3ss 2.0 supports multi-file projects with imports and modules.

### Module Declaration

Declare a module at the top of your file:

```n3
module "app.main"

# Rest of the file...
```

Module names use dot notation matching file structure:
- `"app.main"` → `app/main.ai`
- `"app.shared.types"` → `app/shared/types.ai`
- `"lib.utils"` → `lib/utils.ai`

### Import Statements

Import other modules:

```n3
module "app.main"

# Import specific module
import "app.shared.types"
import "app.models.user"

# Use imported symbols
app "My Application" {
  # Use types from app.shared.types
  # Use models from app.models.user
}
```

### Multi-File Project Structure

```
my_project/
├── app/
│   ├── main.ai           # module "app.main"
│   ├── config.ai         # module "app.config"
│   ├── models/
│   │   ├── user.ai       # module "app.models.user"
│   │   └── product.ai    # module "app.models.product"
│   └── shared/
│       └── types.ai      # module "app.shared.types"
├── lib/
│   └── utils.ai          # module "lib.utils"
└── README.md
```

### Example: Shared Types Module

`app/shared/types.ai`:
```n3
module "app.shared.types"

# Define reusable schemas
schema UserSchema {
  id: number
  name: text
  email: text
  role: one_of("admin", "user", "guest")
  created_at: text
}

schema ProductSchema {
  id: number
  name: text
  price: number
  in_stock: boolean
}

# Define reusable functions
fn format_price(price: number): text =>
  "$" + str(int(price * 100) / 100)

fn validate_email(email: text): boolean =>
  # Simple validation
  len(email) > 0 and email contains "@"
```

### Example: User Model Module

`app/models/user.ai`:
```n3
module "app.models.user"

import "app.shared.types"

dataset "active_users" from postgres table users {
  filter: fn(user) => user.status == "active"
  schema: UserSchema  # From app.shared.types
}

fn get_user_display_name(user: UserSchema): text => {
  if user.role == "admin" {
    return "[Admin] " + user.name
  }
  return user.name
}
```

### Example: Main Application Module

`app/main.ai`:
```n3
module "app.main"

import "app.shared.types"
import "app.models.user"
import "app.config"

app "Customer Portal" connects to postgres "customer_db" {
  description: "Multi-module application example"
  version: "2.0.0"
}

page "Users" at "/users" {
  show table {
    title: "Active Users"
    source: dataset("active_users")  # From app.models.user
    columns: [
      {field: "name", label: "Name"},
      {field: "email", label: "Email"},
      {field: "role", label: "Role"}
    ]
  }
}
```

### Circular Dependencies

The module system detects circular dependencies:

```n3
# ❌ Error: Circular dependency detected
# app.a imports app.b
# app.b imports app.c
# app.c imports app.a
```

### Loading Multi-Module Projects

Use the module system API:

```python
from namel3ss.modules.system import load_multi_module_project

# Load entire project starting from entry point
modules, errors = load_multi_module_project(
    entry_module="app.main",
    project_root="./my_project"
)

if errors:
    for error in errors:
        print(f"Error: {error.message}")
else:
    for module in modules:
        print(f"Loaded: {module.name} ({module.path})")
```

---

## Editor API

Namel3ss 2.0 provides an API for building IDE integrations and LSP servers.

### Parsing and Analysis

```python
from namel3ss.tools.editor_api import parse_source, analyze_module

# Parse source code
source = """
app "Example" {
  # ... code ...
}
"""

result = parse_source(source, uri="file:///path/to/app.ai")

if result.parse_success:
    print(f"Parsed successfully! Found {len(result.symbols)} symbols")
    for symbol in result.symbols:
        print(f"  {symbol.kind}: {symbol.name}")
else:
    for diag in result.diagnostics:
        print(f"[{diag.severity}] {diag.message}")

# Full analysis with type checking
result = analyze_module(source, uri="file:///path/to/app.ai", run_type_check=True)

for diag in result.diagnostics:
    print(f"Line {diag.range.start.line}: {diag.message}")
```

### Symbol Information

```python
from namel3ss.tools.editor_api import find_symbol_at_position

# Find symbol at cursor position
symbol = find_symbol_at_position(
    source=source_code,
    line=10,
    character=15,
    uri="file:///path/to/app.ai"
)

if symbol:
    print(f"Symbol: {symbol.name}")
    print(f"Kind: {symbol.kind}")
    print(f"Type: {symbol.type_info}")
```

### Hover Information

```python
from namel3ss.tools.editor_api import get_hover_info

# Get hover information for symbol
hover_text = get_hover_info(
    source=source_code,
    line=10,
    character=15,
    uri="file:///path/to/app.ai"
)

if hover_text:
    print(hover_text)  # Markdown-formatted
```

### Complete API Example

```python
from namel3ss.tools.editor_api import EditorAPI

api = EditorAPI(project_root="./my_project")

# Analyze file
result = api.analyze_module(source_code, uri="file:///app.ai")

# Get symbol at position
symbol = api.get_symbol_at_position(uri="file:///app.ai", position=Position(10, 15))

# Find all references to symbol
references = api.find_references(uri="file:///app.ai", position=Position(10, 15))

# Find definition
definition = api.find_definition(uri="file:///app.ai", position=Position(10, 15))

# Get completion context
context = api.get_completion_context(uri="file:///app.ai", position=Position(10, 15))
visible_symbols = context["visible_symbols"]
```

### LSP Server Integration

The Editor API is designed for Language Server Protocol integration:

```python
# In your LSP server:

from namel3ss.tools.editor_api import EditorAPI, Position

class Namel3ssLanguageServer:
    def __init__(self):
        self.api = EditorAPI()
    
    def on_did_open(self, uri: str, text: str):
        result = self.api.analyze_module(text, uri)
        self.publish_diagnostics(uri, result.diagnostics)
    
    def on_hover(self, uri: str, line: int, character: int):
        pos = Position(line, character)
        return self.api.get_hover_information(uri, pos)
    
    def on_definition(self, uri: str, line: int, character: int):
        pos = Position(line, character)
        return self.api.find_definition(uri, pos)
    
    def on_references(self, uri: str, line: int, character: int):
        pos = Position(line, character)
        return self.api.find_references(uri, pos)
```

---

## Migration Guide

### From 1.x to 2.0

#### Enable Static Type Checking

Add type annotations gradually:

```n3
# Before (1.x)
fn calculate_total(items, tax_rate) =>
  sum(items) * (1 + tax_rate)

# After (2.0) - recommended
fn calculate_total(items: array<number>, tax_rate: number): number =>
  sum(items) * (1 + tax_rate)
```

#### Use Enhanced Expressions

Replace verbose patterns with concise expressions:

```n3
# Before (1.x)
let doubled = []
for item in items {
  doubled.push(item * 2)
}

# After (2.0)
let doubled = [x * 2 for x in items]
# or
let doubled = map(items, fn(x) => x * 2)
```

#### Organize into Modules

Split large applications:

```n3
# Before (1.x) - single file
# Everything in one large app.ai file

# After (2.0) - modular
# app/main.ai
module "app.main"
import "app.models.user"
import "app.shared.types"

# app/models/user.ai
module "app.models.user"
import "app.shared.types"
# ... user-related code ...

# app/shared/types.ai
module "app.shared.types"
# ... shared schemas and types ...
```

#### Backwards Compatibility

All 1.x code continues to work in 2.0:
- Type annotations are optional
- Enhanced expressions are opt-in
- Module system is optional (single-file projects work)
- Existing syntax unchanged

---

## Summary

Namel3ss 2.0 elevates the language from working prototype to world-class DSL:

✅ **Static Type Checking** - Catch errors before runtime  
✅ **Enhanced Expressions** - Lambdas, subscripts, comprehensions  
✅ **Module System** - Multi-file projects with imports  
✅ **Editor API** - Foundation for LSP and IDE integration  

These features maintain Namel3ss's AI-native focus while providing the tooling and expressiveness expected from production languages.
