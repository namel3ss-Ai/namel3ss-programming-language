# Static Type Checker Guide

The Namel3ss static type checker validates your code before execution, catching type errors early and providing confidence in code correctness.

## Quick Start

### Basic Usage

```python
from namel3ss.types.static_checker import check_module_static
from namel3ss.lang.parser import parse_module

# Parse your code
source = '''
app "MyApp" {
    let x: number = 42
    let y: text = "hello"
}
'''

module = parse_module(source)

# Run type checking
errors = check_module_static(module, path="myapp.ai")

if errors:
    for error in errors:
        print(f"{error.path}:{error.line}:{error.column}")
        print(f"  [{error.code}] {error.message}")
else:
    print("✓ No type errors found")
```

## Type System

### Primitive Types

```namel3ss
let name: text = "Alice"         // String values
let age: number = 30             // Numeric values (int/float)
let active: boolean = true       // Boolean values
let nothing: null = null         // Null value
```

### Array Types

```namel3ss
let numbers: array<number> = [1, 2, 3, 4, 5]
let names: array<text> = ["Alice", "Bob", "Charlie"]
let nested: array<array<number>> = [[1, 2], [3, 4]]
```

### Object Types

```namel3ss
let user: {name: text, age: number} = {
    name: "Alice",
    age: 30
}

// Structural typing - extra fields OK
let detailed_user: {name: text, age: number, email: text} = {
    name: "Bob",
    age: 25,
    email: "bob@example.com"
}

// Can assign to less specific type
let basic: {name: text} = detailed_user  // ✓ OK
```

### Union Types

```namel3ss
let id: text | number = "user123"  // Can be text OR number
id = 456                            // ✓ Also valid

let optional: text | null = "value"
optional = null                     // ✓ Also valid
```

### Function Types

```namel3ss
// Function signature: (param types) => return type
let add: (number, number) => number = fn(x, y) => x + y

// No parameters
let greet: () => text = fn() => "Hello"

// Higher-order functions
let apply: ((number) => number, number) => number = 
    fn(f, x) => f(x)
```

### Enum Types

```namel3ss
let role: one_of("admin", "user", "guest") = "admin"

// Type error:
role = "superuser"  // ✗ Not in enum
```

### Any Type

```namel3ss
let dynamic: any = 42
dynamic = "text"      // ✓ OK
dynamic = [1, 2, 3]   // ✓ OK
```

## Type Inference

The type checker infers types when not explicitly annotated:

```namel3ss
let x = 42              // Inferred: number
let y = "hello"         // Inferred: text
let z = true            // Inferred: boolean
let arr = [1, 2, 3]     // Inferred: array<number>
let obj = {a: 1, b: 2}  // Inferred: {a: number, b: number}
```

## Operators

### Arithmetic Operators

```namel3ss
let a: number = 10 + 5      // ✓ Addition
let b: number = 10 - 5      // ✓ Subtraction
let c: number = 10 * 5      // ✓ Multiplication
let d: number = 10 / 5      // ✓ Division
let e: number = 10 % 3      // ✓ Modulo
let f: number = 2 ** 8      // ✓ Exponentiation

// String concatenation
let greeting: text = "Hello " + "World"  // ✓ OK

// Type error:
let invalid = 10 + "hello"  // ✗ Cannot add number and text
```

### Comparison Operators

```namel3ss
let gt: boolean = 10 > 5       // ✓ Greater than
let lt: boolean = 5 < 10       // ✓ Less than
let gte: boolean = 10 >= 10    // ✓ Greater or equal
let lte: boolean = 5 <= 10     // ✓ Less or equal
let eq: boolean = 10 == 10     // ✓ Equal
let neq: boolean = 10 != 5     // ✓ Not equal
```

### Logical Operators

```namel3ss
let and_result: boolean = true and false   // ✓ Logical AND
let or_result: boolean = true or false     // ✓ Logical OR
let not_result: boolean = not true         // ✓ Logical NOT

// Type error:
let invalid = 10 and 5  // ✗ Operands must be boolean
```

## Lambda Expressions

### Basic Lambdas

```namel3ss
// With type annotations
let double: (number) => number = fn(x: number) => x * 2

// Without annotations (inferred)
let triple = fn(x) => x * 3

// Multiple parameters
let add = fn(x: number, y: number) => x + y
```

### Type Checking

```namel3ss
// Parameter types are checked
let square = fn(x: number) => x * x
square(5)       // ✓ OK
square("text")  // ✗ Type error: expected number

// Return type is inferred
let is_positive = fn(x: number) => x > 0  // Inferred: (number) => boolean
```

## Array Operations

### Indexing

```namel3ss
let numbers: array<number> = [1, 2, 3, 4, 5]

let first: number = numbers[0]        // ✓ OK
let second: number = numbers[1]       // ✓ OK

// Type error:
let invalid = numbers["zero"]  // ✗ Array index must be number
```

### Slicing

```namel3ss
let numbers = [1, 2, 3, 4, 5]

let subset = numbers[1:4]      // [2, 3, 4]
let from_start = numbers[:3]   // [1, 2, 3]
let to_end = numbers[2:]       // [3, 4, 5]
let with_step = numbers[::2]   // [1, 3, 5]
```

## Object Operations

### Property Access

```namel3ss
let user: {name: text, age: number} = {name: "Alice", age: 30}

let name: text = user["name"]      // ✓ OK
let age: number = user["age"]      // ✓ OK

// Type error:
let email = user["email"]  // ✗ Property 'email' does not exist
```

## Built-in Functions

### Type Conversion

```namel3ss
let num_to_str: text = str(42)           // "42"
let str_to_num: number = int("123")      // 123
let to_bool: boolean = bool(1)           // true
```

### Array Functions

```namel3ss
let numbers = [1, 2, 3, 4, 5]

// len: array<T> => number
let count: number = len(numbers)  // 5

// sum: array<number> => number
let total: number = sum(numbers)  // 15
```

### Higher-Order Functions

#### map

```namel3ss
// map<T, R>(array<T>, (T) => R) => array<R>
let numbers = [1, 2, 3, 4, 5]
let doubled: array<number> = map(numbers, fn(x) => x * 2)
// [2, 4, 6, 8, 10]

// Transform to different type
let strings: array<text> = map(numbers, fn(x) => str(x))
// ["1", "2", "3", "4", "5"]
```

#### filter

```namel3ss
// filter<T>(array<T>, (T) => boolean) => array<T>
let numbers = [1, 2, 3, 4, 5]
let evens: array<number> = filter(numbers, fn(x) => x % 2 == 0)
// [2, 4]

let positives = filter(numbers, fn(x) => x > 0)
// [1, 2, 3, 4, 5]

// Type error: filter function must return boolean
let invalid = filter(numbers, fn(x) => x * 2)  // ✗ Returns number
```

## Type Checking Examples

### Example 1: Function Type Checking

```namel3ss
// Define a typed function
let add: (number, number) => number = fn(x: number, y: number) => x + y

// Valid calls
add(5, 10)      // ✓ OK
add(1.5, 2.5)   // ✓ OK

// Type errors
add(5, "10")    // ✗ Argument 2: expected number, got text
add(5)          // ✗ Expected 2 arguments, got 1
```

### Example 2: Array Type Checking

```namel3ss
let numbers: array<number> = [1, 2, 3]
numbers[0] = 42  // ✓ OK

// Type errors
numbers[0] = "text"  // ✗ Cannot assign text to number
let mixed = [1, "two", 3]  // ✗ Array elements must have same type
```

### Example 3: Object Type Checking

```namel3ss
let user: {name: text, age: number} = {
    name: "Alice",
    age: 30
}

// Structural subtyping
let detailed = {name: "Bob", age: 25, email: "bob@example.com"}
let basic: {name: text} = detailed  // ✓ OK (extra fields ignored)

// Type error
let invalid: {name: text, age: number} = {name: "Charlie"}  // ✗ Missing 'age'
```

## Error Messages

### Type Mismatch

```
[TYPE_MISMATCH] Function 'add' argument 1: expected number, got text
  at line 10, column 5
```

### Undefined Variable

```
[UNDEFINED_VARIABLE] Undefined variable 'x'
  at line 5, column 10
```

### Wrong Argument Count

```
[WRONG_ARG_COUNT] Function expects 2 arguments, got 3
  at line 15, column 8
```

### Not Callable

```
[NOT_CALLABLE] Cannot call non-function type number
  at line 20, column 5
```

## Python API

### Check Module

```python
from namel3ss.types.static_checker import check_module_static
from namel3ss.lang.parser import parse_module

source = '''
app "MyApp" {
    let x: number = "wrong type"
}
'''

module = parse_module(source)
errors = check_module_static(module, path="app.ai")

for error in errors:
    print(f"{error.path}:{error.line}:{error.column}")
    print(f"  [{error.code}] {error.message}")
```

### Manual Type Checking

```python
from namel3ss.types.static_checker import StaticTypeChecker, NUMBER, TEXT
from namel3ss.ast.expressions import LiteralExpr, BinaryOp

checker = StaticTypeChecker(path="test.ai")

# Check expression
left = LiteralExpr(value=10)
right = LiteralExpr(value=20)
expr = BinaryOp(op="+", left=left, right=right)

result_type = checker.check_expression(expr)
print(f"Expression type: {result_type}")  # number

# Get errors
if checker.errors:
    for error in checker.errors:
        print(f"Error: {error.message}")
```

### Type Environment

```python
from namel3ss.types.static_checker import TypeEnvironment, NUMBER, TEXT

env = TypeEnvironment()

# Bind variables
env.bind("x", NUMBER)
env.bind("name", TEXT)

# Lookup variables
binding = env.lookup("x")
print(f"x has type: {binding.type}")  # number

# Create nested scope
child_env = env.child_scope()
child_env.bind("local", TEXT)

# Child can see parent bindings
parent_binding = child_env.lookup("x")  # Found
local_binding = child_env.lookup("local")  # Found

# Parent cannot see child bindings
child_in_parent = env.lookup("local")  # None
```

## Best Practices

### 1. Use Type Annotations

```namel3ss
// Good - explicit types
let user: {name: text, age: number} = {name: "Alice", age: 30}

// Also OK - inferred correctly
let user = {name: "Alice", age: 30}
```

### 2. Leverage Type Inference

Let the type checker infer types when obvious:

```namel3ss
let numbers = [1, 2, 3]  // Inferred: array<number>
let doubled = map(numbers, fn(x) => x * 2)  // Inferred: array<number>
```

### 3. Handle Optional Values

Use union types for optional values:

```namel3ss
let name: text | null = get_user_name()

if name != null {
    // Type narrowing - name is text here
    let uppercase = str.upper(name)
}
```

### 4. Document Function Signatures

```namel3ss
// Clear function signature
let calculate_total: (array<number>) => number = fn(items) => {
    sum(items)
}
```

## Advanced Topics

### Generic Functions

The type checker supports generic built-in functions:

```namel3ss
// map is generic: <T, R>(array<T>, (T) => R) => array<R>
let numbers: array<number> = [1, 2, 3]
let strings: array<text> = map(numbers, fn(x) => str(x))

// T=number, R=text, result is array<text>
```

### Structural Subtyping

Objects use structural subtyping (duck typing):

```namel3ss
let display: ({name: text}) => text = fn(obj) => obj.name

// Can pass objects with extra fields
display({name: "Alice", age: 30})  // ✓ OK
```

### Function Variance

Functions are contravariant in parameters, covariant in return types:

```namel3ss
// More specific parameter type = less specific function
// More specific return type = more specific function
```

## Troubleshooting

### "Type mismatch" errors
- Check operand types match operator requirements
- Verify function arguments match parameter types
- Ensure array elements are homogeneous

### "Undefined variable" errors
- Declare variables before use
- Check variable names for typos
- Verify scope (variables in parent scope are visible)

### "Wrong argument count" errors
- Check function signature
- Provide all required parameters
- Remove extra arguments

## See Also

- [Module System Guide](./MODULE_SYSTEM_GUIDE.md)
- [Enhanced Expressions Guide](./ENHANCED_EXPRESSIONS_GUIDE.md)
- [Editor API Documentation](./EDITOR_API_GUIDE.md)
