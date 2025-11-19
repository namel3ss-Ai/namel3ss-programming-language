# Symbolic Expressions in Namel3ss

Namel3ss now includes a powerful symbolic expression system that brings functional and logic programming capabilities to the DSL. This system integrates seamlessly with existing N3 constructs while providing advanced features like pattern matching, recursion, and rule-based reasoning.

## Overview

The symbolic expression system extends N3 with:

- **Named & Anonymous Functions**: Define reusable functions with `fn` keyword
- **Pattern Matching**: Destructure data with `match` expressions
- **Higher-Order Functions**: Built-in `map`, `filter`, `reduce`, and more
- **Rule-Based Logic**: Prolog-style rules and queries for symbolic reasoning
- **Local Bindings**: `let` expressions for scoped variables
- **Conditional Expressions**: Expression-level `if` constructs
- **Safety Guarantees**: Bounded recursion and evaluation limits

All features are:
- **Safe**: No arbitrary code execution, strict sandboxing
- **Composable**: Mix with existing N3 syntax seamlessly
- **Validated**: Full resolver integration with scope checking
- **Production-Ready**: Battle-tested with comprehensive test coverage

## Quick Start

### Simple Function

```n3
app "MyApp".

fn double(x) => x * 2
fn greet(name) => "Hello, " + name + "!"

dataset "users" from memory:
    add column doubled_age = double(age)
    add column greeting = greet(name)
```

### Lambda Expressions

```n3
dataset "processed" from memory:
    add column squares = map(numbers, fn(x) => x * x)
    add column evens = filter(numbers, fn(x) => x % 2 == 0)
```

### Pattern Matching

```n3
fn classify(score) => match score:
    case s if s >= 90: "A"
    case s if s >= 80: "B"
    case s if s >= 70: "C"
    else: "F"
```

### Rule-Based Logic

```n3
rule adult(person) :- person.age >= 18.
rule eligible(person) :- adult(person), person.verified == True.

dataset "eligible_users" from memory:
    filter by: query eligible(user)
```

## Function Definitions

### Named Functions

Define functions at the app level:

```n3
fn function_name(param1, param2, ...) => expression
```

With default parameters:

```n3
fn greet(name, greeting = "Hello") => greeting + ", " + name
```

With type hints:

```n3
fn add(x: int, y: int) => x + y
```

Multi-line functions:

```n3
fn process(data) => {
    let cleaned = filter(data, fn(x) => x != null)
    let transformed = map(cleaned, fn(x) => x * 2)
    reduce(transformed, fn(acc, x) => acc + x, 0)
}
```

### Anonymous Functions (Lambdas)

Lambda syntax: `fn(params) => expression`

```n3
let double = fn(x) => x * 2
let add = fn(a, b) => a + b
```

### Recursion

Functions can call themselves recursively:

```n3
fn factorial(n) => if n <= 1: 1 else: n * factorial(n - 1)

fn fibonacci(n) => match n:
    case 0: 0
    case 1: 1
    else: fibonacci(n - 1) + fibonacci(n - 2)
```

**Safety**: Recursion is limited by `NAMEL3SS_EXPR_MAX_DEPTH` (default: 100).

## Pattern Matching

### Basic Match

```n3
match value:
    case pattern1: result1
    case pattern2: result2
    else: default_result
```

### Literal Patterns

```n3
match status:
    case "pending": "In Progress"
    case "completed": "Done"
    case "cancelled": "Aborted"
    else: "Unknown"
```

### Variable Binding

```n3
match event:
    case {type: "order", id: order_id}:
        process_order(order_id)
    case {type: "refund", amount: amt}:
        process_refund(amt)
    else:
        log_unknown()
```

### List Patterns

```n3
match items:
    case []: "empty"
    case [first]: "single: " + str(first)
    case [first, second, ...rest]: "multiple"
```

### Constructor Patterns

```n3
match result:
    case Ok(value): value
    case Err(error): "Error: " + error
```

### Wildcard Pattern

```n3
match data:
    case {type: "user", _}: "User data"
    case _: "Other"
```

## Higher-Order Functions

### Built-in Functions

#### `map(list, fn)`
Apply function to each element:

```n3
map([1, 2, 3], fn(x) => x * 2)  # [2, 4, 6]
```

#### `filter(list, fn)`
Keep elements where function returns true:

```n3
filter([1, 2, 3, 4], fn(x) => x % 2 == 0)  # [2, 4]
```

#### `reduce(list, fn, initial)`
Accumulate values:

```n3
reduce([1, 2, 3, 4], fn(acc, x) => acc + x, 0)  # 10
```

#### `fold_left(list, fn, initial)`
Left-associative fold:

```n3
fold_left([1, 2, 3], fn(acc, x) => acc - x, 10)  # 4
```

#### `fold_right(list, fn, initial)`
Right-associative fold:

```n3
fold_right([1, 2, 3], fn(acc, x) => x - acc, 0)  # 2
```

#### `zip(list1, list2)`
Combine two lists:

```n3
zip([1, 2, 3], ['a', 'b', 'c'])  # [[1, 'a'], [2, 'b'], [3, 'c']]
```

#### `flat_map(list, fn)`
Map then flatten:

```n3
flat_map([1, 2, 3], fn(x) => [x, x * 2])  # [1, 2, 2, 4, 3, 6]
```

### Composition

```n3
fn compose(f, g) => fn(x) => f(g(x))

let double = fn(x) => x * 2
let increment = fn(x) => x + 1
let double_then_increment = compose(increment, double)
```

## Rule-Based Logic

### Rule Definitions

Define facts and rules:

```n3
# Fact
rule parent(tom, bob).

# Rule with body
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

### Rule Bodies

Clauses are comma-separated (conjunction):

```n3
rule eligible(user) :- 
    user.age >= 18,
    user.verified == True,
    user.status == "active".
```

### Negation

Use `not` for negation:

```n3
rule inactive(user) :- not user.status == "active".
```

### Queries

Query rules in expressions:

```n3
dataset "eligible_users" from memory:
    filter by: query eligible(user)

let parents = query parent(X, "alice")
```

### Query with Limits

```n3
let first_10_parents = query parent(X, Y) limit 10
```

## Local Bindings

### Let Expressions

Introduce local variables:

```n3
let x = 10: let y = 20: x + y  # 30
```

Multiple bindings:

```n3
let result = 
    let tax_rate = 0.08:
    let subtotal = price * quantity:
    let tax = subtotal * tax_rate:
    subtotal + tax
```

## Conditional Expressions

### If Expressions

```n3
if condition: true_value else: false_value
```

Nested:

```n3
if score >= 90:
    "A"
else if score >= 80:
    "B"
else if score >= 70:
    "C"
else:
    "F"
```

### Ternary-style

```n3
let status = if is_premium: "Premium" else: "Standard"
```

## Integration with N3

### Dataset Filters

Use symbolic expressions in filters:

```n3
fn is_eligible(user) => user.age >= 18 and user.verified

dataset "users" from postgres:
    filter by: is_eligible(user) and user.status == "active"
```

### Computed Columns

```n3
dataset "products" from memory:
    add column discounted = if price > 100: price * 0.9 else: price
    add column category = match type:
        case "book": "Books"
        case "electronic": "Electronics"
        else: "Other"
```

### Control Flow

Use symbolic expressions in `if` conditions:

```n3
fn can_view(user) => user.role == "admin" or "view" in user.permissions

page dashboard:
    if can_view(ctx.user):
        show data "metrics"
```

### AI Workflow Conditions

```n3
fn requires_review(confidence) => confidence < 0.8

workflow classify:
    if requires_review(result.confidence):
        human_review result
```

## Safety & Configuration

### Recursion Limits

**Default**: 100 levels
**Configure**: Set `NAMEL3SS_EXPR_MAX_DEPTH` environment variable

```bash
export NAMEL3SS_EXPR_MAX_DEPTH=50
```

### Evaluation Step Limits

**Default**: 10,000 steps
**Configure**: Set `NAMEL3SS_EXPR_MAX_STEPS` environment variable

```bash
export NAMEL3SS_EXPR_MAX_STEPS=5000
```

### Workspace Configuration

In `namel3ss.toml`:

```toml
[defaults]
expr_max_depth = 50
expr_max_steps = 5000
```

Per-app:

```toml
[[apps]]
name = "my_app"
file = "app.n3"
expr_max_depth = 100
expr_max_steps = 20000
```

### Error Handling

When limits are exceeded:

```
RuntimeError: Maximum recursion depth exceeded (limit: 100)
RuntimeError: Maximum evaluation steps exceeded (limit: 10000)
```

## Built-in Functions Reference

### List Operations
- `len(list)` - Length of list
- `head(list)` - First element
- `tail(list)` - All but first element
- `take(list, n)` - First n elements
- `drop(list, n)` - All but first n elements
- `reverse(list)` - Reverse list
- `sort(list)` - Sort list
- `concat(list1, list2)` - Concatenate lists
- `append(list, item)` - Append item to list
- `flatten(nested_list)` - Flatten nested lists

### Type Checking
- `is_int(x)` - Check if integer
- `is_float(x)` - Check if float
- `is_str(x)` - Check if string
- `is_bool(x)` - Check if boolean
- `is_list(x)` - Check if list
- `is_dict(x)` - Check if dictionary
- `is_none(x)` - Check if None

### Type Conversion
- `int(x)` - Convert to integer
- `float(x)` - Convert to float
- `str(x)` - Convert to string
- `bool(x)` - Convert to boolean

### String Operations
- `upper(s)` - Uppercase
- `lower(s)` - Lowercase
- `strip(s)` - Remove whitespace
- `split(s, sep)` - Split string
- `join(list, sep)` - Join strings
- `replace(s, old, new)` - Replace substring
- `starts_with(s, prefix)` - Check prefix
- `ends_with(s, suffix)` - Check suffix
- `contains(s, substring)` - Check contains

### Math Operations
- `abs(x)` - Absolute value
- `min(a, b)` - Minimum
- `max(a, b)` - Maximum
- `pow(x, y)` - Power
- `sqrt(x)` - Square root
- `floor(x)` - Floor
- `ceil(x)` - Ceiling
- `round(x)` - Round

### Logic Operations
- `and(a, b)` - Logical AND
- `or(a, b)` - Logical OR
- `not(x)` - Logical NOT
- `all(list)` - All truthy
- `any(list)` - Any truthy

## Best Practices

### 1. Use Functions for Reusability

Instead of:
```n3
dataset "users" from memory:
    filter by: age >= 18 and verified == True
    
dataset "customers" from memory:
    filter by: age >= 18 and verified == True
```

Do:
```n3
fn is_eligible(entity) => entity.age >= 18 and entity.verified

dataset "users" from memory:
    filter by: is_eligible(user)
    
dataset "customers" from memory:
    filter by: is_eligible(customer)
```

### 2. Prefer Pattern Matching for Complex Conditionals

Instead of:
```n3
if type == "a": 1 else if type == "b": 2 else if type == "c": 3 else: 0
```

Do:
```n3
match type:
    case "a": 1
    case "b": 2
    case "c": 3
    else: 0
```

### 3. Use Let for Complex Calculations

Instead of:
```n3
add column total = price * quantity * (1 + tax_rate)
```

Do:
```n3
add column total = 
    let subtotal = price * quantity:
    let tax = subtotal * tax_rate:
    subtotal + tax
```

### 4. Leverage Higher-Order Functions

Instead of manual loops (when possible):
```n3
fn process(items) => map(
    filter(items, fn(x) => x.active),
    fn(x) => x.value * 2
)
```

### 5. Document Rule Dependencies

```n3
# Base rules
rule adult(person) :- person.age >= 18.
rule verified(person) :- person.verified == True.

# Derived rules
rule eligible(person) :- adult(person), verified(person).
rule premium_eligible(person) :- eligible(person), person.premium == True.
```

## Backward Compatibility

All existing N3 applications continue to work without modification. Symbolic expressions are opt-in:

- Legacy expressions work as before
- Mix legacy and symbolic expressions freely
- No breaking changes to existing syntax

## Performance Considerations

- **Recursion**: Limited by depth, use iterative built-ins when possible
- **Pattern Matching**: O(1) for simple patterns, O(n) for deep nesting
- **Higher-Order Functions**: Lazy evaluation where possible
- **Rule Queries**: Use `limit` for large result sets

## Examples

See `examples/` directory for complete applications:
- `examples/symbolic_functions.n3` - Function examples
- `examples/pattern_matching.n3` - Pattern matching showcase
- `examples/rules_logic.n3` - Rule-based reasoning
- `examples/higher_order.n3` - Functional programming patterns

## Troubleshooting

### "Maximum recursion depth exceeded"
- Reduce recursion depth or use iterative alternatives
- Check for infinite recursion bugs
- Increase `NAMEL3SS_EXPR_MAX_DEPTH` if needed

### "Unknown name 'X' in expression"
- Verify function is defined before use
- Check variable scope in let expressions
- Ensure rule predicates are defined

### "Expected identifier but got..."
- Check syntax: `fn(x) => expr` not `fn (x) => expr`
- Verify parentheses in function calls
- Check comma separation in lists

## Summary

The symbolic expression system transforms Namel3ss into a true AI programming language with:

✅ **Functional Programming**: First-class functions, lambdas, higher-order functions
✅ **Pattern Matching**: Destructure and match complex data structures  
✅ **Logic Programming**: Prolog-style rules and queries
✅ **Type Safety**: Runtime type checking and validation
✅ **Production Ready**: Safe, tested, and fully integrated

Start using symbolic expressions today to build more expressive, maintainable N3 applications!
