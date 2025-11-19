# Symbolic Expression Language Reference

The Namel3ss (N3) expression language extends the declarative DSL with full functional and logic programming capabilities. This document describes the syntax, semantics, and available features.

## Table of Contents

1. [Overview](#overview)
2. [Basic Expressions](#basic-expressions)
3. [Functions](#functions)
4. [Pattern Matching](#pattern-matching)
5. [Rule-Based Reasoning](#rule-based-reasoning)
6. [Built-in Functions](#built-in-functions)
7. [Safety and Limits](#safety-and-limits)
8. [Integration with N3](#integration-with-n3)

## Overview

The expression language provides:

- **First-class functions**: Named functions and lambda expressions
- **Higher-order functions**: Functions that take or return functions
- **Pattern matching**: Structural matching with destructuring
- **Rule-based reasoning**: Prolog-style logic programming
- **Safe evaluation**: Bounded recursion and execution time
- **Immutable semantics**: Pure functional programming model

### Design Principles

1. **No side effects**: Expressions are pure and deterministic
2. **Type safety**: Runtime type checking with clear error messages
3. **Resource limits**: Recursion depth and step count limits
4. **Composability**: Expressions compose naturally

## Basic Expressions

### Literals

```n3
42                  // Integer
3.14                // Float
"hello"             // String
true                // Boolean
false               // Boolean
null                // Null value
```

### Collections

```n3
[1, 2, 3]           // List
{a: 1, b: 2}        // Dict
(1, 2, 3)           // Tuple
```

### Variables

```n3
x                   // Variable reference
_                   // Wildcard (in patterns)
```

### Operators

```n3
x + y               // Addition
x - y               // Subtraction
x * y               // Multiplication
x / y               // Division
x == y              // Equality
x != y              // Inequality
x < y               // Less than
x > y               // Greater than
x <= y              // Less or equal
x >= y              // Greater or equal
```

### Indexing and Slicing

```n3
list[0]             // First element
list[1:3]           // Slice from index 1 to 3
dict["key"]         // Dictionary lookup
```

## Functions

### Named Functions

Define reusable functions with explicit parameters:

```n3
fn factorial(n) =>
  if n <= 1 then 1
  else n * factorial(n - 1)

fn fibonacci(n) =>
  if n <= 1 then n
  else fibonacci(n - 1) + fibonacci(n - 2)

fn sum_list(lst) =>
  if is_empty(lst) then 0
  else head(lst) + sum_list(tail(lst))
```

### Lambda Expressions

Anonymous functions for inline use:

```n3
fn(x) => x * 2                    // Double
fn(x, y) => x + y                 // Add
fn(x) => fn(y) => x + y           // Curried add
```

### Higher-Order Functions

Functions that operate on other functions:

```n3
map(fn(x) => x * 2, [1, 2, 3])                    // [2, 4, 6]
filter(fn(x) => x > 0, [-1, 0, 1, 2])             // [1, 2]
reduce(fn(a, b) => a + b, [1, 2, 3, 4], 0)        // 10
```

### Closures

Functions capture their environment:

```n3
fn make_adder(n) =>
  fn(x) => x + n

let add5 = make_adder(5) in
  add5(10)  // Returns 15
```

## Pattern Matching

Pattern matching allows structural decomposition of data:

### Basic Patterns

```n3
match value {
  case 42 => "the answer"
  case 0 => "zero"
  case _ => "something else"
}
```

### List Patterns

```n3
match list {
  case [] => "empty"
  case [x] => "one element: " + str(x)
  case [x, y] => "two elements"
  case [first, ...rest] => "first: " + str(first)
}
```

### Dictionary Patterns

```n3
match dict {
  case {type: "user", id: user_id} =>
    "User: " + str(user_id)
  case {type: "admin", ...rest} =>
    "Admin with extra fields"
  case _ =>
    "Unknown type"
}
```

### Tuple Patterns

```n3
match tuple {
  case (x, y) => x + y
  case (x, y, z) => x + y + z
}
```

### Guards

Add conditions to patterns:

```n3
match value {
  case x if x > 0 => "positive"
  case x if x < 0 => "negative"
  case _ => "zero"
}
```

## Rule-Based Reasoning

Define logical rules in Prolog style:

### Facts

```n3
rule parent(tom, bob).
rule parent(bob, ann).
rule parent(bob, joe).
```

### Rules with Clauses

```n3
rule ancestor(X, Y) :-
  parent(X, Y).

rule ancestor(X, Y) :-
  parent(X, Z),
  ancestor(Z, Y).
```

### Queries

```n3
query ancestor(tom, X)
// Returns: [{X: bob}, {X: ann}, {X: joe}]
```

### Negation

```n3
rule not_parent(X, Y) :-
  not parent(X, Y).
```

### Multiple Clauses

```n3
rule sibling(X, Y) :-
  parent(P, X),
  parent(P, Y),
  X != Y.
```

## Built-in Functions

### Higher-Order Functions

```n3
map(fn, list)                     // Apply function to each element
filter(fn, list)                  // Keep elements matching predicate
reduce(fn, list, init)            // Fold list with accumulator
fold(fn, init, list)              // Alias for reduce
zip(list1, list2)                 // Combine lists into pairs
enumerate(list)                   // Add indices: [(0, x), (1, y), ...]
```

### List Operations

```n3
head(list)                        // First element
tail(list)                        // All but first
cons(elem, list)                  // Prepend element
append(list1, list2)              // Concatenate lists
reverse(list)                     // Reverse order
sort(list)                        // Sort ascending
length(list)                      // Number of elements
nth(list, index)                  // Element at index
take(list, n)                     // First n elements
drop(list, n)                     // All but first n elements
```

### Numeric Functions

```n3
sum(list)                         // Sum of elements
product(list)                     // Product of elements
min(list)                         // Minimum value
max(list)                         // Maximum value
abs(n)                            // Absolute value
round(n)                          // Round to nearest integer
floor(n)                          // Round down
ceil(n)                           // Round up
range(start, end)                 // Generate range
```

### String Operations

```n3
concat(str1, str2)                // Concatenate strings
split(str, sep)                   // Split into list
join(list, sep)                   // Join list with separator
lower(str)                        // Convert to lowercase
upper(str)                        // Convert to uppercase
strip(str)                        // Remove whitespace
replace(str, old, new)            // Replace substring
```

### Dictionary Operations

```n3
keys(dict)                        // List of keys
values(dict)                      // List of values
items(dict)                       // List of (key, value) pairs
get(dict, key, default)           // Safe lookup with default
merge(dict1, dict2)               // Combine dictionaries
```

### Predicates

```n3
all(fn, list)                     // True if all match
any(fn, list)                     // True if any matches
not(value)                        // Logical negation
is_empty(collection)              // True if empty
is_none(value)                    // True if null
is_list(value)                    // Type check
is_dict(value)                    // Type check
is_int(value)                     // Type check
is_float(value)                   // Type check
is_str(value)                     // Type check
is_bool(value)                    // Type check
```

### Utilities

```n3
identity(x)                       // Return x unchanged
const(x, y)                       // Return x, ignore y
compose(f, g)                     // Create f(g(x))
pipe(list, fns)                   // Apply functions left-to-right
assert(condition, message)        // Raise error if false
```

### Type Conversions

```n3
int(value)                        // Convert to integer
float(value)                      // Convert to float
str(value)                        // Convert to string
bool(value)                       // Convert to boolean
list(value)                       // Convert to list
dict(value)                       // Convert to dictionary
```

## Safety and Limits

### Recursion Limit

Default: 100 levels

```python
# Raises RecursionLimitError
fn infinite() => infinite()
```

### Step Limit

Default: 10,000 operations

```python
# Raises StepLimitError if too many operations
```

### Configuring Limits

```python
from namel3ss.codegen.backend.core.runtime.symbolic_evaluator import evaluate_expression_tree

result = evaluate_expression_tree(
    expr,
    limits={
        'max_recursion': 200,
        'max_steps': 50000
    }
)
```

### Error Handling

All errors include:
- Error message
- Call stack
- Current recursion depth
- Step count

```python
try:
    result = evaluate_expression_tree(expr)
except RecursionLimitError as e:
    print(f"Recursion limit hit: {e}")
except StepLimitError as e:
    print(f"Step limit exceeded: {e}")
except EvaluationError as e:
    print(f"Evaluation error: {e}")
```

## Integration with N3

### Dataset Filters

Use expressions to filter dataset rows:

```n3
DATASET sales {
  SOURCE postgres.orders
  FILTER fn(row) => row.amount > 100 and row.status == "completed"
}
```

### Computed Columns

Add derived columns using expressions:

```n3
DATASET enriched_sales {
  SOURCE sales
  TRANSFORM fn(row) => {
    ...row,
    profit: row.revenue - row.cost,
    margin: (row.revenue - row.cost) / row.revenue
  }
}
```

### Page Computed Properties

Compute page state dynamically:

```n3
PAGE analytics {
  DATA {
    total: reduce(fn(a, row) => a + row.amount, sales, 0)
    average: total / length(sales)
  }
}
```

### AI Prompt Composition

Build prompts with expressions:

```n3
PROMPT analyze_sentiment {
  SYSTEM "You are a sentiment analyzer"
  USER fn(text) =>
    "Analyze: " + text + "\nProvide score from -1 to 1."
}
```

### Rule-Based Insights

Generate insights using logical rules:

```n3
rule high_value_customer(customer_id) :-
  total_purchases(customer_id, Total),
  Total > 10000.

rule at_risk(customer_id) :-
  high_value_customer(customer_id),
  days_since_purchase(customer_id, Days),
  Days > 90.
```

## Examples

### Fibonacci with Memoization

```n3
fn fib_helper(n, memo) =>
  match get(memo, n, null) {
    case null =>
      let result = if n <= 1 then n
                   else fib_helper(n-1, memo) + fib_helper(n-2, memo)
      in merge(memo, {n: result})
    case cached => cached
  }

fn fibonacci(n) => fib_helper(n, {})
```

### List Comprehension Pattern

```n3
fn list_comp(list, pred, transform) =>
  map(transform, filter(pred, list))

// Use: list_comp([1,2,3,4,5], fn(x) => x > 2, fn(x) => x * 2)
// Returns: [6, 8, 10]
```

### Tree Traversal

```n3
fn traverse(tree) =>
  match tree {
    case {value: v, left: l, right: r} =>
      concat(
        traverse(l),
        concat([v], traverse(r))
      )
    case null => []
  }
```

### Query Builder

```n3
fn build_query(filters) =>
  let clauses = map(
    fn(f) => f.field + " " + f.op + " " + str(f.value),
    filters
  ) in
  "SELECT * FROM table WHERE " + join(clauses, " AND ")
```

## See Also

- [Pattern Matching Guide](PATTERN_MATCHING.md)
- [Rule Engine Guide](RULE_ENGINE.md)
- [N3 Language Specification](spec/)
