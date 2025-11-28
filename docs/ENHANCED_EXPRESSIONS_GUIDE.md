# Enhanced Expressions Guide

Namel3ss now supports modern functional programming features including lambda expressions, subscript operations, and list comprehensions.

## Lambda Expressions

### Basic Syntax

```namel3ss
// Simple lambda
let double = fn(x) => x * 2

// With type annotations
let add: (number, number) => number = fn(x: number, y: number) => x + y

// Multiple parameters
let multiply = fn(a, b, c) => a * b * c

// No parameters
let get_greeting = fn() => "Hello, World!"
```

### Using Lambdas

```namel3ss
// Immediately invoked
let result = (fn(x) => x * 2)(5)  // 10

// Passed to higher-order functions
let numbers = [1, 2, 3, 4, 5]
let doubled = map(numbers, fn(x) => x * 2)

// Stored in variables
let is_positive = fn(x) => x > 0
let has_positives = filter(numbers, is_positive)
```

### Lambda Type Checking

```namel3ss
// Parameter types are validated
let safe_divide: (number, number) => number = fn(a, b) => a / b

safe_divide(10, 2)      // ✓ OK: 5
safe_divide("10", 2)    // ✗ Type error: expected number

// Return type is inferred
let compare = fn(x, y) => x > y  // Inferred: (number, number) => boolean
```

### Closures

Lambdas capture variables from their enclosing scope:

```namel3ss
let multiplier = 10
let scale = fn(x) => x * multiplier

scale(5)  // 50

// Update captured variable
multiplier = 20
scale(5)  // 100
```

## Subscript Operations

### Array Indexing

```namel3ss
let numbers = [10, 20, 30, 40, 50]

// Access by index (0-based)
let first = numbers[0]      // 10
let second = numbers[1]     // 20
let last = numbers[4]       // 50

// Negative indices (from end)
let last_item = numbers[-1]     // 50
let second_last = numbers[-2]   // 40
```

### Array Slicing

```namel3ss
let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

// Basic slice [start:end]
let subset = numbers[2:5]      // [3, 4, 5]

// From start
let first_three = numbers[:3]  // [1, 2, 3]

// To end
let last_three = numbers[7:]   // [8, 9, 10]

// With step [start:end:step]
let evens = numbers[1::2]      // [2, 4, 6, 8, 10]
let odds = numbers[0::2]       // [1, 3, 5, 7, 9]

// Reverse
let reversed = numbers[::-1]   // [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
```

### Object Property Access

```namel3ss
let user = {
    name: "Alice",
    age: 30,
    email: "alice@example.com"
}

// Access properties
let name = user["name"]        // "Alice"
let age = user["age"]          // 30

// Dynamic property access
let field = "email"
let value = user[field]        // "alice@example.com"
```

### Nested Access

```namel3ss
let data = {
    users: [
        {name: "Alice", age: 30},
        {name: "Bob", age: 25}
    ]
}

// Chain subscript operations
let first_user = data["users"][0]           // {name: "Alice", age: 30}
let first_name = data["users"][0]["name"]   // "Alice"
```

## List Comprehensions

### Basic Syntax

```namel3ss
// [expression for variable in iterable]
let numbers = [1, 2, 3, 4, 5]
let doubled = [x * 2 for x in numbers]
// [2, 4, 6, 8, 10]

// With filtering
let evens = [x for x in numbers if x % 2 == 0]
// [2, 4]

// Transform and filter
let even_squares = [x * x for x in numbers if x % 2 == 0]
// [4, 16]
```

### Under the Hood

List comprehensions are syntactic sugar for `map` and `filter`:

```namel3ss
// This comprehension:
let result = [x * 2 for x in items if x > 0]

// Is equivalent to:
let filtered = filter(items, fn(x) => x > 0)
let result = map(filtered, fn(x) => x * 2)
```

### Complex Examples

```namel3ss
// Multiple conditions
let filtered = [x for x in numbers if x > 2 if x < 8]
// Equivalent to: if x > 2 and x < 8

// String manipulation
let names = ["alice", "bob", "charlie"]
let uppercase = [str.upper(name) for name in names]
// ["ALICE", "BOB", "CHARLIE"]

// Working with objects
let users = [
    {name: "Alice", age: 30},
    {name: "Bob", age: 25},
    {name: "Charlie", age: 35}
]

let adult_names = [user["name"] for user in users if user["age"] >= 30]
// ["Alice", "Charlie"]
```

## Higher-Order Functions

### map

Transform every element in an array:

```namel3ss
let numbers = [1, 2, 3, 4, 5]

// Using lambda
let squared = map(numbers, fn(x) => x * x)
// [1, 4, 9, 16, 25]

// Convert types
let strings = map(numbers, fn(x) => str(x))
// ["1", "2", "3", "4", "5"]

// Complex transformation
let users = map(
    [1, 2, 3],
    fn(id) => {name: "User " + str(id), id: id}
)
// [{name: "User 1", id: 1}, {name: "User 2", id: 2}, {name: "User 3", id: 3}]
```

### filter

Keep only elements matching a condition:

```namel3ss
let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

// Keep evens
let evens = filter(numbers, fn(x) => x % 2 == 0)
// [2, 4, 6, 8, 10]

// Keep positives
let positives = filter(numbers, fn(x) => x > 0)
// [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

// Complex conditions
let special = filter(numbers, fn(x) => x > 3 and x < 8)
// [4, 5, 6, 7]
```

### Chaining Operations

Combine map and filter for complex transformations:

```namel3ss
let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

// Filter then map
let even_squares = map(
    filter(numbers, fn(x) => x % 2 == 0),
    fn(x) => x * x
)
// [4, 16, 36, 64, 100]

// Multiple transformations
let result = map(
    map(
        filter(numbers, fn(x) => x > 5),
        fn(x) => x * 2
    ),
    fn(x) => x + 1
)
// [13, 15, 17, 19, 21]
```

## Practical Examples

### Example 1: Data Processing Pipeline

```namel3ss
let transactions = [
    {amount: 100, type: "debit"},
    {amount: 50, type: "credit"},
    {amount: 200, type: "debit"},
    {amount: 75, type: "credit"}
]

// Get all debit amounts over 50
let large_debits = map(
    filter(
        transactions,
        fn(t) => t["type"] == "debit" and t["amount"] > 50
    ),
    fn(t) => t["amount"]
)
// [100, 200]

// Or using comprehension
let large_debits_alt = [
    t["amount"] 
    for t in transactions 
    if t["type"] == "debit" 
    if t["amount"] > 50
]
// [100, 200]
```

### Example 2: User Validation

```namel3ss
let users = [
    {name: "Alice", age: 30, active: true},
    {name: "Bob", age: 17, active: true},
    {name: "Charlie", age: 25, active: false},
    {name: "Diana", age: 35, active: true}
]

// Get names of active adult users
let valid_users = [
    user["name"]
    for user in users
    if user["active"]
    if user["age"] >= 18
]
// ["Alice", "Diana"]
```

### Example 3: Matrix Operations

```namel3ss
let matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

// Double all values
let doubled_matrix = map(
    matrix,
    fn(row) => map(row, fn(x) => x * 2)
)
// [[2, 4, 6], [8, 10, 12], [14, 16, 18]]

// Flatten matrix
let flattened = []
for row in matrix {
    for value in row {
        flattened.append(value)
    }
}
// [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### Example 4: String Processing

```namel3ss
let sentences = [
    "hello world",
    "the quick brown fox",
    "jumps over the lazy dog"
]

// Capitalize first letter of each word
let capitalized = map(
    sentences,
    fn(s) => join(
        map(
            split(s, " "),
            fn(word) => upper(word[0]) + word[1:]
        ),
        " "
    )
)
// ["Hello World", "The Quick Brown Fox", "Jumps Over The Lazy Dog"]
```

## Type Safety

All enhanced expressions are fully type-checked:

### Lambda Type Checking

```namel3ss
// Parameter types validated
let add: (number, number) => number = fn(x, y) => x + y
add(5, 10)      // ✓ OK
add(5, "ten")   // ✗ Type error

// Return type inferred
let is_adult = fn(age) => age >= 18  // Inferred: (number) => boolean
```

### Subscript Type Checking

```namel3ss
let numbers: array<number> = [1, 2, 3]

let first: number = numbers[0]      // ✓ OK
let invalid = numbers["zero"]       // ✗ Index must be number

let user: {name: text, age: number} = {name: "Alice", age: 30}
let name: text = user["name"]       // ✓ OK
let missing = user["email"]         // ✗ Property doesn't exist
```

### Comprehension Type Checking

```namel3ss
let numbers: array<number> = [1, 2, 3, 4, 5]

// Result type inferred
let doubled: array<number> = [x * 2 for x in numbers]  // ✓ OK

// Filter function must return boolean
let evens = [x for x in numbers if x % 2]  // ✗ Must return boolean
```

## Performance Tips

### 1. Use Native Operations When Possible

```namel3ss
// Good - single pass
let result = map(items, fn(x) => x * 2)

// Less efficient - multiple passes
let temp = map(items, fn(x) => x)
let result = map(temp, fn(x) => x * 2)
```

### 2. Filter Before Map

```namel3ss
// Good - fewer items to transform
let result = map(
    filter(items, fn(x) => x > 0),
    fn(x) => expensive_operation(x)
)

// Less efficient - transforms all items first
let result = filter(
    map(items, fn(x) => expensive_operation(x)),
    fn(x) => x > 0
)
```

### 3. Use Comprehensions for Readability

```namel3ss
// Clear intent
let valid_items = [
    process(item)
    for item in items
    if is_valid(item)
]

// Less clear
let valid_items = map(
    filter(items, fn(x) => is_valid(x)),
    fn(x) => process(x)
)
```

## Migration Guide

### From Old Style to Lambdas

```namel3ss
// Old: Named functions
function double(x) {
    return x * 2
}
let result = map(numbers, double)

// New: Inline lambdas
let result = map(numbers, fn(x) => x * 2)
```

### From Loops to Comprehensions

```namel3ss
// Old: Explicit loops
let result = []
for x in numbers {
    if x % 2 == 0 {
        result.append(x * x)
    }
}

// New: Comprehension
let result = [x * x for x in numbers if x % 2 == 0]
```

## Best Practices

### 1. Keep Lambdas Simple

```namel3ss
// Good - single expression
let doubled = map(numbers, fn(x) => x * 2)

// Consider extracting to named function
let complex_result = map(numbers, fn(x) => {
    let temp = x * 2
    let adjusted = temp + 10
    return adjusted / 3
})
```

### 2. Use Meaningful Variable Names

```namel3ss
// Good
let active_users = [user for user in users if user["active"]]

// Less clear
let result = [u for u in users if u["active"]]
```

### 3. Chain Operations Logically

```namel3ss
// Good - clear pipeline
let result = map(
    filter(items, fn(x) => is_valid(x)),
    fn(x) => transform(x)
)

// Consider breaking into steps for complex logic
let valid = filter(items, fn(x) => is_valid(x))
let transformed = map(valid, fn(x) => transform(x))
let result = map(transformed, fn(x) => finalize(x))
```

## Troubleshooting

### Lambda Syntax Errors
- Ensure arrow `=>` is used
- Check parameter list is in parentheses (even for single param)
- Verify expression or block is after arrow

### Subscript Errors
- Array indices must be numbers
- Object keys must be strings
- Check index is within bounds

### Comprehension Errors
- Filter conditions must return boolean
- Ensure variable names match
- Check iterable is actually an array

## See Also

- [Static Type Checker Guide](./TYPE_CHECKER_GUIDE.md)
- [Module System Guide](./MODULE_SYSTEM_GUIDE.md)
- [Editor API Documentation](./EDITOR_API_GUIDE.md)
