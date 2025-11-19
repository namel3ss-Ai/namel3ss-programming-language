# Logic Programming in Namel3ss

## Overview

Namel3ss includes first-class support for logic programming with Prolog-style syntax. The knowledge and inference system enables:

- **Facts**: Ground statements about your domain
- **Rules**: Logical implications with backtracking
- **Queries**: Questions against your knowledge base  
- **Unification**: Pattern matching with variables
- **Backtracking**: Systematic exploration of solution spaces

## Knowledge Modules

Knowledge modules contain facts and rules that define relationships in your domain.

### Syntax

```n3
knowledge <module_name>:
    fact <predicate>(<args>).
    rule <head> :- <body>.
```

### Example: Family Relationships

```n3
app "FamilyTree".

knowledge family:
    # Facts: parent relationships
    fact parent(alice, bob).
    fact parent(bob, charlie).
    fact parent(charlie, dave).
    
    # Rules: ancestor is transitive closure of parent
    rule ancestor(X, Y) :- parent(X, Y).
    rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
    
    # Rules: siblings share a parent
    rule sibling(X, Y) :- parent(P, X), parent(P, Y).

query find_ancestors:
    from knowledge family:
    goal ancestor(alice, Descendant)
    variables: Descendant
```

## Terms

### Variables

Variables start with an uppercase letter or underscore:

```n3
X, Y, Name, _Result, _
```

Anonymous variables (`_`) are used when you don't care about the value.

### Atoms

Atoms are symbolic constants (lowercase identifiers):

```n3
alice, bob, engineering, active
```

### Numbers

Integer and floating-point literals:

```n3
42, 3.14, -17, 0.5
```

### Strings

Double-quoted string literals:

```n3
"Alice", "Engineering Department", "123-45-6789"
```

### Structures

Compound terms with a functor and arguments:

```n3
person(alice, 30, engineer)
point(10, 20, 30)
employee(1, "Alice", dept("Engineering", floor(3)))
```

### Lists

Ordered collections of terms:

```n3
[1, 2, 3]
[alice, bob, charlie]
[X, Y | Rest]  # List pattern with head and tail
```

## Facts

Facts are ground (variable-free) statements about your domain:

```n3
knowledge employees:
    fact employee(1, "Alice", "Engineering").
    fact employee(2, "Bob", "Sales").
    fact salary(1, 100000).
    fact salary(2, 80000).
    fact reports_to(2, 1).
```

## Rules

Rules define logical implications. A rule head is true if all goals in the body are true:

```n3
rule <head> :- <goal1>, <goal2>, ..., <goalN>.
```

### Simple Rules

```n3
knowledge company:
    # Direct manager relationship
    rule manager(ManagerId, EmployeeId) :- 
        reports_to(EmployeeId, ManagerId).
    
    # Employee in engineering
    rule engineer(Id, Name) :- 
        employee(Id, Name, "Engineering").
    
    # High earner
    rule high_earner(Id) :- 
        salary(Id, Amount), 
        Amount > 90000.
```

### Recursive Rules

Rules can reference themselves to express transitive relationships:

```n3
knowledge graph:
    # Direct edges
    fact edge(a, b).
    fact edge(b, c).
    fact edge(c, d).
    
    # Transitive closure: path
    rule path(X, Y) :- edge(X, Y).
    rule path(X, Z) :- edge(X, Y), path(Y, Z).
```

### Multiple Clauses

Multiple rules with the same head define alternative ways to prove a goal:

```n3
knowledge animals:
    fact mammal(dog).
    fact mammal(cat).
    fact bird(eagle).
    fact bird(sparrow).
    
    # An animal is either a mammal or a bird
    rule animal(X) :- mammal(X).
    rule animal(X) :- bird(X).
```

## Variable Scoping

Variables in rules are scoped to that rule:

```n3
# Each rule has independent X and Y variables
rule foo(X) :- bar(X).
rule foo(Y) :- baz(Y).
```

**Unsafe variables**: Variables in the head must appear in the body:

```n3
# ✗ INVALID: Y doesn't appear in body
rule bad(X, Y) :- foo(X).

# ✓ VALID: All head variables in body
rule good(X, Y) :- foo(X), bar(Y).
```

## Unification

Unification is pattern matching that binds variables to terms:

```n3
# Unifying person(X, 30) with person(alice, 30)
# Binds: X = alice

# Unifying person(X, Age) with person(bob, 25)
# Binds: X = bob, Age = 25

# Unifying [H|T] with [1, 2, 3]
# Binds: H = 1, T = [2, 3]
```

### Occurs Check

Unification includes an occurs check to prevent infinite structures:

```n3
# ✗ FAILS: Would create X = f(X) (infinite term)
X = f(X)

# ✗ FAILS: Would create X = [a, X, b] (infinite list)
X = [a, X, b]
```

## Backtracking

The logic engine uses depth-first search with backtracking to find all solutions:

```n3
knowledge options:
    fact color(red).
    fact color(blue).
    fact size(small).
    fact size(large).
    
    # Generates all combinations via backtracking
    rule item(C, S) :- color(C), size(S).

query all_items:
    from knowledge options:
    goal item(Color, Size)
    variables: Color, Size

# Results:
# {Color: red, Size: small}
# {Color: red, Size: large}
# {Color: blue, Size: small}
# {Color: blue, Size: large}
```

## Safety Limits

The logic engine includes configurable safety limits:

- **Max depth**: 100 (recursion depth limit)
- **Max steps**: 10,000 (total inference steps)
- **Timeout**: 10 seconds

These prevent infinite loops and resource exhaustion:

```n3
# This would loop forever without limits
rule loop(X) :- loop(X).
```

## Best Practices

### 1. Ground your facts

Facts should not contain variables:

```n3
# ✓ GOOD
fact employee(1, "Alice", "Engineering").

# ✗ BAD
fact employee(X, "Alice", "Engineering").
```

### 2. Avoid singleton variables

Variables that appear only once are often typos:

```n3
# ✗ WARNING: Y is singleton (typo?)
rule has_typo(X) :- foo(X, Y).

# ✓ GOOD: Use _ for intentionally ignored values
rule good(X) :- foo(X, _).
```

### 3. Check predicate arity

Use predicates consistently:

```n3
# ✗ BAD: Inconsistent arity
fact likes(alice, pizza).
fact likes(bob, burgers, "a lot").  # Different arity!

# ✓ GOOD: Consistent arity
fact likes(alice, pizza).
fact likes(bob, burgers).
```

### 4. Use descriptive names

```n3
# ✓ GOOD
rule is_manager(EmployeeId) :- 
    reports_to(_, EmployeeId).

# ✗ LESS CLEAR
rule m(X) :- r(_, X).
```

### 5. Structure complex rules clearly

```n3
# ✓ GOOD: Multi-line for clarity
rule qualified_candidate(PersonId, JobId) :-
    person(PersonId, Name, Skills),
    job(JobId, Title, RequiredSkills),
    has_skills(Skills, RequiredSkills),
    available(PersonId).

# ✗ HARDER TO READ: All on one line
rule qualified_candidate(P, J) :- person(P, N, S), job(J, T, R), has_skills(S, R), available(P).
```

## Common Patterns

### Transitive Closure

```n3
# Base case: direct relationship
rule reachable(X, Y) :- connected(X, Y).

# Recursive case: indirect through intermediate
rule reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
```

### Join Pattern

```n3
# Combine information from multiple sources
rule employee_with_salary(Id, Name, Dept, Salary) :-
    employee(Id, Name, Dept),
    salary(Id, Salary).
```

### Filter Pattern

```n3
# Select subset based on condition
rule senior_employee(Id, Name) :-
    employee(Id, Name, Dept),
    years_of_service(Id, Years),
    Years > 5.
```

### Aggregation Pattern (via counting)

```n3
# Count using recursive helper
rule count_employees(Dept, Count) :-
    findall(Id, employee(Id, _, Dept), Employees),
    length(Employees, Count).
```

## Debugging Tips

### 1. Test rules incrementally

Start with simple goals and add complexity:

```n3
# Test direct facts first
goal employee(1, Name, Dept)

# Then test simple rules
goal engineer(Id, Name)

# Finally test complex recursive rules
goal ancestor(alice, X)
```

### 2. Use ground queries for verification

```n3
# Check if specific fact/relationship holds
goal parent(alice, bob)  # Should succeed
goal parent(alice, dave)  # Should fail
```

### 3. Limit result sets during development

```n3
query test_query:
    from knowledge family:
    goal ancestor(X, Y)
    variables: X, Y
    limit: 10  # Prevent runaway queries during development
```

### 4. Check validation errors

The validator catches common mistakes:

- Arity inconsistencies
- Unsafe variables
- Singleton variables
- Undefined predicates
- Undefined knowledge modules

## Next Steps

- See [QUERIES.md](QUERIES.md) for query syntax and execution
- See examples in `examples/` directory
- Check existing tests in `tests/test_logic_*.py`

## References

- Robinson, J. A. (1965). "A Machine-Oriented Logic Based on the Resolution Principle"
- Clocksin, W. F., & Mellish, C. S. (2003). "Programming in Prolog"
- Sterling, L., & Shapiro, E. (1994). "The Art of Prolog"
