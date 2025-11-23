# Queries in Namel3ss

## Overview

Queries allow you to ask questions against your knowledge base and retrieve results. They support:

- **Multiple knowledge sources**: Combine facts from multiple modules
- **Dataset adapters**: Query external data as if it were facts
- **Variable projection**: Specify which variables to return
- **Result limiting**: Control how many solutions to generate
- **HTTP API generation**: Automatic REST endpoints for queries

## Query Syntax

```n3
query <query_name>:
    from knowledge <module_name>:
    [from dataset <dataset_name>:]
    goal <term>
    [goal <term>]...
    variables: <var1>, <var2>, ...
    [limit: <number>]
```

## Simple Queries

### Basic Fact Retrieval

```n3
app "EmployeeDB".

knowledge employees:
    fact employee(1, "Alice", "Engineering").
    fact employee(2, "Bob", "Sales").
    fact employee(3, "Charlie", "Engineering").

query all_employees:
    from knowledge employees:
    goal employee(Id, Name, Dept)
    variables: Id, Name, Dept
```

**Results:**
```json
[
    {"Id": 1, "Name": "Alice", "Dept": "Engineering"},
    {"Id": 2, "Name": "Bob", "Dept": "Sales"},
    {"Id": 3, "Name": "Charlie", "Dept": "Engineering"}
]
```

### Filtering with Ground Terms

```n3
query engineers_only:
    from knowledge employees:
    goal employee(Id, Name, "Engineering")
    variables: Id, Name
```

**Results:**
```json
[
    {"Id": 1, "Name": "Alice"},
    {"Id": 3, "Name": "Charlie"}
]
```

## Multiple Goals (Joins)

Queries can have multiple goals, creating natural joins:

```n3
knowledge company:
    fact employee(1, "Alice", "Engineering").
    fact employee(2, "Bob", "Sales").
    fact salary(1, 100000).
    fact salary(2, 80000).

query employee_salaries:
    from knowledge company:
    goal employee(Id, Name, Dept)
    goal salary(Id, Amount)
    variables: Id, Name, Dept, Amount
```

**Results:**
```json
[
    {"Id": 1, "Name": "Alice", "Dept": "Engineering", "Amount": 100000},
    {"Id": 2, "Name": "Bob", "Dept": "Sales", "Amount": 80000}
]
```

## Querying with Rules

Queries can invoke rules, triggering inference:

```n3
knowledge family:
    fact parent(alice, bob).
    fact parent(bob, charlie).
    fact parent(charlie, dave).
    
    rule ancestor(X, Y) :- parent(X, Y).
    rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

query descendants_of_alice:
    from knowledge family:
    goal ancestor(alice, Descendant)
    variables: Descendant
```

**Results:**
```json
[
    {"Descendant": "bob"},
    {"Descendant": "charlie"},
    {"Descendant": "dave"}
]
```

## Multiple Knowledge Sources

Queries can combine facts from multiple knowledge modules:

```n3
knowledge employees:
    fact employee(1, "Alice").
    fact employee(2, "Bob").

knowledge departments:
    fact dept_assignment(1, "Engineering").
    fact dept_assignment(2, "Sales").

query employee_departments:
    from knowledge employees:
    from knowledge departments:
    goal employee(Id, Name)
    goal dept_assignment(Id, Dept)
    variables: Id, Name, Dept
```

## Dataset Adapters

Queries can access external datasets as if they were facts:

```n3
query employees_with_salary:
    from knowledge employees:
    from dataset salaries:
    goal employee(Id, Name)
    goal row_salaries(Id, Amount)
    variables: Id, Name, Amount
```

### Dataset Adapter Facts

Datasets expose two types of predicates:

1. **Row facts**: `row_<dataset>(Index, ...columns)`
2. **Field facts**: `field_<dataset>(Index, FieldName, Value)`

Example with a salaries dataset:

```python
# Python side: Define dataset
class SalariesDataset:
    def get_rows(self):
        return [
            {"id": 1, "name": "Alice", "amount": 100000},
            {"id": 2, "name": "Bob", "amount": 80000},
        ]

# Automatically creates facts:
# row_salaries(0, 1, "Alice", 100000)
# row_salaries(1, 2, "Bob", 80000)
# field_salaries(0, "id", 1)
# field_salaries(0, "name", "Alice")
# field_salaries(0, "amount", 100000)
# ...
```

## Variable Projection

Only variables listed in the `variables` clause are returned:

```n3
query manager_names:
    from knowledge company:
    goal employee(ManagerId, ManagerName, _)
    goal reports_to(EmployeeId, ManagerId)
    goal employee(EmployeeId, EmployeeName, _)
    variables: ManagerName, EmployeeName
    # ManagerId and EmployeeId are not returned
```

## Result Limiting

Control how many solutions are generated:

```n3
query first_five_ancestors:
    from knowledge family:
    goal ancestor(alice, X)
    variables: X
    limit: 5
```

This is useful for:

- **Development**: Test queries without generating huge result sets
- **Performance**: Avoid expensive exhaustive searches
- **UI pagination**: Implement "show more" functionality

## Query Execution

### Programmatic Execution

```python
from namel3ss.codegen.backend.core.runtime.query_compiler import (
    QueryCompiler, QueryContext
)

# Setup context with knowledge modules
context = QueryContext()
context.add_knowledge_module(knowledge_module)

# Compile query
compiler = QueryCompiler(context)
compiled_query = compiler.compile(query_ast)

# Execute and get all results
results = list(compiled_query.execute_all())

# Or execute with limit
results = list(compiled_query.execute(limit=10))
```

### HTTP API

Queries automatically generate FastAPI endpoints:

```n3
query find_employees:
    from knowledge employees:
    goal employee(Id, Name, Dept)
    variables: Id, Name, Dept
```

Generates endpoints:

```
GET  /queries/              # List all queries
POST /queries/find_employees/execute  # Execute query
```

Example request:

```bash
curl -X POST http://localhost:8000/queries/find_employees/execute \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {},
    "limit": 10
  }'
```

Example response:

```json
{
  "results": [
    {"Id": 1, "Name": "Alice", "Dept": "Engineering"},
    {"Id": 2, "Name": "Bob", "Dept": "Sales"}
  ],
  "count": 2,
  "limited": false
}
```

## Advanced Patterns

### Parameterized Queries (via dataset adapters)

While queries don't have direct parameters, you can use dataset adapters to inject runtime values:

```python
# Python side: Create adapter with current user context
class UserContextAdapter(LogicAdapter):
    def __init__(self, current_user_id):
        self.current_user_id = current_user_id
    
    def get_facts(self):
        yield LogicFact(head=LogicStruct(
            functor="current_user",
            args=[LogicNumber(value=self.current_user_id)]
        ))

# Add to context before query execution
context.add_adapter(UserContextAdapter(current_user_id=42))
```

```n3
query my_reports:
    from knowledge employees:
    from dataset context:
    goal current_user(UserId)
    goal reports_to(ReportId, UserId)
    goal employee(ReportId, Name, Dept)
    variables: Name, Dept
```

### Recursive Queries

Queries automatically handle recursive rules:

```n3
knowledge graph:
    fact edge(a, b).
    fact edge(b, c).
    fact edge(c, d).
    
    rule path(X, Y) :- edge(X, Y).
    rule path(X, Z) :- edge(X, Y), path(Y, Z).

query all_paths:
    from knowledge graph:
    goal path(From, To)
    variables: From, To
    limit: 100  # Prevent exponential explosion
```

### Existential Queries

Check if any solution exists:

```n3
query has_manager:
    from knowledge employees:
    goal reports_to(EmployeeId, ManagerId)
    variables: EmployeeId
    limit: 1  # Just need one result
```

```python
results = list(compiled_query.execute(limit=1))
has_manager = len(results) > 0
```

### Counting (via result length)

```python
# Count all solutions
results = list(compiled_query.execute_all())
count = len(results)
```

## Performance Considerations

### 1. Use Selective Goals First

Place more selective goals early to prune the search space:

```n3
# ✓ GOOD: Filter by department first (selective)
query engineers:
    goal employee(Id, Name, "Engineering")
    goal salary(Id, Amount)

# ✗ LESS EFFICIENT: Enumerate all employees first
query engineers_slow:
    goal salary(Id, Amount)
    goal employee(Id, Name, "Engineering")
```

### 2. Limit Recursive Queries

Always set limits on potentially recursive queries:

```n3
query transitive_closure:
    from knowledge graph:
    goal path(X, Y)
    variables: X, Y
    limit: 1000  # Prevent exponential blowup
```

### 3. Index Common Queries

For frequently accessed patterns, consider materializing results:

```n3
# Instead of querying ancestor repeatedly, materialize it:
knowledge cached:
    fact ancestor_cached(alice, bob).
    fact ancestor_cached(alice, charlie).
    # ... precomputed results
```

### 4. Monitor Query Complexity

Track query execution time and step counts:

```python
import time

start = time.time()
results = list(compiled_query.execute_all())
elapsed = time.time() - start

print(f"Found {len(results)} results in {elapsed:.3f}s")
```

## Validation

Queries are validated at parse time:

- **Undefined variables**: Variables in `variables` clause must appear in goals
- **Undefined knowledge modules**: Referenced modules must exist
- **Undefined predicates**: Goals must reference defined predicates or facts

## Debugging Queries

### 1. Test goals incrementally

```n3
# Start simple
query test1:
    goal employee(Id, Name, Dept)
    variables: Id, Name, Dept

# Add join
query test2:
    goal employee(Id, Name, Dept)
    goal salary(Id, Amount)
    variables: Id, Name, Amount

# Add filtering
query test3:
    goal employee(Id, Name, "Engineering")
    goal salary(Id, Amount)
    variables: Id, Name, Amount
```

### 2. Use small limits during development

```n3
query debug_query:
    from knowledge large_dataset:
    goal some_predicate(X, Y)
    variables: X, Y
    limit: 5  # Remove after debugging
```

### 3. Check intermediate bindings

```python
# Execute step by step
for result in compiled_query.execute_all():
    print(f"Solution: {result}")
    if len(result) > 10:  # Stop after 10 solutions
        break
```

## Examples

See these files for complete examples:

- `examples/rag_demo.ai` - Retrieval-augmented generation with logic
- `test_full_integration.py` - Complete integration pipeline
- `test_end_to_end_queries.py` - Query compilation and execution
- `tests/test_logic_integration.py` - Complex query patterns

## Next Steps

- See [LOGIC.md](LOGIC.md) for knowledge module syntax
- See [llm-provider-guide.md](llm-provider-guide.md) for AI integration
- Check the `namel3ss/codegen/backend/core/runtime/` directory for implementation details

## References

- Datalog: Declarative query language for deductive databases
- Prolog: Logic programming with SLD resolution
- SQL: Relational query model (for comparison)
