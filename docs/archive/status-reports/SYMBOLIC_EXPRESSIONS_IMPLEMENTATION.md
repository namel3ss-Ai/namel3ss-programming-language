# Symbolic Programming Enhancement for Namel3ss

## Implementation Status

### ✅ Completed

#### 1. Core Expression AST (`namel3ss/ast/expressions.py`)
- **LiteralExpr, VarExpr, CallExpr**: Basic expression nodes
- **LambdaExpr**: Anonymous functions with parameters and body
- **IfExpr**: Conditional expressions
- **LetExpr**: Local bindings
- **ListExpr, DictExpr, TupleExpr**: Collection literals
- **IndexExpr, SliceExpr**: Indexing and slicing operations
- **MatchExpr, MatchCase**: Pattern matching with guards
- **Patterns**: LiteralPattern, VarPattern, ListPattern, DictPattern, TuplePattern, ConstructorPattern, WildcardPattern
- **FunctionDef, Parameter**: Named function definitions
- **RuleDef, RuleHead, RuleBody, RuleClause**: Prolog-style rules
- **QueryExpr**: Rule queries
- **UnifyExpr**: Explicit unification operator

All nodes are immutable dataclasses integrated into the AST and exported via `namel3ss/ast/__init__.py`.

#### 2. Parser (`namel3ss/parser/symbolic.py`)
Created `SymbolicExpressionParser` with methods for:
- Function definitions: `fn name(params) { body }` or `fn name(params) => expr`
- Lambda expressions: `fn(params) => expr`
- If expressions: `if cond then expr else expr`
- Let bindings: `let x = val in body`
- Pattern matching: `match expr { case pat => body, ... }`
- Rules: `rule pred(args) :- body.` or `rule pred(args).` (facts)
- Queries: `query pred(args)` or `query pred(args) limit n`
- Collections: `[1, 2, 3]`, `{a: 1, b: 2}`, `(1, 2, 3)`
- Indexing/slicing: `list[0]`, `list[1:3]`
- Unification: `x ~ y`

### 🚧 In Progress / Pending

#### 3. Symbolic Evaluator (Priority: HIGH)
**File**: `namel3ss/codegen/backend/core/runtime/expression_sandbox.py`

**Required Changes**:
1. Extend `SandboxedExpressionEvaluator` to handle new AST nodes
2. Add evaluation methods:
   - `eval_literal_expr`, `eval_var_expr`, `eval_call_expr`
   - `eval_lambda_expr`, `eval_if_expr`, `eval_let_expr`
   - `eval_list_expr`, `eval_dict_expr`, `eval_tuple_expr`
   - `eval_index_expr`, `eval_slice_expr`
   - `eval_match_expr` (with pattern matching engine)
   - `eval_function_def` (store in environment)
   - `eval_query_expr` (query rule engine)

3. Add recursion safety:
   ```python
   MAX_RECURSION_DEPTH = 100
   MAX_EVAL_STEPS = 10000
   ```

4. Track evaluation state:
   ```python
   class EvalState:
       step_count: int
       recursion_depth: int
       function_env: Dict[str, FunctionDef]
       rule_db: RuleDatabase
   ```

#### 4. Pattern Matching Engine (Priority: HIGH)
**File**: `namel3ss/codegen/backend/core/runtime/pattern_matching.py` (new)

**Required Implementation**:
```python
def match_pattern(pattern: Pattern, value: Any, bindings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Attempt to match pattern against value.
    Returns updated bindings dict on success, None on failure.
    """
    # Implement structural matching for:
    # - Literals (exact equality)
    # - Variables (bind to any value)
    # - Lists (recursive matching with rest patterns)
    # - Dicts (key-based matching with rest patterns)
    # - Tuples (fixed-arity matching)
    # - Constructors (symbolic terms)
    # - Wildcards (match anything, no binding)
```

#### 5. Rule Engine (Priority: MEDIUM)
**File**: `namel3ss/codegen/backend/core/runtime/rule_engine.py` (new)

**Required Implementation**:
```python
class RuleDatabase:
    def __init__(self, rules: List[RuleDef], max_depth: int = 50):
        self.rules = rules
        self.max_depth = max_depth
    
    def query(self, predicate: str, args: List[Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query rules using unification and backtracking.
        Returns list of variable bindings for each solution.
        """
        # Implement:
        # 1. Unification algorithm
        # 2. Backtracking search with depth limit
        # 3. Negation-as-failure for 'not' clauses
        # 4. Guard evaluation
```

#### 6. Resolver Integration (Priority: MEDIUM)
**File**: `namel3ss/resolver.py`

**Required Changes**:
1. Add scope tracking for function parameters and let bindings
2. Validate:
   - No use of undefined variables
   - Function arity matches on calls
   - Rule head/body variable consistency
   - Pattern exhaustiveness (warn if no catch-all case)
3. Optional type inference/checking

#### 7. Integration with Existing Features (Priority: HIGH)
**Files**:
- `namel3ss/codegen/backend/core/runtime/datasets.py`
- `namel3ss/codegen/backend/core/runtime/pages.py`
- Other runtime modules using expressions

**Required Changes**:
Replace string-based expression evaluation with symbolic evaluator:
```python
# Before:
result = eval(filter_expression, scope)

# After:
from namel3ss.ast.expressions import parse_expression
from namel3ss.codegen.backend.core.runtime.expression_sandbox import evaluate_expression_tree

ast = parse_expression(filter_expression)
result = evaluate_expression_tree(ast, scope, limits={
    'max_recursion': 100,
    'max_steps': 10000
})
```

#### 8. Built-in Library Functions (Priority: HIGH)
**File**: `namel3ss/codegen/backend/core/runtime/builtins.py` (new)

**Required Functions**:
```python
BUILTIN_FUNCTIONS = {
    # Higher-order functions
    'map': lambda func, seq: [func(x) for x in seq],
    'filter': lambda pred, seq: [x for x in seq if pred(x)],
    'reduce': lambda func, seq, init: functools.reduce(func, seq, init),
    'fold': ...,  # Right fold
    
    # List operations
    'head': lambda xs: xs[0] if xs else None,
    'tail': lambda xs: xs[1:] if xs else [],
    'cons': lambda x, xs: [x] + xs,
    'append': lambda xs, ys: xs + ys,
    'reverse': lambda xs: list(reversed(xs)),
    'sort': lambda xs: sorted(xs),
    'length': len,
    'nth': lambda xs, n: xs[n] if 0 <= n < len(xs) else None,
    
    # Numeric
    'sum': sum,
    'product': lambda xs: functools.reduce(operator.mul, xs, 1),
    'min': min,
    'max': max,
    'range': range,
    
    # String operations
    'concat': lambda *strs: ''.join(str(s) for s in strs),
    'split': str.split,
    'join': lambda sep, xs: sep.join(str(x) for x in xs),
    
    # Utilities
    'print': print,
    'assert': lambda cond, msg='': ... if cond else raise_error(msg),
}
```

#### 9. Tests (Priority: HIGH)
**File**: `tests/test_symbolic_expressions.py` (new)

**Test Coverage Needed**:
1. **Expression Evaluation**:
   - Literals, variables, operators
   - Lists, dicts, tuples
   - Indexing and slicing
   
2. **Functions**:
   - Named functions
   - Lambdas
   - Recursion (factorial, fibonacci, list operations)
   - Higher-order functions (map, filter, reduce)
   - Closures and lexical scoping
   
3. **Pattern Matching**:
   - Literal matching
   - Variable binding
   - List/dict/tuple destructuring
   - Guards
   - Nested patterns
   
4. **Rules and Queries**:
   - Facts and simple rules
   - Recursive rules (ancestors, paths)
   - Negation-as-failure
   - Multiple solutions
   
5. **Safety Limits**:
   - Max recursion depth exceeded
   - Max evaluation steps exceeded
   - Proper error messages
   
6. **Integration**:
   - Dataset filters using new expressions
   - AI prompt arguments
   - Computed columns
   - Insight conditions

#### 10. Documentation (Priority: MEDIUM)
**File**: `docs/EXPRESSION_LANGUAGE.md` (new)

**Required Sections**:
1. Overview and motivation
2. Syntax reference:
   - Functions and lambdas
   - Pattern matching
   - Rules and queries
3. Built-in functions
4. Safety constraints and limits
5. Integration with N3 DSL
6. Examples:
   - Data transformations
   - Business rules
   - Symbolic reasoning
   
**File**: `docs/PATTERN_MATCHING.md` (new)
- Detailed pattern syntax
- Matching semantics
- Examples

**File**: `docs/RULE_ENGINE.md` (new)
- Rule syntax
- Query semantics
- Unification and backtracking
- Use cases

## Architecture Decisions

### 1. Safety First
- **No eval()**: All evaluation through AST interpretation
- **Bounded execution**: Configurable limits on recursion and steps
- **Deterministic errors**: Clear messages when limits exceeded
- **No side effects in rules**: Pure logic only

### 2. Backward Compatibility
- Existing `.ai` files work unchanged
- Old string-based expressions gradually migrated
- New syntax is opt-in

### 3. Integration Points
The expression system plugs into:
- **Dataset filters**: `filter by: complex_function(row)`
- **Computed columns**: `column result = map(items, transform)`
- **AI prompts**: `prompt with args: {result: compute_summary(data)}`
- **Insight conditions**: `if match data { case pattern => true, else => false }`
- **Page logic**: Any expression-valued field

### 4. Configuration
Runtime settings (add to `runtime.py`):
```python
EXPRESSION_LIMITS = {
    'max_recursion_depth': os.getenv('N3_EXPR_MAX_DEPTH', 100),
    'max_evaluation_steps': os.getenv('N3_EXPR_MAX_STEPS', 10000),
    'enable_rules': os.getenv('N3_ENABLE_RULES', '1') == '1',
    'rule_query_limit': os.getenv('N3_RULE_QUERY_LIMIT', 1000),
}
```

## Next Steps (Priority Order)

1. **Implement symbolic evaluator** with recursion/step limits
2. **Implement pattern matching engine**
3. **Add built-in function library**
4. **Wire evaluator to dataset filters** (key integration point)
5. **Implement basic rule engine**
6. **Write comprehensive tests**
7. **Update resolver** for scope validation
8. **Write documentation**
9. **Wire evaluator to other expression points** (AI prompts, insights, etc.)
10. **Performance optimization** (if needed)

## File Structure

```
namel3ss/
├── ast/
│   ├── expressions.py          [✅ Complete]
│   └── __init__.py             [✅ Updated]
├── parser/
│   ├── symbolic.py             [✅ Complete]
│   └── expressions.py          [Needs integration]
├── codegen/backend/core/runtime/
│   ├── expression_sandbox.py   [🚧 Needs extension]
│   ├── pattern_matching.py     [❌ To create]
│   ├── rule_engine.py          [❌ To create]
│   ├── builtins.py             [❌ To create]
│   └── datasets.py             [🚧 Needs integration]
├── resolver.py                 [🚧 Needs extension]
└── tests/
    └── test_symbolic_expressions.py  [❌ To create]
docs/
├── EXPRESSION_LANGUAGE.md      [❌ To create]
├── PATTERN_MATCHING.md         [❌ To create]
└── RULE_ENGINE.md              [❌ To create]
```

## Example Usage (After Full Implementation)

### Dataset with Complex Filter
```n3
dataset "qualified_leads" from table leads:
  let score = fn(lead) => (
    lead.revenue * 0.5 + lead.engagement * 0.3 + lead.fit * 0.2
  )
  filter by: score(row) > 75 and match row.status {
    case "active" => true,
    case "pending" if row.days_since_contact < 7 => true,
    else => false
  }
```

### Rule-Based Business Logic
```n3
rule eligible_for_discount(Customer, Discount) :-
  customer_tier(Customer, "gold"),
  Discount = 0.2.

rule eligible_for_discount(Customer, Discount) :-
  customer_tier(Customer, "silver"),
  order_count(Customer, Count),
  Count > 10,
  Discount = 0.15.

insight "Discount Eligibility":
  let customers = query eligible_for_discount(C, D)
  show "Found {length(customers)} eligible customers"
```

### Higher-Order Transformations
```n3
dataset "processed_orders" from table orders:
  let enrich = fn(order) => {
    ...order,
    total_with_tax: order.subtotal * 1.1,
    priority: if order.value > 1000 then "high" else "normal"
  }
  
  let result = map(rows, enrich)
  let sorted = sort(result, fn(a, b) => a.total_with_tax > b.total_with_tax)
  
  computed column "enriched" = sorted
```

## Conclusion

The AST and parser foundation is complete. The main work remaining is:
1. Evaluator implementation (critical path)
2. Pattern matching and rule engine
3. Integration with existing runtime
4. Comprehensive testing

This design provides a powerful, safe, and composable symbolic programming layer that transforms N3 from a declarative DSL into a full-featured AI programming language while maintaining all existing functionality.
