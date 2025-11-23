# Namel3ss Control Flow Implementation Summary

## Overview
Successfully extended the Namel3ss (N3) language with if/else conditionals and for loops, enabling conditional logic and iteration over datasets and tables.

## Language Syntax Added

### If/Else Blocks
```n3
if <condition>:
  <statements>
else:
  <statements>
```

Example:
```n3
if user.role == "admin":
  show text "Welcome, admin!"
else:
  show text "Access denied."
```

### For Loops
```n3
for <variable> in dataset <dataset_name>:
  <statements>

for <variable> in table <table_name>:
  <statements>
```

Example:
```n3
for order in dataset latest_orders:
  show text "{order.id} – {order.total}"

for row in table orders:
  show text "{row.id} – {row.status}"
```

## Implementation Details

### 1. AST Changes (`namel3ss/ast.py`)

**New Classes:**
- `Expression`: Captures condition expressions as raw strings (e.g., `user.role == "admin"`)
- `IfBlock`: Represents if/else conditional blocks with condition, body, and optional else_body
- `ForLoop`: Represents for loops with loop_var, source_kind (dataset/table), source_name, and body

**Updated Type Alias:**
```python
PageStatement = Union[ShowText, ShowTable, ShowChart, ShowForm, Action, IfBlock, ForLoop]
```

### 2. Parser Changes (`namel3ss/parser.py`)

**New Methods:**
- `_parse_if_block()`: Parses if/else blocks with proper indentation handling
  - Extracts condition from `if <condition>:`
  - Parses indented body statements
  - Optionally parses `else:` clause at same indentation level
  - Validates at least one statement in body

- `_parse_for_loop()`: Parses for loop constructs
  - Matches pattern: `for <var> in dataset|table <name>:`
  - Extracts loop variable, source kind, and source name
  - Parses indented loop body statements
  - Validates at least one statement in body

**Updated Method:**
- `_parse_page_statement()`: Now dispatches to `_parse_if_block()` and `_parse_for_loop()` when encountering `if` or `for` keywords

**Error Handling:**
- All parsing uses existing `self._error()` mechanism
- Line numbers and raw line content included in error messages
- Validates syntax (trailing colons, proper indentation, non-empty bodies)

### 3. Code Generation Changes (`namel3ss/codegen/frontend.py`)

**New Helper Function:**
- `_render_statements()`: Recursive statement renderer that handles all PageStatement types including nested IfBlock and ForLoop

**Rendering Behavior:**

**IfBlock Rendering:**
- Displays both if and else branches in static HTML for preview purposes
- Uses colored borders (green for if, orange for else) to distinguish branches
- Shows condition text as label: "If <condition>:"
- Recursively renders nested statements

**ForLoop Rendering:**
- Shows loop header: "For each <var> in <source_kind> <source_name>:"
- Renders loop body 3 times as demonstration iterations
- Uses blue border to distinguish loop blocks
- Each iteration is labeled for clarity
- Recursively renders nested statements

**Updated Function:**
- `_generate_page_html()`: Refactored to use `_render_statements()` helper with counters dictionary

### 4. Test Coverage (`tests/parser/test_pages.py`)

The parser control-flow coverage now lives alongside the rest of the page tests:

1. `test_parse_if_block()` – basic if block parsing
2. `test_parse_if_else_block()` – if/else handling
3. `test_parse_if_with_multiple_statements()` – mixed statement bodies
4. `test_parse_for_loop_dataset()` – dataset-backed loops
5. `test_parse_for_loop_table()` – table-backed loops
6. `test_parse_nested_if_in_for()` – nested control flow
7. `test_parse_mixed_statements()` – interleaving control flow with regular statements
8. `test_parse_page_reactive_and_predict_statement()` – reactive metadata and predictions
9. `test_parse_variable_assignment()` – page-level variables
10. `test_parse_chart_layout_metadata()` – chart layout parsing

**Test Results:**
```
pytest tests/parser/test_pages.py
12 passed in 0.05s
```
The full suite (`python3 -m pytest`) continues to pass, ensuring backward compatibility.

## Example Usage

See `examples/control_flow_demo.ai` for a comprehensive demonstration including:
- Role-based conditional access
- Iteration over datasets and tables
- Nested control flow (if inside for)
- Mixed statements (control flow + regular statements)

## Key Features

✅ **Indentation-aware parsing**: Properly handles Python-like indentation for block structure
✅ **Nested control flow**: Supports if blocks inside for loops and vice versa
✅ **Multiple statement types**: Any PageStatement can be used inside control flow blocks
✅ **Error handling**: Clear syntax errors with line numbers and context
✅ **Frontend preview**: Static HTML generation shows control flow structure
✅ **Backward compatible**: All existing tests pass without modification
✅ **Extensible**: Simple Expression type can be extended for more complex conditions

## Future Enhancements

Potential improvements for future iterations:
- Parse expressions into structured AST (not just raw strings)
- Support for `elif` clauses
- While loops
- Break/continue statements
- Dynamic evaluation in frontend (currently shows both branches)
- Backend route generation for conditional data fetching
- Support for nested loops with different sources
- Variable scoping and interpolation in loop bodies
- **4. Real-Time / Reactive Layer**
  - Introduce FastAPI WebSocket endpoints `/ws/pages/{slug}` broadcasting dataset/insight updates.
  - Build frontend subscription runtime for live updates and optimistic UI states with rollback.
  - Extend syntax (`page ... reactive`, auto-refresh directives) and CLI (`namel3ss run ... --realtime`).
  - Add tests covering broadcast flow and optimistic rollback.
