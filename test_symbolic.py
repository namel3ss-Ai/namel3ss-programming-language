#!/usr/bin/env python3
"""Test symbolic expression parsing in isolation."""

from namel3ss.parser.symbolic import SymbolicExpressionParser

# Test case from simple_functional.n3
test_expr = "fn(user) => user.active == true and user.last_login_days < 30"

print(f"Testing: {test_expr}")
print(f"Length: {len(test_expr)}")

parser = SymbolicExpressionParser(test_expr)
print(f"Tokens: {parser.tokens}")
print(f"Token count: {len(parser.tokens)}")
print(f"Token pos before parse: {parser.token_pos}")

try:
    result = parser.parse_extended_expression()
    print(f"Token pos after parse: {parser.token_pos}")
    print(f"✅ Success: {result}")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
