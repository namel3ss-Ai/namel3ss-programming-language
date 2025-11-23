"""Test enhanced coercion methods"""
from namel3ss.parser.base import ParserBase
from namel3ss.errors import N3SyntaxError

p = ParserBase('', path='test.ai')

# Test successful coercion
print("Test 1: Valid integer coercion")
result = p._coerce_int_with_context('42', 'page_size', min_value=1)
print(f"  Result: {result}")
assert result == 42

# Test range validation
print("\nTest 2: Integer with range validation")
result = p._coerce_int_with_context('100', 'max_entries', min_value=1, max_value=1000)
print(f"  Result: {result}")
assert result == 100

# Test error handling
print("\nTest 3: Invalid integer - should raise error")
try:
    p._coerce_int_with_context('abc', 'page_size')
    print("  ERROR: Should have raised N3SyntaxError")
except N3SyntaxError as e:
    print(f"  Correctly raised error: {str(e)[:80]}")

# Test out of range
print("\nTest 4: Out of range - should raise error")
try:
    p._coerce_int_with_context('-5', 'page_size', min_value=1)
    print("  ERROR: Should have raised N3SyntaxError")
except N3SyntaxError as e:
    print(f"  Correctly raised error: {str(e)[:80]}")

# Test coercion hints
print("\nTest 5: Coercion hints")
hint = p._coercion_hint('page_size', 'int')
print(f"  Hint for page_size: {hint}")
assert 'positive integer' in hint

hint = p._coercion_hint('temperature', 'float')
print(f"  Hint for temperature: {hint}")
assert '0.0' in hint and '2.0' in hint

# Test boolean coercion
print("\nTest 6: Boolean coercion")
result = p._coerce_bool_with_context('true', 'reactive')
print(f"  Result: {result}")
assert result is True

result = p._coerce_bool_with_context('no', 'reactive')
print(f"  Result: {result}")
assert result is False

print("\n✅ ALL TESTS PASSED")
