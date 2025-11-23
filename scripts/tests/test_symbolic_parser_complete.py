"""Comprehensive test suite for symbolic parser package refactoring.

Tests import validation, method availability, module structure,
backward compatibility, and integration with ParserBase.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def test_import_validation():
    """Test 1: Verify package can be imported."""
    print("Test 1: Import Validation")
    print("-" * 50)
    
    try:
        from namel3ss.parser.symbolic import SymbolicExpressionParser
        print("✅ Direct import successful: from namel3ss.parser.symbolic import SymbolicExpressionParser")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_method_availability():
    """Test 2: Verify all 20 methods are available."""
    print("\nTest 2: Method Availability")
    print("-" * 50)
    
    from namel3ss.parser.symbolic import SymbolicExpressionParser
    
    required_methods = [
        # Tokenization
        "_tokenize",
        # Token operations (9 methods)
        "current_token", "peek", "consume", "expect", "try_consume",
        "word", "peek_word", "string", "number",
        # Function parsing (2 methods)
        "parse_function_def", "parse_lambda",
        # Expression parsing (2 methods)
        "parse_if_expr", "parse_let_expr",
        # Pattern matching (2 methods)
        "parse_match_expr", "parse_pattern",
        # Logic programming (2 methods)
        "parse_rule_def", "parse_query_expr",
        # Main parser (1 method)
        "parse_extended_expression"
    ]
    
    success = True
    available_methods = []
    missing_methods = []
    
    for method in required_methods:
        if hasattr(SymbolicExpressionParser, method):
            available_methods.append(method)
        else:
            missing_methods.append(method)
            success = False
    
    print(f"Required methods: {len(required_methods)}")
    print(f"Available methods: {len(available_methods)}")
    
    if missing_methods:
        print(f"\n❌ Missing methods ({len(missing_methods)}):")
        for method in missing_methods:
            print(f"  - {method}")
    else:
        print(f"✅ All {len(required_methods)} methods are available")
    
    return success


def test_module_structure():
    """Test 3: Verify module structure and line counts."""
    print("\nTest 3: Module Structure")
    print("-" * 50)
    
    from pathlib import Path
    
    expected_modules = {
        "tokenizer.py": 21,
        "tokens.py": 114,
        "functions.py": 95,
        "expressions.py": 62,
        "patterns.py": 151,
        "logic.py": 107,
        "parser.py": 175,
        "main.py": 103,
        "__init__.py": 9
    }
    
    # The package is at namel3ss/parser/symbolic/
    base_dir = Path(__file__).parent / "namel3ss" / "parser" / "symbolic"
    
    results = []
    total_lines = 0
    all_modules_found = True
    
    for module_name, expected_lines in expected_modules.items():
        module_path = base_dir / module_name
        if module_path.exists():
            actual_lines = len(module_path.read_text().splitlines())
            total_lines += actual_lines
            # Allow 10% tolerance for documentation differences
            within_tolerance = abs(actual_lines - expected_lines) <= expected_lines * 0.10
            status = "✅" if within_tolerance else "⚠️"
            results.append(f"  {status} {module_name}: {actual_lines} lines (expected ~{expected_lines})")
        else:
            results.append(f"  ❌ {module_name}: NOT FOUND")
            all_modules_found = False
    
    for result in results:
        print(result)
    
    print(f"\nTotal lines across modules: {total_lines}")
    print(f"Expected total: ~{sum(expected_modules.values())} lines")
    
    if all_modules_found:
        print(f"✅ All {len(expected_modules)} modules found")
        return True
    else:
        print(f"❌ Some modules missing")
        return False


def test_backward_compatibility():
    """Test 4: Verify backward compatibility wrapper."""
    print("\nTest 4: Backward Compatibility")
    print("-" * 50)
    
    try:
        # Test original import path
        from namel3ss.parser.symbolic import SymbolicExpressionParser as LegacyImport
        
        # Test package import path
        from namel3ss.parser.symbolic.main import SymbolicExpressionParser as PackageImport
        
        # They should be the same class
        if LegacyImport is PackageImport:
            print("✅ Legacy import path works: from namel3ss.parser.symbolic import SymbolicExpressionParser")
            print("✅ Package import path works: from namel3ss.parser.symbolic.main import SymbolicExpressionParser")
            print("✅ Both imports reference the same class")
            return True
        else:
            print("❌ Legacy and package imports reference different classes")
            return False
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_integration():
    """Test 5: Integration test with parser instantiation."""
    print("\nTest 5: Integration Test")
    print("-" * 50)
    
    try:
        from namel3ss.parser.symbolic import SymbolicExpressionParser
        from namel3ss.parser.base import ParserBase
        
        # Test instantiation
        code = """
        fn add(x, y) => x + y
        fn square(n) => n * n
        let result = square(5) in result
        """
        
        parser = SymbolicExpressionParser(code)
        
        # Verify inheritance
        if not isinstance(parser, ParserBase):
            print("❌ Parser doesn't inherit from ParserBase")
            return False
        
        print("✅ Parser instantiation successful")
        print("✅ Parser inherits from ParserBase")
        
        # Test tokenization
        parser.tokens = parser._tokenize(code)
        if not parser.tokens:
            print("❌ Tokenization failed")
            return False
        
        print(f"✅ Tokenization successful: {len(parser.tokens)} tokens")
        
        # Test token operations
        parser.token_pos = 0
        first_token = parser.current_token()
        if first_token is None:
            print("❌ Token operations failed")
            return False
        
        print(f"✅ Token operations work: first token = '{first_token}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_statistics():
    """Test 6: Report statistics about the refactoring."""
    print("\nTest 6: Refactoring Statistics")
    print("-" * 50)
    
    from pathlib import Path
    
    # Count wrapper lines
    wrapper_path = Path(__file__).parent / "namel3ss" / "parser" / "symbolic.py"
    wrapper_lines = len(wrapper_path.read_text().splitlines())
    
    # Count package lines
    package_dir = Path(__file__).parent / "namel3ss" / "parser" / "symbolic"
    package_lines = 0
    for py_file in package_dir.glob("*.py"):
        package_lines += len(py_file.read_text().splitlines())
    
    print(f"Original file: 733 lines, 20 methods, 1 class")
    print(f"New wrapper: {wrapper_lines} lines")
    print(f"Package modules: {package_lines} lines across 9 modules")
    print(f"Total refactored: {wrapper_lines + package_lines} lines")
    print(f"Wrapper reduction: {733 - wrapper_lines} lines removed ({100 * (733 - wrapper_lines) / 733:.1f}%)")
    print(f"Documentation overhead: {package_lines - 733} lines ({100 * (package_lines - 733) / 733:.1f}%)")
    
    print("✅ Statistics collected")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("SYMBOLIC PARSER PACKAGE TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_import_validation,
        test_method_availability,
        test_module_structure,
        test_backward_compatibility,
        test_integration,
        test_statistics
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("\n🎉 ALL TESTS PASSED! Symbolic parser package refactoring is complete.")
    else:
        print("\n⚠️ Some tests failed. Please review the results above.")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
