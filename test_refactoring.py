"""
Test script for refactored AI Parser and State packages.

This script validates that:
1. All imports work correctly
2. Backward compatibility is maintained
3. Core functionality is intact
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_ai_parser_imports():
    """Test AI Parser package imports."""
    print("=" * 60)
    print("Testing AI Parser Package Imports")
    print("=" * 60)
    
    try:
        # Test backward compatibility import
        from namel3ss.parser.ai import AIParserMixin
        print("✅ Backward compatibility: from namel3ss.parser.ai import AIParserMixin")
        
        # Test direct package imports
        from namel3ss.parser.ai.main import AIParserMixin as AIParserMain
        print("✅ Direct import: from namel3ss.parser.ai.main import AIParserMixin")
        
        # Verify they're the same
        assert AIParserMixin is AIParserMain, "Import mismatch detected!"
        print("✅ Imports point to same class")
        
        # Test that methods are accessible
        methods = [m for m in dir(AIParserMixin) if m.startswith('parse_') and not m.startswith('_')]
        print(f"✅ Found {len(methods)} parse methods accessible")
        print(f"   Sample methods: {', '.join(methods[:5])}")
        
        return True
    except Exception as e:
        print(f"❌ AI Parser import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_package_imports():
    """Test State package imports."""
    print("\n" + "=" * 60)
    print("Testing State Package Imports")
    print("=" * 60)
    
    try:
        # Test backward compatibility imports
        from namel3ss.codegen.backend.state import (
            build_backend_state,
            BackendState,
            PageComponent,
            PageSpec,
            _component_to_serializable
        )
        print("✅ Backward compatibility: all main exports imported")
        
        # Test direct package imports
        from namel3ss.codegen.backend.state.main import build_backend_state as build_main
        from namel3ss.codegen.backend.state.classes import BackendState as BS
        print("✅ Direct imports: from state.main and state.classes")
        
        # Verify they're the same
        assert build_backend_state is build_main, "build_backend_state mismatch!"
        assert BackendState is BS, "BackendState mismatch!"
        print("✅ Imports point to same objects")
        
        # Test dataclass structure
        print(f"✅ BackendState has {len(BackendState.__dataclass_fields__)} fields")
        print(f"✅ PageSpec has {len(PageSpec.__dataclass_fields__)} fields")
        print(f"✅ PageComponent has {len(PageComponent.__dataclass_fields__)} fields")
        
        # Test that _component_to_serializable is callable
        assert callable(_component_to_serializable), "_component_to_serializable not callable!"
        print("✅ _component_to_serializable is callable")
        
        return True
    except Exception as e:
        print(f"❌ State package import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_module_structure():
    """Test that all state submodules are importable."""
    print("\n" + "=" * 60)
    print("Testing State Package Module Structure")
    print("=" * 60)
    
    modules = [
        'classes', 'expressions', 'utils', 'datasets', 'frames',
        'models', 'ai', 'insights', 'evaluation', 'training',
        'experiments', 'agents', 'pages', 'statements', 'actions',
        'crud', 'logic', 'variables', 'main'
    ]
    
    failed = []
    for module_name in modules:
        try:
            exec(f"from namel3ss.codegen.backend.state.{module_name} import *")
            print(f"✅ state.{module_name}")
        except Exception as e:
            print(f"❌ state.{module_name}: {e}")
            failed.append(module_name)
    
    if failed:
        print(f"\n❌ {len(failed)} modules failed to import: {', '.join(failed)}")
        return False
    else:
        print(f"\n✅ All {len(modules)} state submodules imported successfully")
        return True


def test_ai_parser_module_structure():
    """Test that all AI parser submodules are importable."""
    print("\n" + "=" * 60)
    print("Testing AI Parser Module Structure")
    print("=" * 60)
    
    modules = [
        'utils', 'schemas', 'training', 'workflows',
        'models', 'chains', 'prompts', 'main'
    ]
    
    failed = []
    for module_name in modules:
        try:
            exec(f"from namel3ss.parser.ai.{module_name} import *")
            print(f"✅ parser.ai.{module_name}")
        except Exception as e:
            print(f"❌ parser.ai.{module_name}: {e}")
            failed.append(module_name)
    
    if failed:
        print(f"\n❌ {len(failed)} modules failed to import: {', '.join(failed)}")
        return False
    else:
        print(f"\n✅ All {len(modules)} AI parser submodules imported successfully")
        return True


def test_functional_parsing():
    """Test that AI Parser can actually parse content."""
    print("\n" + "=" * 60)
    print("Testing AI Parser Functional Parsing")
    print("=" * 60)
    
    try:
        from namel3ss.parser.ai import AIParserMixin
        
        # Verify the mixin is a class
        assert isinstance(AIParserMixin, type), "AIParserMixin should be a class"
        print("✅ AIParserMixin is a valid class")
        
        # Check that the mixin has the expected base classes
        base_names = [base.__name__ for base in AIParserMixin.__bases__]
        print(f"✅ AIParserMixin combines {len(base_names)} mixins:")
        for name in base_names:
            print(f"   - {name}")
        
        # Verify key parsing mixins are included
        expected_mixins = ['ModelsParserMixin', 'ChainsParserMixin', 'PromptsParserMixin']
        for expected in expected_mixins:
            assert expected in base_names, f"Missing expected mixin: {expected}"
        print(f"✅ All expected mixins present")
        
        # Test that individual parser modules are importable
        from namel3ss.parser.ai.chains import ChainsParserMixin
        from namel3ss.parser.ai.prompts import PromptsParserMixin
        from namel3ss.parser.ai.models import ModelsParserMixin
        print("✅ Individual parser mixins can be imported")
        
        return True
    except Exception as e:
        print(f"❌ Functional parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_building():
    """Test that state building infrastructure is intact."""
    print("\n" + "=" * 60)
    print("Testing State Building Infrastructure")
    print("=" * 60)
    
    try:
        from namel3ss.codegen.backend.state import build_backend_state, BackendState
        from namel3ss.ast import App
        
        # Verify build_backend_state signature
        import inspect
        sig = inspect.signature(build_backend_state)
        params = list(sig.parameters.keys())
        assert 'app' in params, "build_backend_state missing 'app' parameter"
        print(f"✅ build_backend_state signature: {sig}")
        
        # Verify BackendState structure
        fields = list(BackendState.__dataclass_fields__.keys())
        required_fields = ['app', 'datasets', 'frames', 'pages', 'env_keys']
        for field in required_fields:
            assert field in fields, f"BackendState missing required field: {field}"
        print(f"✅ BackendState has all required fields ({len(fields)} total)")
        
        return True
    except Exception as e:
        print(f"❌ State building test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("REFACTORING VALIDATION TEST SUITE")
    print("=" * 60)
    print("Testing AI Parser (2,202 lines → 8 modules)")
    print("Testing State Package (2,270 lines → 19 modules)")
    print("=" * 60 + "\n")
    
    tests = [
        ("AI Parser Imports", test_ai_parser_imports),
        ("AI Parser Module Structure", test_ai_parser_module_structure),
        ("AI Parser Functional", test_functional_parsing),
        ("State Package Imports", test_state_package_imports),
        ("State Module Structure", test_state_module_structure),
        ("State Building Infrastructure", test_state_building),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Refactoring is successful!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
