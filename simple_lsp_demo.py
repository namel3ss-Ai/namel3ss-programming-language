"""
Simple demo showing enhanced LSP features without errors.
"""

from namel3ss.lsp.enhanced_completion import EnhancedCompletionProvider
from namel3ss.lsp.enhanced_diagnostics import EnhancedDiagnosticsProvider
from lsprotocol.types import Position


def simple_lsp_demo():
    """Demo the enhanced LSP features."""
    
    print("🚀 N3 Enhanced LSP Features Demo")
    print("=" * 40)
    
    # Test completion provider
    print("\\n1. Enhanced Completions")
    print("-" * 25)
    
    completion_provider = EnhancedCompletionProvider()
    
    # Test completions at start of file
    completions = completion_provider.get_completions("", Position(line=0, character=0), "", "")
    print(f"Available keyword completions: {len(completions.items)}")
    
    for item in completions.items[:5]:  # Show first 5
        print(f"  • {item.label}: {item.detail}")
    
    # Test diagnostics provider  
    print("\\n2. Enhanced Diagnostics")
    print("-" * 25)
    
    diagnostics_provider = EnhancedDiagnosticsProvider()
    
    # Test legacy syntax detection
    legacy_content = '''page "Test" at "/test" {
    show text "Legacy syntax here"
}'''
    
    warnings = diagnostics_provider.check_for_legacy_syntax_warnings(legacy_content)
    print(f"Legacy syntax warnings found: {len(warnings)}")
    
    for warning in warnings:
        line_num = warning.range.start.line + 1
        print(f"  • Line {line_num}: {warning.message}")
    
    # Test error enhancement
    print("\\n3. Error Message Enhancement")  
    print("-" * 30)
    
    # Simulate some common errors
    test_cases = [
        ("Expected: colon", "context_line: value"),
        ("Unexpected token", "some invalid syntax"),
        ("Indentation error", "  bad_indent")
    ]
    
    for error_msg, context in test_cases:
        enhanced = diagnostics_provider._enhance_error_message(
            error_msg, context, [context], 0
        )
        print(f"  Original: {error_msg}")
        print(f"  Enhanced: {enhanced[:100]}...")
        print()
    
    print("\\n✅ Enhanced LSP Features:")
    print("   • Smart keyword completion with snippets")
    print("   • Context-aware suggestions")
    print("   • Legacy syntax warnings")
    print("   • Enhanced error messages with context")
    print("   • Performance optimized with parser caching")
    
    return True


if __name__ == "__main__":
    simple_lsp_demo()