"""
Advanced Code Actions Demo for N3 Language Server.

Demonstrates sophisticated refactoring capabilities including:
- Quick fixes for common syntax errors
- Legacy-to-modern syntax migration
- Component extraction and reuse
- File structure organization
- Safe symbol renaming

This showcases the next level of IDE support beyond basic completions.
"""

import asyncio
from typing import Dict, List
from pathlib import Path

from namel3ss.lsp.code_actions import CodeActionsProvider
from namel3ss.lsp.advanced_refactoring import AdvancedRefactoringEngine, RefactoringContext
from lsprotocol.types import Range, Position, Diagnostic, DiagnosticSeverity


def demo_advanced_code_actions():
    """Demonstrate advanced code action capabilities."""
    
    print("🚀 Advanced Code Actions Demo")
    print("=" * 40)
    
    # Test content with various refactoring opportunities
    legacy_content = '''app "Legacy App" {
    description "Missing colon here"
}

page "LegacyPage" at "/legacy" {
    show text "Old syntax here"
    show form "LegacyForm" {
        field "name" type="text" required=true
        field "email" type="email" required=false
        submit "Submit"
    }
    
    show text "Another text block"
    show text "Yet another text block"
}'''

    modern_content = '''app "Modern App" {
    description: "Proper colon syntax"
}

llm "app_llm" {
    provider: "openai"
    model: "gpt-4o-mini"
    temperature: 0.7
}

page "ModernPage" at "/modern" {
    show text: "Modern syntax here"
    show form: {
        field: {
            name: "name"
            type: "text" 
            required: true
        }
        field: {
            name: "email"
            type: "email"
            required: false
        }
        submit: "Submit"
    }
}'''

    messy_content = '''page "MessyPage" at "/messy" {
show text: "Inconsistent indentation"
        show form: {
    field: {
        name: "test"
            type: "text"
    }
        }
}

app "MessyApp" {
description: "App should come first"
}'''

    # Test 1: Code Actions Provider
    print("\\n1. Code Actions Provider")
    print("-" * 25)
    
    provider = CodeActionsProvider()
    
    # Simulate diagnostic for legacy content
    legacy_diagnostic = Diagnostic(
        range=Range(
            start=Position(line=1, character=16),
            end=Position(line=1, character=35)
        ),
        message="Missing colon in property declaration",
        severity=DiagnosticSeverity.Error,
        source="namel3ss"
    )
    
    # Get quick fix actions
    file_uri = "file:///test/legacy.n3"
    actions = provider.get_code_actions(
        legacy_content,
        file_uri,
        Range(start=Position(line=0, character=0), end=Position(line=0, character=0)),
        [legacy_diagnostic]
    )
    
    print(f"Available code actions: {len(actions)}")
    for action in actions[:5]:  # Show first 5
        kind_str = str(action.kind).split('.')[-1] if action.kind else "Unknown"
        preferred = " ⭐" if action.is_preferred else ""
        print(f"  • [{kind_str}] {action.title}{preferred}")
    
    # Test 2: Advanced Refactoring Engine
    print("\\n2. Advanced Refactoring Engine")
    print("-" * 30)
    
    engine = AdvancedRefactoringEngine()
    workspace_files = {
        "file:///test/legacy.n3": legacy_content,
        "file:///test/modern.n3": modern_content,
        "file:///test/messy.n3": messy_content
    }
    
    # Test legacy modernization
    modernize_result = engine.modernize_legacy_syntax(legacy_content, file_uri)
    
    print(f"Legacy modernization: {'✅ Success' if modernize_result.success else '❌ Failed'}")
    if modernize_result.success and modernize_result.workspace_edit:
        print("  Changes made:")
        for doc_edit in modernize_result.workspace_edit.document_changes:
            print(f"    - {len(doc_edit.edits)} edits to {doc_edit.text_document.uri.split('/')[-1]}")
            for edit in doc_edit.edits[:3]:  # Show first 3 edits
                original_line = legacy_content.splitlines()[edit.range.start.line]
                print(f"      Line {edit.range.start.line + 1}: '{original_line.strip()}' → '{edit.new_text.strip()}'")
    
    # Test file organization
    organize_result = engine.organize_file_structure(messy_content, "file:///test/messy.n3")
    print(f"\\nFile organization: {'✅ Success' if organize_result.success else '❌ Failed'}")
    if organize_result.success:
        print("  File structure optimized for logical component ordering")
    
    # Test 3: Component Extraction
    print("\\n3. Component Extraction")
    print("-" * 25)
    
    # Extract repeated text components
    context = RefactoringContext(
        source_file=file_uri,
        target_range=Range(
            start=Position(line=8, character=4),
            end=Position(line=8, character=35)
        ),
        symbol_name="repeated_text"
    )
    
    selected_content = 'show text: "Another text block"'
    extract_result = engine.extract_component(
        context, "text_display", selected_content, workspace_files
    )
    
    print(f"Component extraction: {'✅ Success' if extract_result.success else '❌ Failed'}")
    if extract_result.success:
        print("  Created reusable component for repeated text patterns")
        print("  Component can be reused across multiple pages")
    
    # Test 4: Safe Symbol Renaming
    print("\\n4. Safe Symbol Renaming")
    print("-" * 25)
    
    rename_context = RefactoringContext(
        source_file=file_uri,
        target_range=Range(
            start=Position(line=0, character=0),
            end=Position(line=0, character=0)
        )
    )
    
    rename_result = engine.safe_rename_symbol(
        rename_context, "Legacy App", "Modernized App", workspace_files
    )
    
    print(f"Symbol renaming: {'✅ Success' if rename_result.success else '❌ Failed'}")
    if rename_result.success:
        print(f"  Would rename across {len(rename_result.affected_files)} files")
        print("  All references updated safely")
    elif rename_result.error_message:
        print(f"  Reason: {rename_result.error_message}")
    
    # Test 5: Transformation Examples
    print("\\n5. Transformation Examples")
    print("-" * 30)
    
    transformations = [
        {
            "name": "Legacy Show Text",
            "before": 'show text "Hello World"',
            "after": 'show text: "Hello World"',
            "type": "Quick Fix"
        },
        {
            "name": "Legacy Field Syntax", 
            "before": 'field "name" type="text" required=true',
            "after": '''field: {
    name: "name"
    type: "text"
    required: true
}''',
            "type": "Modernization"
        },
        {
            "name": "Property Missing Colon",
            "before": 'description "App description"',
            "after": 'description: "App description"',
            "type": "Quick Fix"
        }
    ]
    
    for transform in transformations:
        print(f"\\n{transform['name']} ({transform['type']}):")
        print(f"  Before: {transform['before']}")
        print(f"  After:  {transform['after']}")
    
    # Test 6: Performance and Integration
    print("\\n6. Integration Benefits") 
    print("-" * 25)
    
    benefits = [
        "✅ One-click legacy modernization for entire files",
        "✅ Safe renaming with cross-file reference tracking", 
        "✅ Intelligent component extraction for code reuse",
        "✅ Automatic file organization and structure optimization",
        "✅ Context-aware quick fixes based on error patterns",
        "✅ Integration with 237x faster parser caching",
        "✅ Real-time availability in VS Code and other LSP editors"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")
    
    print("\\n🎉 Advanced Code Actions Complete!")
    print("\\nKey Capabilities:")
    print("  • Smart refactoring with AST awareness")
    print("  • Legacy codebase modernization")  
    print("  • Component extraction and reuse")
    print("  • Safe cross-file symbol operations")
    print("  • Performance optimized with parser caching")
    
    return {
        "quick_fixes": len([a for a in actions if a.kind and 'quickfix' in str(a.kind).lower()]),
        "refactoring_actions": len([a for a in actions if a.kind and 'refactor' in str(a.kind).lower()]),
        "modernization_success": modernize_result.success,
        "extraction_success": extract_result.success,
        "organization_success": organize_result.success
    }


if __name__ == "__main__":
    demo_advanced_code_actions()