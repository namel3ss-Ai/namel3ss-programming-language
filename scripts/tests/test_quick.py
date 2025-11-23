#!/usr/bin/env python3
"""Quick test script for debugging parser issues."""

from namel3ss.parser import Parser

try:
    print("Testing simple_functional.ai...")
    with open('examples/simple_functional.ai', 'r') as f:
        content = f.read()
    
    print(f"Content length: {len(content)} chars")
    print(f"First 100 chars: {content[:100]}")
    
    parser = Parser(content)
    print("Parser created successfully")
    
    ast = parser.parse_app()
    print(f"✅ Success! App name: {ast.name}")
    print(f"   Datasets: {len(ast.datasets)}")
    print(f"   Functions: {len(getattr(ast, 'functions', []))}")
    
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
