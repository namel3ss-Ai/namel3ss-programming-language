#!/usr/bin/env python3
"""Debug test to verify basic structure works."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from namel3ss.lang.parser import parse_module

def test_basic_structure():
    """Test that basic app+page structure parses."""
    
    source = '''app "Test Dashboard".

page "Home" at "/":
  show text "Orders"
'''
    
    try:
        module = parse_module(source, "test.ai")
        print(f"✅ Basic structure parsed. Found {len(module.pages)} pages")
        if module.pages:
            print(f"Page statements: {len(module.pages[0].statements)}")
            for i, stmt in enumerate(module.pages[0].statements):
                print(f"Statement {i}: {type(stmt).__name__}")
    except Exception as e:
        print(f"❌ Basic structure failed: {e}")

if __name__ == "__main__":
    test_basic_structure()