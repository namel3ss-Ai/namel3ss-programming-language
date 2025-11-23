#!/usr/bin/env python3
"""Test which parser is being used."""

from namel3ss.lang.grammar import parse_module
from namel3ss.lang.grammar.helpers import GrammarUnsupportedError

with open('examples/simple_functional.ai', 'r') as f:
    content = f.read()

print("Attempting grammar parser...")
try:
    ast = parse_module(content, path="examples/simple_functional.ai")
    print(f"✅ Grammar parser succeeded: {type(ast)}")
except GrammarUnsupportedError as e:
    print(f"⚠️  Grammar parser unsupported: {e}")
    print("Will fall back to legacy parser")
except Exception as e:
    print(f"❌ Grammar parser error: {type(e).__name__}: {e}")
