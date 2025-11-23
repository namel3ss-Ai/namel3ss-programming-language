#!/usr/bin/env python3
"""Debug script with detailed tracing."""

import sys

# Patch to see what's being parsed
original_advance = None

def trace_advance(self):
    """Trace _advance calls."""
    if hasattr(self, 'pos'):
        print(f"  _advance: pos={self.pos}/{len(self.lines)}", file=sys.stderr)
        if self.pos < len(self.lines) and self.pos < 20:
            line = self.lines[self.pos] if self.pos < len(self.lines) else "EOF"
            print(f"    Current line: {repr(line[:80] if len(line) > 80 else line)}", file=sys.stderr)
    return original_advance(self)

# Monkey patch
from namel3ss.parser.base import ParserBase
original_advance = ParserBase._advance
ParserBase._advance = trace_advance

print("Starting parse with tracing enabled...", file=sys.stderr)

from namel3ss.parser import Parser

with open('examples/simple_functional.ai', 'r') as f:
    content = f.read()

parser = Parser(content)
print("About to call parse_app()...", file=sys.stderr)

try:
    ast = parser.parse_app()
    print(f"✅ Success: {ast.name}")
except KeyboardInterrupt:
    print("\n❌ Interrupted - infinite loop detected")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
