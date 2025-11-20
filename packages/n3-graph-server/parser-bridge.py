#!/usr/bin/env python3
"""
N3 Parser Bridge for Node.js

This script bridges the Python N3 parser with Node.js by:
1. Accepting a .n3 file path or stdin input
2. Parsing the file using namel3ss.lang.grammar.parse_module
3. Converting the AST to JSON
4. Outputting JSON to stdout for Node.js consumption
"""

import sys
import json
import os
from typing import Any, Dict, List
from dataclasses import asdict, is_dataclass

# Add the parent directory to path to import namel3ss
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from namel3ss.lang.grammar import parse_module
from namel3ss.ast.program import Module


def serialize_ast_node(obj: Any) -> Any:
    """
    Recursively serialize AST nodes to JSON-compatible dictionaries.
    Handles dataclasses, lists, dicts, and primitive types.
    """
    if obj is None:
        return None
    
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    if isinstance(obj, list):
        return [serialize_ast_node(item) for item in obj]
    
    if isinstance(obj, dict):
        return {k: serialize_ast_node(v) for k, v in obj.items()}
    
    if is_dataclass(obj):
        # Convert dataclass to dict and add type information
        result = {'type': obj.__class__.__name__}
        for key, value in asdict(obj).items():
            # Skip internal/location fields that aren't needed for graph
            if key in ('location', 'metadata') and not value:
                continue
            result[key] = serialize_ast_node(value)
        return result
    
    # For other objects, try to get their string representation
    try:
        return str(obj)
    except:
        return None


def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            'success': False,
            'error': 'Usage: parser-bridge.py <file.n3> or parser-bridge.py --stdin <filename>'
        }), file=sys.stderr)
        sys.exit(1)
    
    try:
        # Check if reading from stdin
        if sys.argv[1] == '--stdin':
            file_name = sys.argv[2] if len(sys.argv) > 2 else 'stdin.n3'
            source = sys.stdin.read()
        else:
            file_path = sys.argv[1]
            file_name = file_path
            
            if not os.path.exists(file_path):
                print(json.dumps({
                    'success': False,
                    'error': f'File not found: {file_path}'
                }), file=sys.stderr)
                sys.exit(1)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        
        # Parse the N3 source
        module = parse_module(source, path=file_name)
        
        # Serialize to JSON
        result = serialize_ast_node(module)
        
        # Output JSON to stdout
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': f'Parse error: {str(e)}'
        }), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
