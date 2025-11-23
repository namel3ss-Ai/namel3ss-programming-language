#!/usr/bin/env python3
"""Test parsing test_structured_app.ai with grammar parser."""

from namel3ss.lang.grammar import parse_module

# Parse the test app
with open('test_structured_app.ai') as f:
    source = f.read()

result = parse_module(source, 'test_structured_app.ai')

print('✓ Parse successful!')
print(f'Module: {result.name}')
print(f'Extra nodes: {len(result.extra_nodes)}')

if result.extra_nodes:
    prompt = result.extra_nodes[0]
    print(f'\nPrompt: {prompt.name}')
    print(f'Has args: {hasattr(prompt, "args")}')
    print(f'Has output_schema: {hasattr(prompt, "output_schema")}')
    
    if hasattr(prompt, 'args'):
        print(f'Args ({len(prompt.args)}):')
        for arg in prompt.args:
            print(f'  - {arg.name}: {arg.arg_type}')
    
    if hasattr(prompt, 'output_schema'):
        print(f'Output schema ({len(prompt.output_schema.fields)} fields):')
        for field in prompt.output_schema.fields:
            print(f'  - {field.name}: {field.field_type.base_type}')
