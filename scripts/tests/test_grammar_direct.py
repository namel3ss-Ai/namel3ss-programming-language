#!/usr/bin/env python3
"""Direct test of parsing without going through CLI."""

import sys
sys.path.insert(0, r"c:\Users\SBW\OneDrive - Axon Group\Documents\GitHub\namel3ss-programming-language")

from namel3ss.lang.grammar import parse_module

source = '''app "Text Classifier".

llm openai_gpt:
    provider: openai
    model: gpt-4

prompt classify:
    args:
        text: string
    output_schema:
        category: enum("positive", "negative", "neutral")
        confidence: float
    template: "Classify the sentiment of this text: {text}"
    model: openai_gpt
'''

try:
    result = parse_module(source, 'app.ai')
    print('✓ Parse successful!')
    print(f'Module: {result.name}')
    print(f'LLMs: {len(result.app.llms)}')
    print(f'Prompts: {len(result.app.prompts)}')
    
    if result.app.prompts:
        prompt = result.app.prompts[0]
        print(f'\nPrompt: {prompt.name}')
        print(f'Model: {prompt.model}')
        print(f'Template: {prompt.template}')
        print(f'Has args attr: {hasattr(prompt, "args")}')
        print(f'Has output_schema attr: {hasattr(prompt, "output_schema")}')
        
        if hasattr(prompt, 'args') and prompt.args:
            print(f'\nArgs ({len(prompt.args)}):')
            for arg in prompt.args:
                print(f'  - {arg.name}: {arg.arg_type} (required={arg.required})')
        
        if hasattr(prompt, 'output_schema') and prompt.output_schema:
            print(f'\nOutput schema ({len(prompt.output_schema.fields)} fields):')
            for field in prompt.output_schema.fields:
                print(f'  - {field.name}: {field.field_type.base_type}')
                if field.field_type.enum_values:
                    print(f'    enum values: {field.field_type.enum_values}')
    
    print('\n✓ Grammar integration successful!')
    
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
