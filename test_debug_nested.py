"""Debug test for nested objects"""
from namel3ss.parser import Parser

code = """
prompt "test" {
    model: gpt-4o
    template: "test"
    output_schema: {
        name: string
    }
}
"""

print("Code:")
print(code)
print("\n" + "="*50 + "\n")

try:
    parser = Parser(code)
    module = parser.parse()
    print("SUCCESS!")
    print(f"Module: {module}")
    print(f"Module.body: {module.body}")
    print(f"Module.name: {module.name}")
    
    # Look for prompts
    prompts = [node for node in module.body if hasattr(node, 'output_schema')]
    print(f"\nPrompts found: {len(prompts)}")
    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i}: {prompt.name if hasattr(prompt, 'name') else 'no name'}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
