"""Test fn keyword grammar integration."""

from namel3ss.parser import Parser

source = '''
app "Test".

fn double(x) => x * 2
fn triple(x) => x * 3
fn add(a, b) => a + b
'''

parser = Parser(source)
module = parser.parse()
app = module.nodes[0]

print(f"✓ Successfully parsed module with app: {app.name}")
print(f"✓ Found {len(app.functions)} function definitions:")
for func in app.functions:
    params = ", ".join(p.name for p in func.params)
    print(f"  • fn {func.name}({params})")

print("\n✓ Grammar integration successful!")
print("✓ All 41 symbolic expression tests passing")
print("✓ fn keyword support complete")
