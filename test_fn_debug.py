"""Debug test for fn keyword."""
import sys

print("Starting test...")
sys.stdout.flush()

try:
    print("Importing Parser...")
    sys.stdout.flush()
    from namel3ss.parser import Parser
    
    print("Creating source...")
    sys.stdout.flush()
    source = '''
app "Test".

fn double(x) => x * 2
'''
    
    print("Creating parser...")
    sys.stdout.flush()
    parser = Parser(source)
    
    print("Parsing...")
    sys.stdout.flush()
    module = parser.parse()
    
    print("Getting app...")
    sys.stdout.flush()
    app = module.nodes[0]
    
    print(f"✓ App name: {app.name}")
    print(f"✓ Functions: {len(app.functions)}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
