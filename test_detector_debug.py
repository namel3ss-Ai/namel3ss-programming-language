"""Debug script to test detector"""
from namel3ss.deps.detector import FeatureDetector
from namel3ss.parser import Parser

source = """
app "TestApp" {
    description: "Test"
}

dataset users from table users
"""

detector = FeatureDetector()
result = detector.detect_from_source(source)

print(f"Features: {result.features}")
print(f"Warnings: {result.warnings}")

# Check what IR was created
try:
    parser = Parser(source)
    parsed = parser.parse()
    print(f"\nParsed type: {type(parsed)}")
    print(f"Body: {parsed.body}")
    print(f"Body length: {len(parsed.body)}")
    
    for i, item in enumerate(parsed.body):
        print(f"\nBody[{i}] type: {type(item)}")
        print(f"Body[{i}] dir: {[x for x in dir(item) if not x.startswith('_')]}")
        
        # Check if it's a dataset
        if hasattr(item, 'name') and hasattr(item, 'source'):
            print(f"  Found dataset: {item.name}")
            print(f"  Source: {item.source}")
            print(f"  Source type: {type(item.source)}")
            
except Exception as e:
    print(f"Parse error: {e}")
    import traceback
    traceback.print_exc()
