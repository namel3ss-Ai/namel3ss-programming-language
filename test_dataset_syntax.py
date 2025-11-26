from namel3ss.parser import Parser

# Test 1: Quoted name with from clause
test1 = '''
dataset "users" from "db://users"
'''

# Test 2: With app context
test2 = '''
app "Test"

dataset "users" from "db://users"
'''

# Test 3: With inline data
test3 = '''
dataset "users" from inline:
  fields:
    - name: id
    - name: email
'''

for i, source in enumerate([test1, test2, test3], 1):
    try:
        parser = Parser(source)
        module = parser.parse()
        print(f"✓ Test {i} passed")
    except Exception as e:
        print(f"✗ Test {i} failed: {e}")
