from namel3ss.parser import Parser

source = '''app "Test Forms"

page "Contact" at "/contact":
  show form "Contact Us":
    fields:
      - name: email
        component: text_input
'''

try:
    parser = Parser(source)
    result = parser.parse()
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
