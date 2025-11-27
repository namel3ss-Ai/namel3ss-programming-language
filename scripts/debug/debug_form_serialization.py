"""Debug form field serialization."""
from namel3ss.parser import Parser
from namel3ss.codegen.backend.state import build_backend_state

code = open('examples/forms_demo.ai').read()
parser = Parser(code)
module = parser.parse()
app = module.body[0]

state = build_backend_state(app)

# Check page components
for page in state.pages:
    print(f"Page: {page.slug}")
    for idx, comp in enumerate(page.components):
        print(f"  Component {idx}: {comp.type}")
        if comp.type == "form":
            print(f"    Payload keys: {list(comp.payload.keys())}")
            print(f"    Fields: {comp.payload.get('fields', [])}")
            if comp.payload.get('fields'):
                for field in comp.payload.get('fields', []):
                    print(f"      - {field.get('name')}: {field.get('component')}")
                    print(f"        validation: {field.get('validation')}")
                    print(f"        required: {field.get('required')}")
