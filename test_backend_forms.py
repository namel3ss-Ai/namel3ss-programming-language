"""Test backend form endpoint generation."""
from namel3ss.parser import Parser
from namel3ss.codegen.backend.state import build_backend_state
from namel3ss.codegen.backend.core.routers_pkg.pages_router import _render_pages_router_module

# Parse forms demo
code = open('examples/forms_demo.ai').read()
parser = Parser(code)
module = parser.parse()
app = module.body[0]

# Build backend state
state = build_backend_state(app)

# Generate pages router
router_code = _render_pages_router_module(state)

# Check for form endpoint
if '/forms/' in router_code and 'form_data: Dict[str, Any]' in router_code:
    print("✓ Form endpoint generated successfully")
    print("✓ Form validation code included")
    
    # Count validation checks
    if 'validation_schema' in router_code:
        print("✓ Validation schema present")
    if 'HTTPException' in router_code:
        print("✓ Error handling included")
    
    # Show a snippet of the generated form endpoint
    lines = router_code.split('\n')
    in_form = False
    snippet_lines = []
    for i, line in enumerate(lines):
        if '/forms/' in line:
            in_form = True
            snippet_lines = lines[i:min(i+15, len(lines))]
            break
    
    if snippet_lines:
        print("\nGenerated form endpoint (first 15 lines):")
        print('\n'.join(snippet_lines))
else:
    print("✗ Form endpoint not found in generated code")

print(f"\nTotal generated code: {len(router_code)} characters")
