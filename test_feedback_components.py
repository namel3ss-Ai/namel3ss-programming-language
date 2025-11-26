#!/usr/bin/env python3
"""Quick test of Modal and Toast parsing and IR building"""

import textwrap
from namel3ss.parser.program import LegacyProgramParser
from namel3ss.ir.builder import build_frontend_ir

test_code = '''
app "Test App"

page "Test" at "/test":
  modal "confirm_delete":
    title: "Confirm Delete"
    description: "Are you sure?"
    size: "md"
    dismissible: true
    content:
      show text "This action cannot be undone"
    actions:
      action "Cancel" variant "ghost"
      action "Delete" variant "destructive" action "do_delete"
  
  toast "success_toast":
    title: "Success"
    description: "Item saved"
    variant: "success"
    duration: 3000
    position: "top-right"
  
  show text "Test page content"
'''

def test_parse():
    parser = LegacyProgramParser(textwrap.dedent(test_code))
    try:
        module = parser.parse()
        app = module.body[0]
        print("✓ Parsing successful")
        print(f"  App name: {app.name}")
        print(f"  Pages: {len(app.pages)}")
        
        page = app.pages[0]
        print(f"  Page statements: {len(page.body)}")
        
        # Check for Modal
        modal_found = any(stmt.__class__.__name__ == 'Modal' for stmt in page.body)
        print(f"  Modal found: {modal_found}")
        
        # Check for Toast
        toast_found = any(stmt.__class__.__name__ == 'Toast' for stmt in page.body)
        print(f"  Toast found: {toast_found}")
        
        # Test IR building
        print("\n✓ Testing IR building...")
        frontend_ir = build_frontend_ir(app)
        print(f"  Frontend IR pages: {len(frontend_ir.pages)}")
        
        if frontend_ir.pages:
            page_spec = frontend_ir.pages[0]
            print(f"  Page components: {len(page_spec.components)}")
            
            # Check component types
            component_types = [comp.type for comp in page_spec.components]
            print(f"  Component types: {component_types}")
            
            modal_in_ir = 'modal' in component_types
            toast_in_ir = 'toast' in component_types
            print(f"  Modal in IR: {modal_in_ir}")
            print(f"  Toast in IR: {toast_in_ir}")
        
        return True
    except Exception as e:
        print(f"X Parsing/IR failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_parse()
    exit(0 if success else 1)

