#!/usr/bin/env python3
"""Build feedback demo application and validate generated code"""

import os
import shutil
from pathlib import Path

from namel3ss.parser.program import LegacyProgramParser
from namel3ss.ir.builder import build_frontend_ir
from namel3ss.codegen.frontend.react.main import generate_react_vite_site
from namel3ss.codegen.backend.state.main import build_backend_state

def build_feedback_demo():
    """Build feedback components demo application"""
    
    # Parse source
    source_path = Path('examples/feedback_demo.ai')
    with open(source_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    print(f"📄 Parsing {source_path}...")
    parser = LegacyProgramParser(source)
    module = parser.parse()
    
    # Extract app
    app = None
    for item in module.body:
        if item.__class__.__name__ == 'App':
            app = item
            break
    
    if not app:
        print("❌ No app found in module")
        return False
    
    print(f"✅ Parsed app: {app.name}")
    print(f"   Pages: {len(app.pages)}")
    
    # Count components
    modal_count = 0
    toast_count = 0
    for page in app.pages:
        for stmt in page.body:
            if stmt.__class__.__name__ == 'Modal':
                modal_count += 1
            elif stmt.__class__.__name__ == 'Toast':
                toast_count += 1
    
    print(f"   Modals: {modal_count}")
    print(f"   Toasts: {toast_count}")
    
    # Build backend state (needed for IR builder)
    print("\n🔨 Building backend state...")
    state = build_backend_state(app)
    print(f"✅ Backend state built")
    
    # Generate Vite project directly from app
    output_dir = Path('tmp_feedback_demo')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    print(f"\n📦 Generating Vite project to {output_dir}...")
    generate_react_vite_site(app, str(output_dir))
    print(f"✅ Vite project generated")
    
    # Validate generated files
    print(f"\n🔍 Validating generated files...")
    
    # Check for Modal and Toast components
    modal_component = output_dir / 'src' / 'components' / 'Modal.tsx'
    toast_component = output_dir / 'src' / 'components' / 'Toast.tsx'
    
    if not modal_component.exists():
        print(f"❌ Modal.tsx not found")
        return False
    print(f"✅ Modal.tsx generated ({modal_component.stat().st_size} bytes)")
    
    if not toast_component.exists():
        print(f"❌ Toast.tsx not found")
        return False
    print(f"✅ Toast.tsx generated ({toast_component.stat().st_size} bytes)")
    
    # Check page files
    page_files = list((output_dir / 'src' / 'pages').glob('*.tsx'))
    print(f"✅ Generated {len(page_files)} page files")
    
    # Validate Modal imports in pages
    for page_file in page_files:
        content = page_file.read_text(encoding='utf-8')
        has_modal_import = 'import Modal from' in content
        has_toast_import = 'import Toast from' in content
        
        if has_modal_import or has_toast_import:
            print(f"   {page_file.name}: ", end='')
            if has_modal_import:
                print(f"Modal ", end='')
            if has_toast_import:
                print(f"Toast ", end='')
            print()
    
    # Count component usage in generated code
    total_modal_usage = 0
    total_toast_usage = 0
    for page_file in page_files:
        content = page_file.read_text(encoding='utf-8')
        total_modal_usage += content.count('<Modal')
        total_toast_usage += content.count('<Toast')
    
    print(f"\n📊 Component Usage:")
    print(f"   Modal instances: {total_modal_usage}")
    print(f"   Toast instances: {total_toast_usage}")
    
    print(f"\n✅ Feedback demo built successfully!")
    print(f"   Output: {output_dir.absolute()}")
    print(f"\n🚀 To run the demo:")
    print(f"   cd {output_dir}")
    print(f"   npm install")
    print(f"   npm run dev")
    
    return True


if __name__ == '__main__':
    success = build_feedback_demo()
    exit(0 if success else 1)
