#!/usr/bin/env python3
"""Build chrome demo application and validate generated code"""

import os
import shutil
from pathlib import Path

from namel3ss.parser.program import LegacyProgramParser
from namel3ss.ir.builder import build_frontend_ir
from namel3ss.codegen.frontend.react.main import generate_react_vite_site
from namel3ss.codegen.backend.state.main import build_backend_state

def build_chrome_demo():
    """Build chrome demo application"""
    
    # Parse source
    source_path = Path('examples/chrome_demo_clean.ai')
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
    
    # Build backend state (needed for IR builder)
    print("\n🔨 Building backend state...")
    state = build_backend_state(app)
    print(f"✅ Backend state built")
    
    # Generate Vite project directly from app
    output_dir = Path('tmp_chrome_demo')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    print(f"\n📦 Generating Vite project to {output_dir}...")
    generate_react_vite_site(app, str(output_dir))
    print(f"✅ Vite project generated")
    
    # Check generated files
    print("\n📂 Checking generated files...")
    
    components_dir = output_dir / 'src' / 'components'
    pages_dir = output_dir / 'src' / 'pages'
    
    # Chrome components
    chrome_files = [
        'Sidebar.tsx',
        'Navbar.tsx',
        'Breadcrumbs.tsx',
        'CommandPalette.tsx'
    ]
    
    for chrome_file in chrome_files:
        path = components_dir / chrome_file
        if path.exists():
            size = path.stat().st_size
            print(f"   ✅ {chrome_file} ({size} bytes)")
        else:
            print(f"   ❌ {chrome_file} NOT FOUND")
            return False
    
    # Page components
    expected_pages = ['Dashboard', 'Analytics', 'Reports', 'Sales Report', 'Profile', 'Security']
    for page_name in expected_pages:
        # Convert to filename (spaces to underscores, lowercase)
        filename = page_name.replace(' ', '_').lower() + '.tsx'
        path = pages_dir / filename
        if path.exists():
            size = path.stat().st_size
            print(f"   ✅ {filename} ({size} bytes)")
        else:
            print(f"   ⚠️  {filename} not found (may use different naming)")
    
    print(f"\n✅ Chrome demo application built successfully!")
    print(f"   Output: {output_dir.absolute()}")
    
    return True

if __name__ == '__main__':
    success = build_chrome_demo()
    exit(0 if success else 1)
