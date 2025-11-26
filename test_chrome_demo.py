#!/usr/bin/env python3
"""Test script to validate chrome_demo.ai example"""

from namel3ss.parser.program import LegacyProgramParser

def test_chrome_demo():
    """Parse and validate the chrome demo example"""
    
    with open('examples/chrome_demo_clean.ai', 'r', encoding='utf-8') as f:
        source = f.read()
    
    parser = LegacyProgramParser(source)
    program = parser.parse()
    
    print(f"✅ Successfully parsed chrome_demo_clean.ai")
    print(f"   Module name: {program.name or 'unnamed'}")
    print(f"   Body items: {len(program.body)}")
    
    # Check what's in the body
    for item in program.body:
        print(f"   - {item.__class__.__name__}")
        if hasattr(item, 'pages'):
            print(f"     Has {len(item.pages)} pages")
    
    # Find pages (might be nested in app)
    pages = []
    for item in program.body:
        if item.__class__.__name__ == 'Page':
            pages.append(item)
        elif hasattr(item, 'pages'):
            pages.extend(item.pages)
    
    print(f"   Total Pages: {len(pages)}")
    
    # Count chrome components
    sidebar_count = 0
    navbar_count = 0
    breadcrumbs_count = 0
    command_palette_count = 0
    
    for page in pages:
        for stmt in page.statements:
            stmt_type = stmt.__class__.__name__
            if stmt_type == 'Sidebar':
                sidebar_count += 1
            elif stmt_type == 'Navbar':
                navbar_count += 1
            elif stmt_type == 'Breadcrumbs':
                breadcrumbs_count += 1
            elif stmt_type == 'CommandPalette':
                command_palette_count += 1
    
    print(f"\nChrome Components Found:")
    print(f"   Sidebars: {sidebar_count}")
    print(f"   Navbars: {navbar_count}")
    print(f"   Breadcrumbs: {breadcrumbs_count}")
    print(f"   Command Palettes: {command_palette_count}")
    
    return program

if __name__ == '__main__':
    test_chrome_demo()
