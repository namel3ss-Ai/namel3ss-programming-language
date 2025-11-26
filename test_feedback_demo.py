#!/usr/bin/env python3
"""Test feedback demo validation"""

import textwrap
from pathlib import Path
from namel3ss.parser.program import LegacyProgramParser

def test_feedback_demo():
    """Validate feedback demo can be parsed and has correct components"""
    
    source_path = Path('examples/feedback_demo.ai')
    with open(source_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    print("✓ Testing feedback demo parsing...")
    parser = LegacyProgramParser(source)
    module = parser.parse()
    
    # Extract app
    app = None
    for item in module.body:
        if item.__class__.__name__ == 'App':
            app = item
            break
    
    assert app is not None, "App should be parsed"
    assert app.name == "Feedback Demo"
    print(f"  ✓ App name: {app.name}")
    
    # Count pages
    assert len(app.pages) == 3, f"Expected 3 pages, got {len(app.pages)}"
    print(f"  ✓ Pages: {len(app.pages)}")
    
    # Count modals and toasts
    modal_count = 0
    toast_count = 0
    modal_configs = {
        'sizes': set(),
        'dismissible': set(),
        'with_actions': 0,
        'with_content': 0
    }
    toast_configs = {
        'variants': set(),
        'positions': set(),
        'with_actions': 0,
        'durations': set()
    }
    
    for page in app.pages:
        for stmt in page.body:
            if stmt.__class__.__name__ == 'Modal':
                modal_count += 1
                modal_configs['sizes'].add(stmt.size)
                modal_configs['dismissible'].add(stmt.dismissible)
                if stmt.actions:
                    modal_configs['with_actions'] += 1
                if stmt.content:
                    modal_configs['with_content'] += 1
            elif stmt.__class__.__name__ == 'Toast':
                toast_count += 1
                toast_configs['variants'].add(stmt.variant)
                toast_configs['positions'].add(stmt.position)
                toast_configs['durations'].add(stmt.duration)
                if stmt.action_label:
                    toast_configs['with_actions'] += 1
    
    assert modal_count == 6, f"Expected 6 modals, got {modal_count}"
    print(f"  ✓ Modals: {modal_count}")
    
    assert toast_count == 9, f"Expected 9 toasts, got {toast_count}"
    print(f"  ✓ Toasts: {toast_count}")
    
    # Validate modal configurations
    print(f"\n✓ Modal configurations:")
    print(f"  Sizes: {sorted(modal_configs['sizes'])}")
    assert 'sm' in modal_configs['sizes']
    assert 'md' in modal_configs['sizes']
    assert 'lg' in modal_configs['sizes']
    assert 'xl' in modal_configs['sizes']
    print(f"  Dismissible options: {sorted(modal_configs['dismissible'])}")
    assert True in modal_configs['dismissible']
    assert False in modal_configs['dismissible']
    print(f"  With actions: {modal_configs['with_actions']}")
    assert modal_configs['with_actions'] == 6
    print(f"  With content: {modal_configs['with_content']}")
    assert modal_configs['with_content'] == 6
    
    # Validate toast configurations
    print(f"\n✓ Toast configurations:")
    print(f"  Variants: {sorted(toast_configs['variants'])}")
    assert 'default' in toast_configs['variants']
    assert 'success' in toast_configs['variants']
    assert 'error' in toast_configs['variants']
    assert 'warning' in toast_configs['variants']
    assert 'info' in toast_configs['variants']
    print(f"  Positions: {sorted(toast_configs['positions'])}")
    assert 'top' in toast_configs['positions']
    assert 'top-right' in toast_configs['positions']
    assert 'bottom' in toast_configs['positions']
    assert 'bottom-right' in toast_configs['positions']
    assert 'bottom-left' in toast_configs['positions']
    print(f"  Durations: {sorted(toast_configs['durations'])}")
    assert 0 in toast_configs['durations']  # Persistent
    assert 2000 in toast_configs['durations']
    assert 3000 in toast_configs['durations']
    assert 4000 in toast_configs['durations']
    assert 5000 in toast_configs['durations']
    print(f"  With actions: {toast_configs['with_actions']}")
    assert toast_configs['with_actions'] >= 4
    
    print(f"\n✓ All feedback demo validation tests passed!")
    return True

if __name__ == "__main__":
    success = test_feedback_demo()
    exit(0 if success else 1)
