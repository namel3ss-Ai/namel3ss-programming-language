"""Quick validation test for data display components and layouts."""

import textwrap
import pytest
from namel3ss.parser import Parser


def test_show_data_table_works():
    """Verify ShowDataTable parses and works."""
    source = textwrap.dedent('''
        app "Test"
        
        dataset "users" from "db://users"
        
        page "Users" at "/users":
          show data_table "All Users" from dataset users:
            columns:
              - field: name
                header: "Name"
    ''').strip()
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.pages) == 1
    assert len(app.pages[0].body) == 1
    assert app.pages[0].body[0].__class__.__name__ == "ShowDataTable"
    print("✓ ShowDataTable works")


def test_stack_layout_works():
    """Verify StackLayout parses and works."""
    source = textwrap.dedent('''
        app "Test"
        
        dataset "items" from "db://items"
        
        page "Dashboard" at "/dashboard":
          layout stack:
            direction: vertical
            gap: medium
            children:
              - show card "Card 1" from dataset items
              - show card "Card 2" from dataset items
    ''').strip()
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.pages) == 1
    assert len(app.pages[0].body) == 1
    layout = app.pages[0].body[0]
    assert layout.__class__.__name__ == "StackLayout"
    assert layout.direction == "vertical"
    assert len(layout.children) == 2
    print("✓ StackLayout works")


def test_tabs_layout_works():
    """Verify TabsLayout parses and works."""
    source = textwrap.dedent('''
        app "Test"
        
        dataset "data" from "db://data"
        
        page "Tabs" at "/tabs":
          layout tabs:
            tabs:
              - id: tab1
                label: "Tab 1"
                content:
                  - show card "Content" from dataset data
    ''').strip()
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    layout = app.pages[0].body[0]
    assert layout.__class__.__name__ == "TabsLayout"
    assert len(layout.tabs) == 1
    assert layout.tabs[0].id == "tab1"
    print("✓ TabsLayout works")


if __name__ == "__main__":
    test_show_data_table_works()
    test_stack_layout_works()
    test_tabs_layout_works()
    print("\n✅ All critical components verified working!")
