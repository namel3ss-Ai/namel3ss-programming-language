#!/usr/bin/env python3
"""
Quick test script for declarative UI parser.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from namel3ss.parser import Parser
from namel3ss.ast.pages import ShowCard, ShowList, EmptyStateConfig

def test_basic_card():
    """Test basic show card parsing."""
    print("Testing basic card parsing...")
    
    source = '''
dataset test_data:
  fields:
    - id: int
    - title: text

page test:
  path: "/test"
  title: "Test"
  
  show card "Items" from dataset test_data:
    empty_state:
      icon: inbox
      title: "No items"
      message: "Nothing here yet"
'''
    
    try:
        parser = Parser()
        app = parser.parse(source)
        
        assert len(app.pages) == 1, f"Expected 1 page, got {len(app.pages)}"
        page = app.pages[0]
        
        assert len(page.statements) == 1, f"Expected 1 statement, got {len(page.statements)}"
        statement = page.statements[0]
        
        assert isinstance(statement, ShowCard), f"Expected ShowCard, got {type(statement)}"
        assert statement.title == "Items", f"Expected title 'Items', got '{statement.title}'"
        assert statement.source_type == "dataset", f"Expected source_type 'dataset', got '{statement.source_type}'"
        assert statement.source == "test_data", f"Expected source 'test_data', got '{statement.source}'"
        
        assert statement.empty_state is not None, "empty_state is None"
        assert isinstance(statement.empty_state, EmptyStateConfig), f"Expected EmptyStateConfig, got {type(statement.empty_state)}"
        assert statement.empty_state.icon == "inbox", f"Expected icon 'inbox', got '{statement.empty_state.icon}'"
        assert statement.empty_state.title == "No items", f"Expected title 'No items', got '{statement.empty_state.title}'"
        
        print("✅ Basic card parsing: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Basic card parsing: FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_card_with_sections():
    """Test card with sections and actions."""
    print("\nTesting card with sections and actions...")
    
    source = '''
dataset appointments:
  fields:
    - id: int
    - status: text
    - date: date
    - provider: text

page test:
  path: "/test"
  title: "Test"
  
  show card "Appointments" from dataset appointments:
    item:
      type: card
      
      header:
        badges:
          - field: status
            transform: humanize
      
      sections:
        - type: info_grid
          columns: 2
          items:
            - icon: calendar
              label: "Date"
              values:
                - field: date
                  format: "MMMM DD, YYYY"
      
      actions:
        - label: "View"
          icon: eye
          action: view_item
          params: "{{ id }}"
          condition: "status == 'active'"
    
    filter_by: "status != 'cancelled'"
'''
    
    try:
        parser = Parser()
        app = parser.parse(source)
        
        page = app.pages[0]
        statement = page.statements[0]
        
        assert isinstance(statement, ShowCard), f"Expected ShowCard, got {type(statement)}"
        assert statement.filter_by == "status != 'cancelled'", f"Expected filter_by, got '{statement.filter_by}'"
        
        assert statement.item_config is not None, "item_config is None"
        assert statement.item_config.type == "card", f"Expected type 'card', got '{statement.item_config.type}'"
        
        # Check header badges
        assert statement.item_config.header is not None, "header is None"
        assert statement.item_config.header.badges is not None, "badges is None"
        assert len(statement.item_config.header.badges) == 1, f"Expected 1 badge, got {len(statement.item_config.header.badges)}"
        badge = statement.item_config.header.badges[0]
        assert badge.field == "status", f"Expected badge field 'status', got '{badge.field}'"
        assert badge.transform == "humanize", f"Expected transform 'humanize', got '{badge.transform}'"
        
        # Check sections
        assert statement.item_config.sections is not None, "sections is None"
        assert len(statement.item_config.sections) == 1, f"Expected 1 section, got {len(statement.item_config.sections)}"
        section = statement.item_config.sections[0]
        assert section.type == "info_grid", f"Expected type 'info_grid', got '{section.type}'"
        assert section.columns == 2, f"Expected columns 2, got {section.columns}"
        
        # Check actions
        assert statement.item_config.actions is not None, "actions is None"
        assert len(statement.item_config.actions) == 1, f"Expected 1 action, got {len(statement.item_config.actions)}"
        action = statement.item_config.actions[0]
        assert action.label == "View", f"Expected label 'View', got '{action.label}'"
        assert action.condition == "status == 'active'", f"Expected condition, got '{action.condition}'"
        
        print("✅ Card with sections and actions: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Card with sections and actions: FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_show_list():
    """Test show list parsing."""
    print("\nTesting show list parsing...")
    
    source = '''
dataset messages:
  fields:
    - id: int
    - subject: text

page test:
  path: "/test"
  title: "Test"
  
  show list "Messages" from dataset messages:
    list_type: conversation
    enable_search: true
    columns: 1
'''
    
    try:
        parser = Parser()
        app = parser.parse(source)
        
        page = app.pages[0]
        statement = page.statements[0]
        
        assert isinstance(statement, ShowList), f"Expected ShowList, got {type(statement)}"
        assert statement.title == "Messages", f"Expected title 'Messages', got '{statement.title}'"
        assert statement.list_type == "conversation", f"Expected list_type 'conversation', got '{statement.list_type}'"
        assert statement.enable_search is True, f"Expected enable_search True, got {statement.enable_search}"
        assert statement.columns == 1, f"Expected columns 1, got {statement.columns}"
        
        print("✅ Show list parsing: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Show list parsing: FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Declarative UI Parser Tests")
    print("=" * 60)
    
    results = []
    results.append(test_basic_card())
    results.append(test_card_with_sections())
    results.append(test_show_list())
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed")
        sys.exit(1)
