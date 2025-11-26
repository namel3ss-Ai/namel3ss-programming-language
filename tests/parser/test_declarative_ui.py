"""
Test parsing of declarative UI components (show card, show list).
"""

import pytest
from namel3ss.parser import Parser
from namel3ss.ast.pages import ShowCard, ShowList, EmptyStateConfig, CardItemConfig


def test_parse_show_card_basic():
    """Test parsing basic show card statement."""
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
'''
    parser = Parser()
    app = parser.parse(source)
    
    assert len(app.pages) == 1
    page = app.pages[0]
    
    assert len(page.statements) == 1
    statement = page.statements[0]
    
    assert isinstance(statement, ShowCard)
    assert statement.title == "Items"
    assert statement.source_type == "dataset"
    assert statement.source == "test_data"
    assert statement.empty_state is not None
    assert isinstance(statement.empty_state, EmptyStateConfig)
    assert statement.empty_state.icon == "inbox"
    assert statement.empty_state.title == "No items"


def test_parse_show_card_with_item_config():
    """Test parsing show card with item configuration."""
    source = '''
dataset test_data:
  fields:
    - id: int
    - status: text

page test:
  path: "/test"
  title: "Test"
  
  show card "Items" from dataset test_data:
    item:
      type: card
      style: detail
      
      header:
        badges:
          - field: status
            transform: humanize
      
      sections:
        - type: info_grid
          columns: 2
          items:
            - icon: tag
              label: "Status"
              values:
                - field: status
      
      actions:
        - label: "View"
          icon: eye
          action: view_item
          condition: "status == 'active'"
'''
    parser = Parser()
    app = parser.parse(source)
    
    page = app.pages[0]
    statement = page.statements[0]
    
    assert isinstance(statement, ShowCard)
    assert statement.item_config is not None
    assert isinstance(statement.item_config, CardItemConfig)
    assert statement.item_config.type == "card"
    assert statement.item_config.style == "detail"
    
    # Check header
    assert statement.item_config.header is not None
    assert statement.item_config.header.badges is not None
    assert len(statement.item_config.header.badges) == 1
    assert statement.item_config.header.badges[0].field == "status"
    assert statement.item_config.header.badges[0].transform == "humanize"
    
    # Check sections
    assert statement.item_config.sections is not None
    assert len(statement.item_config.sections) == 1
    section = statement.item_config.sections[0]
    assert section.type == "info_grid"
    assert section.columns == 2
    assert section.items is not None
    assert len(section.items) == 1
    
    # Check actions
    assert statement.item_config.actions is not None
    assert len(statement.item_config.actions) == 1
    action = statement.item_config.actions[0]
    assert action.label == "View"
    assert action.icon == "eye"
    assert action.action == "view_item"
    assert action.condition == "status == 'active'"


def test_parse_show_list_basic():
    """Test parsing basic show list statement."""
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
    parser = Parser()
    app = parser.parse(source)
    
    page = app.pages[0]
    statement = page.statements[0]
    
    assert isinstance(statement, ShowList)
    assert statement.title == "Messages"
    assert statement.source_type == "dataset"
    assert statement.source == "messages"
    assert statement.list_type == "conversation"
    assert statement.enable_search is True
    assert statement.columns == 1


def test_parse_card_with_group_by_and_filter():
    """Test parsing card with group_by and filter_by options."""
    source = '''
dataset items:
  fields:
    - id: int
    - category: text
    - status: text

page test:
  path: "/test"
  title: "Test"
  
  show card "Items" from dataset items:
    group_by: "category"
    filter_by: "status == 'active'"
    sort_by: "id desc"
'''
    parser = Parser()
    app = parser.parse(source)
    
    page = app.pages[0]
    statement = page.statements[0]
    
    assert isinstance(statement, ShowCard)
    assert statement.group_by == "category"
    assert statement.filter_by == "status == 'active'"
    assert statement.sort_by == "id desc"


def test_parse_card_footer():
    """Test parsing card with footer configuration."""
    source = '''
dataset items:
  fields:
    - id: int
    - confirmation: text

page test:
  path: "/test"
  title: "Test"
  
  show card "Items" from dataset items:
    item:
      type: card
      
      footer:
        text: "Confirmation: {{ confirmation }}"
        condition: "confirmation != null"
        style: info
'''
    parser = Parser()
    app = parser.parse(source)
    
    page = app.pages[0]
    statement = page.statements[0]
    
    assert isinstance(statement, ShowCard)
    assert statement.item_config is not None
    assert statement.item_config.footer is not None
    footer = statement.item_config.footer
    assert footer.text == "Confirmation: {{ confirmation }}"
    assert footer.condition == "confirmation != null"
    assert footer.style == "info"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
