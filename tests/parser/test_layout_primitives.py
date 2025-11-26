"""Tests for layout primitive parsing (stack, grid, split, tabs, accordion)."""

import pytest
from namel3ss.parser import Parser
from namel3ss.ast import (
    StackLayout,
    GridLayout,
    SplitLayout,
    TabsLayout,
    AccordionLayout,
    ShowCard,
    ShowChart,
)


def test_parse_stack_layout_basic():
    """Test parsing basic stack layout."""
    source = """
page test_page:
  path: "/test"
  
  layout stack:
    direction: vertical
    gap: medium
    align: center
    justify: start
    wrap: false
    children:
      - show card "Data" from dataset items
"""
    parser = Parser(source)
    app = parser.parse()
    
    assert len(app.body) == 1
    page = app.body[0]
    assert len(page.body) == 1
    
    stack = page.body[0]
    assert isinstance(stack, StackLayout)
    assert stack.direction == "vertical"
    assert stack.gap == "medium"
    assert stack.align == "center"
    assert stack.justify == "start"
    assert stack.wrap is False
    assert len(stack.children) == 1
    assert isinstance(stack.children[0], ShowCard)


def test_parse_stack_layout_horizontal_with_numeric_gap():
    """Test horizontal stack with numeric gap."""
    source = """
page test_page:
  path: "/test"
  
  layout stack:
    direction: horizontal
    gap: 24
    align: start
    children:
      - show card "Card 1" from dataset data1
      - show card "Card 2" from dataset data2
"""
    parser = Parser(source)
    app = parser.parse()
    
    stack = app.body[0].body[0]
    assert isinstance(stack, StackLayout)
    assert stack.direction == "horizontal"
    assert stack.gap == 24
    assert len(stack.children) == 2


def test_parse_grid_layout_basic():
    """Test parsing basic grid layout."""
    source = """
page test_page:
  path: "/test"
  
  layout grid:
    columns: 3
    gap: large
    responsive: true
    children:
      - show card "Sales" from dataset sales
      - show card "Leads" from dataset leads
      - show chart "Growth" from dataset metrics
"""
    parser = Parser(source)
    app = parser.parse()
    
    grid = app.body[0].body[0]
    assert isinstance(grid, GridLayout)
    assert grid.columns == 3
    assert grid.gap == "large"
    assert grid.responsive is True
    assert len(grid.children) == 3


def test_parse_grid_layout_with_min_column_width():
    """Test grid with min column width."""
    source = """
page test_page:
  path: "/test"
  
  layout grid:
    columns: auto
    min_column_width: 300px
    gap: medium
    children:
      - show card "Card" from dataset data
"""
    parser = Parser(source)
    app = parser.parse()
    
    grid = app.body[0].body[0]
    assert isinstance(grid, GridLayout)
    assert grid.columns == "auto"
    assert grid.min_column_width == "300px"


def test_parse_split_layout():
    """Test parsing split layout."""
    source = """
page test_page:
  path: "/test"
  
  layout split:
    ratio: 0.3
    resizable: true
    orientation: horizontal
    left:
      - show card "Orders" from dataset orders
    right:
      - show card "Details" from dataset order_details
"""
    parser = Parser(source)
    app = parser.parse()
    
    split = app.body[0].body[0]
    assert isinstance(split, SplitLayout)
    assert split.ratio == 0.3
    assert split.resizable is True
    assert split.orientation == "horizontal"
    assert len(split.left) == 1
    assert len(split.right) == 1
    assert isinstance(split.left[0], ShowCard)
    assert isinstance(split.right[0], ShowCard)


def test_parse_tabs_layout():
    """Test parsing tabs layout."""
    source = """
page test_page:
  path: "/test"
  
  layout tabs:
    default_tab: overview
    persist_state: true
    tabs:
      - id: overview
        label: "Overview"
        icon: home
        badge: "3"
        content:
          - show card "Summary" from dataset summary
          
      - id: details
        label: "Details"
        icon: list
        content:
          - show card "Data" from dataset items
"""
    parser = Parser(source)
    app = parser.parse()
    
    tabs = app.body[0].body[0]
    assert isinstance(tabs, TabsLayout)
    assert tabs.default_tab == "overview"
    assert tabs.persist_state is True
    assert len(tabs.tabs) == 2
    
    # Check first tab
    tab1 = tabs.tabs[0]
    assert tab1.id == "overview"
    assert tab1.label == "Overview"
    assert tab1.icon == "home"
    assert tab1.badge == "3"
    assert len(tab1.content) == 1
    
    # Check second tab
    tab2 = tabs.tabs[1]
    assert tab2.id == "details"
    assert tab2.label == "Details"
    assert tab2.icon == "list"


def test_parse_accordion_layout():
    """Test parsing accordion layout."""
    source = """
page test_page:
  path: "/test"
  
  layout accordion:
    multiple: true
    items:
      - id: section1
        title: "Personal Information"
        description: "Your profile details"
        icon: user
        default_open: true
        content:
          - show card "Profile" from dataset profile
          
      - id: section2
        title: "Settings"
        icon: settings
        default_open: false
        content:
          - show card "Preferences" from dataset preferences
"""
    parser = Parser(source)
    app = parser.parse()
    
    accordion = app.body[0].body[0]
    assert isinstance(accordion, AccordionLayout)
    assert accordion.multiple is True
    assert len(accordion.items) == 2
    
    # Check first item
    item1 = accordion.items[0]
    assert item1.id == "section1"
    assert item1.title == "Personal Information"
    assert item1.description == "Your profile details"
    assert item1.icon == "user"
    assert item1.default_open is True
    assert len(item1.content) == 1
    
    # Check second item
    item2 = accordion.items[1]
    assert item2.id == "section2"
    assert item2.title == "Settings"
    assert item2.default_open is False


def test_parse_nested_layouts():
    """Test parsing nested layout primitives."""
    source = """
page test_page:
  path: "/test"
  
  layout stack:
    direction: vertical
    gap: large
    children:
      - layout grid:
          columns: 2
          children:
            - show card "Card 1" from dataset data1
            - show card "Card 2" from dataset data2
      
      - layout split:
          ratio: 0.5
          left:
            - show card "Left" from dataset left_data
          right:
            - show card "Right" from dataset right_data
"""
    parser = Parser(source)
    app = parser.parse()
    
    stack = app.body[0].body[0]
    assert isinstance(stack, StackLayout)
    assert len(stack.children) == 2
    
    # First child is a grid
    grid = stack.children[0]
    assert isinstance(grid, GridLayout)
    assert grid.columns == 2
    assert len(grid.children) == 2
    
    # Second child is a split
    split = stack.children[1]
    assert isinstance(split, SplitLayout)
    assert split.ratio == 0.5


def test_parse_tabs_validation_error():
    """Test that tabs layout validates default_tab."""
    source = """
page test_page:
  path: "/test"
  
  layout tabs:
    default_tab: nonexistent
    tabs:
      - id: tab1
        label: "Tab 1"
        content:
          - show card "Data" from dataset data
"""
    parser = Parser(source)
    
    with pytest.raises(Exception) as exc_info:
        app = parser.parse()
    
    assert "default_tab" in str(exc_info.value).lower()


def test_parse_stack_invalid_direction():
    """Test that stack validates direction values."""
    source = """
page test_page:
  path: "/test"
  
  layout stack:
    direction: diagonal
    children:
      - show card "Data" from dataset data
"""
    parser = Parser(source)
    
    with pytest.raises(Exception) as exc_info:
        app = parser.parse()
    
    assert "direction" in str(exc_info.value).lower()


def test_parse_split_invalid_ratio():
    """Test that split validates ratio range."""
    source = """
page test_page:
  path: "/test"
  
  layout split:
    ratio: 1.5
    left:
      - show card "Left" from dataset left_data
    right:
      - show card "Right" from dataset right_data
"""
    parser = Parser(source)
    
    with pytest.raises(Exception) as exc_info:
        app = parser.parse()
    
    assert "ratio" in str(exc_info.value).lower()
