"""Comprehensive validation tests for data display components and layouts WITH datasets.

These tests validate that critical "non-negotiable" components work correctly with datasets:
- Data Display: ShowDataTable, ShowDataList, ShowStatSummary, ShowTimeline, ShowAvatarGroup, ShowDataChart
- Layouts: StackLayout, GridLayout, SplitLayout, TabsLayout, AccordionLayout
"""

import pytest
from namel3ss.parser import Parser


def test_show_data_table_with_dataset():
    """Verify ShowDataTable works with external datasets."""
    source = '''
app "Test App"

dataset "users" from "db://users"

page "Users Page" at "/users":
  show data_table "All Users" from dataset users:
    columns:
      - field: name
        header: "Name"
      - field: email
        header: "Email"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    # Verify dataset
    assert len(app.datasets) == 1
    assert app.datasets[0].name == "users"
    assert app.datasets[0].source == "db://users"
    
    # Verify page and table
    assert len(app.pages) == 1
    page = app.pages[0]
    assert page.name == "Users Page"
    assert len(page.body) == 1
    
    table = page.body[0]
    assert table.__class__.__name__ == "ShowDataTable"
    assert table.title == "All Users"
    print(f"✓ ShowDataTable parses with dataset reference")


def test_show_data_list_with_dataset():
    """Verify ShowDataList works with datasets."""
    source = '''
app "Test App"

dataset "tasks" from "api://tasks"

page "Tasks" at "/tasks":
  show data_list "Task List" from dataset tasks:
    item:
      title: item.name
      description: item.description
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.datasets) == 1
    assert app.datasets[0].name == "tasks"
    assert len(app.pages) == 1
    
    list_comp = app.pages[0].body[0]
    assert list_comp.__class__.__name__ == "ShowDataList"
    assert list_comp.title == "Task List"
    print(f"✓ ShowDataList parses with dataset reference")


def test_show_stat_summary():
    """Verify ShowStatSummary works."""
    source = '''
app "Test App"

dataset "metrics" from "db://metrics"

page "Dashboard" at "/dashboard":
  show stat_summary "Total Users" from dataset metrics:
    value: metrics.total_users
    format: number
    trend: up
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    stat = app.pages[0].body[0]
    assert stat.__class__.__name__ == "ShowStatSummary"
    assert stat.title == "Total Users"
    print(f"✓ ShowStatSummary parses with dataset reference")


def test_show_data_chart():
    """Verify ShowDataChart works with datasets."""
    source = '''
app "Test App"

dataset "sales" from "db://sales"

page "Analytics" at "/analytics":
  show data_chart "Sales Trend" from dataset sales:
    chart_type: line
    x_axis: date
    y_axis: revenue
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    chart = app.pages[0].body[0]
    assert chart.__class__.__name__ == "ShowDataChart"
    assert chart.title == "Sales Trend"
    print(f"✓ ShowDataChart parses with dataset reference")


def test_show_timeline():
    """Verify ShowTimeline works with datasets."""
    source = '''
app "Test App"

dataset "events" from "db://events"

page "Timeline" at "/timeline":
  show timeline "Event History" from dataset events:
    timestamp_field: created_at
    title_field: title
    description_field: body
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    timeline = app.pages[0].body[0]
    assert timeline.__class__.__name__ == "ShowTimeline"
    assert timeline.title == "Event History"
    print(f"✓ ShowTimeline parses with dataset reference")


def test_show_avatar_group():
    """Verify ShowAvatarGroup works with datasets."""
    source = '''
app "Test App"

dataset "team" from "db://team_members"

page "Team" at "/team":
  show avatar_group "Team Members" from dataset team:
    name_field: name
    avatar_field: avatar_url
    max_visible: 5
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    avatar_group = app.pages[0].body[0]
    assert avatar_group.__class__.__name__ == "ShowAvatarGroup"
    assert avatar_group.title == "Team Members"
    print(f"✓ ShowAvatarGroup parses with dataset reference")


def test_stack_layout():
    """Verify StackLayout works."""
    source = '''
app "Test App"

dataset "items" from "db://items"

page "Dashboard" at "/dashboard":
  layout stack:
    direction: vertical
    gap: medium
    children:
      - show card "Card 1" from dataset items
      - show card "Card 2" from dataset items
      - show card "Card 3" from dataset items
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    layout = app.pages[0].body[0]
    assert layout.__class__.__name__ == "StackLayout"
    assert layout.direction == "vertical"
    assert layout.gap == "medium"
    assert len(layout.children) == 3
    print(f"✓ StackLayout parses with {len(layout.children)} children")


def test_grid_layout():
    """Verify GridLayout works."""
    source = '''
app "Test App"

dataset "data" from "db://data"

page "Grid" at "/grid":
  layout grid:
    columns: 3
    gap: large
    children:
      - show card "A" from dataset data
      - show card "B" from dataset data
      - show card "C" from dataset data
      - show card "D" from dataset data
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    layout = app.pages[0].body[0]
    assert layout.__class__.__name__ == "GridLayout"
    assert layout.columns == 3
    assert layout.gap == "large"
    assert len(layout.children) == 4
    print(f"✓ GridLayout parses with {layout.columns} columns and {len(layout.children)} children")


def test_tabs_layout():
    """Verify TabsLayout works with multiple tabs."""
    source = '''
app "Test App"

dataset "data" from "db://data"

page "Tabs Page" at "/tabs":
  layout tabs:
    default_tab: overview
    tabs:
      - id: overview
        label: "Overview"
        content:
          - show card "Summary" from dataset data
      - id: details
        label: "Details"
        content:
          - show card "Details" from dataset data
      - id: settings
        label: "Settings"
        content:
          - show card "Config" from dataset data
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    layout = app.pages[0].body[0]
    assert layout.__class__.__name__ == "TabsLayout"
    assert layout.default_tab == "overview"
    assert len(layout.tabs) == 3
    
    # Verify tab structure
    tab1 = layout.tabs[0]
    assert tab1.id == "overview"
    assert tab1.label == "Overview"
    assert len(tab1.content) == 1
    print(f"✓ TabsLayout parses with {len(layout.tabs)} tabs")


def test_accordion_layout():
    """Verify AccordionLayout works."""
    source = '''
app "Test App"

dataset "data" from "db://data"

page "Accordion" at "/accordion":
  layout accordion:
    multiple: true
    items:
      - id: section1
        title: "Section 1"
        default_open: true
        content:
          - show card "Content 1" from dataset data
      - id: section2
        title: "Section 2"
        default_open: false
        content:
          - show card "Content 2" from dataset data
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    layout = app.pages[0].body[0]
    assert layout.__class__.__name__ == "AccordionLayout"
    assert layout.multiple == True
    assert len(layout.items) == 2
    
    item1 = layout.items[0]
    assert item1.id == "section1"
    assert item1.title == "Section 1"
    assert item1.default_open == True
    print(f"✓ AccordionLayout parses with {len(layout.items)} sections")


def test_split_layout():
    """Verify SplitLayout works."""
    source = '''
app "Test App"

dataset "data" from "db://data"

page "Split" at "/split":
  layout split:
    orientation: horizontal
    ratio: 0.3
    resizable: true
    left:
      - show card "Left Panel" from dataset data
    right:
      - show card "Right Panel" from dataset data
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    layout = app.pages[0].body[0]
    assert layout.__class__.__name__ == "SplitLayout"
    assert layout.orientation == "horizontal"
    assert layout.ratio == 0.3
    assert layout.resizable == True
    assert len(layout.left) == 1
    assert len(layout.right) == 1
    print(f"✓ SplitLayout parses with {layout.orientation} orientation")


def test_complex_nested_layouts():
    """Verify complex nested layouts work correctly."""
    source = '''
app "Test App"

dataset "items" from "db://items"
dataset "stats" from "db://stats"

page "Complex" at "/complex":
  layout stack:
    direction: vertical
    gap: large
    children:
      - show card "Header" from dataset items
      - layout grid:
          columns: 2
          gap: medium
          children:
            - show stat_summary "Metric 1" from dataset stats:
                value: 100
                format: number
            - show stat_summary "Metric 2" from dataset stats:
                value: 200
                format: number
      - layout tabs:
          tabs:
            - id: tab1
              label: "Tab 1"
              content:
                - show card "Tab Content" from dataset items
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    # Verify multiple datasets
    assert len(app.datasets) == 2
    
    stack = app.pages[0].body[0]
    assert stack.__class__.__name__ == "StackLayout"
    assert len(stack.children) == 3
    
    # First child is a card
    assert stack.children[0].__class__.__name__ == "ShowCard"
    
    # Second child is a grid layout
    grid = stack.children[1]
    assert grid.__class__.__name__ == "GridLayout"
    assert grid.columns == 2
    assert len(grid.children) == 2
    
    # Third child is tabs layout
    tabs = stack.children[2]
    assert tabs.__class__.__name__ == "TabsLayout"
    assert len(tabs.tabs) == 1
    print(f"✓ Complex nested layouts parse correctly")


def test_all_data_display_components_together():
    """Verify all data display components can coexist."""
    source = '''
app "Test App"

dataset "users" from "db://users"
dataset "events" from "db://events"
dataset "sales" from "db://sales"
dataset "team" from "db://team"

page "Dashboard" at "/dashboard":
  layout stack:
    direction: vertical
    gap: medium
    children:
      - show data_table "Users" from dataset users:
          columns:
            - field: name
              header: "Name"
      - show data_list "Events" from dataset events:
          item:
            title: item.title
      - show stat_summary "Total" from dataset sales:
          value: 1000
          format: currency
      - show data_chart "Trend" from dataset sales:
          chart_type: line
          x_axis: date
          y_axis: amount
      - show timeline "History" from dataset events:
          timestamp_field: created_at
          title_field: title
      - show avatar_group "Team" from dataset team:
          name_field: name
          avatar_field: avatar
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    # Verify datasets
    assert len(app.datasets) == 4
    
    # Verify page with stack layout
    layout = app.pages[0].body[0]
    assert layout.__class__.__name__ == "StackLayout"
    assert len(layout.children) == 6
    
    # Verify each component type
    assert layout.children[0].__class__.__name__ == "ShowDataTable"
    assert layout.children[1].__class__.__name__ == "ShowDataList"
    assert layout.children[2].__class__.__name__ == "ShowStatSummary"
    assert layout.children[3].__class__.__name__ == "ShowDataChart"
    assert layout.children[4].__class__.__name__ == "ShowTimeline"
    assert layout.children[5].__class__.__name__ == "ShowAvatarGroup"
    print(f"✓ All 6 data display components work together!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
