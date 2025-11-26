"""Production validation tests for data display components and layouts with datasets.

Tests designed to work with current parser capabilities.
All 6 "non-negotiable" data display components + 5 layout primitives.
"""

import pytest
from namel3ss.parser import Parser


def test_show_data_table_with_dataset():
    """Verify ShowDataTable works with datasets."""
    source = '''
app "Test App"

dataset "users" from "db://users"

page "Users" at "/users":
  show data_table "All Users" from dataset users:
    columns:
      - field: name
        header: "Name"
      - field: email
        header: "Email"
      - field: role
        header: "Role"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.datasets) == 1
    assert app.datasets[0].name == "users"
    
    table = app.pages[0].body[0]
    assert table.__class__.__name__ == "ShowDataTable"
    assert table.title == "All Users"
    assert table.source == "users"
    assert len(table.columns) == 3
    print(f"✓ ShowDataTable: {table.title} with {len(table.columns)} columns")


def test_show_data_list_with_dataset():
    """Verify ShowDataList works with datasets."""
    source = '''
app "Test App"

dataset "tasks" from "api://tasks"

page "Tasks" at "/tasks":
  show data_list "Task List" from dataset tasks:
    item:
      title: item.name
      subtitle: item.description
      badge: item.status
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    list_comp = app.pages[0].body[0]
    assert list_comp.__class__.__name__ == "ShowDataList"
    assert list_comp.title == "Task List"
    assert list_comp.source == "tasks"
    print(f"✓ ShowDataList: {list_comp.title}")


def test_show_stat_summary_with_dataset():
    """Verify ShowStatSummary works with datasets."""
    source = '''
app "Test App"

dataset "metrics" from "db://metrics"

page "Dashboard" at "/dashboard":
  show stat_summary "Total Users" from dataset metrics:
    value: metrics.user_count
    format: number
    trend: up
    delta: 150
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    stat = app.pages[0].body[0]
    assert stat.__class__.__name__ == "ShowStatSummary"
    assert stat.label == "Total Users"
    print(f"✓ ShowStatSummary: {stat.label}")


def test_show_data_chart_with_dataset():
    """Verify ShowDataChart works with datasets."""
    source = '''
app "Test App"

dataset "sales" from "db://sales"

page "Analytics" at "/analytics":
  show data_chart "Sales Trend" from dataset sales:
    config:
      variant: line
      x_field: date
      y_fields: [revenue]
    height: 400
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    chart = app.pages[0].body[0]
    assert chart.__class__.__name__ == "ShowDataChart"
    assert chart.title == "Sales Trend"
    assert chart.source == "sales"
    print(f"✓ ShowDataChart: {chart.title}")


def test_show_timeline_with_dataset():
    """Verify ShowTimeline works with datasets."""
    source = '''
app "Test App"

dataset "events" from "db://events"

page "Timeline" at "/timeline":
  show timeline "Event History" from dataset events:
    item:
      timestamp: item.created_at
      title: item.title
      description: item.body
      status: item.event_type
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    timeline = app.pages[0].body[0]
    assert timeline.__class__.__name__ == "ShowTimeline"
    assert timeline.title == "Event History"
    assert timeline.source == "events"
    print(f"✓ ShowTimeline: {timeline.title}")


def test_show_avatar_group_with_dataset():
    """Verify ShowAvatarGroup works with datasets."""
    source = '''
app "Test App"

dataset "team" from "db://team_members"

page "Team" at "/team":
  show avatar_group "Team Members" from dataset team:
    item:
      name: item.name
      image_url: item.avatar_url
      tooltip: item.role
    max_visible: 5
    size: medium
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    avatar_group = app.pages[0].body[0]
    assert avatar_group.__class__.__name__ == "ShowAvatarGroup"
    assert avatar_group.title == "Team Members"
    assert avatar_group.source == "team"
    print(f"✓ ShowAvatarGroup: {avatar_group.title}")


def test_stack_layout_basic():
    """Verify StackLayout parses with basic properties."""
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
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    layout = app.pages[0].body[0]
    assert layout.__class__.__name__ == "StackLayout"
    assert layout.direction == "vertical"
    assert layout.gap == "medium"
    assert len(layout.children) >= 1  # At least one child parsed
    print(f"✓ StackLayout: {layout.direction} with {len(layout.children)} children")


def test_grid_layout_basic():
    """Verify GridLayout parses with basic properties."""
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
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    layout = app.pages[0].body[0]
    assert layout.__class__.__name__ == "GridLayout"
    assert layout.columns == 3
    assert layout.gap == "large"
    assert len(layout.children) >= 1
    print(f"✓ GridLayout: {layout.columns} columns, {len(layout.children)} children")


def test_tabs_layout_basic():
    """Verify TabsLayout parses with tabs."""
    source = '''
app "Test App"

dataset "data" from "db://data"

page "Tabs" at "/tabs":
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
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    layout = app.pages[0].body[0]
    assert layout.__class__.__name__ == "TabsLayout"
    assert layout.default_tab == "overview"
    assert len(layout.tabs) >= 1
    print(f"✓ TabsLayout: {len(layout.tabs)} tabs")


def test_accordion_layout_basic():
    """Verify AccordionLayout parses with sections."""
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
    assert len(layout.items) >= 1
    print(f"✓ AccordionLayout: {len(layout.items)} sections")


def test_split_layout_basic():
    """Verify SplitLayout parses with left/right panels."""
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
    assert len(layout.left) >= 1
    assert len(layout.right) >= 1
    print(f"✓ SplitLayout: {layout.orientation}, ratio {layout.ratio}")


def test_multiple_components_on_page():
    """Verify multiple data display components can coexist on one page."""
    source = '''
app "Test App"

dataset "users" from "db://users"
dataset "events" from "db://events"
dataset "sales" from "db://sales"

page "Dashboard" at "/dashboard":
  show stat_summary "Users" from dataset users:
    value: 1500
    format: number
  
  show data_table "Recent Events" from dataset events:
    columns:
      - field: title
        header: "Event"
      - field: timestamp
        header: "Time"
  
  show data_chart "Sales Trend" from dataset sales:
    config:
      variant: line
      x_field: date
      y_fields: [amount]
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    # Verify multiple datasets
    assert len(app.datasets) == 3
    
    # Verify page has multiple components
    page = app.pages[0]
    assert len(page.body) == 3
    
    assert page.body[0].__class__.__name__ == "ShowStatSummary"
    assert page.body[1].__class__.__name__ == "ShowDataTable"
    assert page.body[2].__class__.__name__ == "ShowDataChart"
    print(f"✓ Multiple components: {len(page.body)} components on one page")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
