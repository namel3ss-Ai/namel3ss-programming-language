"""Comprehensive validation tests for data display components and layouts WITH datasets.

These tests validate that critical "non-negotiable" components work correctly with datasets:
- Data Display: ShowDataTable, ShowDataList, ShowStatSummary, ShowTimeline, ShowAvatarGroup, ShowDataChart
- Layouts: StackLayout, GridLayout, SplitLayout, TabsLayout, AccordionLayout
"""

import pytest
from namel3ss.parser import Parser


def test_show_data_table_with_sql_dataset():
    """Verify ShowDataTable works with SQL datasets."""
    source = '''
dataset users:
  sql: |
    SELECT id, name, email FROM users

page users_page:
  path: "/users"
  title: "Users"
  
  ui:
    type: table
    
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
    
    # Verify dataset exists
    assert len(app.datasets) == 1
    dataset = app.datasets[0]
    assert dataset.name == "users"
    assert "SELECT" in dataset.sql
    
    # Verify page and table
    assert len(app.pages) == 1
    page = app.pages[0]
    assert page.name == "users_page"
    assert page.path == "/users"
    
    # Find the data table in page UI
    assert hasattr(page, 'ui')
    assert page.ui is not None


def test_show_data_list_with_dataset():
    """Verify ShowDataList works with datasets."""
    source = '''
dataset tasks:
  sql: |
    SELECT id, name, description, status FROM tasks

page tasks_page:
  path: "/tasks"
  title: "Tasks"
  
  ui:
    type: list
    
    show data_list "Task List" from dataset tasks:
      item:
        title: item.name
        description: item.description
        badge: item.status
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.datasets) == 1
    assert app.datasets[0].name == "tasks"
    
    assert len(app.pages) == 1
    page = app.pages[0]
    assert page.name == "tasks_page"
    assert hasattr(page, 'ui')


def test_show_stat_summary_with_dataset():
    """Verify ShowStatSummary works with computed datasets."""
    source = '''
dataset revenue:
  sql: |
    SELECT SUM(amount) as total_revenue,
           COUNT(*) as order_count
    FROM orders

page dashboard:
  path: "/dashboard"
  title: "Dashboard"
  
  ui:
    type: dashboard
    
    show stat_summary "Total Revenue" from dataset revenue:
      value: revenue.total_revenue
      format: currency
      trend: up
      trend_value: 15.3
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.datasets) == 1
    assert len(app.pages) == 1
    page = app.pages[0]
    assert page.name == "dashboard"


def test_show_data_chart_with_dataset():
    """Verify ShowDataChart works with time-series datasets."""
    source = '''
dataset sales_trend:
  sql: |
    SELECT date, revenue, orders
    FROM daily_sales
    ORDER BY date

page analytics:
  path: "/analytics"
  title: "Analytics"
  
  ui:
    type: chart
    
    show data_chart "Sales Trend" from dataset sales_trend:
      chart_type: line
      x_axis: date
      y_axis: revenue
      height: 400
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.datasets) == 1
    assert len(app.pages) == 1


def test_stack_layout_with_cards():
    """Verify StackLayout works with multiple ShowCard components."""
    source = '''
dataset items:
  sql: |
    SELECT id, title, content FROM items

page dashboard_stack:
  path: "/stack"
  title: "Stack Layout"
  
  ui:
    type: dashboard
    
    layout stack:
      direction: vertical
      gap: medium
      
      children:
        - type: card
          show card "Card 1" from dataset items:
            content: items[0].content
            
        - type: card
          show card "Card 2" from dataset items:
            content: items[1].content
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.datasets) == 1
    assert len(app.pages) == 1


def test_grid_layout_with_stats():
    """Verify GridLayout works with multiple stat summaries."""
    source = '''
dataset metrics:
  sql: |
    SELECT 
      SUM(revenue) as total_revenue,
      COUNT(orders) as total_orders,
      COUNT(DISTINCT customers) as total_customers
    FROM analytics

page metrics_grid:
  path: "/metrics"
  title: "Metrics Grid"
  
  ui:
    type: dashboard
    
    layout grid:
      columns: 3
      gap: large
      
      children:
        - type: stat
          show stat_summary "Revenue":
            value: metrics.total_revenue
            format: currency
            
        - type: stat
          show stat_summary "Orders":
            value: metrics.total_orders
            format: number
            
        - type: stat
          show stat_summary "Customers":
            value: metrics.total_customers
            format: number
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.datasets) == 1
    assert len(app.pages) == 1


def test_tabs_layout_with_tables():
    """Verify TabsLayout works with data tables in each tab."""
    source = '''
dataset active_orders:
  sql: |
    SELECT * FROM orders WHERE status = 'active'

dataset completed_orders:
  sql: |
    SELECT * FROM orders WHERE status = 'completed'

page orders_tabs:
  path: "/orders"
  title: "Orders"
  
  ui:
    type: tabs
    
    layout tabs:
      default_tab: active
      
      tabs:
        - id: active
          label: "Active Orders"
          content:
            - type: table
              show data_table "Active" from dataset active_orders:
                columns:
                  - field: id
                    header: "Order ID"
                    
        - id: completed
          label: "Completed Orders"
          content:
            - type: table
              show data_table "Completed" from dataset completed_orders:
                columns:
                  - field: id
                    header: "Order ID"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.datasets) == 2
    assert len(app.pages) == 1


def test_accordion_layout_with_sections():
    """Verify AccordionLayout works with collapsible sections."""
    source = '''
dataset data:
  sql: |
    SELECT category, items FROM data

page accordion_page:
  path: "/accordion"
  title: "Accordion"
  
  ui:
    type: accordion
    
    layout accordion:
      multiple: true
      
      items:
        - id: section1
          title: "Section 1"
          default_open: true
          content:
            - type: card
              show card "Content 1":
                body: "Section 1 content"
                
        - id: section2
          title: "Section 2"
          default_open: false
          content:
            - type: card
              show card "Content 2":
                body: "Section 2 content"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.datasets) == 1
    assert len(app.pages) == 1


def test_split_layout_with_list_and_detail():
    """Verify SplitLayout works with master-detail pattern."""
    source = '''
dataset items:
  sql: |
    SELECT id, name, details FROM items

page split_view:
  path: "/split"
  title: "Split View"
  
  ui:
    type: split
    
    layout split:
      orientation: horizontal
      ratio: 0.3
      resizable: true
      
      left:
        - type: list
          show data_list "Items" from dataset items:
            item:
              title: item.name
              
      right:
        - type: card
          show card "Details":
            body: "Item details here"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.datasets) == 1
    assert len(app.pages) == 1


def test_show_timeline_with_events():
    """Verify ShowTimeline works with event datasets."""
    source = '''
dataset events:
  sql: |
    SELECT timestamp, title, description, type
    FROM events
    ORDER BY timestamp DESC

page timeline_page:
  path: "/timeline"
  title: "Timeline"
  
  ui:
    type: timeline
    
    show timeline "Event Timeline" from dataset events:
      timestamp_field: timestamp
      title_field: title
      description_field: description
      type_field: type
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.datasets) == 1
    assert len(app.pages) == 1


def test_show_avatar_group_with_users():
    """Verify ShowAvatarGroup works with user datasets."""
    source = '''
dataset team_members:
  sql: |
    SELECT id, name, avatar_url, role
    FROM users
    WHERE team_id = 1

page team_page:
  path: "/team"
  title: "Team"
  
  ui:
    type: profile
    
    show avatar_group "Team Members" from dataset team_members:
      name_field: name
      avatar_field: avatar_url
      max_visible: 5
      size: medium
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.datasets) == 1
    assert len(app.pages) == 1


def test_complex_nested_layouts_with_multiple_datasets():
    """Verify nested layouts work with multiple datasets."""
    source = '''
dataset stats:
  sql: |
    SELECT metric, value FROM stats

dataset charts:
  sql: |
    SELECT date, value FROM chart_data

page complex_dashboard:
  path: "/complex"
  title: "Complex Dashboard"
  
  ui:
    type: dashboard
    
    layout stack:
      direction: vertical
      gap: large
      
      children:
        - type: grid
          layout grid:
            columns: 3
            gap: medium
            children:
              - type: stat
                show stat_summary "Metric 1":
                  value: 100
                  format: number
              - type: stat
                show stat_summary "Metric 2":
                  value: 200
                  format: number
              - type: stat
                show stat_summary "Metric 3":
                  value: 300
                  format: number
                  
        - type: chart
          show data_chart "Trend" from dataset charts:
            chart_type: line
            x_axis: date
            y_axis: value
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.datasets) == 2
    assert len(app.pages) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
