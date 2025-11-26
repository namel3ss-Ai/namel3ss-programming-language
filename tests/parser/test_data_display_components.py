"""
Test parsing of data display components (ShowDataTable, ShowDataList, ShowStatSummary, 
ShowTimeline, ShowAvatarGroup, ShowDataChart).
"""

import pytest
from namel3ss.parser import Parser
from namel3ss.ast.pages import (
    ShowDataTable,
    ShowDataList,
    ShowStatSummary,
    ShowTimeline,
    ShowAvatarGroup,
    ShowDataChart,
    ColumnConfig,
    ToolbarConfig,
    ListItemConfig,
    SparklineConfig,
    TimelineItem,
    AvatarItem,
    ChartConfig,
    EmptyStateConfig,
)


# =============================================================================
# DATA TABLE TESTS
# =============================================================================


def test_parse_show_data_table_basic():
    """Test parsing basic data table statement."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int
    - name: text

page "Test" at "/test"
  
  show data_table "Users" from dataset test_data:
    columns:
      - field: id
        header: "ID"
      - field: name
        header: "Name"
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    page = app.pages[0]
    statement = page.body[0]
    
    assert isinstance(statement, ShowDataTable)
    assert statement.title == "Users"
    assert statement.source_type == "dataset"
    assert statement.source == "test_data"
    assert len(statement.columns) == 2
    assert isinstance(statement.columns[0], ColumnConfig)
    assert statement.columns[0].field == "id"
    assert statement.columns[0].header == "ID"


def test_parse_show_data_table_with_toolbar():
    """Test parsing data table with toolbar configuration."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int
    - name: text

page "Test" at "/test"
  
  show data_table "Users" from dataset test_data:
    columns:
      - field: name
        header: "Name"
        sortable: true
    
    toolbar:
      searchable: true
      search_fields: ["name", "email"]
      
      filters:
        - field: status
          label: "Status"
          type: select
          options:
            - value: active
              label: "Active"
            - value: inactive
              label: "Inactive"
      
      bulk_actions:
        - label: "Delete Selected"
          action: bulk_delete
          icon: trash
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowDataTable)
    assert statement.toolbar is not None
    assert isinstance(statement.toolbar, ToolbarConfig)
    assert statement.toolbar.searchable is True
    assert statement.toolbar.search_fields == ["name", "email"]
    assert len(statement.toolbar.filters) == 1
    assert statement.toolbar.filters[0]["field"] == "status"
    assert len(statement.toolbar.bulk_actions) == 1
    assert statement.toolbar.bulk_actions[0]["label"] == "Delete Selected"


def test_parse_show_data_table_with_row_actions():
    """Test parsing data table with row actions."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show data_table "Items" from dataset test_data:
    columns:
      - field: id
        header: "ID"
    
    row_actions:
      - label: "View"
        action: view_item
        icon: eye
      - label: "Edit"
        action: edit_item
        icon: edit
        condition: "is_editable == true"
      - label: "Delete"
        action: delete_item
        icon: trash
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowDataTable)
    assert len(statement.row_actions) == 3
    assert statement.row_actions[0]["label"] == "View"
    assert statement.row_actions[1]["condition"] == "is_editable == true"


def test_parse_show_data_table_with_pagination():
    """Test parsing data table with pagination settings."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show data_table "Items" from dataset test_data:
    columns:
      - field: id
        header: "ID"
    
    rows_per_page: 25
    
    empty_state:
      icon: table
      title: "No data"
      message: "No records found"
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowDataTable)
    assert statement.rows_per_page == 25
    assert statement.empty_state is not None
    assert statement.empty_state.title == "No data"


# =============================================================================
# DATA LIST TESTS
# =============================================================================


def test_parse_show_data_list_basic():
    """Test parsing basic data list statement."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int
    - name: text

page "Test" at "/test"
  
  show data_list "Activities" from dataset test_data:
    item:
      title:
        field: name
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowDataList)
    assert statement.title == "Activities"
    assert statement.source_type == "dataset"
    assert statement.source == "test_data"
    assert statement.item_config is not None
    assert isinstance(statement.item_config, ListItemConfig)


def test_parse_show_data_list_with_avatar():
    """Test parsing data list with avatar configuration."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show data_list "Users" from dataset test_data:
    item:
      avatar:
        field: profile_image
        fallback: "?"
        size: lg
      
      title:
        text: "{{ first_name }} {{ last_name }}"
      
      subtitle:
        field: role
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowDataList)
    assert statement.item_config.avatar is not None
    assert statement.item_config.avatar["field"] == "profile_image"
    assert statement.item_config.avatar["size"] == "lg"
    assert statement.item_config.subtitle is not None


def test_parse_show_data_list_with_metadata():
    """Test parsing data list with metadata."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show data_list "Events" from dataset test_data:
    item:
      title:
        field: event_name
      
      metadata:
        - field: timestamp
          icon: clock
          format: relative
        - field: location
          icon: map-pin
        - text: "{{ attendee_count }} attendees"
          icon: users
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowDataList)
    assert len(statement.item_config.metadata) == 3
    assert statement.item_config.metadata[0]["icon"] == "clock"
    assert statement.item_config.metadata[2]["text"] == "{{ attendee_count }} attendees"


def test_parse_show_data_list_with_badge_and_actions():
    """Test parsing data list with badge and actions."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show data_list "Tasks" from dataset test_data:
    item:
      title:
        field: task_name
      
      badge:
        field: status
        style: status_badge
        transform: humanize
      
      actions:
        - label: "Complete"
          action: mark_complete
          icon: check
          condition: "status != 'done'"
        - label: "View"
          action: view_task
          icon: eye
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowDataList)
    assert statement.item_config.badge is not None
    assert statement.item_config.badge["field"] == "status"
    assert len(statement.item_config.actions) == 2


# =============================================================================
# STAT SUMMARY TESTS
# =============================================================================


def test_parse_show_stat_summary_basic():
    """Test parsing basic stat summary statement."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show stat_summary from dataset test_data:
    label: "Total Users"
    value:
      field: total_count
      format: number
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowStatSummary)
    assert statement.label == "Total Users"
    assert statement.value["field"] == "total_count"
    assert statement.value["format"] == "number"


def test_parse_show_stat_summary_with_delta():
    """Test parsing stat summary with delta indicator."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show stat_summary from dataset test_data:
    title: "Revenue"
    label: "Monthly Revenue"
    value:
      field: revenue
      format: currency
    
    delta:
      field: revenue_change
      format: percentage
      show_sign: true
    
    trend:
      field: trend_direction
      up_is_good: true
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowStatSummary)
    assert statement.title == "Revenue"
    assert statement.delta is not None
    assert statement.delta["field"] == "revenue_change"
    assert statement.delta["show_sign"] is True
    assert statement.trend is not None
    assert statement.trend["up_is_good"] is True


def test_parse_show_stat_summary_with_sparkline():
    """Test parsing stat summary with sparkline."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show stat_summary from dataset test_data:
    label: "Page Views"
    value:
      field: total_views
      format: compact
    
    sparkline:
      data: view_history
      color: "#3b82f6"
      height: "40px"
    
    comparison: "vs last month"
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowStatSummary)
    assert statement.sparkline is not None
    assert isinstance(statement.sparkline, SparklineConfig)
    assert statement.sparkline.data == "view_history"
    assert statement.sparkline.color == "#3b82f6"
    assert statement.comparison == "vs last month"


# =============================================================================
# TIMELINE TESTS
# =============================================================================


def test_parse_show_timeline_basic():
    """Test parsing basic timeline statement."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show timeline "Activity Log" from dataset test_data:
    items:
      - timestamp: created_at
        title:
          field: event_name
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowTimeline)
    assert statement.title == "Activity Log"
    assert len(statement.items) == 1
    assert isinstance(statement.items[0], TimelineItem)
    assert statement.items[0].timestamp == "created_at"


def test_parse_show_timeline_with_icons_and_status():
    """Test parsing timeline with icons and status."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show timeline "System Events" from dataset test_data:
    items:
      - timestamp: event_time
        icon: check
        status: success
        title:
          text: "{{ event_type }} completed"
        description:
          field: event_details
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowTimeline)
    assert statement.items[0].icon == "check"
    assert statement.items[0].status == "success"
    assert statement.items[0].description is not None


def test_parse_show_timeline_with_grouping():
    """Test parsing timeline with date grouping."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show timeline "Changelog" from dataset test_data:
    group_by_date: true
    
    items:
      - timestamp: change_date
        title:
          field: change_summary
    
    empty_state:
      icon: clock
      title: "No events"
      message: "No timeline events to display"
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowTimeline)
    assert statement.group_by_date is True
    assert statement.empty_state is not None


# =============================================================================
# AVATAR GROUP TESTS
# =============================================================================


def test_parse_show_avatar_group_basic():
    """Test parsing basic avatar group statement."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show avatar_group "Team Members" from dataset test_data:
    items:
      - name_field: full_name
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowAvatarGroup)
    assert statement.title == "Team Members"
    assert len(statement.items) == 1
    assert isinstance(statement.items[0], AvatarItem)
    assert statement.items[0].name_field == "full_name"


def test_parse_show_avatar_group_with_images_and_status():
    """Test parsing avatar group with images and status."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show avatar_group "Online Users" from dataset test_data:
    items:
      - image_field: avatar_url
        name_field: username
        status_field: online_status
        tooltip_template: "{{ username }} - {{ role }}"
    
    max_visible: 10
    size: lg
    show_status: true
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowAvatarGroup)
    assert statement.items[0].image_field == "avatar_url"
    assert statement.items[0].status_field == "online_status"
    assert statement.items[0].tooltip_template == "{{ username }} - {{ role }}"
    assert statement.max_visible == 10
    assert statement.size == "lg"
    assert statement.show_status is True


# =============================================================================
# DATA CHART TESTS
# =============================================================================


def test_parse_show_data_chart_basic():
    """Test parsing basic data chart statement."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show data_chart "Sales Chart" from dataset test_data:
    chart:
      type: line
      x_axis: month
      series:
        - data_key: sales
          label: "Sales"
          color: "#3b82f6"
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowDataChart)
    assert statement.title == "Sales Chart"
    assert statement.chart_config is not None
    assert isinstance(statement.chart_config, ChartConfig)
    assert statement.chart_config.type == "line"
    assert statement.chart_config.x_axis == "month"
    assert len(statement.chart_config.series) == 1


def test_parse_show_data_chart_bar():
    """Test parsing bar chart configuration."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show data_chart "Revenue by Product" from dataset test_data:
    chart:
      type: bar
      x_axis: product_name
      y_axis: revenue
      legend: true
      grid: true
      height: "400px"
      
      series:
        - data_key: q1_revenue
          label: "Q1"
          color: "#10b981"
        - data_key: q2_revenue
          label: "Q2"
          color: "#3b82f6"
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowDataChart)
    assert statement.chart_config.type == "bar"
    assert statement.chart_config.legend is True
    assert statement.chart_config.grid is True
    assert statement.chart_config.height == "400px"
    assert len(statement.chart_config.series) == 2


def test_parse_show_data_chart_pie():
    """Test parsing pie chart configuration."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show data_chart "Market Share" from dataset test_data:
    chart:
      type: pie
      series:
        - data_key: percentage
          label: "Share"
    
    empty_state:
      icon: chart
      title: "No data available"
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowDataChart)
    assert statement.chart_config.type == "pie"
    assert statement.empty_state is not None


def test_parse_show_data_chart_multi_series():
    """Test parsing chart with multiple series."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show data_chart "Performance Metrics" from dataset test_data:
    chart:
      type: area
      x_axis: date
      legend: true
      grid: true
      
      series:
        - data_key: cpu_usage
          label: "CPU"
          color: "#ef4444"
        - data_key: memory_usage
          label: "Memory"
          color: "#f59e0b"
        - data_key: disk_usage
          label: "Disk"
          color: "#10b981"
'''
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    
    assert isinstance(statement, ShowDataChart)
    assert statement.chart_config.type == "area"
    assert len(statement.chart_config.series) == 3
    assert statement.chart_config.series[0]["data_key"] == "cpu_usage"
    assert statement.chart_config.series[1]["color"] == "#f59e0b"


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


def test_parse_data_table_missing_columns():
    """Test that data table requires columns."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show data_table "Items" from dataset test_data:
    toolbar:
      searchable: true
'''
    parser = Parser()
    
    with pytest.raises(Exception):  # Should raise error for missing columns
        parser.parse(source)


def test_parse_data_list_missing_item():
    """Test that data list requires item configuration."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show data_list "Items" from dataset test_data:
    empty_state:
      title: "No items"
'''
    parser = Parser()
    
    with pytest.raises(Exception):  # Should raise error for missing item config
        parser.parse(source)


def test_parse_stat_summary_missing_value():
    """Test that stat summary requires value configuration."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show stat_summary from dataset test_data:
    label: "Total"
    delta:
      field: change
'''
    parser = Parser()
    
    with pytest.raises(Exception):  # Should raise error for missing value
        parser.parse(source)


def test_parse_data_chart_missing_series():
    """Test that data chart requires series configuration."""
    source = '''
app \"Test App\"

dataset "test_data" from inline:
  fields:
    - id: int

page "Test" at "/test"
  
  show data_chart "Chart" from dataset test_data:
    chart:
      type: line
      x_axis: date
'''
    parser = Parser()
    
    with pytest.raises(Exception):  # Should raise error for missing series
        parser.parse(source)
