"""Tests for data display and layout components WITHOUT datasets at top level.

Since the modern parser has issues with dataset + page combinations,
these tests validate components work by using datasets defined AFTER pages,
or by testing component parsing directly without datasets.
"""

import pytest
from namel3ss.parser import Parser


def test_show_stat_summary_parses():
    """Verify ShowStatSummary parses correctly."""
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
    assert stat.label == "Total Users"  # It's 'label', not 'title'
    print(f"✓ ShowStatSummary parses with label: {stat.label}")


def test_show_form_with_dataset():
    """Verify forms work with datasets (baseline test)."""
    source = '''
app "Test App"

dataset "users" from "db://users"

page "Form Page" at "/form":
  show form "User Form":
    fields:
      - name: email
        component: text_input
        label: "Email"
        required: true
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    # Verify dataset
    assert len(app.datasets) == 1
    assert app.datasets[0].name == "users"
    
    # Verify page
    assert len(app.pages) == 1
    page = app.pages[0]
    assert page.name == "Form Page"
    
    # Verify form
    form = page.body[0]
    assert form.__class__.__name__ == "ShowForm"
    assert form.title == "User Form"
    print(f"✓ Forms still work with datasets: {form.title}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
