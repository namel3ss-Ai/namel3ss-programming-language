"""
Test suite for slug-based routing and dataset/table fallback behavior.

This test verifies:
1. Page routes use slugs (/api/pages/{slug}) instead of route-based paths
2. Dataset lookup gracefully handles table sources not in DATASETS
3. Frontend and backend routing are consistent
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

# Test the routing changes
def test_page_api_path_uses_slug():
    """Verify that page API paths are generated using slugs."""
    from namel3ss.codegen.backend.state import _encode_page
    from namel3ss.ast import Page
    
    # Create a test page with route "/"
    page = Page(
        name="Home",
        route="/",
        statements=[],
        reactive=False,
        refresh_policy=None,
        layout=None,
    )
    
    # Encode the page
    page_spec = _encode_page(0, page, set(), {})
    
    # Verify the api_path uses slug, not "root"
    assert page_spec.slug == "home"
    assert page_spec.api_path == "/api/pages/home"
    assert page_spec.api_path != "/api/pages/root"


def test_page_api_path_with_different_routes():
    """Verify that different routes get correct slug-based API paths."""
    from namel3ss.codegen.backend.state import _encode_page
    from namel3ss.ast import Page
    
    test_cases = [
        ("Home", "/", "home", "/api/pages/home"),
        ("Admin", "/admin", "admin", "/api/pages/admin"),
        ("Feedback", "/feedback", "feedback", "/api/pages/feedback"),
        ("User Profile", "/profile", "user_profile", "/api/pages/user_profile"),
    ]
    
    for name, route, expected_slug, expected_api_path in test_cases:
        page = Page(
            name=name,
            route=route,
            statements=[],
            reactive=False,
            refresh_policy=None,
            layout=None,
        )
        
        page_spec = _encode_page(0, page, set(), {})
        
        assert page_spec.slug == expected_slug, f"Expected slug '{expected_slug}' for page '{name}'"
        assert page_spec.api_path == expected_api_path, f"Expected API path '{expected_api_path}' for page '{name}'"


def test_render_page_endpoint_uses_slug():
    """Verify that rendered page endpoints use slug in the route."""
    from namel3ss.codegen.backend.core.routers import _render_page_endpoint
    from namel3ss.codegen.backend.state import PageSpec
    
    page_spec = PageSpec(
        name="Home",
        route="/",
        slug="home",
        index=0,
        api_path="/api/pages/home",
        reactive=False,
        refresh_policy=None,
        components=[],
        layout={},
    )
    
    lines = _render_page_endpoint(page_spec)
    
    # Verify the route uses the slug
    assert any("/api/pages/home" in line for line in lines), "Route should use slug 'home'"
    assert not any("/api/pages/root" in line for line in lines), "Route should not use 'root'"
    
    # Verify function name uses slug
    assert any("page_home_0" in line for line in lines), "Function name should use slug"


@pytest.mark.asyncio
@pytest.mark.skip(reason="SQLAlchemy compatibility issue with Python 3.13")
async def test_fetch_dataset_rows_table_fallback():
    """Verify that fetch_dataset_rows falls back to querying tables when dataset is not in DATASETS."""
    from namel3ss.codegen.backend.core.runtime.datasets import fetch_dataset_rows
    
    # Mock session with execute method
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [
        MagicMock(_mapping={"id": 1, "name": "Test"}),
        MagicMock(_mapping={"id": 2, "name": "Test2"}),
    ]
    mock_session.execute.return_value = mock_result
    
    # Empty datasets dict (key not found)
    datasets = {}
    context = {}
    
    # Mock all required dependencies
    def mock_resolve_connector(dataset, context):
        return {}
    
    def mock_dataset_cache_settings(dataset, context):
        return ("global", False, None)
    
    def mock_make_dataset_cache_key(key, scope, context):
        return f"{scope}:{key}"
    
    def mock_clone_rows(rows):
        return list(rows)
    
    # Call fetch_dataset_rows with a key not in DATASETS
    rows = await fetch_dataset_rows(
        "orders",
        mock_session,
        context,
        datasets=datasets,
        resolve_connector=mock_resolve_connector,
        dataset_cache_settings=mock_dataset_cache_settings,
        make_dataset_cache_key=mock_make_dataset_cache_key,
        dataset_cache_index={},
        cache_get=None,
        clone_rows=mock_clone_rows,
        load_dataset_source=None,
        execute_dataset_pipeline=None,
        cache_set=None,
        broadcast_dataset_refresh=None,
        schedule_dataset_refresh=None,
    )
    
    # Verify that it attempted to query the table
    assert mock_session.execute.called, "Should have attempted to execute SQL query"
    
    # Verify we got rows back
    assert len(rows) == 2
    assert rows[0]["id"] == 1
    assert rows[1]["name"] == "Test2"


@pytest.mark.asyncio
@pytest.mark.skip(reason="SQLAlchemy compatibility issue with Python 3.13")
async def test_fetch_dataset_rows_returns_empty_on_table_error():
    """Verify that fetch_dataset_rows returns empty list if table query fails."""
    from namel3ss.codegen.backend.core.runtime.datasets import fetch_dataset_rows
    
    # Mock session that raises an error
    mock_session = AsyncMock()
    mock_session.execute.side_effect = Exception("Table not found")
    
    # Empty datasets dict (key not found)
    datasets = {}
    context = {}
    
    # Mock required dependencies
    def mock_resolve_connector(dataset, context):
        return {}
    
    def mock_dataset_cache_settings(dataset, context):
        return ("global", False, None)
    
    def mock_make_dataset_cache_key(key, scope, context):
        return f"{scope}:{key}"
    
    def mock_clone_rows(rows):
        return list(rows)
    
    # Call fetch_dataset_rows with a key not in DATASETS
    rows = await fetch_dataset_rows(
        "nonexistent_table",
        mock_session,
        context,
        datasets=datasets,
        resolve_connector=mock_resolve_connector,
        dataset_cache_settings=mock_dataset_cache_settings,
        make_dataset_cache_key=mock_make_dataset_cache_key,
        dataset_cache_index={},
        cache_get=None,
        clone_rows=mock_clone_rows,
        load_dataset_source=None,
        execute_dataset_pipeline=None,
        cache_set=None,
        broadcast_dataset_refresh=None,
        schedule_dataset_refresh=None,
    )
    
    # Verify that it returns empty list instead of crashing
    assert rows == []


@pytest.mark.asyncio
@pytest.mark.skip(reason="SQLAlchemy compatibility issue with Python 3.13")
async def test_fetch_dataset_rows_uses_dataset_when_available():
    """Verify that fetch_dataset_rows uses the dataset definition when available."""
    from namel3ss.codegen.backend.core.runtime.datasets import fetch_dataset_rows
    
    # Mock session
    mock_session = AsyncMock()
    
    # Dataset exists in DATASETS
    datasets = {
        "monthly_sales": {
            "name": "monthly_sales",
            "source_type": "table",
            "source": "sales",
            "operations": [],
            "connector": {"type": "table"},
            "sample_rows": [{"id": 1, "revenue": 100}],
        }
    }
    context = {}
    
    # Mock dependencies
    def mock_resolve_connector(dataset, context):
        return dataset.get("connector", {})
    
    def mock_dataset_cache_settings(dataset, context):
        return ("global", False, None)
    
    def mock_make_dataset_cache_key(key, scope, context):
        return f"{scope}:{key}"
    
    def mock_clone_rows(rows):
        return list(rows)
    
    async def mock_load_dataset_source(dataset, connector, session, context):
        return dataset.get("sample_rows", [])
    
    async def mock_execute_dataset_pipeline(dataset, rows, context):
        return rows
    
    # Call fetch_dataset_rows with a key that IS in DATASETS
    rows = await fetch_dataset_rows(
        "monthly_sales",
        mock_session,
        context,
        datasets=datasets,
        resolve_connector=mock_resolve_connector,
        dataset_cache_settings=mock_dataset_cache_settings,
        make_dataset_cache_key=mock_make_dataset_cache_key,
        dataset_cache_index={},
        cache_get=None,
        clone_rows=mock_clone_rows,
        load_dataset_source=mock_load_dataset_source,
        execute_dataset_pipeline=mock_execute_dataset_pipeline,
        cache_set=None,
        broadcast_dataset_refresh=None,
        schedule_dataset_refresh=None,
    )
    
    # Verify we got the dataset's rows
    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["revenue"] == 100


def test_component_endpoint_uses_slug_based_path():
    """Verify that component endpoints use slug-based paths."""
    from namel3ss.codegen.backend.core.routers import _render_component_endpoint
    from namel3ss.codegen.backend.state import PageSpec, PageComponent
    
    page_spec = PageSpec(
        name="Home",
        route="/",
        slug="home",
        index=0,
        api_path="/api/pages/home",
        reactive=False,
        refresh_policy=None,
        components=[],
        layout={},
    )
    
    component = PageComponent(
        type="table",
        payload={
            "title": "Test Table",
            "source": "orders",
            "source_type": "table",
            "columns": ["id", "name"],
        },
        index=0,
    )
    
    lines = _render_component_endpoint(page_spec, component, 0)
    
    # Verify the route uses slug-based path
    assert any("/api/pages/home" in line for line in lines), "Component route should use slug"
    assert not any("/api/pages/root" in line for line in lines), "Component route should not use 'root'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
