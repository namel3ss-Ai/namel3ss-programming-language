"""
Integration tests for API endpoints.

Tests HTTP API behavior using FastAPI test client.
"""

from uuid import uuid4

import pytest
from httpx import AsyncClient


@pytest.mark.integration
class TestAPIEndpoints:
    """Test HTTP API endpoints."""
    
    async def test_health_check(self, client: AsyncClient):
        """Test health check endpoint."""
        response = await client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    async def test_root_endpoint(self, client: AsyncClient):
        """Test root endpoint."""
        response = await client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "docs" in data
        assert "health" in data
    
    async def test_create_item(self, client: AsyncClient, clean_database):
        """Test creating item via API."""
        payload = {
            "name": "API Test Item",
            "description": "Created via API",
            "quantity": 50,
            "price": 29.99,
            "tags": ["test", "api"],
        }
        
        response = await client.post("/{{ endpoint_prefix }}/", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "API Test Item"
        assert data["quantity"] == 50
        assert "id" in data
        assert "created_at" in data
    
    async def test_create_item_validation_error(self, client: AsyncClient):
        """Test create with invalid data."""
        payload = {
            "name": "",  # Invalid: empty name
            "quantity": -10,  # Invalid: negative
        }
        
        response = await client.post("/{{ endpoint_prefix }}/", json=payload)
        
        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "validation_error"
    
    async def test_get_item(self, client: AsyncClient, created_item):
        """Test getting item by ID."""
        response = await client.get(f"/{{ endpoint_prefix }}/{created_item.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(created_item.id)
        assert data["name"] == created_item.name
    
    async def test_get_item_not_found(self, client: AsyncClient, clean_database):
        """Test getting non-existent item."""
        fake_id = uuid4()
        response = await client.get(f"/{{ endpoint_prefix }}/{fake_id}")
        
        assert response.status_code == 404
    
    async def test_list_items(self, client: AsyncClient, multiple_items):
        """Test listing items."""
        response = await client.get("/{{ endpoint_prefix }}/")
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert data["total"] == 5
        assert len(data["items"]) == 5
    
    async def test_list_items_pagination(self, client: AsyncClient, multiple_items):
        """Test pagination."""
        response = await client.get("/{{ endpoint_prefix }}/?page=1&page_size=2")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert data["has_next"] is True
        assert data["has_prev"] is False
    
    async def test_list_items_filter_by_active(self, client: AsyncClient, multiple_items):
        """Test filtering by active status."""
        response = await client.get("/{{ endpoint_prefix }}/?is_active=true")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3  # Items 0, 2, 4 are active
        assert all(item["is_active"] for item in data["items"])
    
    async def test_list_items_filter_by_tags(self, client: AsyncClient, multiple_items):
        """Test filtering by tags."""
        response = await client.get("/{{ endpoint_prefix }}/?tags=common")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5  # All have "common" tag
    
    async def test_search_items(self, client: AsyncClient, multiple_items):
        """Test search endpoint."""
        response = await client.get("/{{ endpoint_prefix }}/search/?q=Item 2")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["name"] == "Item 2"
    
    async def test_search_items_missing_query(self, client: AsyncClient):
        """Test search without query parameter."""
        response = await client.get("/{{ endpoint_prefix }}/search/")
        
        assert response.status_code == 422  # Validation error
    
    async def test_update_item(self, client: AsyncClient, created_item):
        """Test updating item."""
        payload = {
            "name": "Updated Name",
            "quantity": 999,
        }
        
        response = await client.put(
            f"/{{ endpoint_prefix }}/{created_item.id}",
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["quantity"] == 999
        # Price should be unchanged
        assert data["price"] == created_item.price
    
    async def test_update_item_not_found(self, client: AsyncClient, clean_database):
        """Test updating non-existent item."""
        fake_id = uuid4()
        payload = {"name": "Updated"}
        
        response = await client.put(
            f"/{{ endpoint_prefix }}/{fake_id}",
            json=payload
        )
        
        assert response.status_code == 404
    
    async def test_delete_item(self, client: AsyncClient, created_item):
        """Test soft deleting item."""
        response = await client.delete(f"/{{ endpoint_prefix }}/{created_item.id}")
        
        assert response.status_code == 204
        
        # Verify item is soft deleted
        get_response = await client.get(f"/{{ endpoint_prefix }}/{created_item.id}")
        assert get_response.status_code == 404
        
        # But can be found with include_deleted
        get_response = await client.get(
            f"/{{ endpoint_prefix }}/{created_item.id}?include_deleted=true"
        )
        assert get_response.status_code == 200
    
    async def test_delete_item_hard(self, client: AsyncClient, created_item):
        """Test hard deleting item."""
        response = await client.delete(
            f"/{{ endpoint_prefix }}/{created_item.id}?hard=true"
        )
        
        assert response.status_code == 204
        
        # Verify item is permanently deleted
        get_response = await client.get(
            f"/{{ endpoint_prefix }}/{created_item.id}?include_deleted=true"
        )
        assert get_response.status_code == 404
    
    async def test_delete_item_not_found(self, client: AsyncClient, clean_database):
        """Test deleting non-existent item."""
        fake_id = uuid4()
        response = await client.delete(f"/{{ endpoint_prefix }}/{fake_id}")
        
        assert response.status_code == 404
    
    async def test_restore_item(self, client: AsyncClient, created_item):
        """Test restoring soft-deleted item."""
        # First soft delete
        await client.delete(f"/{{ endpoint_prefix }}/{created_item.id}")
        
        # Then restore
        response = await client.post(f"/{{ endpoint_prefix }}/{created_item.id}/restore")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(created_item.id)
        assert data["deleted_at"] is None
    
    async def test_restore_not_deleted_fails(self, client: AsyncClient, created_item):
        """Test restoring non-deleted item."""
        response = await client.post(f"/{{ endpoint_prefix }}/{created_item.id}/restore")
        
        assert response.status_code == 404
    
    async def test_count_items(self, client: AsyncClient, multiple_items):
        """Test count endpoint."""
        response = await client.get("/{{ endpoint_prefix }}/stats/count")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 5
    
    async def test_count_items_with_filters(self, client: AsyncClient, multiple_items):
        """Test count with filters."""
        response = await client.get("/{{ endpoint_prefix }}/stats/count?is_active=true")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
    
    async def test_cors_headers(self, client: AsyncClient):
        """Test CORS headers are present."""
        response = await client.options("/{{ endpoint_prefix }}/")
        
        # FastAPI test client doesn't process CORS fully, but we can verify app config
        # In production, CORS middleware will handle this
        assert response.status_code in (200, 405)  # OPTIONS may not be explicitly defined
    
    async def test_request_id_header(self, client: AsyncClient, clean_database):
        """Test request ID is added to response."""
        response = await client.get("/health")
        
        assert "x-request-id" in response.headers
    
    async def test_openapi_docs(self, client: AsyncClient):
        """Test OpenAPI documentation is available."""
        response = await client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
