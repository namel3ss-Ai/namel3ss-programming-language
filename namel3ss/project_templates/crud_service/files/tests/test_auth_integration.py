"""
Integration tests for authentication with API endpoints.

Tests protected routes with valid/invalid tokens, and authorization checks.
"""

from datetime import timedelta

import pytest
from httpx import AsyncClient

from api.security import create_access_token
from config.settings import get_settings


@pytest.fixture
def auth_token():
    """Create a valid authentication token for tests."""
    settings = get_settings()
    return create_access_token(
        user_id="test_user_123",
        settings=settings,
        email="test@example.com",
        username="testuser",
        roles=["user"],
        tenant_id="test_tenant",
    )


@pytest.fixture
def admin_token():
    """Create a valid admin authentication token for tests."""
    settings = get_settings()
    return create_access_token(
        user_id="admin_user_456",
        settings=settings,
        email="admin@example.com",
        username="adminuser",
        roles=["user", "admin"],
        tenant_id="test_tenant",
    )


@pytest.fixture
def expired_token():
    """Create an expired authentication token for tests."""
    settings = get_settings()
    return create_access_token(
        user_id="test_user_123",
        settings=settings,
        expires_delta=timedelta(seconds=-10),  # Expired 10 seconds ago
    )


@pytest.mark.integration
class TestAuthenticatedEndpoints:
    """Test endpoints requiring authentication."""
    
    async def test_create_item_without_auth(self, client: AsyncClient, clean_database):
        """Test creating item without authentication fails."""
        payload = {
            "name": "Test Item",
            "description": "Should fail",
            "quantity": 10,
            "price": 99.99,
        }
        
        response = await client.post("/{{ endpoint_prefix }}/", json=payload)
        
        assert response.status_code == 401
        data = response.json()
        assert "authentication" in data["detail"].lower() or "unauthorized" in data["detail"].lower()
    
    async def test_create_item_with_valid_auth(
        self, client: AsyncClient, clean_database, auth_token
    ):
        """Test creating item with valid authentication succeeds."""
        payload = {
            "name": "Authenticated Item",
            "description": "Created with auth",
            "quantity": 10,
            "price": 99.99,
        }
        
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = await client.post("/{{ endpoint_prefix }}/", json=payload, headers=headers)
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Authenticated Item"
        assert "id" in data
    
    async def test_create_item_with_expired_token(
        self, client: AsyncClient, clean_database, expired_token
    ):
        """Test creating item with expired token fails."""
        payload = {
            "name": "Test Item",
            "description": "Should fail",
            "quantity": 10,
            "price": 99.99,
        }
        
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = await client.post("/{{ endpoint_prefix }}/", json=payload, headers=headers)
        
        assert response.status_code == 401
        data = response.json()
        assert "expired" in data["detail"].lower()
    
    async def test_create_item_with_invalid_token(self, client: AsyncClient, clean_database):
        """Test creating item with invalid token fails."""
        payload = {
            "name": "Test Item",
            "description": "Should fail",
            "quantity": 10,
            "price": 99.99,
        }
        
        headers = {"Authorization": "Bearer invalid.jwt.token"}
        response = await client.post("/{{ endpoint_prefix }}/", json=payload, headers=headers)
        
        assert response.status_code == 401
    
    async def test_create_item_with_malformed_header(self, client: AsyncClient, clean_database):
        """Test creating item with malformed authorization header fails."""
        payload = {
            "name": "Test Item",
            "description": "Should fail",
            "quantity": 10,
            "price": 99.99,
        }
        
        # Missing 'Bearer' prefix
        headers = {"Authorization": "some_token"}
        response = await client.post("/{{ endpoint_prefix }}/", json=payload, headers=headers)
        
        assert response.status_code == 401
    
    async def test_update_item_without_auth(self, client: AsyncClient, created_item):
        """Test updating item without authentication fails."""
        payload = {
            "name": "Updated Name",
            "price": 149.99,
        }
        
        response = await client.put(f"/{{ endpoint_prefix }}/{created_item.id}", json=payload)
        
        assert response.status_code == 401
    
    async def test_update_item_with_auth(self, client: AsyncClient, created_item, auth_token):
        """Test updating item with authentication succeeds."""
        payload = {
            "name": "Updated Name",
            "price": 149.99,
        }
        
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = await client.put(
            f"/{{ endpoint_prefix }}/{created_item.id}",
            json=payload,
            headers=headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["price"] == 149.99
    
    async def test_delete_item_without_auth(self, client: AsyncClient, created_item):
        """Test deleting item without authentication fails."""
        response = await client.delete(f"/{{ endpoint_prefix }}/{created_item.id}")
        
        assert response.status_code == 401
    
    async def test_delete_item_with_auth(self, client: AsyncClient, created_item, auth_token):
        """Test deleting item with authentication succeeds."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = await client.delete(
            f"/{{ endpoint_prefix }}/{created_item.id}",
            headers=headers,
        )
        
        assert response.status_code == 204
    
    async def test_restore_item_without_auth(self, client: AsyncClient, created_item, auth_token):
        """Test restoring item without authentication fails."""
        # First delete the item
        headers = {"Authorization": f"Bearer {auth_token}"}
        await client.delete(f"/{{ endpoint_prefix }}/{created_item.id}", headers=headers)
        
        # Try to restore without auth
        response = await client.post(f"/{{ endpoint_prefix }}/{created_item.id}/restore")
        
        assert response.status_code == 401
    
    async def test_restore_item_with_auth(self, client: AsyncClient, created_item, auth_token):
        """Test restoring item with authentication succeeds."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # First delete the item
        await client.delete(f"/{{ endpoint_prefix }}/{created_item.id}", headers=headers)
        
        # Restore with auth
        response = await client.post(
            f"/{{ endpoint_prefix }}/{created_item.id}/restore",
            headers=headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(created_item.id)


@pytest.mark.integration
class TestPublicEndpoints:
    """Test endpoints that don't require authentication."""
    
    async def test_get_item_without_auth(self, client: AsyncClient, created_item):
        """Test getting item without authentication succeeds (public endpoint)."""
        response = await client.get(f"/{{ endpoint_prefix }}/{created_item.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(created_item.id)
    
    async def test_list_items_without_auth(self, client: AsyncClient, multiple_items):
        """Test listing items without authentication succeeds (public endpoint)."""
        response = await client.get("/{{ endpoint_prefix }}/")
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) > 0
    
    async def test_search_items_without_auth(self, client: AsyncClient, created_item):
        """Test searching items without authentication succeeds (public endpoint)."""
        response = await client.get(f"/{{ endpoint_prefix }}/search/?q={created_item.name}")
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
    
    async def test_count_items_without_auth(self, client: AsyncClient, multiple_items):
        """Test counting items without authentication succeeds (public endpoint)."""
        response = await client.get("/{{ endpoint_prefix }}/stats/count")
        
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert data["count"] > 0


@pytest.mark.integration
class TestAuthenticationFlow:
    """Test complete authentication flows."""
    
    async def test_multiple_requests_with_same_token(
        self, client: AsyncClient, clean_database, auth_token
    ):
        """Test making multiple authenticated requests with same token."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Create item
        payload1 = {"name": "Item 1", "quantity": 10, "price": 99.99}
        response1 = await client.post("/{{ endpoint_prefix }}/", json=payload1, headers=headers)
        assert response1.status_code == 201
        
        # Create another item
        payload2 = {"name": "Item 2", "quantity": 20, "price": 149.99}
        response2 = await client.post("/{{ endpoint_prefix }}/", json=payload2, headers=headers)
        assert response2.status_code == 201
        
        # Update first item
        item1_id = response1.json()["id"]
        update_payload = {"quantity": 15}
        response3 = await client.put(
            f"/{{ endpoint_prefix }}/{item1_id}",
            json=update_payload,
            headers=headers,
        )
        assert response3.status_code == 200
        assert response3.json()["quantity"] == 15
    
    async def test_mixed_public_and_protected_requests(
        self, client: AsyncClient, created_item, auth_token
    ):
        """Test mixing public (no auth) and protected (auth required) requests."""
        # Public request - list items
        response1 = await client.get("/{{ endpoint_prefix }}/")
        assert response1.status_code == 200
        
        # Public request - get item
        response2 = await client.get(f"/{{ endpoint_prefix }}/{created_item.id}")
        assert response2.status_code == 200
        
        # Protected request without auth - should fail
        response3 = await client.delete(f"/{{ endpoint_prefix }}/{created_item.id}")
        assert response3.status_code == 401
        
        # Protected request with auth - should succeed
        headers = {"Authorization": f"Bearer {auth_token}"}
        response4 = await client.delete(
            f"/{{ endpoint_prefix }}/{created_item.id}",
            headers=headers,
        )
        assert response4.status_code == 204


@pytest.mark.integration
class TestTokenValidation:
    """Test token validation edge cases."""
    
    async def test_token_with_wrong_secret(self, client: AsyncClient, clean_database):
        """Test token signed with wrong secret fails."""
        import jwt
        from datetime import datetime
        
        # Create token with wrong secret
        claims = {
            "sub": "user_123",
            "exp": datetime.utcnow() + timedelta(minutes=30),
            "iat": datetime.utcnow(),
        }
        wrong_token = jwt.encode(claims, "wrong-secret-key", algorithm="HS256")
        
        payload = {"name": "Test Item", "quantity": 10, "price": 99.99}
        headers = {"Authorization": f"Bearer {wrong_token}"}
        response = await client.post("/{{ endpoint_prefix }}/", json=payload, headers=headers)
        
        assert response.status_code == 401
    
    async def test_token_without_required_claims(self, client: AsyncClient, clean_database):
        """Test token missing required claims fails."""
        import jwt
        from datetime import datetime
        
        settings = get_settings()
        
        # Create token without 'sub' claim
        claims = {
            "exp": datetime.utcnow() + timedelta(minutes=30),
            "iat": datetime.utcnow(),
        }
        invalid_token = jwt.encode(
            claims,
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )
        
        payload = {"name": "Test Item", "quantity": 10, "price": 99.99}
        headers = {"Authorization": f"Bearer {invalid_token}"}
        response = await client.post("/{{ endpoint_prefix }}/", json=payload, headers=headers)
        
        assert response.status_code == 401
