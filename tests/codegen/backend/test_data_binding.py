"""
Test dataset router CRUD operations and realtime broadcasting.

Tests the generated FastAPI dataset endpoints for:
- Pagination, sorting, filtering
- Create, update, delete operations  
- Realtime event broadcasting
- Error handling and validation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient
from fastapi.testclient import TestClient

from namel3ss.codegen.backend.runtime.sql_compiler import (
    compile_dataset_to_sql,
    compile_dataset_insert,
    compile_dataset_update,
    compile_dataset_delete,
)


class TestSQLCompiler:
    """Test SQL query generation for dataset operations."""
    
    def test_compile_dataset_to_sql_basic(self):
        """Test basic dataset query compilation."""
        result = compile_dataset_to_sql("users", page=1, page_size=10)
        
        assert "SELECT * FROM users" in result["query"]
        assert "LIMIT 10 OFFSET 0" in result["query"]
        assert "SELECT COUNT(*) FROM users" in result["count_query"]
        assert result["params"] == {}
    
    def test_compile_dataset_to_sql_with_search(self):
        """Test dataset query with search functionality."""
        result = compile_dataset_to_sql(
            "users",
            search="john",
            searchable_fields=["name", "email"]
        )
        
        assert "WHERE" in result["query"]
        assert "name ILIKE" in result["query"]
        assert "email ILIKE" in result["query"]
        assert "OR" in result["query"]
        assert "%john%" in result["params"]["search_0"]
        assert "%john%" in result["params"]["search_1"]
    
    def test_compile_dataset_to_sql_with_sorting(self):
        """Test dataset query with sorting."""
        result = compile_dataset_to_sql(
            "users",
            sort_by="name",
            sort_order="desc"
        )
        
        assert "ORDER BY name DESC" in result["query"]
    
    def test_compile_dataset_to_sql_with_pagination(self):
        """Test dataset query with pagination."""
        result = compile_dataset_to_sql("users", page=3, page_size=25)
        
        assert "LIMIT 25 OFFSET 50" in result["query"]
    
    def test_compile_dataset_insert(self):
        """Test dataset insert compilation."""
        data = {"name": "John Doe", "email": "john@example.com"}
        result = compile_dataset_insert("users", data)
        
        assert "INSERT INTO users" in result["query"]
        assert "(name, email)" in result["query"]
        assert "VALUES (%(name)s, %(email)s)" in result["query"]
        assert "RETURNING id" in result["query"]
        assert result["params"] == data
        assert "SELECT * FROM users WHERE id = %(id)s" in result["select_query"]
    
    def test_compile_dataset_update(self):
        """Test dataset update compilation."""
        data = {"name": "Jane Doe", "email": "jane@example.com"}
        result = compile_dataset_update("users", "123", data)
        
        assert "UPDATE users" in result["query"]
        assert "SET name = %(name)s, email = %(email)s" in result["query"]
        assert "WHERE id = %(id)s" in result["query"]
        assert result["params"]["name"] == "Jane Doe"
        assert result["params"]["email"] == "jane@example.com"
        assert result["params"]["id"] == "123"
    
    def test_compile_dataset_delete(self):
        """Test dataset delete compilation."""
        result = compile_dataset_delete("users", "123")
        
        assert "DELETE FROM users" in result["query"]
        assert "WHERE id = %(id)s" in result["query"]
        assert result["params"] == {"id": "123"}
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention in table/column names."""
        with pytest.raises(ValueError):
            compile_dataset_to_sql("users; DROP TABLE users; --")
        
        with pytest.raises(ValueError):
            compile_dataset_to_sql("users", sort_by="name'; DROP TABLE users; --")


@pytest.mark.asyncio
class TestRealtimeBroadcasting:
    """Test realtime broadcasting functionality."""
    
    @patch('namel3ss.codegen.backend.runtime.realtime._redis_pool')
    async def test_broadcast_dataset_change_success(self, mock_redis):
        """Test successful dataset change broadcast."""
        from namel3ss.codegen.backend.runtime.realtime import broadcast_dataset_change
        
        mock_redis.publish = AsyncMock(return_value=2)  # 2 subscribers
        
        result = await broadcast_dataset_change(
            dataset_name="users",
            event_type="create",
            data={"id": 1, "name": "John"}
        )
        
        assert result is True
        assert mock_redis.publish.call_count == 3  # all, dataset, dataset:event channels
    
    @patch('namel3ss.codegen.backend.runtime.realtime._redis_pool', None)
    async def test_broadcast_dataset_change_no_redis(self):
        """Test broadcast when Redis is not available."""
        from namel3ss.codegen.backend.runtime.realtime import broadcast_dataset_change
        
        result = await broadcast_dataset_change(
            dataset_name="users",
            event_type="create",
            data={"id": 1}
        )
        
        assert result is False
    
    @patch('namel3ss.codegen.backend.runtime.realtime._redis_pool')
    async def test_subscribe_to_dataset_changes(self, mock_redis):
        """Test dataset change subscription."""
        from namel3ss.codegen.backend.runtime.realtime import subscribe_to_dataset_changes
        
        mock_pubsub = MagicMock()
        mock_redis.pubsub.return_value = mock_pubsub
        
        pubsub = await subscribe_to_dataset_changes("users")
        
        assert pubsub == mock_pubsub
        mock_pubsub.subscribe.assert_called_once_with("dataset_updates:users")
    
    @patch('namel3ss.codegen.backend.runtime.realtime._redis_pool')
    async def test_dataset_change_handler(self, mock_redis):
        """Test DatasetChangeHandler functionality."""
        from namel3ss.codegen.backend.runtime.realtime import DatasetChangeHandler
        
        # Create handler with custom callback
        events_received = []
        
        class TestHandler(DatasetChangeHandler):
            async def on_dataset_change(self, dataset_name, event_type, data, timestamp):
                events_received.append({
                    "dataset": dataset_name,
                    "event": event_type,
                    "data": data
                })
        
        handler = TestHandler("users")
        
        # Mock message handling
        test_message = '{"dataset": "users", "event_type": "create", "data": {"id": 1}}'
        await handler._handle_message(test_message)
        
        assert len(events_received) == 1
        assert events_received[0]["dataset"] == "users"
        assert events_received[0]["event"] == "create"


class TestDatasetRouter:
    """Test the generated dataset router endpoints."""
    
    @pytest.fixture
    def mock_backend_ir(self):
        """Create mock BackendIR for testing."""
        from namel3ss.ir import DatasetSpec
        
        dataset = MagicMock()
        dataset.name = "users"
        dataset.primary_key = "id"
        dataset.schema = [
            {"name": "id", "type": "integer", "nullable": False},
            {"name": "name", "type": "string", "nullable": False},
            {"name": "email", "type": "string", "nullable": True},
        ]
        dataset.access_policy = {
            "read_only": False,
            "allow_create": True,
            "allow_update": True,
            "allow_delete": True,
            "sortable_fields": ["name", "email"],
            "searchable_fields": ["name", "email"],
        }
        dataset.realtime_enabled = True
        
        backend_ir = MagicMock()
        backend_ir.datasets = [dataset]
        
        return backend_ir
    
    def test_render_dataset_router_module(self, mock_backend_ir):
        """Test dataset router module generation."""
        from namel3ss.codegen.backend.core.dataset_router import _render_dataset_router_module
        
        result = _render_dataset_router_module(mock_backend_ir)
        
        assert "router = APIRouter(prefix=\"/api/datasets\"" in result
        assert "get_users_dataset" in result
        assert "create_users_item" in result
        assert "update_users_item" in result
        assert "delete_users_item" in result
        assert "broadcast_dataset_change" in result
    
    def test_dataset_models_generation(self, mock_backend_ir):
        """Test Pydantic model generation for datasets."""
        from namel3ss.codegen.backend.core.dataset_router import _generate_dataset_models
        
        dataset = mock_backend_ir.datasets[0]
        result = _generate_dataset_models(dataset)
        
        assert "class UsersItem(BaseModel):" in result
        assert "class UsersCreateRequest(BaseModel):" in result
        assert "class UsersUpdateRequest(BaseModel):" in result
        assert "id: int" in result
        assert "name: str" in result
        assert "email: str | None" in result


@pytest.mark.integration
class TestDatasetEndpointsIntegration:
    """Integration tests for dataset endpoints."""
    
    @pytest.fixture
    def test_app(self):
        """Create test FastAPI app with dataset router."""
        from fastapi import FastAPI
        from namel3ss.codegen.backend.core.dataset_router import _render_dataset_router_module
        
        # Mock minimal backend IR
        mock_ir = MagicMock()
        mock_ir.datasets = []
        
        app = FastAPI()
        
        # In a real scenario, this would be imported and added to the app
        # For now, we just test the generation doesn't crash
        router_code = _render_dataset_router_module(mock_ir)
        assert "APIRouter" in router_code
        
        return app
    
    def test_empty_datasets_router_generation(self, test_app):
        """Test router generation with no datasets."""
        from namel3ss.codegen.backend.core.dataset_router import _render_dataset_router_module
        
        mock_ir = MagicMock()
        mock_ir.datasets = []
        
        result = _render_dataset_router_module(mock_ir)
        
        # Should still generate basic router structure
        assert "router = APIRouter(prefix=\"/api/datasets\"" in result
        assert "DatasetResponse" in result
        assert "DatasetQuery" in result