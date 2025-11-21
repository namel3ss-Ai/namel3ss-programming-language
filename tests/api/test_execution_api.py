"""Tests for execution API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient
from fastapi import status

from n3_server.api.main import app
from n3_server.db.models import Project


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_graph_data():
    """Sample graph data for testing."""
    return {
        "nodes": [
            {
                "id": "start-1",
                "type": "start",
                "label": "START",
                "data": {},
            },
            {
                "id": "prompt-1",
                "type": "prompt",
                "label": "Greeting",
                "data": {
                    "name": "greeting",
                    "text": "Say hello to {{name}}",
                    "model": "gpt-4",
                    "arguments": ["name"],
                    "outputSchema": {"greeting": "string"},
                },
            },
            {
                "id": "end-1",
                "type": "end",
                "label": "END",
                "data": {},
            },
        ],
        "edges": [
            {"id": "e1", "source": "start-1", "target": "prompt-1"},
            {"id": "e2", "source": "prompt-1", "target": "end-1"},
        ],
    }


@pytest.fixture
def mock_project(sample_graph_data):
    """Mock project with graph data."""
    project = MagicMock(spec=Project)
    project.id = "test-project"
    project.name = "Test Project"
    project.graph_data = sample_graph_data
    project.metadata = {}
    return project


@pytest.fixture
def mock_db_session(mock_project):
    """Mock database session."""
    session = MagicMock()
    
    # Mock query result
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=mock_project)
    
    session.execute = AsyncMock(return_value=result)
    
    return session


# ============================================================================
# Execution API Tests
# ============================================================================

class TestExecutionEndpoint:
    """Tests for POST /api/execution/graphs/{project_id}/execute."""
    
    @pytest.mark.asyncio
    async def test_execute_graph_success(self, mock_db_session):
        """Test successful graph execution."""
        from n3_server.execution.executor import ExecutionSpan, SpanAttribute, SpanType
        from datetime import datetime, timezone
        
        # Mock database dependency
        async def override_get_db():
            yield mock_db_session
        
        app.dependency_overrides = {
            "n3_server.db.session.get_db": override_get_db
        }
        
        # Mock LLM registry
        with patch("n3_server.api.execution.build_llm_registry") as mock_llm_reg:
            mock_llm = MagicMock()
            mock_llm.model_name = "gpt-4"
            mock_llm_reg.return_value = {"gpt-4": mock_llm}
            
            # Mock RuntimeRegistry
            with patch("n3_server.api.execution.RuntimeRegistry.from_conversion_context") as mock_reg:
                mock_registry = MagicMock()
                mock_reg.return_value = mock_registry
                
                # Mock GraphExecutor
                with patch("n3_server.api.execution.GraphExecutor") as MockExecutor:
                    mock_executor = MockExecutor.return_value
                    mock_executor.execute_chain = AsyncMock(return_value={"greeting": "Hello Alice"})
                    
                    # Make a request
                    async with AsyncClient(app=app, base_url="http://test") as client:
                        response = await client.post(
                            "/api/execution/graphs/test-project/execute",
                            json={
                                "entry": "start",
                                "input": {"name": "Alice"},
                                "options": {
                                    "max_steps": 50,
                                    "trace_level": "full"
                                }
                            }
                        )
        
        # Cleanup
        app.dependency_overrides = {}
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "success"
        assert data["projectId"] == "test-project"
        assert "executionId" in data
        assert "result" in data
        assert "trace" in data
        assert "metrics" in data
    
    @pytest.mark.asyncio
    async def test_execute_graph_project_not_found(self):
        """Test execution with non-existent project."""
        # Mock database to return None
        async def override_get_db():
            session = MagicMock()
            result = MagicMock()
            result.scalar_one_or_none = MagicMock(return_value=None)
            session.execute = AsyncMock(return_value=result)
            yield session
        
        app.dependency_overrides = {
            "n3_server.db.session.get_db": override_get_db
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/execution/graphs/nonexistent/execute",
                json={
                    "entry": "start",
                    "input": {}
                }
            )
        
        app.dependency_overrides = {}
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_execute_graph_invalid_structure(self, mock_db_session):
        """Test execution with invalid graph structure."""
        # Mock project with invalid graph data
        mock_project = MagicMock()
        mock_project.id = "test-project"
        mock_project.name = "Test"
        mock_project.graph_data = {
            "nodes": [],  # Empty nodes - invalid
            "edges": [],
        }
        
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=mock_project)
        mock_db_session.execute = AsyncMock(return_value=result)
        
        async def override_get_db():
            yield mock_db_session
        
        app.dependency_overrides = {
            "n3_server.db.session.get_db": override_get_db
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/execution/graphs/test-project/execute",
                json={
                    "entry": "start",
                    "input": {}
                }
            )
        
        app.dependency_overrides = {}
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    @pytest.mark.asyncio
    async def test_execute_graph_validation_options(self):
        """Test execution request validation."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Invalid max_steps
            response = await client.post(
                "/api/execution/graphs/test-project/execute",
                json={
                    "entry": "start",
                    "input": {},
                    "options": {
                        "max_steps": 0,  # Invalid: must be >= 1
                    }
                }
            )
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            
            # Invalid trace_level
            response = await client.post(
                "/api/execution/graphs/test-project/execute",
                json={
                    "entry": "start",
                    "input": {},
                    "options": {
                        "trace_level": "invalid",  # Must be: full, summary, none
                    }
                }
            )
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestValidationEndpoint:
    """Tests for POST /api/execution/graphs/{project_id}/validate."""
    
    @pytest.mark.asyncio
    async def test_validate_graph_success(self, mock_db_session):
        """Test successful graph validation."""
        async def override_get_db():
            yield mock_db_session
        
        app.dependency_overrides = {
            "n3_server.db.session.get_db": override_get_db
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/execution/graphs/test-project/validate"
            )
        
        app.dependency_overrides = {}
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "valid" in data
        assert "errors" in data
        assert isinstance(data["errors"], list)
    
    @pytest.mark.asyncio
    async def test_validate_graph_with_errors(self):
        """Test validation with graph errors."""
        # Mock project with cycle in graph
        mock_project = MagicMock()
        mock_project.id = "test-project"
        mock_project.name = "Test"
        mock_project.graph_data = {
            "nodes": [
                {"id": "start-1", "type": "start", "label": "START", "data": {}},
                {
                    "id": "prompt-1",
                    "type": "prompt",
                    "label": "P1",
                    "data": {
                        "name": "p1",
                        "text": "test",
                        "model": "gpt-4",
                        "arguments": [],
                        "outputSchema": {"result": "string"},
                    },
                },
                {
                    "id": "prompt-2",
                    "type": "prompt",
                    "label": "P2",
                    "data": {
                        "name": "p2",
                        "text": "test",
                        "model": "gpt-4",
                        "arguments": [],
                        "outputSchema": {"result": "string"},
                    },
                },
            ],
            "edges": [
                {"id": "e1", "source": "start-1", "target": "prompt-1"},
                {"id": "e2", "source": "prompt-1", "target": "prompt-2"},
                {"id": "e3", "source": "prompt-2", "target": "prompt-1"},  # Cycle
            ],
        }
        
        session = MagicMock()
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=mock_project)
        session.execute = AsyncMock(return_value=result)
        
        async def override_get_db():
            yield session
        
        app.dependency_overrides = {
            "n3_server.db.session.get_db": override_get_db
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/execution/graphs/test-project/validate"
            )
        
        app.dependency_overrides = {}
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_validate_empty_graph(self):
        """Test validation with empty graph."""
        mock_project = MagicMock()
        mock_project.id = "test-project"
        mock_project.name = "Test"
        mock_project.graph_data = {}  # Empty graph data
        
        session = MagicMock()
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=mock_project)
        session.execute = AsyncMock(return_value=result)
        
        async def override_get_db():
            yield session
        
        app.dependency_overrides = {
            "n3_server.db.session.get_db": override_get_db
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/execution/graphs/test-project/validate"
            )
        
        app.dependency_overrides = {}
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["valid"] is False


class TestAPIResponseModels:
    """Tests for API response models."""
    
    def test_execution_response_serialization(self):
        """Test ExecutionResponse can be serialized."""
        from n3_server.api.execution import (
            ExecutionResponse,
            ExecutionMetrics,
            ExecutionSpanResponse,
            SpanAttributeResponse,
        )
        
        response = ExecutionResponse(
            execution_id="test-exec-1",
            project_id="test-project",
            status="success",
            result={"greeting": "Hello"},
            trace=[
                ExecutionSpanResponse(
                    span_id="span-1",
                    parent_span_id=None,
                    name="test_span",
                    type="CHAIN",
                    start_time="2024-01-01T00:00:00Z",
                    end_time="2024-01-01T00:00:01Z",
                    duration_ms=1000.0,
                    status="ok",
                    attributes=SpanAttributeResponse(
                        model="gpt-4",
                        tokens_prompt=100,
                        tokens_completion=50,
                        cost=0.015,
                    ),
                )
            ],
            metrics=ExecutionMetrics(
                total_duration_ms=1000.0,
                total_tokens_prompt=100,
                total_tokens_completion=50,
                total_cost=0.015,
                span_count=1,
            ),
        )
        
        # Should serialize without errors
        json_data = response.model_dump(by_alias=True)
        assert json_data["executionId"] == "test-exec-1"
        assert json_data["status"] == "success"
        assert json_data["metrics"]["totalCost"] == 0.015
    
    def test_validation_response_serialization(self):
        """Test ValidationResponse can be serialized."""
        from n3_server.api.execution import ValidationResponse, ValidationError
        
        response = ValidationResponse(
            valid=False,
            errors=[
                ValidationError(
                    node_id="node-1",
                    message="Missing required field",
                    details={"field": "system_prompt"},
                )
            ],
        )
        
        json_data = response.model_dump(by_alias=True)
        assert json_data["valid"] is False
        assert len(json_data["errors"]) == 1
        assert json_data["errors"][0]["nodeId"] == "node-1"


class TestAPIErrorHandling:
    """Tests for API error handling."""
    
    @pytest.mark.asyncio
    async def test_internal_server_error(self):
        """Test handling of unexpected errors."""
        # Mock database to raise exception
        async def override_get_db():
            raise Exception("Database connection failed")
            yield  # pragma: no cover
        
        app.dependency_overrides = {
            "n3_server.db.session.get_db": override_get_db
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/execution/graphs/test-project/execute",
                json={"entry": "start", "input": {}}
            )
        
        app.dependency_overrides = {}
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
