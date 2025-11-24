"""Test suite for N3 adapters.

Tests all adapter implementations with mocks for external services.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from namel3ss.adapters import (
    # Base
    AdapterConfig,
    AdapterType,
    AdapterValidationError,
    AdapterExecutionError,
    RetryPolicy,
    # Python
    PythonAdapter,
    PythonAdapterConfig,
    # HTTP
    HttpAdapter,
    HttpAdapterConfig,
    HttpMethod,
    # Database
    DatabaseAdapter,
    DatabaseAdapterConfig,
    DatabaseEngine,
    QueryType,
    # Queue
    CeleryQueueAdapter,
    QueueAdapterConfig,
    QueueBackend,
    # Model
    ModelAdapter,
    ModelAdapterConfig,
    ModelProvider,
)


class TestPythonAdapter:
    """Test Python FFI adapter."""
    
    def test_call_simple_function(self):
        """Test calling a simple Python function."""
        config = PythonAdapterConfig(
            name="test_func",
            module="builtins",
            function="len",
        )
        adapter = PythonAdapter(config)
        
        result = adapter.execute(obj=[1, 2, 3])
        assert result.success
        assert result.output == 3
    
    def test_call_with_kwargs(self):
        """Test function call with keyword arguments."""
        config = PythonAdapterConfig(
            name="test_sorted",
            module="builtins",
            function="sorted",
        )
        adapter = PythonAdapter(config)
        
        result = adapter.execute(iterable=[3, 1, 2], reverse=True)
        assert result.success
        assert result.output == [3, 2, 1]
    
    def test_timeout_enforcement(self):
        """Test timeout on slow function."""
        def slow_func():
            import time
            time.sleep(10)
        
        config = PythonAdapterConfig(
            name="slow",
            callable=slow_func,
            timeout=0.1,
        )
        adapter = PythonAdapter(config)
        
        with pytest.raises(AdapterExecutionError):
            adapter.execute()
    
    def test_validation_error(self):
        """Test input validation."""
        def typed_func(x: int) -> int:
            return x * 2
        
        config = PythonAdapterConfig(
            name="typed",
            callable=typed_func,
        )
        adapter = PythonAdapter(config)
        
        # Should fail validation (string instead of int)
        with pytest.raises(AdapterValidationError):
            adapter.execute(x="not_an_int")


class TestHttpAdapter:
    """Test HTTP adapter."""
    
    @patch('httpx.Client')
    def test_get_request(self, mock_client_class):
        """Test GET request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        
        mock_client = Mock()
        mock_client.request.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        config = HttpAdapterConfig(
            name="test_api",
            base_url="https://api.example.com",
            endpoint="/users",
            method=HttpMethod.GET,
        )
        adapter = HttpAdapter(config)
        
        result = adapter.execute()
        assert result.success
        assert result.output == {"data": "test"}
        mock_client.request.assert_called_once()
    
    @patch('httpx.Client')
    def test_post_with_body(self, mock_client_class):
        """Test POST with JSON body."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 123}
        
        mock_client = Mock()
        mock_client.request.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        config = HttpAdapterConfig(
            name="create_user",
            base_url="https://api.example.com",
            endpoint="/users",
            method=HttpMethod.POST,
        )
        adapter = HttpAdapter(config)
        
        result = adapter.execute(name="Alice", email="alice@example.com")
        assert result.success
        assert result.output["id"] == 123
    
    @patch('httpx.Client')
    def test_bearer_auth(self, mock_client_class):
        """Test Bearer token authentication."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"authorized": True}
        
        mock_client = Mock()
        mock_client.request.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        config = HttpAdapterConfig(
            name="auth_api",
            base_url="https://api.example.com",
            endpoint="/protected",
            method=HttpMethod.GET,
            auth_token="test_token_123",
        )
        adapter = HttpAdapter(config)
        
        result = adapter.execute()
        assert result.success


class TestDatabaseAdapter:
    """Test database adapter."""
    
    @patch('sqlalchemy.create_engine')
    def test_select_query(self, mock_create_engine):
        """Test SELECT query execution."""
        # Mock database engine and connection
        mock_result = Mock()
        mock_result.keys.return_value = ["id", "name"]
        mock_result.fetchmany.return_value = [
            (1, "Alice"),
            (2, "Bob"),
        ]
        
        mock_conn = Mock()
        mock_conn.execute.return_value = mock_result
        mock_conn.begin.return_value.__enter__.return_value = mock_conn
        
        mock_engine = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        config = DatabaseAdapterConfig(
            name="test_db",
            connection_url="sqlite:///:memory:",
            engine_type=DatabaseEngine.SQLITE,
        )
        adapter = DatabaseAdapter(config)
        
        result = adapter.execute(
            query="SELECT * FROM users WHERE status = :status",
            params={"status": "active"}
        )
        
        assert result.success
        assert len(result.output) == 2
        assert result.output[0]["name"] == "Alice"
    
    def test_sql_injection_prevention(self):
        """Test SQL injection pattern detection."""
        config = DatabaseAdapterConfig(
            name="secure_db",
            connection_url="sqlite:///:memory:",
            engine_type=DatabaseEngine.SQLITE,
            allow_raw_sql=False,
        )
        
        # Should detect dangerous patterns
        dangerous_queries = [
            "SELECT * FROM users WHERE id = 1; DROP TABLE users;",
            "SELECT * FROM users WHERE name = 'a' OR '1'='1'",
            "SELECT * FROM users /* comment */ WHERE id = 1",
        ]
        
        adapter = DatabaseAdapter(config)
        for query in dangerous_queries:
            with pytest.raises(AdapterValidationError):
                adapter._validate_query_safety(query)
    
    @patch('sqlalchemy.create_engine')
    def test_parameterized_query(self, mock_create_engine):
        """Test parameterized query with :param syntax."""
        mock_result = Mock()
        mock_result.keys.return_value = ["count"]
        mock_result.fetchmany.return_value = [(5,)]
        
        mock_conn = Mock()
        mock_conn.execute.return_value = mock_result
        mock_conn.begin.return_value.__enter__.return_value = mock_conn
        
        mock_engine = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        config = DatabaseAdapterConfig(
            name="param_db",
            connection_url="sqlite:///:memory:",
            engine_type=DatabaseEngine.SQLITE,
        )
        adapter = DatabaseAdapter(config)
        
        result = adapter.execute(
            query="SELECT COUNT(*) as count FROM orders WHERE user_id = :user_id",
            params={"user_id": 123}
        )
        
        assert result.success


class TestQueueAdapter:
    """Test queue adapters."""
    
    @patch('celery.Celery')
    def test_celery_enqueue(self, mock_celery_class):
        """Test Celery task enqueue."""
        mock_result = Mock()
        mock_result.id = "task-123"
        
        mock_app = Mock()
        mock_app.send_task.return_value = mock_result
        mock_celery_class.return_value = mock_app
        
        config = QueueAdapterConfig(
            name="celery_queue",
            backend=QueueBackend.CELERY,
            broker_url="redis://localhost:6379/0",
            task_name="tasks.process_data",
        )
        adapter = CeleryQueueAdapter(config)
        
        result = adapter.execute(data={"key": "value"})
        assert result.success
        assert result.output["task_id"] == "task-123"
        assert result.output["status"] == "queued"
    
    @patch('celery.Celery')
    def test_celery_task_status(self, mock_celery_class):
        """Test checking Celery task status."""
        mock_task = Mock()
        mock_task.state = "SUCCESS"
        mock_task.ready.return_value = True
        mock_task.successful.return_value = True
        mock_task.result = {"output": "processed"}
        
        mock_async_result = Mock(return_value=mock_task)
        
        mock_app = Mock()
        mock_celery_class.return_value = mock_app
        
        config = QueueAdapterConfig(
            name="celery_queue",
            backend=QueueBackend.CELERY,
            broker_url="redis://localhost:6379/0",
            task_name="tasks.process",
        )
        adapter = CeleryQueueAdapter(config)
        
        with patch('celery.result.AsyncResult', return_value=mock_task):
            status = adapter.get_task_status("task-123")
            assert status["state"] == "SUCCESS"
            assert status["ready"] is True


class TestModelAdapter:
    """Test model adapter."""
    
    @patch('openai.OpenAI')
    def test_openai_chat(self, mock_openai_class):
        """Test OpenAI chat completion."""
        mock_choice = Mock()
        mock_choice.message.content = "Hello! How can I help?"
        mock_choice.message.role = "assistant"
        mock_choice.finish_reason = "stop"
        
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 8
        mock_usage.total_tokens = 18
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        config = ModelAdapterConfig(
            name="gpt4",
            provider=ModelProvider.OPENAI,
            api_key="sk-test",
            model="gpt-4",
        )
        adapter = ModelAdapter(config)
        
        result = adapter.execute(
            messages=[
                {"role": "user", "content": "Hello!"}
            ]
        )
        
        assert result.success
        assert result.output["content"] == "Hello! How can I help?"
        assert result.output["usage"]["total_tokens"] == 18
    
    @patch('anthropic.Anthropic')
    def test_anthropic_chat(self, mock_anthropic_class):
        """Test Anthropic chat completion."""
        mock_content = Mock()
        mock_content.text = "I'm Claude, an AI assistant."
        
        mock_usage = Mock()
        mock_usage.input_tokens = 12
        mock_usage.output_tokens = 9
        
        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = mock_usage
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        config = ModelAdapterConfig(
            name="claude",
            provider=ModelProvider.ANTHROPIC,
            api_key="sk-ant-test",
            model="claude-3-sonnet-20240229",
        )
        adapter = ModelAdapter(config)
        
        result = adapter.execute(
            messages=[
                {"role": "user", "content": "Who are you?"}
            ]
        )
        
        assert result.success
        assert "Claude" in result.output["content"]
    
    def test_token_tracking(self):
        """Test cumulative token usage tracking."""
        with patch('openai.OpenAI'):
            config = ModelAdapterConfig(
                name="gpt4",
                provider=ModelProvider.OPENAI,
                api_key="sk-test",
                model="gpt-4",
                track_tokens=True,
            )
            adapter = ModelAdapter(config)
            
            # Manually set token count
            adapter._total_tokens = 1500
            
            usage = adapter.get_token_usage()
            assert usage["total_tokens"] == 1500
            
            adapter.reset_token_usage()
            assert adapter.get_token_usage()["total_tokens"] == 0


class TestRetryLogic:
    """Test adapter retry logic."""
    
    def test_retry_on_failure(self):
        """Test automatic retry on transient failures."""
        call_count = 0
        
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        config = PythonAdapterConfig(
            name="retry_test",
            callable=failing_func,
            retry_policy=RetryPolicy(
                max_attempts=3,
                backoff_factor=0.1,
            ),
        )
        adapter = PythonAdapter(config)
        
        result = adapter.execute()
        assert result.success
        assert result.output == "success"
        assert call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
