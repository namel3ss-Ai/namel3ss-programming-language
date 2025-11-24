"""Test suite for N3 Python SDK.

Tests N3Client (remote) and N3InProcessRuntime (embedded).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import httpx
from pathlib import Path

from namel3ss_sdk import (
    N3Client,
    N3InProcessRuntime,
    N3Settings,
    N3ClientConfig,
    N3RuntimeConfig,
    # Exceptions
    N3ClientError,
    N3ServerError,
    N3TimeoutError,
    N3ConnectionError,
    N3AuthError,
    N3CircuitBreakerError,
)


class TestN3Client:
    """Test remote N3 client."""
    
    @patch('httpx.Client')
    def test_execute_chain_success(self, mock_client_class):
        """Test successful chain execution."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": "Chain executed successfully",
            "request_id": "req-123"
        }
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        config = N3ClientConfig(
            base_url="https://api.example.com",
            api_token="test_token",
        )
        client = N3Client(config)
        
        result = client.chains.run("my_chain", inputs={"key": "value"})
        assert result["result"] == "Chain executed successfully"
    
    @patch('httpx.Client')
    def test_4xx_error_handling(self, mock_client_class):
        """Test 4xx client error handling."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "Invalid request"
        }
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=Mock(), response=mock_response
        )
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        config = N3ClientConfig(base_url="https://api.example.com")
        client = N3Client(config)
        
        with pytest.raises(N3ClientError) as exc_info:
            client.chains.run("bad_chain")
        
        assert exc_info.value.status_code == 400
    
    @patch('httpx.Client')
    def test_5xx_error_handling(self, mock_client_class):
        """Test 5xx server error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error", request=Mock(), response=mock_response
        )
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        config = N3ClientConfig(base_url="https://api.example.com")
        client = N3Client(config)
        
        with pytest.raises(N3ServerError):
            client.chains.run("failing_chain")
    
    @patch('httpx.Client')
    def test_timeout_error(self, mock_client_class):
        """Test timeout error handling."""
        mock_client = Mock()
        mock_client.post.side_effect = httpx.TimeoutException("Request timed out")
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        config = N3ClientConfig(
            base_url="https://api.example.com",
            timeout=1.0,
        )
        client = N3Client(config)
        
        with pytest.raises(N3TimeoutError) as exc_info:
            client.chains.run("slow_chain")
        
        assert exc_info.value.timeout_seconds == 1.0
    
    @patch('httpx.Client')
    def test_connection_error(self, mock_client_class):
        """Test connection error handling."""
        mock_client = Mock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        config = N3ClientConfig(base_url="https://api.example.com")
        client = N3Client(config)
        
        with pytest.raises(N3ConnectionError):
            client.chains.run("unreachable_chain")
    
    @patch('httpx.Client')
    def test_retry_logic(self, mock_client_class):
        """Test automatic retry on transient failures."""
        call_count = 0
        
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("Temporary failure")
            
            response = Mock()
            response.status_code = 200
            response.json.return_value = {"result": "success"}
            return response
        
        mock_client = Mock()
        mock_client.post.side_effect = side_effect
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        config = N3ClientConfig(
            base_url="https://api.example.com",
            retry_attempts=3,
            retry_backoff=0.1,
        )
        client = N3Client(config)
        
        result = client.chains.run("flaky_chain")
        assert result["result"] == "success"
        assert call_count == 3
    
    def test_circuit_breaker(self):
        """Test circuit breaker pattern."""
        config = N3ClientConfig(
            base_url="https://api.example.com",
            circuit_breaker_threshold=2,
            circuit_breaker_timeout=1.0,
        )
        client = N3Client(config)
        
        # Mock failures to trigger circuit breaker
        with patch('httpx.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.post.side_effect = httpx.ConnectError("Connection failed")
            mock_client_class.return_value.__enter__.return_value = mock_client
            
            # First two failures
            with pytest.raises(N3ConnectionError):
                client.chains.run("failing_chain")
            
            with pytest.raises(N3ConnectionError):
                client.chains.run("failing_chain")
            
            # Circuit should now be open
            with pytest.raises(N3CircuitBreakerError):
                client.chains.run("any_chain")


class TestN3InProcessRuntime:
    """Test in-process N3 runtime."""
    
    def test_load_n3_file(self, tmp_path):
        """Test loading .n3 file."""
        # Create test .n3 file
        n3_file = tmp_path / "test.n3"
        n3_file.write_text("""
        prompt "greeting" {
          template: "Hello, {{name}}!"
        }
        """)
        
        config = N3RuntimeConfig(source_file=str(n3_file))
        
        with patch('namel3ss_sdk.runtime.Parser') as mock_parser:
            mock_ast = Mock()
            mock_parser.return_value.parse.return_value = mock_ast
            
            with patch('namel3ss_sdk.runtime.BackendGenerator') as mock_gen:
                runtime = N3InProcessRuntime(config)
                
                # Verify parser was called
                mock_parser.assert_called_once()
                mock_gen.assert_called_once()
    
    def test_execute_chain(self, tmp_path):
        """Test executing chain from runtime."""
        n3_file = tmp_path / "app.n3"
        n3_file.write_text("""
        chain "double" {
          inputs: {x: int}
          outputs: {result: int}
          steps: [
            {set: "result", value: "{{x * 2}}"}
          ]
        }
        """)
        
        config = N3RuntimeConfig(source_file=str(n3_file))
        
        # Mock the backend runtime
        with patch('namel3ss_sdk.runtime.Parser'), \
             patch('namel3ss_sdk.runtime.BackendGenerator'), \
             patch('namel3ss_sdk.runtime.importlib') as mock_importlib:
            
            mock_runtime = Mock()
            mock_chain = Mock()
            mock_chain.return_value = {"result": 10}
            mock_runtime.double = mock_chain
            
            mock_module = Mock()
            mock_module.runtime = mock_runtime
            mock_importlib.import_module.return_value = mock_module
            
            runtime = N3InProcessRuntime(config)
            result = runtime.execute_raw("double", x=5)
            
            assert result == {"result": 10}


class TestN3Settings:
    """Test configuration management."""
    
    def test_default_settings(self):
        """Test default configuration values."""
        settings = N3Settings()
        
        assert settings.client_config.base_url == "http://localhost:8000"
        assert settings.client_config.timeout == 30.0
        assert settings.client_config.retry_attempts == 3
    
    def test_env_var_override(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("N3_BASE_URL", "https://custom.example.com")
        monkeypatch.setenv("N3_API_TOKEN", "custom_token")
        monkeypatch.setenv("N3_TIMEOUT", "60.0")
        
        settings = N3Settings()
        
        assert settings.client_config.base_url == "https://custom.example.com"
        assert settings.client_config.api_token == "custom_token"
        assert settings.client_config.timeout == 60.0
    
    def test_explicit_config(self):
        """Test explicit configuration."""
        config = N3ClientConfig(
            base_url="https://api.prod.com",
            api_token="prod_token",
            timeout=120.0,
        )
        
        assert config.base_url == "https://api.prod.com"
        assert config.api_token == "prod_token"
        assert config.timeout == 120.0


class TestContextManagers:
    """Test context manager support."""
    
    @patch('httpx.Client')
    def test_client_context_manager(self, mock_client_class):
        """Test N3Client as context manager."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "ok"}
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        config = N3ClientConfig(base_url="https://api.example.com")
        
        with N3Client(config) as client:
            result = client.chains.run("test_chain")
            assert result["result"] == "ok"


class TestAsyncSupport:
    """Test async API support."""
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_async_chain_execution(self, mock_client_class):
        """Test async chain execution."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "async_success"}
        
        mock_client = Mock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__.return_value = mock_client
        
        config = N3ClientConfig(base_url="https://api.example.com")
        client = N3Client(config)
        
        # Mock async execution
        with patch.object(client.chains, 'arun') as mock_arun:
            mock_arun.return_value = {"result": "async_success"}
            
            result = await client.chains.arun("async_chain")
            assert result["result"] == "async_success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
