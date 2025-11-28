"""Tests for enhanced Ollama provider features (caching, metrics, error handling, editor tools)."""

import pytest
import asyncio
import os
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any
import httpx

from namel3ss.providers.local.ollama import (
    OllamaProvider,
    OllamaModelManager,
    OllamaServerManager,
    OllamaError,
    ModelAvailabilityCache,
    OllamaEditorTools,
    OLLAMA_BASE_URL_ENV,
    DEFAULT_OLLAMA_URL,
)
from namel3ss.providers.base import ProviderMessage, ProviderResponse


class TestModelAvailabilityCache:
    """Test cases for ModelAvailabilityCache."""
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test basic cache operations."""
        cache = ModelAvailabilityCache(ttl=60)
        
        # Initially empty
        result = await cache.get("test-key")
        assert result is None
        
        # Set value
        await cache.set("test-key", True)
        result = await cache.get("test-key")
        assert result is True
        
        # Set False
        await cache.set("another-key", False)
        result = await cache.get("another-key")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache entries expire after TTL."""
        cache = ModelAvailabilityCache(ttl=0.1)  # 100ms TTL
        
        await cache.set("test-key", True)
        result = await cache.get("test-key")
        assert result is True
        
        # Wait for expiration
        await asyncio.sleep(0.15)
        
        result = await cache.get("test-key")
        assert result is None  # Expired
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self):
        """Test manual cache invalidation."""
        cache = ModelAvailabilityCache(ttl=60)
        
        await cache.set("test-key", True)
        assert await cache.get("test-key") is True
        
        await cache.invalidate("test-key")
        assert await cache.get("test-key") is None
    
    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test clearing entire cache."""
        cache = ModelAvailabilityCache(ttl=60)
        
        await cache.set("key1", True)
        await cache.set("key2", False)
        await cache.set("key3", True)
        
        await cache.clear()
        
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None


class TestOllamaError:
    """Test cases for OllamaError."""
    
    def test_basic_error(self):
        """Test basic error creation."""
        error = OllamaError("Something went wrong")
        assert "Something went wrong" in str(error)
    
    def test_error_with_context(self):
        """Test error with model and base_url context."""
        error = OllamaError(
            "Model not found",
            model="llama3:8b",
            base_url="http://localhost:11434"
        )
        
        error_str = str(error)
        assert "Model not found" in error_str
        assert "llama3:8b" in error_str
        assert "http://localhost:11434" in error_str
    
    def test_error_with_suggestion(self):
        """Test error includes actionable suggestion."""
        error = OllamaError(
            "Server unreachable",
            base_url="http://localhost:11434",
            suggestion="Run: ollama serve"
        )
        
        error_str = str(error)
        assert "Server unreachable" in error_str
        assert "ollama serve" in error_str
    
    def test_error_attributes(self):
        """Test error attributes are accessible."""
        original = ValueError("original error")
        error = OllamaError(
            "Wrapper error",
            model="test-model",
            base_url="http://test",
            suggestion="Fix it",
            original_error=original
        )
        
        assert error.model == "test-model"
        assert error.base_url == "http://test"
        assert error.suggestion == "Fix it"
        assert error.original_error is original


class TestOllamaModelManagerEnhanced:
    """Test cases for enhanced OllamaModelManager with caching."""
    
    @pytest.fixture
    def model_manager(self):
        """Create OllamaModelManager with test cache."""
        cache = ModelAvailabilityCache(ttl=60)
        return OllamaModelManager("http://localhost:11434", cache=cache)
    
    @pytest.mark.asyncio
    async def test_ensure_model_uses_cache(self, model_manager):
        """Test that model availability check uses cache."""
        model_name = "llama3:8b"
        
        # Pre-populate cache
        cache_key = model_manager._cache_key(model_name)
        await model_manager._cache.set(cache_key, True)
        
        # Should not make network call if cached
        with patch.object(model_manager, '_is_model_available') as mock_check:
            await model_manager.ensure_model_available(model_name, auto_pull=False)
            mock_check.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_ensure_model_helpful_error_when_missing(self, model_manager):
        """Test helpful error when model missing and auto_pull disabled."""
        model_name = "missing-model:latest"
        
        with patch.object(model_manager, '_is_model_available', return_value=False):
            with pytest.raises(OllamaError) as exc_info:
                await model_manager.ensure_model_available(
                    model_name,
                    auto_pull=False
                )
            
            error = exc_info.value
            assert model_name in str(error)
            assert "ollama pull" in str(error)
            assert error.suggestion is not None
    
    @pytest.mark.asyncio
    async def test_delete_model_invalidates_cache(self, model_manager):
        """Test model deletion invalidates cache."""
        model_name = "test-model:latest"
        
        # Pre-populate cache
        cache_key = model_manager._cache_key(model_name)
        await model_manager._cache.set(cache_key, True)
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            
            mock_client_instance = AsyncMock()
            mock_client_instance.delete = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            await model_manager.delete_model(model_name)
        
        # Cache should be invalidated
        assert await model_manager._cache.get(cache_key) is None


class TestOllamaServerManagerEnhanced:
    """Test cases for enhanced OllamaServerManager with health throttling."""
    
    @pytest.fixture
    def server_manager(self):
        """Create OllamaServerManager for testing."""
        config = {
            'host': '127.0.0.1',
            'port': 11434,
            'health_check_interval': 30,
        }
        return OllamaServerManager(config)
    
    @pytest.mark.asyncio
    async def test_health_check_throttling(self, server_manager):
        """Test health checks are throttled."""
        mock_response_data = {'models': []}
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_response_data
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # First call - should hit server
            await server_manager.health_check()
            assert mock_client_instance.get.call_count == 1
            
            # Second call immediately after - should use cache
            await server_manager.health_check()
            assert mock_client_instance.get.call_count == 1  # No additional call
            
            # Force fresh check
            await server_manager.health_check(force=True)
            assert mock_client_instance.get.call_count == 2  # New call
    
    @pytest.mark.asyncio
    async def test_health_check_returns_metrics(self, server_manager):
        """Test health check returns response time metrics."""
        mock_response_data = {'models': [{'name': 'test:latest'}]}
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_response_data
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            health = await server_manager.health_check(force=True)
            
            assert health['status'] == 'healthy'
            assert 'response_time_ms' in health
            assert health['response_time_ms'] >= 0
            assert 'checked_at' in health


class TestOllamaProviderConfiguration:
    """Test cases for OllamaProvider configuration and URL resolution."""
    
    def test_base_url_from_config(self):
        """Test explicit base_url in config takes priority."""
        provider = OllamaProvider(
            name="test",
            model="llama3:8b",
            config={'base_url': 'http://custom:9999'}
        )
        assert provider.base_url == 'http://custom:9999'
    
    def test_base_url_from_env(self, monkeypatch):
        """Test base_url from environment variable."""
        monkeypatch.setenv(OLLAMA_BASE_URL_ENV, 'http://env-server:8080')
        
        provider = OllamaProvider(
            name="test",
            model="llama3:8b",
            config={}
        )
        assert provider.base_url == 'http://env-server:8080'
    
    def test_base_url_from_host_port(self):
        """Test base_url constructed from host and port."""
        provider = OllamaProvider(
            name="test",
            model="llama3:8b",
            config={'host': '192.168.1.100', 'port': 11435}
        )
        assert provider.base_url == 'http://192.168.1.100:11435'
    
    def test_base_url_default(self):
        """Test default base_url when nothing specified."""
        provider = OllamaProvider(
            name="test",
            model="llama3:8b",
            config={}
        )
        assert provider.base_url == DEFAULT_OLLAMA_URL
    
    def test_cache_ttl_configurable(self):
        """Test model cache TTL is configurable."""
        provider = OllamaProvider(
            name="test",
            model="llama3:8b",
            config={'model_cache_ttl': 120}
        )
        assert provider._cache.ttl == 120


class TestOllamaEditorTools:
    """Test cases for OllamaEditorTools."""
    
    @pytest.mark.asyncio
    async def test_check_status_reachable(self):
        """Test check_status when server is reachable."""
        mock_response_data = {
            'models': [
                {'name': 'llama3:8b'},
                {'name': 'mistral:latest'}
            ]
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_response_data
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            status = await OllamaEditorTools.check_status()
            
            assert status['reachable'] is True
            assert status['models_count'] == 2
            assert 'llama3:8b' in status['models']
    
    @pytest.mark.asyncio
    async def test_check_status_unreachable(self):
        """Test check_status when server is unreachable."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            status = await OllamaEditorTools.check_status()
            
            assert status['reachable'] is False
            assert 'error' in status
            assert 'suggestion' in status
    
    @pytest.mark.asyncio
    async def test_list_available_models(self):
        """Test listing models for editor consumption."""
        mock_response_data = {
            'models': [
                {
                    'name': 'llama3:8b',
                    'size': 4661229568,
                    'modified_at': '2024-01-01T00:00:00Z',
                    'digest': 'sha256:abc123'
                }
            ]
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_response_data
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            models = await OllamaEditorTools.list_available_models()
            
            assert len(models) == 1
            assert models[0]['name'] == 'llama3:8b'
            assert models[0]['size'] == 4661229568
    
    def test_get_default_model(self):
        """Test default model recommendation."""
        default = OllamaEditorTools.get_default_model()
        assert default == "llama3:8b"
    
    def test_get_supported_capabilities(self):
        """Test capabilities report."""
        caps = OllamaEditorTools.get_supported_capabilities()
        
        assert caps['chat'] is True
        assert caps['streaming'] is True
        assert caps['local_deployment'] is True
        assert caps['function_calling'] is False
    
    def test_validate_model_name_valid(self):
        """Test model name validation - valid cases."""
        result = OllamaEditorTools.validate_model_name("llama3:8b")
        assert result['is_valid'] is True
        
        result = OllamaEditorTools.validate_model_name("mistral")
        assert result['is_valid'] is True
    
    def test_validate_model_name_invalid(self):
        """Test model name validation - invalid cases."""
        # Empty
        result = OllamaEditorTools.validate_model_name("")
        assert result['is_valid'] is False
        
        # Invalid format
        result = OllamaEditorTools.validate_model_name("model:tag:extra")
        assert result['is_valid'] is False
        
        # Whitespace
        result = OllamaEditorTools.validate_model_name("model with spaces")
        assert result['is_valid'] is False


class TestOllamaProviderErrorHandling:
    """Test cases for improved error handling."""
    
    @pytest.fixture
    def provider(self):
        """Create provider for testing."""
        return OllamaProvider(
            name="test",
            model="llama3:8b",
            config={'auto_pull_model': False, 'auto_start_server': False}
        )
    
    @pytest.mark.asyncio
    async def test_server_unreachable_error(self, provider):
        """Test error message when server is unreachable."""
        messages = [ProviderMessage(role="user", content="test")]
        
        with patch.object(provider, '_ensure_ready', new_callable=AsyncMock), \
             patch.object(provider, '_get_http_client') as mock_client:
            
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection failed")
            )
            mock_client.return_value = mock_client_instance
            
            with pytest.raises(OllamaError) as exc_info:
                await provider.generate(messages)
            
            error = exc_info.value
            assert "Could not reach Ollama" in str(error)
            assert provider.base_url in str(error)
            assert "ollama serve" in str(error)
    
    @pytest.mark.asyncio
    async def test_context_window_error(self, provider):
        """Test error message for context window overflow."""
        messages = [ProviderMessage(role="user", content="test")]
        
        with patch.object(provider, '_ensure_ready', new_callable=AsyncMock), \
             patch.object(provider, '_get_http_client') as mock_client:
            
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Error: context too large"
            
            http_error = httpx.HTTPStatusError(
                "Error",
                request=Mock(),
                response=mock_response
            )
            
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(side_effect=http_error)
            mock_client.return_value = mock_client_instance
            
            with pytest.raises(OllamaError) as exc_info:
                await provider.generate(messages)
            
            error = exc_info.value
            assert "too large" in str(error).lower()
            assert "context window" in str(error).lower()
            assert str(provider.num_ctx) in str(error)


class TestOllamaProviderMetrics:
    """Test cases for metrics recording."""
    
    @pytest.fixture
    def provider(self):
        """Create provider for testing."""
        return OllamaProvider(
            name="test",
            model="llama3:8b",
            config={'auto_pull_model': False, 'auto_start_server': False}
        )
    
    @pytest.mark.asyncio
    async def test_generate_records_metrics(self, provider):
        """Test that generate records comprehensive metrics."""
        messages = [ProviderMessage(role="user", content="test")]
        
        mock_response_data = {
            'message': {'content': 'response'},
            'model': 'llama3:8b',
            'done': True,
            'prompt_eval_count': 10,
            'eval_count': 20,
        }
        
        recorded_metrics = []
        
        def mock_record_metric(name, value, tags):
            recorded_metrics.append((name, value, tags))
        
        with patch.object(provider, '_ensure_ready', new_callable=AsyncMock), \
             patch.object(provider, '_get_http_client') as mock_client, \
             patch('namel3ss.providers.local.ollama.record_metric', side_effect=mock_record_metric):
            
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_response_data
            
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_client_instance
            
            await provider.generate(messages)
        
        # Check metrics were recorded
        metric_names = [m[0] for m in recorded_metrics]
        assert 'ollama.request.duration' in metric_names
        assert 'ollama.tokens.total' in metric_names
        assert 'ollama.tokens.prompt' in metric_names
        assert 'ollama.tokens.completion' in metric_names
        assert 'ollama.throughput.tokens_per_second' in metric_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
