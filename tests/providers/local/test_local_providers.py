"""Tests for local model provider implementations."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import json
import httpx

from namel3ss.providers.local.vllm import VLLMProvider, VLLMDeploymentManager
from namel3ss.providers.local.ollama import OllamaProvider, OllamaModelManager, OllamaServerManager
from namel3ss.providers.local.local_ai import LocalAIProvider, LocalAIServerManager
from namel3ss.providers.base import ProviderMessage, ProviderResponse, ProviderError


class TestVLLMProvider:
    """Test cases for VLLMProvider."""
    
    @pytest.fixture
    def vllm_config(self):
        """Standard vLLM configuration for testing."""
        return {
            'host': '127.0.0.1',
            'port': 8001,
            'temperature': 0.7,
            'max_tokens': 1024,
            'auto_start_server': False,  # Don't auto-start in tests
            'deployment_config': {
                'gpu_memory_utilization': 0.9,
                'tensor_parallel_size': 1,
                'max_model_len': 2048,
                'dtype': 'float16'
            }
        }
    
    @pytest.fixture
    def vllm_provider(self, vllm_config):
        """Create VLLMProvider instance for testing."""
        return VLLMProvider(
            name="test_vllm",
            model="microsoft/DialoGPT-medium",
            config=vllm_config
        )
    
    def test_vllm_provider_initialization(self, vllm_provider, vllm_config):
        """Test VLLMProvider initialization."""
        assert vllm_provider.name == "test_vllm"
        assert vllm_provider.model == "microsoft/DialoGPT-medium"
        assert vllm_provider.temperature == 0.7
        assert vllm_provider.max_tokens == 1024
        assert vllm_provider.base_url == "http://127.0.0.1:8001"
        assert vllm_provider.auto_start_server is False
    
    @pytest.mark.asyncio
    async def test_vllm_generate(self, vllm_provider):
        """Test VLLM text generation."""
        messages = [
            ProviderMessage(role="user", content="Hello, how are you?")
        ]
        
        mock_response_data = {
            'choices': [{
                'message': {'content': 'Hello! I am doing well, thank you.'},
                'finish_reason': 'stop'
            }],
            'model': 'microsoft/DialoGPT-medium',
            'usage': {'completion_tokens': 8, 'prompt_tokens': 5}
        }
        
        with patch.object(vllm_provider, '_ensure_server_running', new_callable=AsyncMock), \
             patch('namel3ss.providers.local.vllm.record_metric', new_callable=AsyncMock), \
             patch('httpx.AsyncClient') as mock_client_class:
            
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_response_data
            
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            result = await vllm_provider.generate(messages)
            
            assert isinstance(result, ProviderResponse)
            assert result.output_text == 'Hello! I am doing well, thank you.'
            assert result.model == 'microsoft/DialoGPT-medium'
            assert result.metadata['provider'] == 'vllm'
    
    @pytest.mark.asyncio
    async def test_vllm_stream(self, vllm_provider):
        """Test VLLM streaming generation."""
        messages = [
            ProviderMessage(role="user", content="Tell me a story")
        ]
        
        mock_stream_chunks = [
            'data: {"choices": [{"delta": {"content": "Once"}}], "model": "test"}',
            'data: {"choices": [{"delta": {"content": " upon"}}], "model": "test"}',
            'data: {"choices": [{"delta": {"content": " a time"}}], "model": "test"}',
            'data: [DONE]'
        ]
        
        # Create async iterator for stream chunks
        async def async_lines_generator():
            for chunk in mock_stream_chunks:
                yield chunk
        
        with patch.object(vllm_provider, '_ensure_server_running', new_callable=AsyncMock), \
             patch('namel3ss.providers.local.vllm.record_metric', new_callable=AsyncMock), \
             patch('httpx.AsyncClient') as mock_client_class:
            
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.aiter_lines = Mock(return_value=async_lines_generator())
            
            # Create a proper async context manager
            async_context_manager = AsyncMock()
            async_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
            async_context_manager.__aexit__ = AsyncMock(return_value=None)
            
            mock_client = AsyncMock()
            mock_client.stream = Mock(return_value=async_context_manager)
            mock_client_class.return_value = mock_client
            
            chunks = []
            async for chunk in vllm_provider.stream(messages):
                chunks.append(chunk)
            
            assert len(chunks) == 3
            assert chunks[0].output_text == "Once"
            assert chunks[1].output_text == " upon"
            assert chunks[2].output_text == " a time"
    
    def test_vllm_build_request_payload(self, vllm_provider):
        """Test VLLM request payload building."""
        messages = [
            ProviderMessage(role="user", content="Hello"),
            ProviderMessage(role="assistant", content="Hi there!"),
            ProviderMessage(role="user", content="How are you?")
        ]
        
        payload = vllm_provider._build_request_payload(messages, stream=False)
        
        assert payload['model'] == "microsoft/DialoGPT-medium"
        assert payload['stream'] is False
        assert payload['temperature'] == 0.7
        assert payload['max_tokens'] == 1024
        assert len(payload['messages']) == 3
        assert payload['messages'][0] == {"role": "user", "content": "Hello"}


class TestOllamaProvider:
    """Test cases for OllamaProvider."""
    
    @pytest.fixture
    def ollama_config(self):
        """Standard Ollama configuration for testing."""
        return {
            'host': '127.0.0.1',
            'port': 11434,
            'temperature': 0.8,
            'auto_pull_model': False,
            'auto_start_server': False,
            'deployment_config': {
                'num_gpu': 1,
                'num_thread': 8
            }
        }
    
    @pytest.fixture
    def ollama_provider(self, ollama_config):
        """Create OllamaProvider instance for testing."""
        return OllamaProvider(
            name="test_ollama",
            model="llama3:8b",
            config=ollama_config
        )
    
    def test_ollama_provider_initialization(self, ollama_provider):
        """Test OllamaProvider initialization."""
        assert ollama_provider.name == "test_ollama"
        assert ollama_provider.model == "llama3:8b"
        assert ollama_provider.temperature == 0.8
        assert ollama_provider.base_url == "http://127.0.0.1:11434"
        assert ollama_provider.auto_pull_model is False
        assert ollama_provider.auto_start_server is False
    
    @pytest.mark.asyncio
    async def test_ollama_generate(self, ollama_provider):
        """Test Ollama text generation."""
        messages = [
            ProviderMessage(role="user", content="What is AI?")
        ]
        
        mock_response_data = {
            'message': {'content': 'AI stands for Artificial Intelligence.'},
            'model': 'llama3:8b',
            'created_at': '2023-01-01T00:00:00Z',
            'done': True
        }
        
        with patch.object(ollama_provider, '_ensure_ready', new_callable=AsyncMock), \
             patch('namel3ss.providers.local.ollama.record_metric', new_callable=AsyncMock), \
             patch('httpx.AsyncClient') as mock_client_class:
            
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_response_data
            
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            result = await ollama_provider.generate(messages)
            
            assert isinstance(result, ProviderResponse)
            assert result.output_text == 'AI stands for Artificial Intelligence.'
            assert result.model == 'llama3:8b'
            assert result.metadata['provider'] == 'ollama'
    
    @pytest.mark.asyncio
    async def test_ollama_stream(self, ollama_provider):
        """Test Ollama streaming generation."""
        messages = [
            ProviderMessage(role="user", content="Explain quantum computing")
        ]
        
        mock_stream_chunks = [
            '{"message": {"content": "Quantum"}, "done": false}',
            '{"message": {"content": " computing"}, "done": false}',
            '{"message": {"content": " is"}, "done": false}',
            '{"message": {"content": ""}, "done": true}'
        ]
        
        # Create async iterator for stream chunks
        async def async_lines_generator():
            for chunk in mock_stream_chunks:
                yield chunk
        
        with patch.object(ollama_provider, '_ensure_ready', new_callable=AsyncMock), \
             patch('namel3ss.providers.local.ollama.record_metric', new_callable=AsyncMock), \
             patch('httpx.AsyncClient') as mock_client_class:
            
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.aiter_lines = Mock(return_value=async_lines_generator())
            
            # Create a proper async context manager
            async_context_manager = AsyncMock()
            async_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
            async_context_manager.__aexit__ = AsyncMock(return_value=None)
            
            mock_client = AsyncMock()
            mock_client.stream = Mock(return_value=async_context_manager)
            mock_client_class.return_value = mock_client
            
            chunks = []
            async for chunk in ollama_provider.stream(messages):
                chunks.append(chunk)
            
            assert len(chunks) == 4
            assert chunks[0].output_text == "Quantum"
            assert chunks[1].output_text == " computing"
            assert chunks[2].output_text == " is"
            assert chunks[3].metadata['done'] is True


class TestLocalAIProvider:
    """Test cases for LocalAIProvider."""
    
    @pytest.fixture
    def localai_config(self):
        """Standard LocalAI configuration for testing."""
        return {
            'host': '127.0.0.1',
            'port': 8080,
            'temperature': 0.7,
            'max_tokens': 1024,
            'auto_start_server': False,
            'deployment_config': {
                'backend': 'llama-cpp',
                'f16': True,
                'threads': 4,
                'gpu_layers': 35
            }
        }
    
    @pytest.fixture
    def localai_provider(self, localai_config):
        """Create LocalAIProvider instance for testing."""
        return LocalAIProvider(
            name="test_localai",
            model="ggml-model",
            config=localai_config
        )
    
    def test_localai_provider_initialization(self, localai_provider):
        """Test LocalAIProvider initialization."""
        assert localai_provider.name == "test_localai"
        assert localai_provider.model == "ggml-model"
        assert localai_provider.temperature == 0.7
        assert localai_provider.max_tokens == 1024
        assert localai_provider.base_url == "http://127.0.0.1:8080"
        assert localai_provider.auto_start_server is False
    
    @pytest.mark.asyncio
    async def test_localai_generate(self, localai_provider):
        """Test LocalAI text generation."""
        messages = [
            ProviderMessage(role="user", content="What is machine learning?")
        ]
        
        mock_response_data = {
            'choices': [{
                'message': {'content': 'Machine learning is a subset of artificial intelligence.'},
                'finish_reason': 'stop'
            }],
            'model': 'ggml-model',
            'usage': {'completion_tokens': 10, 'prompt_tokens': 5}
        }
        
        with patch.object(localai_provider, '_ensure_server_running', new_callable=AsyncMock), \
             patch('namel3ss.providers.local.local_ai.record_metric', new_callable=AsyncMock), \
             patch('httpx.AsyncClient') as mock_client_class:
            
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_response_data
            
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            result = await localai_provider.generate(messages)
            
            assert isinstance(result, ProviderResponse)
            assert result.output_text == 'Machine learning is a subset of artificial intelligence.'
            assert result.model == 'ggml-model'
            assert result.metadata['provider'] == 'local_ai'


class TestVLLMDeploymentManager:
    """Test cases for VLLMDeploymentManager."""
    
    @pytest.fixture
    def deployment_config(self):
        """Standard deployment configuration for testing."""
        return {
            'host': '127.0.0.1',
            'port': 8001,
            'gpu_memory_utilization': 0.9,
            'tensor_parallel_size': 1,
            'max_model_len': 2048
        }
    
    @pytest.fixture
    def deployment_manager(self, deployment_config):
        """Create VLLMDeploymentManager instance for testing."""
        return VLLMDeploymentManager(deployment_config)
    
    def test_deployment_manager_initialization(self, deployment_manager):
        """Test VLLMDeploymentManager initialization."""
        assert deployment_manager.base_url == "http://127.0.0.1:8001"
        assert deployment_manager.config['gpu_memory_utilization'] == 0.9
    
    def test_build_vllm_command(self, deployment_manager):
        """Test vLLM command building."""
        model_name = "microsoft/DialoGPT-medium"
        cmd = deployment_manager._build_vllm_command(model_name)
        
        expected_elements = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--host", "127.0.0.1",
            "--port", "8001",
            "--gpu-memory-utilization", "0.9",
            "--tensor-parallel-size", "1",
            "--max-model-len", "2048"
        ]
        
        for element in expected_elements:
            assert str(element) in [str(c) for c in cmd]
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, deployment_manager):
        """Test successful health check."""
        mock_health_data = {"status": "ready", "version": "0.4.0"}
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_health_data
            mock_response.raise_for_status = Mock()
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            health = await deployment_manager.health_check()
            
            assert health == mock_health_data
    
    @pytest.mark.asyncio 
    async def test_health_check_failure(self, deployment_manager):
        """Test failed health check."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = httpx.ConnectError("Connection failed")
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client.return_value.__aexit__.return_value = None
            
            with pytest.raises(ProviderError, match="vLLM health check failed"):
                await deployment_manager.health_check()


class TestOllamaModelManager:
    """Test cases for OllamaModelManager."""
    
    @pytest.fixture
    def model_manager(self):
        """Create OllamaModelManager instance for testing."""
        return OllamaModelManager("http://localhost:11434")
    
    @pytest.mark.asyncio
    async def test_ensure_model_available_already_available(self, model_manager):
        """Test model availability when model is already available."""
        model_name = "llama3:8b"
        
        with patch.object(model_manager, '_is_model_available', return_value=True) as mock_check:
            await model_manager.ensure_model_available(model_name)
            
            mock_check.assert_called_once_with(model_name)
            assert model_name in model_manager._pulled_models
    
    @pytest.mark.asyncio
    async def test_ensure_model_available_needs_pulling(self, model_manager):
        """Test model availability when model needs to be pulled."""
        model_name = "mistral:7b"
        
        with patch.object(model_manager, '_is_model_available', return_value=False), \
             patch.object(model_manager, '_pull_model', new_callable=AsyncMock) as mock_pull:
            
            await model_manager.ensure_model_available(model_name)
            
            mock_pull.assert_called_once_with(model_name)
            assert model_name in model_manager._pulled_models
    
    @pytest.mark.asyncio
    async def test_list_models(self, model_manager):
        """Test listing available models."""
        mock_models_response = {
            "models": [
                {"name": "llama3:8b", "size": "4.7GB"},
                {"name": "mistral:7b", "size": "4.1GB"}
            ]
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_models_response
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            models = await model_manager.list_models()
            
            assert len(models) == 2
            assert models[0]["name"] == "llama3:8b"
            assert models[1]["name"] == "mistral:7b"


class TestLocalAIServerManager:
    """Test cases for LocalAIServerManager."""
    
    @pytest.fixture
    def server_config(self):
        """Standard server configuration for testing."""
        return {
            'host': '127.0.0.1',
            'port': 8080,
            'use_docker': False,
            'binary_path': 'local-ai',
            'models_path': '/models',
            'debug': True
        }
    
    @pytest.fixture
    def server_manager(self, server_config):
        """Create LocalAIServerManager instance for testing."""
        return LocalAIServerManager(server_config)
    
    def test_server_manager_initialization(self, server_manager):
        """Test LocalAIServerManager initialization."""
        assert server_manager.base_url == "http://127.0.0.1:8080"
        assert server_manager.host == "127.0.0.1"
        assert server_manager.port == 8080
    
    def test_build_binary_command(self, server_manager):
        """Test binary command building."""
        cmd = server_manager._build_binary_command()
        
        expected_elements = [
            "local-ai",
            "--address", "127.0.0.1:8080",
            "--models-path", "/models",
            "--debug"
        ]
        
        for element in expected_elements:
            assert element in cmd
    
    def test_build_docker_command(self, server_manager):
        """Test Docker command building."""
        server_manager.config['use_docker'] = True
        server_manager.config['gpu'] = True
        server_manager.config['container_name'] = 'test-localai'
        
        cmd = server_manager._build_docker_command()
        
        expected_elements = [
            "docker", "run", "-d",
            "--name", "test-localai",
            "-p", "8080:8080",
            "--gpus", "all"
        ]
        
        for element in expected_elements:
            assert element in cmd


@pytest.mark.integration
class TestLocalProvidersIntegration:
    """Integration tests for local providers."""
    
    def test_provider_factory_integration(self):
        """Test that local providers are properly registered in factory."""
        from namel3ss.providers.factory import _PROVIDER_CLASSES, get_provider_class
        
        # Import providers to trigger registration
        from namel3ss.providers.local import vllm, ollama, local_ai
        
        # Test provider registration
        assert 'vllm' in _PROVIDER_CLASSES or get_provider_class('vllm') is not None
        assert 'ollama' in _PROVIDER_CLASSES or get_provider_class('ollama') is not None
        assert 'local_ai' in _PROVIDER_CLASSES or get_provider_class('local_ai') is not None
    
    def test_ai_model_validation(self):
        """Test AI model validation with local deployment configs."""
        from namel3ss.ast.ai.models import AIModel
        from namel3ss.ast.ai.validation import validate_ai_model
        
        # Test valid vLLM model
        vllm_model = AIModel(
            name="test_vllm",
            provider="vllm", 
            model_name="microsoft/DialoGPT-medium",
            config={"temperature": 0.7},
            deployment_config={
                "gpu_memory_utilization": 0.9,
                "tensor_parallel_size": 1
            },
            is_local=True
        )
        
        # Should not raise any exceptions
        validate_ai_model(vllm_model)
        
        # Test valid Ollama model
        ollama_model = AIModel(
            name="test_ollama",
            provider="ollama",
            model_name="llama3:8b", 
            config={"temperature": 0.8},
            deployment_config={
                "num_gpu": 1,
                "num_thread": 8
            },
            is_local=True
        )
        
        # Should not raise any exceptions
        validate_ai_model(ollama_model)
        
        # Test invalid configuration
        invalid_model = AIModel(
            name="test_invalid",
            provider="vllm",
            model_name="test-model",
            deployment_config={
                "gpu_memory_utilization": 1.5  # Invalid: > 1.0
            },
            is_local=True
        )
        
        with pytest.raises(Exception):  # Should raise AIValidationError
            validate_ai_model(invalid_model)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])