import asyncio
import inspect
import tempfile
import json
import os
from unittest.mock import Mock, AsyncMock

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Run async test functions without requiring external plugins."""
    test_function = pyfuncitem.obj
    if inspect.iscoroutinefunction(test_function):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            # Filter funcargs to only include parameters the function expects
            sig = inspect.signature(test_function)
            filtered_args = {k: v for k, v in pyfuncitem.funcargs.items() if k in sig.parameters}
            loop.run_until_complete(test_function(**filtered_args))
        finally:
            loop.close()
            asyncio.set_event_loop(None)
        return True
    return None


def pytest_configure(config):
    """Register markers for pytest."""
    config.addinivalue_line("markers", "asyncio: mark async tests")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "network: mark test as requiring network access")
    config.addinivalue_line("markers", "docker: mark test as requiring Docker")


# Local model deployment test fixtures
@pytest.fixture
def sample_models_config():
    """Sample model configurations for testing."""
    return {
        'vllm': {
            'name': 'vllm_test_model',
            'provider': 'vllm',
            'model_name': 'microsoft/DialoGPT-medium',
            'host': '127.0.0.1',
            'port': 8001,
            'config': {
                'temperature': 0.7,
                'max_tokens': 1024
            },
            'deployment_config': {
                'gpu_memory_utilization': 0.9,
                'tensor_parallel_size': 1,
                'max_model_len': 2048,
                'dtype': 'float16'
            }
        },
        'ollama': {
            'name': 'ollama_test_model',
            'provider': 'ollama',
            'model_name': 'llama3:8b',
            'host': '127.0.0.1',
            'port': 11434,
            'config': {
                'temperature': 0.8
            },
            'deployment_config': {
                'num_gpu': 1,
                'num_thread': 8,
                'num_ctx': 2048
            }
        },
        'local_ai': {
            'name': 'localai_test_model',
            'provider': 'local_ai',
            'model_name': 'ggml-model',
            'host': '127.0.0.1',
            'port': 8080,
            'config': {
                'temperature': 0.7,
                'max_tokens': 1024
            },
            'deployment_config': {
                'backend': 'llama-cpp',
                'f16': True,
                'threads': 4,
                'gpu_layers': 35,
                'context_size': 2048
            }
        }
    }


@pytest.fixture
def temp_config_file(sample_models_config):
    """Create temporary configuration file for testing."""
    def _create_config_file(provider='vllm'):
        config = sample_models_config[provider]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            return f.name
    
    return _create_config_file


@pytest.fixture
def cleanup_temp_files():
    """Cleanup temporary files after tests."""
    temp_files = []
    
    def _add_temp_file(file_path):
        temp_files.append(file_path)
        return file_path
    
    yield _add_temp_file
    
    # Cleanup
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Warning: Failed to cleanup {file_path}: {e}")


@pytest.fixture
def mock_provider_responses():
    """Mock responses for provider API calls."""
    return {
        'vllm_generate': {
            'choices': [{
                'message': {'content': 'Hello! How can I help you today?'},
                'finish_reason': 'stop'
            }],
            'model': 'microsoft/DialoGPT-medium',
            'usage': {'completion_tokens': 8, 'prompt_tokens': 5}
        },
        'ollama_generate': {
            'message': {'content': 'Hello! How can I help you today?'},
            'model': 'llama3:8b',
            'created_at': '2023-01-01T00:00:00Z',
            'done': True
        },
        'localai_generate': {
            'choices': [{
                'message': {'content': 'Hello! How can I help you today?'},
                'finish_reason': 'stop'
            }],
            'model': 'ggml-model',
            'usage': {'completion_tokens': 8, 'prompt_tokens': 5}
        }
    }


def create_mock_provider(provider_type, **kwargs):
    """Create a mock provider for testing."""
    provider = Mock()
    provider.name = kwargs.get('name', f'test_{provider_type}')
    provider.model = kwargs.get('model', 'test-model')
    provider.provider_type = provider_type
    
    # Mock methods
    provider.generate = AsyncMock()
    provider.stream = AsyncMock()
    provider.start_deployment = AsyncMock()
    provider.stop_deployment = AsyncMock()
    provider.get_deployment_info = Mock(return_value={
        'status': 'running',
        'model': provider.model,
        'url': f'http://localhost:8000'
    })
    
    return provider
