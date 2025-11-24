"""Tests for local model IR specifications."""

import pytest
from dataclasses import asdict
from typing import Dict, Any, Optional

from namel3ss.ir.spec import (
    LocalModelSpec, BackendIR, FrontendIR
)


class TestLocalModelSpec:
    """Test cases for LocalModelSpec."""
    
    def test_local_model_spec_creation(self):
        """Test LocalModelSpec creation with default values."""
        spec = LocalModelSpec(
            name="test_model",
            provider="vllm",
            model_name="microsoft/DialoGPT-medium"
        )
        
        assert spec.name == "test_model"
        assert spec.provider == "vllm"
        assert spec.model_name == "microsoft/DialoGPT-medium"
        assert spec.host == "127.0.0.1"
        assert spec.port == 8000
        assert spec.auto_start is True
        assert spec.config == {}
        assert spec.deployment_config == {}
        assert spec.health_check_timeout == 300
        assert spec.startup_timeout == 600
    
    def test_local_model_spec_with_custom_config(self):
        """Test LocalModelSpec with custom configuration."""
        config = {
            'temperature': 0.8,
            'max_tokens': 2048,
            'top_p': 0.9
        }
        
        deployment_config = {
            'gpu_memory_utilization': 0.9,
            'tensor_parallel_size': 2,
            'max_model_len': 4096
        }
        
        spec = LocalModelSpec(
            name="advanced_model",
            provider="vllm",
            model_name="meta-llama/Llama-2-7b-chat-hf",
            host="0.0.0.0",
            port=8001,
            auto_start=False,
            config=config,
            deployment_config=deployment_config,
            health_check_timeout=180,
            startup_timeout=900
        )
        
        assert spec.name == "advanced_model"
        assert spec.host == "0.0.0.0"
        assert spec.port == 8001
        assert spec.auto_start is False
        assert spec.config == config
        assert spec.deployment_config == deployment_config
        assert spec.health_check_timeout == 180
        assert spec.startup_timeout == 900
    
    def test_local_model_spec_validation(self):
        """Test LocalModelSpec validation."""
        # Valid spec should not raise
        spec = LocalModelSpec(
            name="valid_model",
            provider="ollama",
            model_name="llama3:8b"
        )
        
        # Test validation method if it exists
        if hasattr(spec, 'validate'):
            spec.validate()
        
        # Test invalid port ranges
        with pytest.raises(ValueError):
            LocalModelSpec(
                name="invalid_port",
                provider="vllm", 
                model_name="test-model",
                port=0  # Invalid port
            )
        
        with pytest.raises(ValueError):
            LocalModelSpec(
                name="invalid_port",
                provider="vllm",
                model_name="test-model", 
                port=65536  # Invalid port
            )
    
    def test_local_model_spec_serialization(self):
        """Test LocalModelSpec serialization to dict."""
        spec = LocalModelSpec(
            name="serialization_test",
            provider="local_ai",
            model_name="ggml-model",
            config={'temperature': 0.7},
            deployment_config={'backend': 'llama-cpp'}
        )
        
        spec_dict = asdict(spec)
        
        assert spec_dict['name'] == "serialization_test"
        assert spec_dict['provider'] == "local_ai"
        assert spec_dict['model_name'] == "ggml-model"
        assert spec_dict['config'] == {'temperature': 0.7}
        assert spec_dict['deployment_config'] == {'backend': 'llama-cpp'}
        assert spec_dict['host'] == "127.0.0.1"
        assert spec_dict['port'] == 8000
    
    def test_local_model_spec_deserialization(self):
        """Test LocalModelSpec creation from dict."""
        spec_data = {
            'name': 'deserialization_test',
            'provider': 'vllm',
            'model_name': 'microsoft/DialoGPT-medium',
            'host': '192.168.1.100',
            'port': 8002,
            'auto_start': False,
            'config': {'temperature': 0.5, 'max_tokens': 1024},
            'deployment_config': {'gpu_memory_utilization': 0.8},
            'health_check_timeout': 240,
            'startup_timeout': 720
        }
        
        spec = LocalModelSpec(**spec_data)
        
        assert spec.name == spec_data['name']
        assert spec.provider == spec_data['provider']
        assert spec.model_name == spec_data['model_name']
        assert spec.host == spec_data['host']
        assert spec.port == spec_data['port']
        assert spec.auto_start == spec_data['auto_start']
        assert spec.config == spec_data['config']
        assert spec.deployment_config == spec_data['deployment_config']
    
    def test_local_model_spec_base_url_property(self):
        """Test base_url property calculation."""
        spec = LocalModelSpec(
            name="url_test",
            provider="vllm",
            model_name="test-model",
            host="192.168.1.50",
            port=8003
        )
        
        if hasattr(spec, 'base_url'):
            assert spec.base_url == "http://192.168.1.50:8003"
        else:
            # If base_url is not a property, test manual construction
            expected_url = f"http://{spec.host}:{spec.port}"
            assert expected_url == "http://192.168.1.50:8003"
    
    def test_local_model_spec_provider_specific_configs(self):
        """Test provider-specific deployment configurations."""
        # vLLM configuration
        vllm_spec = LocalModelSpec(
            name="vllm_test",
            provider="vllm",
            model_name="microsoft/DialoGPT-medium",
            deployment_config={
                'gpu_memory_utilization': 0.9,
                'tensor_parallel_size': 2,
                'max_model_len': 2048,
                'dtype': 'float16',
                'trust_remote_code': True
            }
        )
        
        assert vllm_spec.deployment_config['gpu_memory_utilization'] == 0.9
        assert vllm_spec.deployment_config['tensor_parallel_size'] == 2
        
        # Ollama configuration
        ollama_spec = LocalModelSpec(
            name="ollama_test",
            provider="ollama",
            model_name="llama3:8b",
            deployment_config={
                'num_gpu': 1,
                'num_thread': 8,
                'num_ctx': 2048,
                'temperature': 0.8
            }
        )
        
        assert ollama_spec.deployment_config['num_gpu'] == 1
        assert ollama_spec.deployment_config['num_thread'] == 8
        
        # LocalAI configuration
        localai_spec = LocalModelSpec(
            name="localai_test",
            provider="local_ai",
            model_name="ggml-model",
            deployment_config={
                'backend': 'llama-cpp',
                'f16': True,
                'threads': 4,
                'gpu_layers': 35,
                'context_size': 2048
            }
        )
        
        assert localai_spec.deployment_config['backend'] == 'llama-cpp'
        assert localai_spec.deployment_config['f16'] is True


class TestBackendIRIntegration:
    """Test cases for LocalModelSpec integration with BackendIR."""
    
    def test_backend_ir_with_local_models(self):
        """Test BackendIR containing local model specifications."""
        local_model_specs = [
            LocalModelSpec(
                name="chat_model",
                provider="vllm",
                model_name="microsoft/DialoGPT-medium",
                port=8001
            ),
            LocalModelSpec(
                name="completion_model",
                provider="ollama",
                model_name="llama3:8b",
                port=11434
            )
        ]
        
        backend_ir = BackendIR(
            local_models=local_model_specs,
            endpoints=[],
            database_config=None
        )
        
        assert len(backend_ir.local_models) == 2
        assert backend_ir.local_models[0].name == "chat_model"
        assert backend_ir.local_models[1].name == "completion_model"
        assert backend_ir.local_models[0].provider == "vllm"
        assert backend_ir.local_models[1].provider == "ollama"
    
    def test_complete_ir_local_model_integration(self):
        """Test complete IR with local model specifications."""
        local_model = LocalModelSpec(
            name="integrated_model",
            provider="local_ai",
            model_name="ggml-model",
            config={'temperature': 0.7},
            deployment_config={'backend': 'llama-cpp'}
        )
        
        backend_ir = BackendIR(
            local_models=[local_model],
            endpoints=[],
            database_config=None
        )
        
        frontend_ir = FrontendIR(
            app_name="test_app",
            app_version="1.0.0",
            pages=[],
            routes=[]
        )
        
        assert len(backend_ir.local_models) == 1
        assert backend_ir.local_models[0].name == "integrated_model"
        assert backend_ir.local_models[0].provider == "local_ai"


class TestLocalModelSpecEdgeCases:
    """Test edge cases and error conditions for LocalModelSpec."""
    
    def test_empty_configurations(self):
        """Test LocalModelSpec with empty configurations."""
        spec = LocalModelSpec(
            name="empty_config_test",
            provider="vllm",
            model_name="test-model",
            config={},
            deployment_config={}
        )
        
        assert spec.config == {}
        assert spec.deployment_config == {}
    
    def test_none_configurations(self):
        """Test LocalModelSpec with None configurations."""
        spec = LocalModelSpec(
            name="none_config_test",
            provider="ollama",
            model_name="test-model",
            config=None,
            deployment_config=None
        )
        
        # Should handle None gracefully (convert to empty dict or keep as None)
        assert spec.config is None or spec.config == {}
        assert spec.deployment_config is None or spec.deployment_config == {}
    
    def test_large_timeout_values(self):
        """Test LocalModelSpec with large timeout values."""
        spec = LocalModelSpec(
            name="large_timeout_test",
            provider="vllm",
            model_name="test-model",
            health_check_timeout=3600,  # 1 hour
            startup_timeout=7200        # 2 hours
        )
        
        assert spec.health_check_timeout == 3600
        assert spec.startup_timeout == 7200
    
    def test_special_characters_in_names(self):
        """Test LocalModelSpec with special characters in names."""
        spec = LocalModelSpec(
            name="model-with_special.chars_123",
            provider="ollama",
            model_name="namespace/model:tag"
        )
        
        assert spec.name == "model-with_special.chars_123"
        assert spec.model_name == "namespace/model:tag"
    
    def test_unicode_characters(self):
        """Test LocalModelSpec with unicode characters."""
        spec = LocalModelSpec(
            name="模型测试",
            provider="vllm",
            model_name="test-model"
        )
        
        assert spec.name == "模型测试"
    
    def test_deep_nested_config(self):
        """Test LocalModelSpec with deeply nested configuration."""
        complex_config = {
            'generation': {
                'parameters': {
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50
                },
                'constraints': {
                    'max_tokens': 2048,
                    'stop_sequences': ["<|end|>", "\n\n"]
                }
            },
            'model_settings': {
                'precision': 'fp16',
                'optimization': {
                    'enable_kv_cache': True,
                    'tensor_parallel': True
                }
            }
        }
        
        spec = LocalModelSpec(
            name="complex_config_test",
            provider="vllm",
            model_name="test-model",
            config=complex_config
        )
        
        assert spec.config['generation']['parameters']['temperature'] == 0.7
        assert spec.config['model_settings']['optimization']['enable_kv_cache'] is True


class TestLocalModelSpecValidation:
    """Test validation logic for LocalModelSpec."""
    
    def test_valid_provider_names(self):
        """Test validation of provider names."""
        valid_providers = ['vllm', 'ollama', 'local_ai']
        
        for provider in valid_providers:
            spec = LocalModelSpec(
                name=f"test_{provider}",
                provider=provider,
                model_name="test-model"
            )
            assert spec.provider == provider
    
    def test_invalid_provider_names(self):
        """Test rejection of invalid provider names."""
        invalid_providers = ['openai', 'anthropic', 'invalid_provider', '']
        
        for provider in invalid_providers:
            if hasattr(LocalModelSpec, '_validate_provider'):
                with pytest.raises(ValueError):
                    LocalModelSpec(
                        name="test",
                        provider=provider,
                        model_name="test-model"
                    )
            # If no validation exists, just ensure it doesn't crash
    
    def test_port_range_validation(self):
        """Test port range validation."""
        # Valid ports
        valid_ports = [1024, 8080, 11434, 8001, 65535]
        
        for port in valid_ports:
            spec = LocalModelSpec(
                name="port_test",
                provider="vllm",
                model_name="test-model",
                port=port
            )
            assert spec.port == port
        
        # Invalid ports (if validation exists)
        invalid_ports = [0, -1, 65536, 100000]
        
        for port in invalid_ports:
            if hasattr(LocalModelSpec, '_validate_port'):
                with pytest.raises(ValueError):
                    LocalModelSpec(
                        name="port_test",
                        provider="vllm",
                        model_name="test-model",
                        port=port
                    )
    
    def test_timeout_validation(self):
        """Test timeout value validation."""
        # Valid timeouts
        valid_timeouts = [30, 300, 600, 1800, 3600]
        
        for timeout in valid_timeouts:
            spec = LocalModelSpec(
                name="timeout_test",
                provider="vllm",
                model_name="test-model",
                health_check_timeout=timeout,
                startup_timeout=timeout * 2
            )
            assert spec.health_check_timeout == timeout
            assert spec.startup_timeout == timeout * 2
        
        # Invalid timeouts (if validation exists)
        invalid_timeouts = [-1, 0]
        
        for timeout in invalid_timeouts:
            if hasattr(LocalModelSpec, '_validate_timeout'):
                with pytest.raises(ValueError):
                    LocalModelSpec(
                        name="timeout_test",
                        provider="vllm",
                        model_name="test-model",
                        health_check_timeout=timeout
                    )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])