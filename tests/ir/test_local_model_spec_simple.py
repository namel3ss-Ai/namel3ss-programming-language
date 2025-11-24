"""Simple tests for LocalModelSpec in IR to validate our understanding."""

import pytest
from dataclasses import asdict

from namel3ss.ir.spec import LocalModelSpec, BackendIR


class TestLocalModelSpecSimple:
    """Basic tests for LocalModelSpec to understand the structure."""
    
    def test_local_model_spec_creation(self):
        """Test creating LocalModelSpec with required fields."""
        spec = LocalModelSpec(
            name="test_model",
            engine_type="vllm",
            model_name="microsoft/DialoGPT-medium"
        )
        
        assert spec.name == "test_model"
        assert spec.engine_type == "vllm"
        assert spec.model_name == "microsoft/DialoGPT-medium"
        assert spec.host == "0.0.0.0"  # Default
        assert spec.port == 8000  # Default
    
    def test_local_model_spec_with_config(self):
        """Test LocalModelSpec with custom configuration."""
        deployment_config = {
            'gpu_memory_utilization': 0.9,
            'tensor_parallel_size': 2
        }
        
        vllm_config = {
            'max_model_len': 2048,
            'dtype': 'float16'
        }
        
        spec = LocalModelSpec(
            name="advanced_model",
            engine_type="vllm",
            model_name="meta-llama/Llama-2-7b-chat-hf",
            host="127.0.0.1",
            port=8001,
            deployment_config=deployment_config,
            vllm_config=vllm_config,
            gpu_required=True,
            min_vram_gb=8.0
        )
        
        assert spec.deployment_config == deployment_config
        assert spec.vllm_config == vllm_config
        assert spec.gpu_required is True
        assert spec.min_vram_gb == 8.0
    
    def test_local_model_spec_serialization(self):
        """Test LocalModelSpec serialization."""
        spec = LocalModelSpec(
            name="serialization_test",
            engine_type="ollama",
            model_name="llama3:8b",
            ollama_config={'num_gpu': 1}
        )
        
        spec_dict = asdict(spec)
        
        assert spec_dict['name'] == "serialization_test"
        assert spec_dict['engine_type'] == "ollama"
        assert spec_dict['ollama_config'] == {'num_gpu': 1}
    
    def test_backend_ir_with_local_models(self):
        """Test BackendIR containing local model specs."""
        local_models = [
            LocalModelSpec(
                name="chat_model",
                engine_type="vllm",
                model_name="microsoft/DialoGPT-medium"
            ),
            LocalModelSpec(
                name="completion_model",
                engine_type="ollama",
                model_name="llama3:8b"
            )
        ]
        
        backend_ir = BackendIR(
            app_name="test_app",
            local_models=local_models
        )
        
        assert len(backend_ir.local_models) == 2
        assert backend_ir.local_models[0].name == "chat_model"
        assert backend_ir.local_models[0].engine_type == "vllm"
        assert backend_ir.local_models[1].name == "completion_model"
        assert backend_ir.local_models[1].engine_type == "ollama"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])