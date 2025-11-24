# Local Model Deployment Guide for Namel3ss

This guide demonstrates how to deploy and manage local AI models using Namel3ss with support for vLLM, Ollama, and LocalAI.

## Overview

Namel3ss provides production-grade support for deploying AI models locally using three powerful inference engines:

- **vLLM**: High-throughput LLM inference with continuous batching
- **Ollama**: Easy-to-use local model runtime with model management
- **LocalAI**: Multi-format local AI inference server (GGML, GGUF, etc.)

## Quick Start

### 1. Install Dependencies

Choose the local model providers you need:

```bash
# For vLLM support
pip install namel3ss[vllm]

# For Ollama support  
pip install namel3ss[ollama]

# For LocalAI support
pip install namel3ss[localai]

# For all local model providers
pip install namel3ss[local-models]
```

### 2. Define Local Models in Namel3ss

Create an `app.n3` file with local model definitions:

```n3
# vLLM deployment example
ai model local_llama {
    provider: vllm
    model: meta-llama/Llama-2-7b-chat-hf
    config: {
        temperature: 0.8
        max_tokens: 4096
    }
    deployment_config: {
        gpu_memory_utilization: 0.95
        max_model_len: 4096
        dtype: "float16"
        tensor_parallel_size: 1
    }
    description: "Local Llama model for private inference"
}

# Ollama deployment example
ai model local_mistral {
    provider: ollama
    model: mistral:7b
    config: {
        temperature: 0.6
        num_ctx: 2048
    }
    deployment_config: {
        num_gpu: 1
        num_thread: 8
    }
    description: "Local Mistral model via Ollama"
}

# LocalAI deployment example
ai model custom_model {
    provider: local_ai
    model: custom-model
    local_model_path: "/models/custom-model.gguf"
    config: {
        temperature: 0.7
        context_length: 2048
    }
    deployment_config: {
        backend: "llama-cpp"
        f16: true
        threads: 4
    }
    description: "Custom local model via LocalAI"
}
```

### 3. Deploy Models

Use the Namel3ss CLI to manage local model deployments:

```bash
# Start a local model deployment
namel3ss deploy local start local_llama

# Check deployment status
namel3ss deploy local status

# Check specific model status
namel3ss deploy local status local_llama

# List all local deployments
namel3ss deploy local list

# Stop a deployment
namel3ss deploy local stop local_llama
```

## Provider-Specific Guides

### vLLM Deployment

vLLM provides high-throughput inference with continuous batching, ideal for production workloads.

#### Model Configuration

```n3
ai model production_llama {
    provider: vllm
    model: meta-llama/Llama-2-13b-chat-hf
    config: {
        temperature: 0.7
        max_tokens: 2048
        top_p: 0.9
    }
    deployment_config: {
        # GPU memory utilization (0.1-1.0)
        gpu_memory_utilization: 0.95
        
        # Maximum sequence length
        max_model_len: 4096
        
        # Model precision (auto, float16, bfloat16)
        dtype: "float16"
        
        # Multi-GPU setup
        tensor_parallel_size: 2
        
        # Quantization (optional)
        quantization: "awq"
        
        # Server configuration
        host: "0.0.0.0"
        port: 8000
    }
}
```

#### Hardware Requirements

- **GPU Memory**: Varies by model size (7B models ~14GB VRAM with float16)
- **System RAM**: 32GB+ recommended for larger models
- **Storage**: Fast SSD for model loading

#### Advanced vLLM Features

```bash
# Start with custom GPU memory utilization
namel3ss deploy local start production_llama --gpu 0.9

# Start on specific host/port
namel3ss deploy local start production_llama --host 0.0.0.0 --port 8001
```

### Ollama Deployment

Ollama provides an easy-to-use interface with automatic model management and pulling.

#### Model Configuration

```n3
ai model chat_assistant {
    provider: ollama
    model: llama3:8b
    config: {
        temperature: 0.8
        top_k: 40
        top_p: 0.9
        repeat_penalty: 1.1
        num_ctx: 4096
    }
    deployment_config: {
        # GPU acceleration
        num_gpu: 1
        
        # CPU threads
        num_thread: 8
        
        # Keep model loaded for 5 minutes after last request
        keep_alive: "5m"
        
        # Server configuration
        host: "127.0.0.1"
        port: 11434
    }
}
```

#### Model Management

Ollama automatically pulls models when needed:

```bash
# Start deployment (automatically pulls model if needed)
namel3ss deploy local start chat_assistant

# Check available models
namel3ss deploy local status chat_assistant
```

### LocalAI Deployment

LocalAI supports multiple model formats and backends, providing flexibility for various model types.

#### Model Configuration

```n3
ai model efficient_model {
    provider: local_ai
    model: efficient-7b
    local_model_path: "/models/efficient-7b.gguf"
    config: {
        temperature: 0.7
        max_tokens: 1024
        context_length: 2048
    }
    deployment_config: {
        # Backend engine (llama-cpp, gpt4all, etc.)
        backend: "llama-cpp"
        
        # Use float16 precision
        f16: true
        
        # CPU threads
        threads: 8
        
        # GPU layers (-1 for all layers)
        gpu_layers: 35
        
        # Memory mapping and locking
        mmap: true
        mlock: true
        
        # Batch size
        batch_size: 512
        
        # Server configuration
        host: "127.0.0.1"
        port: 8080
        use_docker: false
    }
}
```

#### Docker Deployment

For containerized LocalAI deployment:

```n3
ai model containerized_model {
    provider: local_ai
    model: my-model
    config: {
        temperature: 0.7
        max_tokens: 1024
    }
    deployment_config: {
        use_docker: true
        container_name: "localai-my-model"
        image: "quay.io/go-skynet/local-ai:latest"
        models_path: "./models"
        gpu: true
        env: {
            "DEBUG": "true"
            "SINGLE_ACTIVE_BACKEND": "true"
        }
    }
}
```

## Integration Examples

### Using Local Models in Chains

```n3
chain analyze_text {
    input: {text: str}
    output: {sentiment: str, summary: str}
    
    step sentiment_analysis {
        model: local_llama
        prompt: "Analyze the sentiment of this text: {{input.text}}"
        output: sentiment
    }
    
    step summarization {
        model: local_mistral
        prompt: "Summarize this text: {{input.text}}"
        output: summary
    }
}
```

### Multi-Model Setup

Deploy multiple models for different use cases:

```n3
# Fast model for simple tasks
ai model quick_responder {
    provider: ollama
    model: phi3:mini
    config: {
        temperature: 0.5
        max_tokens: 512
    }
}

# Powerful model for complex tasks
ai model deep_thinker {
    provider: vllm
    model: meta-llama/Llama-2-70b-chat-hf
    deployment_config: {
        tensor_parallel_size: 4
        gpu_memory_utilization: 0.95
    }
}

# Efficient model for embedding tasks
ai model text_embedder {
    provider: local_ai
    model: all-MiniLM-L6-v2
    local_model_path: "/models/all-MiniLM-L6-v2.gguf"
    deployment_config: {
        backend: "bert-embeddings"
    }
}
```

## Performance Optimization

### GPU Memory Management

```n3
# Optimize for maximum throughput
ai model high_throughput {
    provider: vllm
    model: mistralai/Mixtral-8x7B-Instruct-v0.1
    deployment_config: {
        gpu_memory_utilization: 0.98
        max_model_len: 8192
        tensor_parallel_size: 2
        swap_space: 4  # GB of CPU memory for offloading
    }
}
```

### CPU Optimization

```n3
# Optimize for CPU inference
ai model cpu_optimized {
    provider: local_ai
    model: efficient-model
    deployment_config: {
        threads: 16
        batch_size: 1024
        mlock: true
        mmap: true
    }
}
```

## Monitoring and Management

### Health Checks

```bash
# Check health of all deployments
namel3ss deploy local status

# Get detailed health information
namel3ss deploy local status local_llama
```

### Resource Monitoring

Monitor resource usage during deployment:

```bash
# Watch GPU usage
nvidia-smi -l 1

# Monitor system resources
htop

# Check deployment logs (when available)
namel3ss deploy local logs local_llama
```

## Troubleshooting

### Common Issues

1. **Out of GPU Memory**
   ```n3
   # Reduce GPU memory utilization
   deployment_config: {
       gpu_memory_utilization: 0.8  # Reduce from 0.95
       max_model_len: 2048          # Reduce sequence length
   }
   ```

2. **Model Loading Errors**
   ```bash
   # Check model availability
   namel3ss doctor
   
   # Verify model path
   ls -la /path/to/model/files
   ```

3. **Port Conflicts**
   ```n3
   # Use different ports for multiple deployments
   deployment_config: {
       port: 8001  # Instead of default 8000
   }
   ```

### Performance Issues

1. **Slow Inference**
   - Check GPU utilization with `nvidia-smi`
   - Increase batch size if memory allows
   - Use quantized models (AWQ, GPTQ)

2. **Memory Issues**
   - Enable memory mapping (`mmap: true`)
   - Use model sharding for large models
   - Reduce context length

### Debugging

Enable verbose logging:

```bash
# Set debug mode
export NAMEL3SS_LOG_LEVEL=DEBUG

# Run with verbose output
namel3ss deploy local start my_model --verbose
```

## Best Practices

### Security

1. **API Access Control**
   ```n3
   deployment_config: {
       api_key_required: true
       allowed_origins: ["http://localhost:3000"]
   }
   ```

2. **Network Security**
   - Use `host: "127.0.0.1"` for local-only access
   - Use `host: "0.0.0.0"` only when needed for external access
   - Consider using reverse proxy for production

### Resource Management

1. **Memory Optimization**
   - Use appropriate precision (float16 for GPU, int8 for CPU)
   - Enable memory mapping for large models
   - Set reasonable context windows

2. **GPU Utilization**
   - Monitor GPU memory usage
   - Use tensor parallelism for multi-GPU setups
   - Consider quantization for memory-constrained environments

### Model Selection

1. **Choose Right Model Size**
   - 7B models: Good balance of performance and resources
   - 13B models: Better quality, more GPU memory required
   - 70B+ models: Best quality, requires multiple GPUs

2. **Optimize for Use Case**
   - Chat models: Llama2-Chat, Mistral-Instruct
   - Code generation: CodeLlama, StarCoder
   - Embeddings: Sentence transformers, BGE models

## Advanced Configuration

### Custom Model Registry

```n3
# Define reusable model configurations
ai model base_config {
    provider: vllm
    config: {
        temperature: 0.7
        max_tokens: 2048
    }
    deployment_config: {
        gpu_memory_utilization: 0.9
        dtype: "float16"
    }
}

# Extend base configuration
ai model specialized_model {
    extends: base_config
    model: "specialized/model-7b"
    deployment_config: {
        max_model_len: 4096
        tensor_parallel_size: 2
    }
}
```

### Multi-Environment Deployment

```n3
# Development configuration
ai model dev_model {
    provider: ollama
    model: phi3:mini
    deployment_config: {
        port: 11434
    }
}

# Production configuration  
ai model prod_model {
    provider: vllm
    model: meta-llama/Llama-2-7b-chat-hf
    deployment_config: {
        host: "0.0.0.0"
        port: 8000
        gpu_memory_utilization: 0.95
        tensor_parallel_size: 2
    }
}
```

This completes the comprehensive local model deployment guide for Namel3ss, covering all three supported providers with practical examples and best practices.