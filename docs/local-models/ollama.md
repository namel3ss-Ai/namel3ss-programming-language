# Ollama Integration Guide

Complete guide to using Ollama for local model deployment in Namel3ss.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Model Management](#model-management)
- [Performance & Limitations](#performance--limitations)
- [Error Handling](#error-handling)
- [Editor Integration](#editor-integration)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Overview

[Ollama](https://ollama.ai) is a local LLM runtime that makes it easy to run open-source models like Llama, Mistral, and CodeLlama on your own hardware.

**Benefits**:
- 🔒 **Privacy**: All data stays on your machine
- 💰 **Cost**: No API fees after initial setup
- ⚡ **Latency**: No network overhead (hardware-dependent)
- 🔧 **Control**: Full control over model selection and configuration

**Trade-offs**:
- Requires local compute resources (GPU recommended)
- Slower than cloud APIs on typical hardware
- Smaller context windows than latest cloud models
- Model downloads require disk space (2-50GB per model)

## Requirements

### System Requirements

- **Operating System**: macOS, Linux, or Windows with WSL2
- **RAM**: 8GB minimum, 16GB+ recommended
- **Disk Space**: 10GB+ free for models
- **GPU** (recommended): NVIDIA with CUDA support or Apple Silicon

### Software Requirements

1. **Ollama** (required)
   - Install from: https://ollama.ai
   - Version: 0.1.0 or later

2. **Python Package** (handled automatically by Namel3ss)
   ```bash
   pip install namel3ss[ollama]
   # or for all local model providers:
   pip install namel3ss[local-models]
   ```

## Installation & Setup

### Step 1: Install Ollama

#### macOS
```bash
brew install ollama
```

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Windows
Download and install from [ollama.ai](https://ollama.ai)

### Step 2: Start Ollama Server

```bash
ollama serve
```

The server starts on `http://localhost:11434` by default.

### Step 3: Pull a Model

```bash
# Popular models:
ollama pull llama3:8b          # Meta's Llama 3 (8B parameters)
ollama pull mistral:latest     # Mistral AI's Mistral (7B)
ollama pull codellama:13b      # Code-specialized Llama (13B)
ollama pull phi:latest         # Microsoft's Phi-2 (3B, fast)
```

**Tip**: Start with smaller models (3B-8B) for testing, then scale up based on your hardware.

### Step 4: Verify Installation

```bash
# Check available models
ollama list

# Test a model
ollama run llama3:8b "Hello, world!"
```

## Configuration

### Basic Configuration (.ai file)

```yaml
ai model chat_model {
    provider: ollama
    model: llama3:8b
    config: {
        temperature: 0.7
        max_tokens: 2048
    }
}
```

### Full Configuration with All Options

```yaml
ai model production_ollama {
    provider: ollama
    model: mistral:7b
    
    # Generation parameters
    config: {
        temperature: 0.8
        top_k: 40
        top_p: 0.9
        repeat_penalty: 1.1
        num_ctx: 4096              # Context window size
        max_tokens: 2048
        
        # Server configuration
        base_url: "http://localhost:11434"  # Override default
        auto_pull_model: true       # Auto-pull missing models
        auto_start_server: false    # Auto-start Ollama (opt-in)
        
        # Cache and performance
        model_cache_ttl: 60         # Cache model checks (seconds)
        health_check_interval: 30   # Min time between health checks
    }
    
    # Deployment configuration
    deployment_config: {
        num_gpu: 1                  # Number of GPUs to use
        num_thread: 8               # CPU threads
        keep_alive: "10m"           # Keep model loaded
    }
}
```

### Environment Variables

Override default settings with environment variables:

```bash
# Set custom Ollama server URL
export NAMEL3SS_OLLAMA_BASE_URL="http://192.168.1.100:11434"

# Then use in your .ai files - will use the env var
ai model remote_ollama {
    provider: ollama
    model: llama3:8b
}
```

**Priority** (highest to lowest):
1. Explicit `base_url` in config
2. `NAMEL3SS_OLLAMA_BASE_URL` environment variable
3. Constructed from `host` and `port` in config
4. Default: `http://localhost:11434`

### Python API Configuration

```python
from namel3ss.providers.local.ollama import OllamaProvider

provider = OllamaProvider(
    name="my_ollama",
    model="llama3:8b",
    config={
        'base_url': 'http://localhost:11434',
        'temperature': 0.7,
        'num_ctx': 4096,
        'auto_pull_model': True,
        'auto_start_server': False,
    }
)
```

## Model Management

### Auto-Pull Behavior

**Enabled** (default):
```yaml
config: {
    auto_pull_model: true
}
```
- Namel3ss automatically pulls missing models
- First use may take several minutes
- Progress logged to console

**Disabled**:
```yaml
config: {
    auto_pull_model: false
}
```
- Fails fast with helpful error if model missing
- Recommended for production (pre-pull models)

### Manual Model Management

```bash
# List local models
ollama list

# Pull a model
ollama pull llama3:8b

# Remove a model
ollama rm llama3:8b

# Show model info
ollama show llama3:8b
```

### Programmatic Model Management

```python
# List available models
models = await provider.list_models()
for model in models:
    print(f"{model['name']} - {model['size']} bytes")

# Pull a specific model
await provider.pull_model("mistral:latest")

# Delete a model
await provider.delete_model("old-model:tag")
```

## Performance & Limitations

### Performance Characteristics

| Aspect | Local (Ollama) | Cloud APIs |
|--------|---------------|------------|
| **Privacy** | ✅ Complete | ⚠️ Data sent to provider |
| **Cost** | ✅ Free after setup | 💰 Pay per token |
| **Latency** | ⚡ No network (hardware-dependent) | 🌐 Network + queue time |
| **Throughput** | 🐌 5-50 tokens/sec (typical) | ⚡ 100+ tokens/sec |
| **Context Window** | 📏 2K-32K tokens | 📏 Up to 200K+ tokens |
| **Model Selection** | 🎯 Limited to downloaded | 🌍 All provider models |

### Hardware Impact

**CPU Only** (8 cores):
- 3B models: ~5-15 tokens/sec
- 7B models: ~2-5 tokens/sec
- 13B+ models: <1 token/sec (not recommended)

**GPU** (NVIDIA RTX 3090, 24GB VRAM):
- 3B models: ~50-100 tokens/sec
- 7B models: ~30-50 tokens/sec
- 13B models: ~15-25 tokens/sec
- 30B models: ~5-10 tokens/sec

**Apple Silicon** (M1/M2 Max):
- 3B models: ~40-80 tokens/sec
- 7B models: ~20-40 tokens/sec
- 13B models: ~10-20 tokens/sec

### Context Window Limits

Most Ollama models have smaller context windows than latest cloud models:

| Model | Default Context | Max Context |
|-------|----------------|-------------|
| llama3:8b | 8K tokens | 32K tokens |
| mistral:7b | 8K tokens | 32K tokens |
| codellama:13b | 16K tokens | 100K tokens |
| phi:latest | 2K tokens | 4K tokens |

**Cloud comparison**: GPT-4 Turbo supports 128K tokens, Claude 3 supports 200K tokens.

### Best Practices

1. **Start Small**: Test with 3B-7B models before scaling up
2. **GPU Acceleration**: Use GPU for 10x+ speedup
3. **Context Management**: Keep prompts under model's context limit
4. **Batch Operations**: Process multiple items in parallel when possible
5. **Model Selection**: Match model size to your hardware capabilities

## Error Handling

### Common Errors & Solutions

#### Error: "Could not reach Ollama server"

**Cause**: Ollama not running or wrong URL

**Solution**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Or check your NAMEL3SS_OLLAMA_BASE_URL
echo $NAMEL3SS_OLLAMA_BASE_URL
```

#### Error: "Ollama model 'llama3:8b' is not available locally"

**Cause**: Model not pulled

**Solution**:
```bash
# Option 1: Pull manually
ollama pull llama3:8b

# Option 2: Enable auto-pull in config
config: {
    auto_pull_model: true
}
```

#### Error: "Input is too large for model's context window"

**Cause**: Prompt exceeds model's context limit

**Solution**:
```yaml
# Option 1: Increase context window (if model supports it)
config: {
    num_ctx: 8192  # or higher
}

# Option 2: Reduce prompt length
# Option 3: Use a model with larger context (e.g., codellama:13b)
```

#### Error: "Ollama executable not found"

**Cause**: Ollama not installed or not in PATH

**Solution**:
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from ollama.ai

# Verify installation
which ollama
ollama --version
```

### Error Message Format

All Ollama errors include:
- **Clear description** of what went wrong
- **Context**: Model name, server URL
- **Actionable suggestion**: Specific command to fix

Example:
```
OllamaError: Ollama model 'mistral:latest' is not available locally.
Model: mistral:latest
Server: http://localhost:11434

Suggestion: Run: ollama pull mistral:latest
Or enable auto_pull in your configuration.
```

## Editor Integration

### VSCode Integration (Coming Soon)

The Ollama provider exposes APIs for editor integrations:

```python
from namel3ss.providers.local.ollama import OllamaEditorTools

# Check server status
status = await OllamaEditorTools.check_status()
if status['reachable']:
    print(f"Ollama running with {status['models_count']} models")

# Get models for autocomplete
models = await OllamaEditorTools.list_available_models()
# Returns: [{'name': 'llama3:8b', 'size': 4661229568, ...}, ...]

# Validate model name
validation = OllamaEditorTools.validate_model_name("llama3:8b")
if not validation['is_valid']:
    print(validation['message'])

# Get default recommendation
default = OllamaEditorTools.get_default_model()  # "llama3:8b"

# Check capabilities
caps = OllamaEditorTools.get_supported_capabilities()
# Returns: {'chat': True, 'streaming': True, ...}
```

### Features for Editor Developers

- **Model autocomplete**: List available models for dropdowns
- **Validation**: Real-time model name validation
- **Status indicator**: Show Ollama server status in status bar
- **Quick actions**: Pull/remove models from command palette
- **Diagnostics**: Helpful error messages in problems panel

## Advanced Usage

### Multiple Ollama Instances

Run multiple Ollama servers on different ports:

```bash
# Terminal 1: Production models
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Terminal 2: Experimental models
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

```yaml
# production.ai
ai model production {
    provider: ollama
    model: llama3:70b
    config: {
        base_url: "http://localhost:11434"
    }
}

# experimental.ai
ai model experimental {
    provider: ollama
    model: mistral:latest
    config: {
        base_url: "http://localhost:11435"
    }
}
```

### GPU Configuration

```bash
# Specify GPU layers (offload to GPU)
export OLLAMA_NUM_GPU=35  # Number of layers to run on GPU

# Run on specific GPU
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
```

```yaml
deployment_config: {
    num_gpu: 1      # Use 1 GPU
    num_thread: 8   # CPU threads for remaining layers
}
```

### Streaming with Token Metrics

```python
async for chunk in provider.stream(messages):
    print(chunk.output_text, end='', flush=True)
    
    if chunk.metadata.get('done'):
        tokens = chunk.metadata.get('tokens', {})
        print(f"\n\nTotal tokens: {tokens.get('total', 0)}")
        print(f"Throughput: {chunk.metadata.get('throughput', {}).get('tokens_per_second', 0):.1f} tok/s")
```

### Health Monitoring

```python
# Get comprehensive health status
health = await provider.health_check(force=True)

print(f"Status: {health['status']}")
print(f"Model available: {health['model_available']}")
print(f"Response time: {health['server_health']['response_time_ms']:.0f}ms")
print(f"Available models: {len(health['available_models'])}")
```

### Metrics & Observability

Ollama provider automatically records:
- `ollama.request.duration` - Request latency
- `ollama.tokens.total` - Total tokens per request
- `ollama.tokens.prompt` - Prompt tokens
- `ollama.tokens.completion` - Completion tokens
- `ollama.throughput.tokens_per_second` - Generation speed
- `ollama.model.check_duration` - Model availability check time
- `ollama.model.pull_duration` - Model download time
- `ollama.stream.chunks` - Number of streaming chunks
- `ollama.health.check_duration` - Health check latency

Access via Namel3ss metrics system:
```python
from namel3ss.observability.metrics import register_metric_listener

def log_metrics(name, values, labels):
    if name.startswith('ollama.'):
        print(f"{name}: {values} {labels}")

register_metric_listener(log_metrics)
```

## Troubleshooting

### Model Downloads Failing

**Symptom**: "Timeout while pulling model" or download stalls

**Solutions**:
1. Check network connection
2. Increase timeout:
   ```python
   await provider.pull_model("large-model:tag", pull_timeout=1200)  # 20 min
   ```
3. Check disk space: `df -h`
4. Try smaller model first to test connectivity

### Slow Generation Speed

**Symptom**: <5 tokens/second

**Solutions**:
1. **Use GPU**: Ensure `CUDA_VISIBLE_DEVICES` is set
2. **Reduce model size**: Try 3B-7B models instead of 13B+
3. **Close other apps**: Free up RAM/VRAM
4. **Check CPU usage**: `top` or Task Manager
5. **Update Ollama**: `ollama --version` and upgrade if old

### Out of Memory Errors

**Symptom**: Ollama crashes or refuses to load model

**Solutions**:
1. **Reduce context window**:
   ```yaml
   config: {
       num_ctx: 2048  # Instead of 8192
   }
   ```
2. **Use smaller model**: 3B or 7B instead of 13B+
3. **Reduce concurrent requests**: Wait for completion before next
4. **Free system memory**: Close other applications

### Port Already in Use

**Symptom**: "Address already in use: 11434"

**Solutions**:
```bash
# Find process using port
lsof -i :11434  # macOS/Linux
netstat -ano | findstr :11434  # Windows

# Kill process or use different port
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

### Model Not Found After Pull

**Symptom**: Model pulled but still shows as unavailable

**Solutions**:
1. **Verify pull**: `ollama list`
2. **Check exact name**: Model names are case-sensitive
3. **Restart Ollama**: `pkill ollama && ollama serve`
4. **Clear cache**:
   ```python
   await provider._cache.clear()
   ```

## Additional Resources

- **Official Ollama**: https://ollama.ai
- **Model Library**: https://ollama.ai/library
- **GitHub**: https://github.com/ollama/ollama
- **Namel3ss Docs**: [LOCAL_MODEL_DEPLOYMENT.md](../LOCAL_MODEL_DEPLOYMENT.md)
- **API Reference**: [API_REFERENCE.md](../API_REFERENCE.md)

## Next Steps

- [vLLM Integration](./vllm.md) - Higher performance alternative
- [LocalAI Integration](./localai.md) - More format flexibility
- [Local Model Comparison](../LOCAL_MODEL_DEPLOYMENT.md#provider-comparison)
- [Production Deployment](../deployment/local-models.md)

---

**Need help?** Check the [Troubleshooting](#troubleshooting) section or open an issue on GitHub.
