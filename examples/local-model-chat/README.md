# Local Model Chat Application

A complete example demonstrating local model deployment with Namel3ss using vLLM, Ollama, and LocalAI.

## Features

- Multi-provider local model support
- Real-time chat interface
- Model switching capabilities  
- Performance monitoring
- Deployment management

## Quick Start

1. Install dependencies:
   ```bash
   pip install namel3ss[local-models]
   ```

2. Deploy a local model:
   ```bash
   # Using Ollama (easiest to get started)
   namel3ss deploy local start chat_model
   
   # Or using vLLM (for production)
   namel3ss deploy local start production_model
   ```

3. Build and run the application:
   ```bash
   namel3ss build app.ai
   cd build && python -m uvicorn main:app --reload
   ```

4. Open http://localhost:8000 in your browser

## Model Configurations

The application demonstrates three different local model deployments:

- **Ollama**: Easy setup with automatic model pulling
- **vLLM**: High-performance inference for production
- **LocalAI**: Flexible multi-format model support

See `app.ai` for complete model definitions and deployment configurations.