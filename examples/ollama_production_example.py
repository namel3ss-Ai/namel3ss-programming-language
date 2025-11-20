"""
Production Example: Using Ollama Provider for Local LLM Inference

This example demonstrates production-ready usage of the Ollama provider
for running local LLMs (Llama 3, Mistral, Mixtral, CodeLlama, etc.)
"""

import asyncio
from namel3ss.ml.providers import OllamaProvider, StreamConfig


async def check_ollama_connection():
    """Verify Ollama is running and accessible."""
    async with OllamaProvider(model="llama3") as provider:
        try:
            models = await provider.list_models()
            print(f"✅ Connected to Ollama. Available models: {len(models)}")
            for model in models[:5]:  # Show first 5
                print(f"   - {model.get('name')} ({model.get('size', 0) / 1e9:.2f} GB)")
            return True
        except Exception as e:
            print(f"❌ Ollama not accessible: {e}")
            print("   Make sure Ollama is running: ollama serve")
            return False


async def pull_model_if_needed(model_name: str):
    """Download model if not already available."""
    async with OllamaProvider(model=model_name) as provider:
        try:
            models = await provider.list_models()
            model_exists = any(model_name in m.get('name', '') for m in models)
            
            if not model_exists:
                print(f"📥 Downloading {model_name}...")
                async for progress in provider.pull_model(model_name):
                    status = progress.get('status', '')
                    if 'total' in progress and 'completed' in progress:
                        pct = (progress['completed'] / progress['total']) * 100
                        print(f"   {status}: {pct:.1f}%")
                    else:
                        print(f"   {status}")
                print(f"✅ Model {model_name} downloaded")
            else:
                print(f"✅ Model {model_name} already available")
        except Exception as e:
            print(f"❌ Failed to pull model: {e}")


async def simple_generation():
    """Example: Simple text generation."""
    print("\n" + "="*60)
    print("Example 1: Simple Generation")
    print("="*60)
    
    async with OllamaProvider(
        model="llama3",
        temperature=0.7,
        max_tokens=200
    ) as provider:
        prompt = "Explain quantum computing in one sentence."
        
        print(f"Prompt: {prompt}")
        print("Generating...\n")
        
        response = await provider.agenerate(prompt)
        
        print(f"Response: {response.content}")
        print(f"\nTokens: {response.usage.get('total_tokens', 0)}")
        print(f"Model: {response.model}")


async def streaming_generation():
    """Example: Streaming text generation with real-time display."""
    print("\n" + "="*60)
    print("Example 2: Streaming Generation")
    print("="*60)
    
    async with OllamaProvider(
        model="llama3",
        temperature=0.8,
        max_tokens=300
    ) as provider:
        prompt = "Write a haiku about coding at night."
        
        print(f"Prompt: {prompt}")
        print("Streaming response:\n")
        
        full_response = ""
        async for chunk in provider.stream_generate(prompt):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
            
            if chunk.finish_reason:
                print(f"\n\n[Finished: {chunk.finish_reason}]")
                if chunk.usage:
                    print(f"Tokens: {chunk.usage.get('total_tokens', 0)}")


async def system_prompt_example():
    """Example: Using system prompts for role-playing."""
    print("\n" + "="*60)
    print("Example 3: System Prompt (Role-Playing)")
    print("="*60)
    
    async with OllamaProvider(
        model="llama3",
        temperature=0.7,
        max_tokens=200
    ) as provider:
        system = "You are a helpful Python programming assistant. Answer concisely with code examples."
        prompt = "How do I read a JSON file in Python?"
        
        print(f"System: {system}")
        print(f"User: {prompt}\n")
        
        response = await provider.agenerate(
            prompt,
            system=system
        )
        
        print(f"Assistant: {response.content}")


async def concurrent_requests():
    """Example: Concurrent requests with automatic concurrency control."""
    print("\n" + "="*60)
    print("Example 4: Concurrent Requests")
    print("="*60)
    
    async with OllamaProvider(
        model="llama3",
        max_concurrent=3,  # Limit concurrent requests
        temperature=0.7
    ) as provider:
        prompts = [
            "What is Python?",
            "What is JavaScript?",
            "What is Rust?",
            "What is Go?",
            "What is TypeScript?",
        ]
        
        print(f"Processing {len(prompts)} prompts concurrently (max 3 at a time)...\n")
        
        # All requests run concurrently with semaphore limiting to 3
        tasks = [
            provider.agenerate(prompt, max_tokens=50)
            for prompt in prompts
        ]
        
        results = await asyncio.gather(*tasks)
        
        for prompt, result in zip(prompts, results):
            print(f"Q: {prompt}")
            print(f"A: {result.content[:100]}...")
            print()


async def streaming_with_backpressure():
    """Example: Streaming with timeout and chunk limits."""
    print("\n" + "="*60)
    print("Example 5: Streaming with Backpressure Control")
    print("="*60)
    
    async with OllamaProvider(
        model="llama3",
        temperature=0.9
    ) as provider:
        prompt = "Write a long story about a programmer's journey."
        
        # Configure streaming with limits
        config = StreamConfig(
            stream_timeout=30.0,  # Max 30 seconds total
            chunk_timeout=5.0,     # Max 5 seconds between chunks
            max_chunks=50          # Stop after 50 chunks
        )
        
        print(f"Prompt: {prompt}")
        print(f"Config: max_chunks={config.max_chunks}, chunk_timeout={config.chunk_timeout}s")
        print("Streaming response:\n")
        
        chunk_count = 0
        async for chunk in provider.stream_generate(prompt, stream_config=config):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                chunk_count += 1
            
            if chunk.finish_reason:
                print(f"\n\n[Finished: {chunk.finish_reason}, chunks: {chunk_count}]")


async def error_handling_example():
    """Example: Proper error handling."""
    print("\n" + "="*60)
    print("Example 6: Error Handling")
    print("="*60)
    
    # Wrong URL - will trigger retry logic
    async with OllamaProvider(
        model="llama3",
        base_url="http://localhost:9999"  # Wrong port
    ) as provider:
        try:
            print("Attempting to connect to wrong port...")
            response = await provider.agenerate("Hello")
            print(f"Response: {response.content}")
        except Exception as e:
            print(f"✅ Caught error (as expected): {type(e).__name__}")
            print(f"   Message: {str(e)[:100]}...")


async def code_generation_example():
    """Example: Code generation with CodeLlama."""
    print("\n" + "="*60)
    print("Example 7: Code Generation (CodeLlama)")
    print("="*60)
    
    # Use CodeLlama for better code generation
    async with OllamaProvider(
        model="codellama",
        temperature=0.3,  # Lower temperature for more deterministic code
        max_tokens=500
    ) as provider:
        system = "You are an expert programmer. Write clean, production-ready code."
        prompt = "Write a Python function to calculate Fibonacci numbers with memoization."
        
        print(f"System: {system}")
        print(f"User: {prompt}\n")
        print("Streaming code:\n")
        
        async for chunk in provider.stream_generate(prompt, system=system):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            if chunk.finish_reason:
                print(f"\n\n[Finished: {chunk.finish_reason}]")


async def main():
    """Run all production examples."""
    print("🚀 Ollama Provider - Production Examples\n")
    
    # Check Ollama connection
    if not await check_ollama_connection():
        print("\n⚠️  Please start Ollama first:")
        print("   $ ollama serve")
        return
    
    # Ensure model is available
    await pull_model_if_needed("llama3")
    
    # Run examples
    try:
        await simple_generation()
        await streaming_generation()
        await system_prompt_example()
        await concurrent_requests()
        await streaming_with_backpressure()
        await error_handling_example()
        
        # Optionally run code generation if codellama is available
        # await pull_model_if_needed("codellama")
        # await code_generation_example()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
    
    print("\n✅ All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
