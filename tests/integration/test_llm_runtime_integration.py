"""Integration tests for LLM runtime with DSL compilation."""

import pytest
from namel3ss.parser import Parser
from namel3ss.codegen.backend.core.generator import generate_backend


def test_llm_block_compilation():
    """Test that LLM blocks are parsed and compiled correctly."""
    source = """
app "TestApp".

llm my_gpt4:
    provider: openai
    model: gpt-4
    temperature: 0.7
    max_tokens: 1024

llm my_claude:
    provider: anthropic
    model: claude-3-opus-20240229
    temperature: 0.5
    max_tokens: 2048
"""
    
    # Parse the source
    module = Parser(source).parse()
    app = module.body[0]
    
    # Verify LLM definitions were parsed
    assert len(app.llms) == 2
    
    # Check first LLM
    gpt4 = app.llms[0]
    assert gpt4.name == "my_gpt4"
    assert gpt4.provider == "openai"
    assert gpt4.model == "gpt-4"
    assert gpt4.temperature == 0.7
    assert gpt4.max_tokens == 1024
    
    # Check second LLM
    claude = app.llms[1]
    assert claude.name == "my_claude"
    assert claude.provider == "anthropic"
    assert claude.model == "claude-3-opus-20240229"
    assert claude.temperature == 0.5
    assert claude.max_tokens == 2048
    
    # Compile backend
    output_dir = "/tmp/test_llm_compilation"
    generate_backend(app, output_dir)
    
    # Check that runtime.py was generated
    import os
    runtime_path = os.path.join(output_dir, "generated", "runtime.py")
    assert os.path.exists(runtime_path)
    
    # Read generated runtime and check for LLM registry
    with open(runtime_path, 'r') as f:
        runtime_code = f.read()
    
    # Verify LLM_REGISTRY is present
    assert "LLM_REGISTRY" in runtime_code
    assert "my_gpt4" in runtime_code
    assert "my_claude" in runtime_code
    
    # Verify LLM initialization code is present
    assert "_LLM_INSTANCES" in runtime_code
    assert "_initialize_llm_instances" in runtime_code
    assert "from namel3ss.llm import create_llm" in runtime_code


def test_llm_registry_structure():
    """Test that LLM registry has correct structure in generated code."""
    source = """
app "TestApp".

llm test_llm:
    provider: openai
    model: gpt-3.5-turbo
    temperature: 0.8
    max_tokens: 512
    top_p: 0.9
    frequency_penalty: 0.5
    presence_penalty: 0.3
"""
    
    module = Parser(source).parse()
    app = module.body[0]
    
    # Compile
    output_dir = "/tmp/test_llm_registry"
    generate_backend(app, output_dir)
    
    # Load and execute runtime to check registry
    import os
    import sys
    runtime_path = os.path.join(output_dir, "generated", "runtime.py")
    
    # This is a basic smoke test - we can't fully execute without mocking
    # But we can verify the structure
    with open(runtime_path, 'r') as f:
        runtime_code = f.read()
    
    # Check that all LLM parameters are included
    assert '"provider": "openai"' in runtime_code or "'provider': 'openai'" in runtime_code
    assert '"model": "gpt-3.5-turbo"' in runtime_code or "'model': 'gpt-3.5-turbo'" in runtime_code
    assert "0.8" in runtime_code  # temperature
    assert "512" in runtime_code  # max_tokens


def test_llm_with_chain():
    """Test LLM compilation with multiple configuration options."""
    source = """
app "TestApp".

llm my_llm:
    provider: openai
    model: gpt-4
    temperature: 0.7
    max_tokens: 1000
    top_p: 0.9

llm another_llm:
    provider: anthropic
    model: claude-3-sonnet-20240229
"""
    
    module = Parser(source).parse()
    app = module.body[0]
    
    # Verify LLMs were parsed
    assert len(app.llms) == 2
    assert app.llms[0].name == "my_llm"
    assert app.llms[1].name == "another_llm"
    
    # Compile
    output_dir = "/tmp/test_llm_chain"
    generate_backend(app, output_dir)
    
    # Verify runtime includes LLMs
    import os
    runtime_path = os.path.join(output_dir, "generated", "runtime.py")
    with open(runtime_path, 'r') as f:
        runtime_code = f.read()
    
    assert "_LLM_INSTANCES" in runtime_code
    assert "my_llm" in runtime_code
    assert "another_llm" in runtime_code
    assert "0.7" in runtime_code  # temperature
    assert "1000" in runtime_code  # max_tokens
    assert "0.9" in runtime_code  # top_p


def test_multiple_llm_providers():
    """Test multiple LLM providers in one app."""
    source = """
app "TestApp".

llm openai_llm:
    provider: openai
    model: gpt-4

llm anthropic_llm:
    provider: anthropic
    model: claude-3-opus-20240229

llm vertex_llm:
    provider: vertex
    model: gemini-pro

llm azure_llm:
    provider: azure_openai
    model: gpt-4

llm local_llm:
    provider: local
    model: llama2
"""
    
    module = Parser(source).parse()
    app = module.body[0]
    
    # Verify all 5 LLMs were parsed
    assert len(app.llms) == 5
    
    providers = {llm.provider for llm in app.llms}
    assert providers == {"openai", "anthropic", "vertex", "azure_openai", "local"}
    
    # Compile
    output_dir = "/tmp/test_multi_provider"
    generate_backend(app, output_dir)
    
    # Verify all providers in runtime
    import os
    runtime_path = os.path.join(output_dir, "generated", "runtime.py")
    with open(runtime_path, 'r') as f:
        runtime_code = f.read()
    
    for llm in app.llms:
        assert llm.name in runtime_code
        assert llm.provider in runtime_code


def test_llm_initialization_error_handling():
    """Test that LLM initialization handles errors gracefully."""
    source = """
app "TestApp".

llm incomplete_llm:
    provider: openai
    model: gpt-4
"""
    
    module = Parser(source).parse()
    app = module.body[0]
    
    # Compile
    output_dir = "/tmp/test_llm_errors"
    generate_backend(app, output_dir)
    
    # Check that error handling is present
    import os
    runtime_path = os.path.join(output_dir, "generated", "runtime.py")
    with open(runtime_path, 'r') as f:
        runtime_code = f.read()
    
    # Verify error handling in initialization
    assert "except Exception as exc:" in runtime_code
    assert "logger.error" in runtime_code or "logger.warning" in runtime_code
    assert "Failed to initialize LLM" in runtime_code or "Skipping LLM" in runtime_code
