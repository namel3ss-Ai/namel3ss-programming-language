"""Integration tests for prompt block runtime generation."""

import os
import tempfile

from namel3ss.parser import Parser
from namel3ss.codegen.backend import generate_backend


def test_prompt_block_compilation():
    """Test that prompt blocks compile and generate proper runtime code."""
    source = """
app "TestApp".

prompt summarize:
    template: "Summarize the following text in {max_length} words: {text}"

prompt classify:
    template: "Classify the sentiment as positive, negative, or neutral: {text}"
"""
    
    module = Parser(source).parse()
    app = module.body[0]
    
    # Verify prompts were parsed
    assert len(app.prompts) == 2
    assert app.prompts[0].name == "summarize"
    assert "Summarize" in app.prompts[0].template
    assert "{max_length}" in app.prompts[0].template
    assert "{text}" in app.prompts[0].template
    
    assert app.prompts[1].name == "classify"
    
    # Compile to backend
    output_dir = "/tmp/test_prompt_compilation"
    generate_backend(app, output_dir)
    
    # Verify runtime was generated
    runtime_path = os.path.join(output_dir, "generated", "runtime.py")
    assert os.path.exists(runtime_path)
    
    # Read generated code
    with open(runtime_path, 'r') as f:
        runtime_code = f.read()
    
    # Verify prompt initialization code was generated
    assert "AI_PROMPTS" in runtime_code
    assert "_PROMPT_INSTANCES" in runtime_code
    assert "_initialize_prompt_instances" in runtime_code
    assert "from namel3ss.prompts import create_prompt" in runtime_code
    assert "summarize" in runtime_code
    assert "classify" in runtime_code


def test_prompt_registry_structure():
    """Test that prompt configurations are properly included in generated code."""
    source = """
app "TestApp".

prompt generate_report:
    template: "Generate a {report_type} report about {topic}. Include: {sections}"
"""
    
    module = Parser(source).parse()
    app = module.body[0]
    
    output_dir = "/tmp/test_prompt_registry"
    generate_backend(app, output_dir)
    
    runtime_path = os.path.join(output_dir, "generated", "runtime.py")
    with open(runtime_path, 'r') as f:
        runtime_code = f.read()
    
    # Verify all prompt properties are in generated code
    assert "generate_report" in runtime_code
    assert "Generate a" in runtime_code


def test_prompt_with_llm():
    """Test prompt blocks alongside LLM blocks."""
    source = """
app "TestApp".

llm my_gpt:
    provider: openai
    model: gpt-4

prompt my_prompt:
    template: "Answer this question: {question}"
    model: my_gpt
"""
    
    module = Parser(source).parse()
    app = module.body[0]
    
    # Verify both were parsed
    assert len(app.llms) == 1
    assert len(app.prompts) == 1
    assert app.prompts[0].model == "my_gpt"
    
    # Compile
    output_dir = "/tmp/test_prompt_with_llm"
    generate_backend(app, output_dir)
    
    runtime_path = os.path.join(output_dir, "generated", "runtime.py")
    with open(runtime_path, 'r') as f:
        runtime_code = f.read()
    
    # Verify both LLM and prompt initialization
    assert "_LLM_INSTANCES" in runtime_code
    assert "_PROMPT_INSTANCES" in runtime_code
    assert "my_gpt" in runtime_code
    assert "my_prompt" in runtime_code


def test_multiple_prompts():
    """Test multiple prompt blocks."""
    source = """
app "TestApp".

prompt prompt1:
    template: "First: {input}"

prompt prompt2:
    template: "Second: {data}"

prompt prompt3:
    template: "Third: {value}"
"""
    
    module = Parser(source).parse()
    app = module.body[0]
    
    # Verify all prompts were parsed
    assert len(app.prompts) == 3
    
    names = {prompt.name for prompt in app.prompts}
    assert names == {"prompt1", "prompt2", "prompt3"}
    
    # Compile
    output_dir = "/tmp/test_multiple_prompts"
    generate_backend(app, output_dir)
    
    runtime_path = os.path.join(output_dir, "generated", "runtime.py")
    with open(runtime_path, 'r') as f:
        runtime_code = f.read()
    
    assert "_PROMPT_INSTANCES" in runtime_code
    assert "prompt1" in runtime_code
    assert "prompt2" in runtime_code
    assert "prompt3" in runtime_code


def test_prompt_initialization_error_handling():
    """Test that prompt initialization handles errors gracefully."""
    source = """
app "TestApp".

prompt valid_prompt:
    template: "Test: {var}"
"""
    
    module = Parser(source).parse()
    app = module.body[0]
    
    output_dir = "/tmp/test_prompt_errors"
    generate_backend(app, output_dir)
    
    runtime_path = os.path.join(output_dir, "generated", "runtime.py")
    with open(runtime_path, 'r') as f:
        runtime_code = f.read()
    
    # Verify error handling is present
    assert "try:" in runtime_code
    assert "except Exception as e:" in runtime_code
    assert "logger.warning" in runtime_code
    assert "Failed to initialize prompt" in runtime_code


if __name__ == "__main__":
    test_prompt_block_compilation()
    test_prompt_registry_structure()
    test_prompt_with_llm()
    test_multiple_prompts()
    test_prompt_initialization_error_handling()
    print("All prompt integration tests passed!")
