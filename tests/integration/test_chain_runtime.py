"""
Integration tests for chain/workflow parsing and runtime generation.

Tests the complete flow:
1. Parse N3 source with chain definitions
2. Generate backend via generate_backend()
3. Verify runtime sees correct chain structure
4. Validate all step kinds, fields, and control flow are properly encoded
"""

import importlib
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from namel3ss.codegen.backend import generate_backend
from namel3ss.parser import Parser


# ============================================================================
# Test Utilities
# ============================================================================

def _generate_backend(app_source: str, tmp_path: Path) -> Path:
    """Generate backend from N3 source."""
    app = Parser(app_source).parse_app()
    backend_dir = tmp_path / "test_chain_backend"
    generate_backend(app, backend_dir)
    return backend_dir


def _load_runtime_module(package_name: str, backend_dir: Path):
    """Dynamically import generated runtime module."""
    # Ensure __init__.py exists
    init_py = backend_dir / "__init__.py"
    if not init_py.exists():
        init_py.write_text("", encoding="utf-8")
    
    # Load the package
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_py,
        submodule_search_locations=[str(backend_dir)],
    )
    assert spec and spec.loader, f"Failed to load {package_name}"
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    
    # Import the runtime module
    return importlib.import_module(f"{package_name}.generated.runtime")


def _cleanup_modules(package_name: str):
    """Remove backend modules from sys.modules."""
    to_remove = [key for key in sys.modules if key.startswith(package_name)]
    for key in to_remove:
        del sys.modules[key]


# ============================================================================
# Basic Chain Integration Tests
# ============================================================================

class TestBasicChainIntegration:
    """Test basic chain parsing and runtime generation."""
    
    def test_simple_chain_generates_runtime(self, tmp_path: Path):
        """Test that a simple chain generates correct runtime structure."""
        source = '''
app "TestApp"

chain "simple_workflow" {
    step "validate" {
        kind: "python"
        target: "validators.check"
    }
    
    step "process" {
        kind: "prompt"
        target: "processor"
    }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_simple_chain"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            
            # Verify AI_CHAINS registry exists
            assert hasattr(runtime, "AI_CHAINS"), "AI_CHAINS not found in runtime"
            chains = runtime.AI_CHAINS
            
            # Verify our chain is in the registry
            assert "simple_workflow" in chains, "Chain 'simple_workflow' not in registry"
            chain = chains["simple_workflow"]
            
            # Verify chain structure
            assert chain["name"] == "simple_workflow"
            assert "steps" in chain
            assert len(chain["steps"]) == 2
            
            # Verify first step
            step1 = chain["steps"][0]
            assert step1["type"] == "step"
            assert step1["kind"] == "python"
            assert step1["target"] == "validators.check"
            assert step1["name"] == "validate"
            assert step1["stop_on_error"] is True  # default
            
            # Verify second step
            step2 = chain["steps"][1]
            assert step2["type"] == "step"
            assert step2["kind"] == "prompt"
            assert step2["target"] == "processor"
            assert step2["name"] == "process"
            
        finally:
            _cleanup_modules(package_name)
    
    def test_chain_with_options_runtime(self, tmp_path: Path):
        """Test chain step with options generates correct runtime."""
        source = '''
app "TestApp"

chain "configured_chain" {
    step "llm_call" {
        kind: "llm"
        target: "gpt-4"
        options: {
            temperature: 0.7
            max_tokens: 500
            system_prompt: "You are a helpful assistant"
        }
    }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_options_chain"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            chains = runtime.AI_CHAINS
            
            assert "configured_chain" in chains
            chain = chains["configured_chain"]
            step = chain["steps"][0]
            
            # Verify options are encoded
            assert "options" in step
            options = step["options"]
            assert options["temperature"] == 0.7
            assert options["max_tokens"] == 500
            assert options["system_prompt"] == "You are a helpful assistant"
            
        finally:
            _cleanup_modules(package_name)
    
    def test_chain_with_evaluation_runtime(self, tmp_path: Path):
        """Test chain step with evaluation config generates correct runtime."""
        source = '''
app "TestApp"

chain "evaluated_chain" {
    step "guarded_step" {
        kind: "prompt"
        target: "user_input_handler"
        evaluation: {
            evaluators: ["toxicity_check", "relevance_score"]
            guardrail: "safety_policy"
        }
        stop_on_error: false
    }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_evaluation_chain"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            chains = runtime.AI_CHAINS
            
            assert "evaluated_chain" in chains
            chain = chains["evaluated_chain"]
            step = chain["steps"][0]
            
            # Verify evaluation config
            assert "evaluation" in step
            evaluation = step["evaluation"]
            assert "evaluators" in evaluation
            assert evaluation["evaluators"] == ["toxicity_check", "relevance_score"]
            assert evaluation["guardrail"] == "safety_policy"
            
            # Verify stop_on_error
            assert step["stop_on_error"] is False
            
        finally:
            _cleanup_modules(package_name)


# ============================================================================
# Step Kinds Integration Tests
# ============================================================================

class TestChainStepKindsIntegration:
    """Test all chain step kinds generate correct runtime."""
    
    def test_python_step_runtime(self, tmp_path: Path):
        """Test Python step kind in runtime."""
        source = '''
app "TestApp"

chain "python_chain" {
    step "compute" {
        kind: "python"
        target: "compute.calculate"
        options: { data: "input" }
    }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_python_step"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            step = runtime.AI_CHAINS["python_chain"]["steps"][0]
            
            assert step["kind"] == "python"
            assert step["target"] == "compute.calculate"
            
        finally:
            _cleanup_modules(package_name)
    
    def test_llm_step_runtime(self, tmp_path: Path):
        """Test LLM step kind in runtime."""
        source = '''
app "TestApp"

chain "llm_chain" {
    step "generate" {
        kind: "llm"
        target: "gpt-4"
        options: { "prompt": "Generate text" }
    }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_llm_step"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            step = runtime.AI_CHAINS["llm_chain"]["steps"][0]
            
            assert step["kind"] == "llm"
            assert step["target"] == "gpt-4"
            
        finally:
            _cleanup_modules(package_name)
    
    def test_rag_step_runtime(self, tmp_path: Path):
        """Test RAG step kind in runtime."""
        source = '''
app "TestApp"

chain "rag_chain" {
    step "retrieve" {
        kind: "rag"
        target: "doc_index"
        options: { "query": "search query", top_k: 5 }
    }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_rag_step"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            step = runtime.AI_CHAINS["rag_chain"]["steps"][0]
            
            assert step["kind"] == "rag"
            assert step["target"] == "doc_index"
            assert step["options"]["top_k"] == 5
            
        finally:
            _cleanup_modules(package_name)
    
    def test_memory_steps_runtime(self, tmp_path: Path):
        """Test memory read/write steps in runtime."""
        source = '''
app "TestApp"

chain "memory_chain" {
    step "read_context" {
        kind: "memory_read"
        target: "conversation_history"
        options: { limit: 10 }
    }
    
    step "save_response" {
        kind: "memory_write"
        target: "conversation_history"
        options: { content: "response data" }
    }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_memory_steps"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            chain = runtime.AI_CHAINS["memory_chain"]
            
            # Verify memory_read step
            read_step = chain["steps"][0]
            assert read_step["kind"] == "memory_read"
            assert read_step["target"] == "conversation_history"
            assert read_step["options"]["limit"] == 10
            
            # Verify memory_write step
            write_step = chain["steps"][1]
            assert write_step["kind"] == "memory_write"
            assert write_step["target"] == "conversation_history"
            
        finally:
            _cleanup_modules(package_name)
    
    def test_tool_step_runtime(self, tmp_path: Path):
        """Test tool step kind in runtime."""
        source = '''
app "TestApp"

chain "tool_chain" {
    step "api_call" {
        kind: "tool"
        target: "weather_api"
        options: { location: "Seattle" }
    }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_tool_step"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            step = runtime.AI_CHAINS["tool_chain"]["steps"][0]
            
            assert step["kind"] == "tool"
            assert step["target"] == "weather_api"
            
        finally:
            _cleanup_modules(package_name)
    
    def test_chain_invocation_step_runtime(self, tmp_path: Path):
        """Test sub-chain invocation step in runtime."""
        source = '''
app "TestApp"

chain "sub_chain" {
    step "sub_process" {
        kind: "python"
        target: "process.data"
    }
}

chain "main_chain" {
    step "invoke_sub" {
        kind: "chain"
        target: "sub_chain"
        options: { input: "data" }
    }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_chain_invocation"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            chains = runtime.AI_CHAINS
            
            # Verify both chains exist
            assert "sub_chain" in chains
            assert "main_chain" in chains
            
            # Verify main chain invokes sub chain
            step = chains["main_chain"]["steps"][0]
            assert step["kind"] == "chain"
            assert step["target"] == "sub_chain"
            
        finally:
            _cleanup_modules(package_name)


# ============================================================================
# Control Flow Integration Tests
# ============================================================================

class TestChainControlFlowIntegration:
    """Test chain control flow generates correct runtime."""
    
    def test_if_block_runtime(self, tmp_path: Path):
        """Test if block generates correct runtime structure."""
        source = '''
app "TestApp"

chain "conditional_chain" {
    step "check" {
        kind: "python"
        target: "validator.check"
    }
    
    if steps.check.success:
        step "on_success" {
            kind: "prompt"
            target: "success_handler"
        }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_if_block"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            chain = runtime.AI_CHAINS["conditional_chain"]
            
            assert len(chain["steps"]) == 2
            
            # Verify regular step
            assert chain["steps"][0]["type"] == "step"
            
            # Verify if block
            if_node = chain["steps"][1]
            assert if_node["type"] == "if"
            assert "condition" in if_node
            assert "then" in if_node
            assert len(if_node["then"]) == 1
            assert if_node["then"][0]["kind"] == "prompt"
            assert if_node["then"][0]["target"] == "success_handler"
            
        finally:
            _cleanup_modules(package_name)
    
    def test_if_else_runtime(self, tmp_path: Path):
        """Test if/else block generates correct runtime."""
        source = '''
app "TestApp"

chain "branching_chain" {
    step "classify" {
        kind: "prompt"
        target: "classifier"
    }
    
    if steps.classify.category == "urgent":
        step "urgent_handler" {
            kind: "tool"
            target: "pager"
        }
    else:
        step "normal_handler" {
            kind: "python"
            target: "queue.add"
        }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_if_else"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            chain = runtime.AI_CHAINS["branching_chain"]
            
            if_node = chain["steps"][1]
            assert if_node["type"] == "if"
            
            # Verify then branch
            assert len(if_node["then"]) == 1
            assert if_node["then"][0]["kind"] == "tool"
            
            # Verify else branch
            assert "else" in if_node
            assert len(if_node["else"]) == 1
            assert if_node["else"][0]["kind"] == "python"
            
        finally:
            _cleanup_modules(package_name)
    
    def test_for_loop_runtime(self, tmp_path: Path):
        """Test for loop generates correct runtime."""
        source = '''
app "TestApp"

dataset "items" from table items

chain "batch_chain" {
    for item in dataset "items":
        step "process_item" {
            kind: "python"
            target: "processor.handle"
            options: { data: "item" }
        }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_for_loop"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            chain = runtime.AI_CHAINS["batch_chain"]
            
            for_node = chain["steps"][0]
            assert for_node["type"] == "for"
            assert for_node["loop_var"] == "item"
            assert for_node["source_kind"] == "dataset"
            assert for_node["source_name"] == "items"
            
            # Verify body
            assert "body" in for_node
            assert len(for_node["body"]) == 1
            assert for_node["body"][0]["kind"] == "python"
            
        finally:
            _cleanup_modules(package_name)
    
    def test_while_loop_runtime(self, tmp_path: Path):
        """Test while loop generates correct runtime."""
        source = '''
app "TestApp"

chain "retry_chain" {
    step "attempt" {
        kind: "tool"
        target: "api"
    }
    
    while context.retry_count < 3:
        step "retry" {
            kind: "python"
            target: "retry_handler"
        }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_while_loop"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            chain = runtime.AI_CHAINS["retry_chain"]
            
            while_node = chain["steps"][1]
            assert while_node["type"] == "while"
            assert "condition" in while_node
            
            # Verify body
            assert "body" in while_node
            assert len(while_node["body"]) == 1
            assert while_node["body"][0]["kind"] == "python"
            
        finally:
            _cleanup_modules(package_name)


# ============================================================================
# Chain Configuration Integration Tests
# ============================================================================

class TestChainConfigIntegration:
    """Test chain configuration generates correct runtime."""
    
    def test_chain_with_input_key_runtime(self, tmp_path: Path):
        """Test chain with custom input_key in runtime."""
        source = '''
app "TestApp"

chain "input_chain" {
    input_key: "user_query"
    
    step "process" {
        kind: "prompt"
        target: "handler"
    }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_input_key"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            chain = runtime.AI_CHAINS["input_chain"]
            
            assert chain["input_key"] == "user_query"
            
        finally:
            _cleanup_modules(package_name)
    
    def test_chain_with_metadata_runtime(self, tmp_path: Path):
        """Test chain with metadata in runtime."""
        source = '''
app "TestApp"

chain "meta_chain" {
    metadata: {
        description: "Test chain"
        version: "1.0"
        tags: ["test", "demo"]
    }
    
    step "work" {
        kind: "python"
        target: "work.do"
    }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_metadata"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            chain = runtime.AI_CHAINS["meta_chain"]
            
            assert "metadata" in chain
            metadata = chain["metadata"]
            assert metadata["description"] == "Test chain"
            assert metadata["version"] == "1.0"
            assert metadata["tags"] == ["test", "demo"]
            
        finally:
            _cleanup_modules(package_name)


# ============================================================================
# Complex Chain Integration Tests
# ============================================================================

class TestComplexChainIntegration:
    """Test complex real-world chain scenarios."""
    
    def test_memory_chat_chain_runtime(self, tmp_path: Path):
        """Test complex memory-based chat chain."""
        source = '''
app "TestApp"

chain "chat_with_memory" {
    input_key: "user_message"
    
    step "load_history" {
        kind: "memory_read"
        target: "chat_history"
        options: { limit: 10 }
    }
    
    step "generate_response" {
        kind: "prompt"
        target: "chat_prompt"
        options: {
            message: "user_input"
            history: "step_history"
        }
    }
    
    step "save_exchange" {
        kind: "memory_write"
        target: "chat_history"
        options: { exchange: "conversation" }
    }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_memory_chat"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            chain = runtime.AI_CHAINS["chat_with_memory"]
            
            assert chain["input_key"] == "user_message"
            assert len(chain["steps"]) == 3
            
            # Verify memory operations
            assert chain["steps"][0]["kind"] == "memory_read"
            assert chain["steps"][0]["target"] == "chat_history"
            assert chain["steps"][2]["kind"] == "memory_write"
            assert chain["steps"][2]["target"] == "chat_history"
            
        finally:
            _cleanup_modules(package_name)
    
    def test_conditional_escalation_chain_runtime(self, tmp_path: Path):
        """Test chain with conditional escalation logic."""
        source = '''
app "TestApp"

chain "smart_routing" {
    step "analyze" {
        kind: "prompt"
        target: "analyzer"
    }
    
    if steps.analyze.confidence > 0.9:
        step "auto_respond" {
            kind: "prompt"
            target: "responder"
        }
    elif steps.analyze.confidence > 0.7:
        step "suggest_response" {
            kind: "prompt"
            target: "suggester"
        }
    else:
        step "escalate_human" {
            kind: "tool"
            target: "notification_service"
        }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_escalation"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            chain = runtime.AI_CHAINS["smart_routing"]
            
            # Verify step
            assert chain["steps"][0]["kind"] == "prompt"
            
            # Verify if/elif/else structure
            if_node = chain["steps"][1]
            assert if_node["type"] == "if"
            assert len(if_node["then"]) == 1
            assert if_node["then"][0]["name"] == "auto_respond"
            
            assert "elif" in if_node
            assert len(if_node["elif"]) == 1
            elif_branch = if_node["elif"][0]
            assert len(elif_branch["steps"]) == 1
            assert elif_branch["steps"][0]["name"] == "suggest_response"
            
            assert len(if_node["else"]) == 1
            assert if_node["else"][0]["name"] == "escalate_human"
            
        finally:
            _cleanup_modules(package_name)
    
    def test_nested_control_flow_runtime(self, tmp_path: Path):
        """Test chain with nested control flow."""
        source = '''
app "TestApp"

dataset "batches" from table batches

chain "nested_processing" {
    for batch in dataset "batches":
        step "validate_batch" {
            kind: "python"
            target: "validator"
        }
        
        if steps.validate_batch.valid:
            step "process_batch" {
                kind: "python"
                target: "processor"
            }
        else:
            step "log_error" {
                kind: "python"
                target: "logger"
            }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_nested_flow"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            chain = runtime.AI_CHAINS["nested_processing"]
            
            # Verify for loop
            for_node = chain["steps"][0]
            assert for_node["type"] == "for"
            assert len(for_node["body"]) == 2
            
            # Verify nested if/else inside for loop
            nested_if = for_node["body"][1]
            assert nested_if["type"] == "if"
            assert len(nested_if["then"]) == 1
            assert len(nested_if["else"]) == 1
            
        finally:
            _cleanup_modules(package_name)


# ============================================================================
# Multiple Chains Integration Test
# ============================================================================

class TestMultipleChainsIntegration:
    """Test multiple chains in same app."""
    
    def test_multiple_chains_in_runtime(self, tmp_path: Path):
        """Test that multiple chains are all registered in runtime."""
        source = '''
app "TestApp"

chain "chain_one" {
    step "step1" {
        kind: "python"
        target: "handler1"
    }
}

chain "chain_two" {
    step "step2" {
        kind: "prompt"
        target: "handler2"
    }
}

chain "chain_three" {
    step "step3" {
        kind: "llm"
        target: "gpt-4"
    }
}
'''
        backend_dir = _generate_backend(source, tmp_path)
        package_name = "test_multiple_chains"
        
        try:
            runtime = _load_runtime_module(package_name, backend_dir)
            chains = runtime.AI_CHAINS
            
            # Verify all chains are registered
            assert "chain_one" in chains
            assert "chain_two" in chains
            assert "chain_three" in chains
            
            # Verify each has correct structure
            assert chains["chain_one"]["steps"][0]["kind"] == "python"
            assert chains["chain_two"]["steps"][0]["kind"] == "prompt"
            assert chains["chain_three"]["steps"][0]["kind"] == "llm"
            
        finally:
            _cleanup_modules(package_name)
