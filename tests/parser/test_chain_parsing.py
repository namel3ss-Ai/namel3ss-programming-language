"""
Comprehensive tests for chain/workflow parsing.

Tests the full chain parsing implementation including:
- Step blocks with all fields (kind, target, options, evaluation, stop_on_error)
- Multiple step kinds (prompt, llm, tool, python, react, rag, memory_read, memory_write)
- Control flow (if/elif/else, for, while)
- Legacy steps list format
- Error handling for invalid chains
"""

import pytest
from namel3ss.lang.parser import parse_module
from namel3ss.ast import Chain, ChainStep, StepEvaluationConfig, WorkflowIfBlock, WorkflowForBlock, WorkflowWhileBlock
from namel3ss.lang.parser.errors import N3SyntaxError


class TestChainStepParsing:
    """Test parsing of individual chain steps."""
    
    def test_parse_simple_step(self):
        """Test parsing a simple step with kind and target."""
        source = '''
chain "test_chain" {
    step "validate" {
        kind: "python"
        target: "validators.check_input"
    }
}
'''
        module = parse_module(source)
        assert len(module.body) > 0
        
        # Find the chain (it should be attached to app, but also in body)
        chain = None
        for item in module.body:
            if isinstance(item, Chain):
                chain = item
                break
        
        assert chain is not None, "Chain not found in module"
        assert chain.name == "test_chain"
        assert len(chain.steps) == 1
        
        step = chain.steps[0]
        assert isinstance(step, ChainStep)
        assert step.kind == "python"
        assert step.target == "validators.check_input"
        assert step.name == "validate"
        assert step.stop_on_error is True  # default
        assert step.options == {}
    
    def test_parse_step_with_options(self):
        """Test parsing a step with options/arguments."""
        source = '''
chain "process_chain" {
    step "call_llm" {
        kind: "llm"
        target: "gpt-4"
        options: {
            temperature: 0.7
            max_tokens: 500
            system_message: "You are a helpful assistant"
        }
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        assert chain is not None
        
        step = chain.steps[0]
        assert step.kind == "llm"
        assert step.target == "gpt-4"
        assert step.options["temperature"] == 0.7
        assert step.options["max_tokens"] == 500
        assert step.options["system_message"] == "You are a helpful assistant"
    
    def test_parse_step_with_evaluation(self):
        """Test parsing a step with evaluation config."""
        source = '''
chain "safe_chain" {
    step "generate" {
        kind: "prompt"
        target: "content_generator"
        options: {
            text: "input text"
        }
        stop_on_error: false
        evaluation: {
            evaluators: ["toxicity_check", "relevance_score"]
            guardrail: "content_policy"
        }
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        assert chain is not None
        
        step = chain.steps[0]
        assert step.kind == "prompt"
        assert step.target == "content_generator"
        assert step.stop_on_error is False
        assert step.evaluation is not None
        assert isinstance(step.evaluation, StepEvaluationConfig)
        assert "toxicity_check" in step.evaluation.evaluators
        assert "relevance_score" in step.evaluation.evaluators
        assert step.evaluation.guardrail == "content_policy"
    
    def test_parse_multiple_steps(self):
        """Test parsing a chain with multiple steps."""
        source = '''
chain "multi_step" {
    step "fetch_data" {
        kind: "python"
        target: "data.fetch"
    }
    
    step "process" {
        kind: "prompt"
        target: "analyzer"
    }
    
    step "save" {
        kind: "python"
        target: "data.save"
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        assert chain is not None
        assert len(chain.steps) == 3
        
        assert chain.steps[0].name == "fetch_data"
        assert chain.steps[0].kind == "python"
        
        assert chain.steps[1].name == "process"
        assert chain.steps[1].kind == "prompt"
        
        assert chain.steps[2].name == "save"
        assert chain.steps[2].kind == "python"


class TestChainStepKinds:
    """Test parsing of different step kinds."""
    
    def test_prompt_step(self):
        """Test prompt step kind."""
        source = '''
chain "test" {
    step "classify" {
        kind: "prompt"
        target: "classifier_prompt"
        options: {
            text: "input data"
        }
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        step = chain.steps[0]
        assert step.kind == "prompt"
        assert step.target == "classifier_prompt"
    
    def test_llm_step(self):
        """Test llm step kind."""
        source = '''
chain "test" {
    step {
        kind: "llm"
        target: "gpt-4o-mini"
        options: {
            prompt: "Summarize this"
        }
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        step = chain.steps[0]
        assert step.kind == "llm"
        assert step.name is None  # unnamed step
    
    def test_tool_step(self):
        """Test tool step kind."""
        source = '''
chain "test" {
    step "search" {
        kind: "tool"
        target: "web_search"
        options: {
            query: "latest news"
        }
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        step = chain.steps[0]
        assert step.kind == "tool"
        assert step.target == "web_search"
    
    def test_python_step(self):
        """Test python step kind."""
        source = '''
chain "test" {
    step "transform" {
        kind: "python"
        target: "transformers.clean_data"
        options: {
            data: "raw_input"
        }
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        step = chain.steps[0]
        assert step.kind == "python"
        assert step.target == "transformers.clean_data"
    
    def test_rag_step(self):
        """Test rag step kind."""
        source = '''
chain "test" {
    step "retrieve" {
        kind: "rag"
        target: "doc_retrieval"
        options: {
            query: "search terms"
            top_k: 5
        }
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        step = chain.steps[0]
        assert step.kind == "rag"
        assert step.options["top_k"] == 5
    
    def test_memory_read_step(self):
        """Test memory_read step kind."""
        source = '''
chain "test" {
    step "load_history" {
        kind: "memory_read"
        target: "conversation_history"
        options: {
            assign_to: "context.history"
        }
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        step = chain.steps[0]
        assert step.kind == "memory_read"
        assert step.target == "conversation_history"
    
    def test_memory_write_step(self):
        """Test memory_write step kind."""
        source = '''
chain "test" {
    step "save_result" {
        kind: "memory_write"
        target: "user_data"
        options: {
            value: "processed result"
            mode: "append"
        }
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        step = chain.steps[0]
        assert step.kind == "memory_write"
        assert step.options["mode"] == "append"
    
    def test_chain_step(self):
        """Test chain step kind (sub-chain invocation)."""
        source = '''
chain "test" {
    step "run_subchain" {
        kind: "chain"
        target: "preprocessing_chain"
        options: {
            input: "raw_data"
        }
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        step = chain.steps[0]
        assert step.kind == "chain"
        assert step.target == "preprocessing_chain"
    
    def test_knowledge_query_step(self):
        """Test knowledge_query step kind."""
        source = '''
chain "test" {
    step "query_kb" {
        kind: "knowledge_query"
        target: "product_knowledge"
        options: {
            query: "product features"
        }
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        step = chain.steps[0]
        assert step.kind == "knowledge_query"


class TestChainControlFlow:
    """Test parsing of control flow in chains."""
    
    def test_if_block(self):
        """Test parsing if block in chain."""
        source = '''
chain "conditional" {
    step "check" {
        kind: "prompt"
        target: "validator"
    }
    
    if context.steps.check.result.valid == true:
        step "process" {
            kind: "python"
            target: "processor"
        }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        assert chain is not None
        assert len(chain.steps) == 2
        
        # First is the check step
        assert isinstance(chain.steps[0], ChainStep)
        
        # Second is the if block
        if_block = chain.steps[1]
        assert isinstance(if_block, WorkflowIfBlock)
        assert if_block.condition is not None
        assert len(if_block.then_steps) == 1
        assert isinstance(if_block.then_steps[0], ChainStep)
    
    def test_if_else_block(self):
        """Test parsing if/else block in chain."""
        source = '''
chain "branching" {
    if context.score > 0.8:
        step "high_confidence" {
            kind: "python"
            target: "handler.high"
        }
    else:
        step "low_confidence" {
            kind: "python"
            target: "handler.low"
        }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        if_block = chain.steps[0]
        
        assert isinstance(if_block, WorkflowIfBlock)
        assert len(if_block.then_steps) == 1
        assert len(if_block.else_steps) == 1
        
        assert if_block.then_steps[0].target == "handler.high"
        assert if_block.else_steps[0].target == "handler.low"
    
    def test_for_loop(self):
        """Test parsing for loop in chain."""
        source = '''
chain "batch_process" {
    for item in dataset "customers":
        step "process_customer" {
            kind: "prompt"
            target: "customer_analyzer"
            options: {
                data: item
            }
        }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        for_block = chain.steps[0]
        
        assert isinstance(for_block, WorkflowForBlock)
        assert for_block.loop_var == "item"
        assert for_block.source_kind == "dataset"
        assert for_block.source_name == "customers"
        assert len(for_block.body) == 1
        assert isinstance(for_block.body[0], ChainStep)
    
    def test_while_loop(self):
        """Test parsing while loop in chain."""
        source = '''
chain "retry_logic" {
    while context.retry_count < 3 and not context.success:
        step "attempt" {
            kind: "python"
            target: "api.call"
        }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        while_block = chain.steps[0]
        
        assert isinstance(while_block, WorkflowWhileBlock)
        assert while_block.condition is not None
        assert len(while_block.body) == 1


class TestChainConfig:
    """Test parsing of chain configuration."""
    
    def test_chain_with_input_key(self):
        """Test chain with custom input_key."""
        source = '''
chain "custom_input" {
    input_key: "user_data"
    
    step "process" {
        kind: "python"
        target: "processor"
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        assert chain.input_key == "user_data"
    
    def test_chain_with_metadata(self):
        """Test chain with metadata."""
        source = '''
chain "documented" {
    metadata: {
        version: "1.0"
        owner: "team_ai"
        description: "Main processing chain"
    }
    
    step "work" {
        kind: "prompt"
        target: "worker"
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        assert chain.metadata["version"] == "1.0"
        assert chain.metadata["owner"] == "team_ai"
        assert chain.metadata["description"] == "Main processing chain"
    
    def test_chain_with_policy(self):
        """Test chain with policy reference."""
        source = '''
chain "safe" {
    policy_name: "content_policy"
    
    step "generate" {
        kind: "llm"
        target: "gpt-4"
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        assert chain.policy_name == "content_policy"


class TestLegacyChainFormat:
    """Test parsing of legacy chain format with steps list."""
    
    def test_legacy_steps_list(self):
        """Test legacy format with steps as list of strings."""
        source = '''
chain "legacy" {
    steps: ["input", "rag:retriever", "prompt:qa_prompt", "llm:gpt4"]
    input_key: "query"
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        assert chain is not None
        assert chain.input_key == "query"
        assert len(chain.steps) == 4
        
        # Legacy format should be converted to ChainStep objects
        assert all(isinstance(step, ChainStep) for step in chain.steps)
        
        # Check parsed steps
        assert chain.steps[0].kind == "input"
        assert chain.steps[1].kind == "rag"
        assert chain.steps[1].target == "retriever"
        assert chain.steps[2].kind == "prompt"
        assert chain.steps[2].target == "qa_prompt"
        assert chain.steps[3].kind == "llm"
        assert chain.steps[3].target == "gpt4"


class TestChainErrorHandling:
    """Test error handling for invalid chain definitions."""
    
    def test_step_missing_kind(self):
        """Test error when step is missing 'kind' field."""
        source = '''
chain "invalid" {
    step "broken" {
        target: "something"
    }
}
'''
        with pytest.raises(N3SyntaxError) as exc_info:
            parse_module(source)
        assert "missing required field 'kind'" in str(exc_info.value).lower()
    
    def test_step_missing_target(self):
        """Test error when step is missing 'target' field."""
        source = '''
chain "invalid" {
    step "broken" {
        kind: "prompt"
    }
}
'''
        with pytest.raises(N3SyntaxError) as exc_info:
            parse_module(source)
        assert "missing required field 'target'" in str(exc_info.value).lower()
    
    def test_step_invalid_kind_type(self):
        """Test error when kind is not a string."""
        source = '''
chain "invalid" {
    step "broken" {
        kind: 123
        target: "something"
    }
}
'''
        with pytest.raises(N3SyntaxError) as exc_info:
            parse_module(source)
        assert "kind" in str(exc_info.value).lower()
        assert "must be a string" in str(exc_info.value).lower()
    
    def test_step_invalid_options_type(self):
        """Test error when options is not a dict."""
        source = '''
chain "invalid" {
    step "broken" {
        kind: "prompt"
        target: "something"
        options: "not a dict"
    }
}
'''
        with pytest.raises(N3SyntaxError) as exc_info:
            parse_module(source)
        assert "options" in str(exc_info.value).lower()
        assert "object" in str(exc_info.value).lower() or "dict" in str(exc_info.value).lower()


class TestComplexChainExamples:
    """Test parsing of complex, realistic chain examples."""
    
    def test_rag_qa_chain(self):
        """Test realistic RAG question-answering chain."""
        source = '''
chain "rag_qa" {
    input_key: "question"
    
    step "retrieve" {
        kind: "rag"
        target: "doc_retrieval"
        options: {
            query: input.question
            top_k: 5
        }
    }
    
    step "answer" {
        kind: "prompt"
        target: "qa_prompt"
        options: {
            question: input.question
            context: steps.retrieve.results
        }
        evaluation: {
            evaluators: ["relevance_check"]
        }
    }
    
    step "format" {
        kind: "python"
        target: "formatter.to_json"
        options: {
            data: steps.answer.output
        }
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        assert chain is not None
        assert chain.name == "rag_qa"
        assert chain.input_key == "question"
        assert len(chain.steps) == 3
        
        # Verify step sequence
        assert chain.steps[0].kind == "rag"
        assert chain.steps[1].kind == "prompt"
        assert chain.steps[1].evaluation is not None
        assert chain.steps[2].kind == "python"
    
    def test_memory_chat_chain(self):
        """Test realistic memory-based chat chain."""
        source = '''
chain "chat" {
    step "load_history" {
        kind: "memory_read"
        target: "conversation_history"
        options: {
            assign_to: "context.history"
        }
    }
    
    step "generate_response" {
        kind: "prompt"
        target: "chat_prompt"
        options: {
            message: input.message
            history: context.history
        }
    }
    
    step "save_message" {
        kind: "memory_write"
        target: "conversation_history"
        options: {
            value: {
                role: "user"
                content: input.message
            }
            mode: "append"
        }
    }
    
    step "save_response" {
        kind: "memory_write"
        target: "conversation_history"
        options: {
            value: {
                role: "assistant"
                content: steps.generate_response.output
            }
            mode: "append"
        }
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        assert chain is not None
        assert len(chain.steps) == 4
        
        # Verify memory operations
        assert chain.steps[0].kind == "memory_read"
        assert chain.steps[2].kind == "memory_write"
        assert chain.steps[3].kind == "memory_write"
    
    def test_conditional_chain_with_escalation(self):
        """Test chain with conditional escalation logic."""
        source = '''
chain "support_triage" {
    step "classify" {
        kind: "prompt"
        target: "ticket_classifier"
        options: {
            ticket: input.text
        }
    }
    
    if steps.classify.result.urgency == "high":
        step "escalate" {
            kind: "tool"
            target: "notify_oncall"
            options: {
                ticket_id: input.id
                priority: "urgent"
            }
        }
    else:
        step "auto_reply" {
            kind: "prompt"
            target: "auto_responder"
            options: {
                ticket: input.text
                category: steps.classify.result.category
            }
        }
    
    step "log" {
        kind: "python"
        target: "logger.log_ticket"
        options: {
            data: steps.classify.result
        }
    }
}
'''
        module = parse_module(source)
        chain = next((item for item in module.body if isinstance(item, Chain)), None)
        assert chain is not None
        assert len(chain.steps) == 3  # classify, if-block, log
        
        # Verify if-else structure
        if_block = chain.steps[1]
        assert isinstance(if_block, WorkflowIfBlock)
        assert len(if_block.then_steps) == 1
        assert len(if_block.else_steps) == 1
        assert if_block.then_steps[0].target == "notify_oncall"
        assert if_block.else_steps[0].target == "auto_responder"
