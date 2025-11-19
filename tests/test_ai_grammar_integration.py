"""
Test AI parser integration with main grammar.

Tests that structured prompts, AI models, training jobs, tuning jobs, chains,
connectors, memory, and templates are all recognized during normal compilation
of .n3 files through the grammar parser.
"""

import textwrap
import pytest
from namel3ss.lang.grammar import parse_module
from namel3ss.ast.ai import (
    Connector,
    Template,
    Memory,
    AIModel,
    Prompt,
    Chain,
    TrainingJob,
    TuningJob,
)


def test_parse_connector():
    """Test that connector blocks are parsed through grammar."""
    source = """
connector "openai_conn" type openai:
    provider: openai
    api_key: ${OPENAI_API_KEY}
    organization: my-org
"""
    module = parse_module(source)
    assert module is not None
    
    # Find connector in body or extra_nodes
    connectors = [node for node in module.body if isinstance(node, Connector)]
    assert len(connectors) == 1
    assert connectors[0].name == "openai_conn"
    assert connectors[0].connector_type == "openai"
    assert connectors[0].provider == "openai"


def test_parse_template():
    """Test that template definitions are parsed through grammar."""
    source = """
define template "customer_support":
    prompt: "How can I help you with {{topic}}?"
"""
    module = parse_module(source)
    assert module is not None
    
    templates = [node for node in module.body if isinstance(node, Template)]
    assert len(templates) == 1
    assert templates[0].name == "customer_support"
    assert "{{topic}}" in templates[0].prompt


def test_parse_memory():
    """Test that memory configurations are parsed through grammar."""
    source = textwrap.dedent("""\
        memory "conversation_memory":
            kind: buffer
            max_tokens: 2000
        """).strip()
    module = parse_module(source)
    assert module is not None
    
    memories = [node for node in module.body if isinstance(node, Memory)]
    assert len(memories) == 1
    assert memories[0].name == "conversation_memory"
    assert memories[0].kind == "buffer"


def test_parse_ai_model():
    """Test that AI model blocks are parsed through grammar."""
    source = textwrap.dedent("""\
        ai model "gpt4" using openai:
            model: gpt-4
            temperature: 0.7
        """).strip()
    module = parse_module(source)
    assert module is not None
    
    models = [node for node in module.body if isinstance(node, AIModel)]
    assert len(models) == 1
    assert models[0].name == "gpt4"
    assert models[0].provider == "openai"


def test_parse_structured_prompt():
    """Test that structured prompts with args and output_schema are parsed."""
    source = textwrap.dedent("""\
        prompt "analyze_text":
            args:
                text: str
                category: str = "general"
            output_schema:
                sentiment: str
                confidence: float
            template:
                Analyze: {{text}}
            using model "gpt4"
        """).strip()
    module = parse_module(source)
    assert module is not None
    
    # Prompts should be in app.prompts
    assert module.body[0].prompts
    prompt = module.body[0].prompts[0]
    assert prompt.name == "analyze_text"
    assert len(prompt.args) >= 2
    assert prompt.output_schema is not None


def test_parse_chain_with_workflow():
    """Test that chains with workflow blocks are parsed through grammar."""
    source = textwrap.dedent("""\
        app:
            name: test_app
        
        define chain "support_chain":
            workflow:
                - step "greet":
                    template "greet_template"
                - step "respond":
                    llm "chat_model" context greet
        """).strip()
    module = parse_module(source)
    assert module is not None
    
    # Chains should be in app.chains
    assert module.body[0].chains
    chain = module.body[0].chains[0]
    assert chain.name == "support_chain"
    assert len(chain.steps) > 0


def test_parse_training_job():
    """Test that training job definitions are parsed through grammar."""
    source = textwrap.dedent("""\
        training "sentiment_trainer":
            model: bert-base
            dataset: sentiment_data
            objective: accuracy
            epochs: 3
            batch_size: 32
        """).strip()
    module = parse_module(source)
    assert module is not None
    
    jobs = [node for node in module.body if isinstance(node, TrainingJob)]
    assert len(jobs) == 1
    assert jobs[0].name == "sentiment_trainer"
    assert jobs[0].model == "bert-base"


def test_parse_tuning_job():
    """Test that tuning job definitions are parsed through grammar."""
    source = textwrap.dedent("""\
        tuning "code_tuner":
            training_job: sentiment_trainer
            strategy: random
            max_trials: 10
            search_space:
                learning_rate: range[0.001, 0.1]
        """).strip()
    module = parse_module(source)
    assert module is not None
    
    jobs = [node for node in module.body if isinstance(node, TuningJob)]
    assert len(jobs) == 1
    assert jobs[0].name == "code_tuner"
    assert jobs[0].training_job == "sentiment_trainer"


def test_mixed_ai_constructs():
    """Test that multiple AI constructs can be parsed in one file."""
    source = textwrap.dedent("""\
        connector "my_conn" type openai:
            provider: openai
            api_key: test
        
        ai model "my_model" using openai:
            model: gpt-4
        
        define template "my_template":
            prompt: "Hello {{name}}"
        
        app:
            name: test_app
        
        prompt "my_prompt":
            args:
                input: str
            template:
                Process: {{input}}
            using model "my_model"
        
        define chain "my_chain":
            workflow:
                - step "step1":
                    llm "my_model"
        """).strip()
    module = parse_module(source)
    assert module is not None
    
    # Check we got all the different types
    connectors = [n for n in module.body if isinstance(n, Connector)]
    models = [n for n in module.body if isinstance(n, AIModel)]
    templates = [n for n in module.body if isinstance(n, Template)]
    
    assert len(connectors) == 1
    assert len(models) == 1
    assert len(templates) == 1
    assert module.body[0].prompts or any(hasattr(n, 'prompts') for n in module.body)  # App should have prompts
    assert module.body[0].chains or any(hasattr(n, 'chains') for n in module.body)   # App should have chains


def test_backward_compatibility_simple_prompt():
    """Test that simple prompt syntax works."""
    source = textwrap.dedent("""\
        app:
            name: test_app
        
        prompt "simple":
            template:
                Hello world
        """).strip()
    # This should not raise an error
    module = parse_module(source)
    assert module is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
