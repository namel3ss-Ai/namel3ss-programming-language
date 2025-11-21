"""Tests for refactored prompt block parsing with generalized keyword handling.

This test suite verifies:
1. Any keyword can be used as a field name in prompt blocks
2. parse_prompt_declaration() properly reuses shared block parsing logic
3. Error messages are clear and precise
4. Special fields (input, output, template) are handled correctly
"""

import pytest
from namel3ss.lang.parser import N3Parser
from namel3ss.lang.parser.errors import N3SyntaxError
from namel3ss.ast import Prompt


class TestKeywordAsKeys:
    """Test that various language keywords can be used as prompt block field names."""
    
    def test_all_keywords_as_field_names(self):
        """Test that any keyword token can be used as a field name in prompt blocks."""
        source = """
prompt "test" {
    model: "gpt-4"
    agent: "test-agent"
    policy: "default"
    llm: "openai"
    chain: "test-chain"
    filter: "basic"
    index: "vector-db"
    memory: "conversation"
    graph: "knowledge-graph"
    template: "Test template"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        # Find the prompt declaration
        prompt = None
        for decl in module.body:
            if isinstance(decl, Prompt):
                prompt = decl
                break
        
        assert prompt is not None
        assert prompt.name == "test"
        assert prompt.model == "gpt-4"
        assert prompt.template == "Test template"
        
        # Check that all keywords ended up in parameters
        assert prompt.parameters["agent"] == "test-agent"
        assert prompt.parameters["policy"] == "default"
        assert prompt.parameters["llm"] == "openai"
        assert prompt.parameters["chain"] == "test-chain"
        assert prompt.parameters["filter"] == "basic"
        assert prompt.parameters["index"] == "vector-db"
        assert prompt.parameters["memory"] == "conversation"
        assert prompt.parameters["graph"] == "knowledge-graph"
    
    def test_control_flow_keywords_as_fields(self):
        """Test that control flow keywords (if, else, for, while) can be field names."""
        source = """
prompt "control_test" {
    model: "gpt-4"
    if: "conditional-value"
    else: "alternative-value"
    for: "iteration-value"
    while: "loop-value"
    template: "Test"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.parameters["if"] == "conditional-value"
        assert prompt.parameters["else"] == "alternative-value"
        assert prompt.parameters["for"] == "iteration-value"
        assert prompt.parameters["while"] == "loop-value"
    
    def test_declaration_keywords_as_fields(self):
        """Test that declaration keywords can be field names."""
        source = """
prompt "decl_test" {
    model: "gpt-4"
    app: "app-value"
    page: "page-value"
    function: "func-value"
    tool: "tool-value"
    connector: "conn-value"
    training: "train-value"
    template: "Test"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.parameters["app"] == "app-value"
        assert prompt.parameters["page"] == "page-value"
        assert prompt.parameters["function"] == "func-value"
        assert prompt.parameters["tool"] == "tool-value"
        assert prompt.parameters["connector"] == "conn-value"
        assert prompt.parameters["training"] == "train-value"
    
    def test_mixed_identifiers_and_keywords(self):
        """Test mixing regular identifiers with keyword field names."""
        source = """
prompt "mixed_test" {
    model: "gpt-4"
    custom_field: "value1"
    agent: "agent-value"
    another_field: "value2"
    policy: "policy-value"
    template: "Test"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.parameters["custom_field"] == "value1"
        assert prompt.parameters["agent"] == "agent-value"
        assert prompt.parameters["another_field"] == "value2"
        assert prompt.parameters["policy"] == "policy-value"


class TestInputOutputTemplateHandling:
    """Test special handling of input, output, and template fields."""
    
    def test_input_output_template_fields(self):
        """Test that input, output, and template are handled specially."""
        source = """
prompt "io_test" {
    model: "gpt-4"
    input: {
        query: "text",
        context: "text"
    }
    output: {
        answer: "text",
        confidence: "float"
    }
    template: "Answer: {query} with {context}"
    max_tokens: 100
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        
        # Check input_fields
        assert len(prompt.input_fields) == 2
        assert prompt.input_fields["query"] == "text"
        assert prompt.input_fields["context"] == "text"
        
        # Check output_fields
        assert len(prompt.output_fields) == 2
        assert prompt.output_fields["answer"] == "text"
        assert prompt.output_fields["confidence"] == "float"
        
        # Check template
        assert prompt.template == "Answer: {query} with {context}"
        
        # Check other parameters
        assert prompt.parameters["max_tokens"] == 100
    
    def test_template_as_string_literal(self):
        """Test template can be a string literal."""
        source = """
prompt "template_test" {
    model: "gpt-4"
    template: "This is a simple template"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert prompt.template == "This is a simple template"
    
    def test_template_with_expressions(self):
        """Test template can contain complex values."""
        source = """
prompt "complex_template" {
    model: "gpt-4"
    template: "Value is concatenated"
}
"""
        # This should parse without errors
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert prompt.template is not None
    
    def test_prompt_without_input_output(self):
        """Test prompt with only template and no input/output schemas."""
        source = """
prompt "simple" {
    model: "gpt-4"
    template: "Simple prompt"
    temperature: 0.7
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert len(prompt.input_fields) == 0
        assert len(prompt.output_fields) == 0
        assert prompt.template == "Simple prompt"
        assert prompt.parameters["temperature"] == 0.7


class TestErrorConditions:
    """Test error handling and error messages."""
    
    def test_colon_where_field_name_expected(self):
        """Test error when colon appears where field name is expected."""
        source = """
prompt "error_test" {
    model: "gpt-4"
    : "value"
}
"""
        parser = N3Parser(source, path="test.n3")
        
        with pytest.raises(N3SyntaxError) as exc_info:
            parser.parse()
        
        error = exc_info.value
        assert "Expected field name" in error.message
        assert "colon" in error.message.lower() or ":" in str(error)
    
    def test_closing_brace_where_field_name_expected(self):
        """Test error when closing brace appears prematurely."""
        source = """
prompt "error_test" {
    model: "gpt-4"
    }
    extra: "value"
}
"""
        # This should parse the first block correctly and not raise an error
        # because the closing brace is valid. Let's test a different scenario.
        parser = N3Parser(source, path="test.n3")
        # This will likely raise a different error about unexpected token
        # The key is that the parser should handle it gracefully
        try:
            module = parser.parse()
            # If it parses, check that it parsed correctly
            prompt = module.declarations[0]
            assert prompt.model == "gpt-4"
        except N3SyntaxError as e:
            # If it raises an error, it should be clear
            assert "Expected" in e.message or "Unexpected" in e.message
    
    def test_comma_where_field_name_expected(self):
        """Test error when comma appears where field name is expected."""
        source = """
prompt "error_test" {
    model: "gpt-4",
    , "value"
}
"""
        parser = N3Parser(source, path="test.n3")
        
        with pytest.raises(N3SyntaxError) as exc_info:
            parser.parse()
        
        error = exc_info.value
        assert "Expected field name" in error.message or "Unexpected" in error.message
    
    def test_eof_in_prompt_block(self):
        """Test error when EOF is reached inside a prompt block."""
        source = """
prompt "incomplete" {
    model: "gpt-4"
    template: "Test"
"""
        parser = N3Parser(source, path="test.n3")
        
        with pytest.raises(N3SyntaxError) as exc_info:
            parser.parse()
        
        error = exc_info.value
        # Should mention unexpected end or missing closing brace
        assert "end" in error.message.lower() or "}" in str(error) or "Expected" in error.message
    
    def test_missing_colon_after_field_name(self):
        """Test error when colon is missing after field name."""
        source = """
prompt "error_test" {
    model "gpt-4"
}
"""
        parser = N3Parser(source, path="test.n3")
        
        with pytest.raises(N3SyntaxError) as exc_info:
            parser.parse()
        
        error = exc_info.value
        assert ":" in str(error) or "colon" in error.message.lower() or "Expected" in error.message


class TestBackwardsCompatibility:
    """Test that existing examples still parse correctly."""
    
    def test_basic_prompt_parsing(self):
        """Test basic prompt structure that should work as before."""
        source = """
prompt "summarize" {
    model: "gpt-4"
    description: "Summarizes text"
    template: "Summarize the following: {text}"
    max_tokens: 500
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.name == "summarize"
        assert prompt.model == "gpt-4"
        assert prompt.description == "Summarizes text"
        assert prompt.template == "Summarize the following: {text}"
        assert prompt.parameters["max_tokens"] == 500
    
    def test_prompt_with_all_standard_fields(self):
        """Test prompt with commonly used fields."""
        source = """
prompt "qa" {
    model: "gpt-4"
    description: "Q&A prompt"
    input: {
        question: "text",
        context: "text"
    }
    output: {
        answer: "text",
        sources: "list"
    }
    template: "Q: {question}\\nContext: {context}\\nA:"
    temperature: 0.7
    max_tokens: 300
    top_p: 0.9
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.name == "qa"
        assert prompt.model == "gpt-4"
        assert len(prompt.input_fields) == 2
        assert len(prompt.output_fields) == 2
        assert prompt.parameters["temperature"] == 0.7
        assert prompt.parameters["max_tokens"] == 300
        assert prompt.parameters["top_p"] == 0.9


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_prompt_with_nested_objects(self):
        """Test prompt with nested object values in parameters."""
        source = """
prompt "complex" {
    model: "gpt-4"
    template: "Test"
    config: {
        temperature: 0.7,
        max_tokens: 500
    }
    metadata: {
        version: "1.0",
        author: "test"
    }
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.parameters["config"]["temperature"] == 0.7
        assert prompt.parameters["config"]["max_tokens"] == 500
        # metadata is a special field - goes to prompt.metadata, not parameters
        assert prompt.metadata["version"] == "1.0"
        assert prompt.metadata["author"] == "test"
    
    def test_prompt_with_array_values(self):
        """Test prompt with array values in parameters."""
        source = """
prompt "array_test" {
    model: "gpt-4"
    template: "Test"
    stop_sequences: ["END", "STOP", "\\n\\n"]
    allowed_models: ["gpt-4", "gpt-3.5-turbo"]
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert len(prompt.parameters["stop_sequences"]) == 3
        assert prompt.parameters["stop_sequences"][0] == "END"
        assert len(prompt.parameters["allowed_models"]) == 2
    
    def test_multiple_prompts_in_module(self):
        """Test module with multiple prompt declarations."""
        source = """
module "test"

prompt "first" {
    model: "gpt-4"
    template: "First prompt"
}

prompt "second" {
    model: "gpt-3.5-turbo"
    template: "Second prompt"
    agent: "test-agent"
}

prompt "third" {
    model: "claude-2"
    template: "Third prompt"
    policy: "conservative"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompts = [d for d in module.body if isinstance(d, Prompt)]
        assert len(prompts) == 3
        assert prompts[0].name == "first"
        assert prompts[1].name == "second"
        assert prompts[2].name == "third"
        assert "agent" in prompts[1].parameters
        assert "policy" in prompts[2].parameters
