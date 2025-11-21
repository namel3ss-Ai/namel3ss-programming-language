"""Tests for modernized prompt parsing with args, output_schema, parameters, metadata, effects.

This test module verifies that the unified config parsing pattern correctly handles
all modern Prompt fields introduced in the AST refactoring, including:

- args: List[PromptArgument] for typed arguments
- output_schema: OutputSchema for structured outputs
- parameters: Dict for model parameters (temperature, max_tokens, etc.)
- metadata: Dict for versioning, tags, ownership
- effects: Set for effect tracking

Also tests validation logic for:
- Legacy vs modern field conflicts (input+args, output+output_schema)
- Model/llm aliasing behavior
- Name field handling
"""

import pytest
from namel3ss.lang.parser import N3Parser
from namel3ss.lang.parser.errors import N3SyntaxError
from namel3ss.ast import Prompt


class TestModernPromptFields:
    """Test modern Prompt field support."""
    
    def test_prompt_with_args(self):
        """Test prompt with typed arguments instead of legacy input."""
        source = """
prompt "extract_entities" {
    model: "gpt-4"
    template: "Extract entities from: {{text}}"
    args: [
        {name: "text", type: "string", required: true}
    ]
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.name == "extract_entities"
        assert prompt.args == [{"name": "text", "type": "string", "required": True}]
        assert prompt.input_fields == []  # Legacy field empty
    
    def test_prompt_with_output_schema(self):
        """Test prompt with structured output schema."""
        source = """
prompt "structured_extract" {
    model: "gpt-4"
    template: "Extract structured data"
    output_schema: {
        people: ["string"],
        organizations: ["string"]
    }
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.output_schema is not None
        assert prompt.output_schema == {
            "people": ["string"],
            "organizations": ["string"]
        }
        assert prompt.output_fields == []  # Legacy field empty
    
    def test_prompt_with_parameters(self):
        """Test prompt with explicit parameters field."""
        source = """
prompt "creative_writer" {
    model: "gpt-4"
    template: "Write a story"
    parameters: {
        temperature: 0.9,
        max_tokens: 1000,
        top_p: 0.95
    }
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        # parameters field should contain the explicit dict
        assert "temperature" in prompt.parameters
        assert prompt.parameters["temperature"] == 0.9
        assert prompt.parameters["max_tokens"] == 1000
        assert prompt.parameters["top_p"] == 0.95
    
    def test_prompt_with_metadata(self):
        """Test prompt with explicit metadata field."""
        source = """
prompt "versioned_prompt" {
    model: "gpt-4"
    template: "Test template"
    metadata: {
        version: "2.1.0",
        author: "team@example.com",
        tags: ["production", "critical"]
    }
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.metadata["version"] == "2.1.0"
        assert prompt.metadata["author"] == "team@example.com"
        assert "production" in prompt.metadata["tags"]
    
    def test_prompt_with_effects(self):
        """Test prompt with effects field."""
        source = """
prompt "stateful_prompt" {
    model: "gpt-4"
    template: "Perform action"
    effects: ["io", "network", "state"]
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        # effects should be parsed as a list
        assert prompt.effects == ["io", "network", "state"]
    
    def test_prompt_with_all_modern_fields(self):
        """Test prompt with all modern fields together."""
        source = """
prompt "comprehensive" {
    model: "gpt-4"
    template: "Complex template with {{input}}"
    description: "A fully-featured modern prompt"
    args: [
        {name: "input", type: "string", required: true}
    ]
    output_schema: {
        result: "string",
        confidence: "float"
    }
    parameters: {
        temperature: 0.7,
        max_tokens: 500
    }
    metadata: {
        version: "1.0.0",
        team: "ai-research"
    }
    effects: ["llm_call"]
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.name == "comprehensive"
        assert prompt.model == "gpt-4"
        assert prompt.description == "A fully-featured modern prompt"
        assert len(prompt.args) == 1
        assert prompt.output_schema["result"] == "string"
        assert prompt.parameters["temperature"] == 0.7
        assert prompt.metadata["version"] == "1.0.0"
        assert "llm_call" in prompt.effects


class TestLegacyVsModernPromptFields:
    """Test that legacy and modern fields don't conflict."""
    
    def test_legacy_input_alone(self):
        """Test that legacy input field works."""
        source = """
prompt "legacy_input" {
    model: "gpt-4"
    template: "Process {{user_input}}"
    input {
        user_input: string
    }
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert len(prompt.input_fields) > 0
        assert prompt.args == []  # Modern field empty
    
    def test_modern_args_alone(self):
        """Test that modern args field works."""
        source = """
prompt "modern_args" {
    model: "gpt-4"
    template: "Process {{user_input}}"
    args: [
        {name: "user_input", type: "string"}
    ]
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert len(prompt.args) > 0
        assert prompt.input_fields == []  # Legacy field empty
    
    def test_legacy_output_alone(self):
        """Test that legacy output field works."""
        source = """
prompt "legacy_output" {
    model: "gpt-4"
    template: "Generate result"
    output {
        result: string
    }
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert len(prompt.output_fields) > 0
        assert prompt.output_schema is None  # Modern field empty
    
    def test_modern_output_schema_alone(self):
        """Test that modern output_schema field works."""
        source = """
prompt "modern_schema" {
    model: "gpt-4"
    template: "Generate result"
    output_schema: {
        result: "string"
    }
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.output_schema is not None
        assert prompt.output_fields == []  # Legacy field empty


class TestPromptModelAliasing:
    """Test model/llm aliasing behavior."""
    
    def test_llm_as_alias_for_model(self):
        """Test that 'llm' can be used as alias for 'model'."""
        source = """
prompt "alias_test" {
    llm: "gpt-4"
    template: "Test"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.model == "gpt-4"  # llm -> model aliasing
    
    def test_model_and_llm_both_present(self):
        """Test that both model and llm can coexist (llm goes to parameters)."""
        source = """
prompt "both_fields" {
    model: "gpt-4"
    llm: "claude"
    template: "Test"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.model == "gpt-4"  # Explicit model field
        assert prompt.parameters["llm"] == "claude"  # llm goes to parameters
    
    def test_model_only(self):
        """Test that model field works alone."""
        source = """
prompt "model_only" {
    model: "gpt-4"
    template: "Test"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.model == "gpt-4"
        assert "llm" not in prompt.parameters


class TestPromptNameHandling:
    """Test that 'name' field inside block is handled correctly."""
    
    def test_name_inside_block_moved_to_metadata(self):
        """Test that name inside block doesn't override declared name."""
        source = """
prompt "canonical_name" {
    model: "gpt-4"
    template: "Test"
    name: "internal_name"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.name == "canonical_name"  # From declaration
        assert prompt.metadata.get("internal_name") == "internal_name"  # Moved to metadata


class TestParametersAndMetadataRouting:
    """Test that unknown fields are correctly routed to parameters."""
    
    def test_unknown_fields_go_to_parameters(self):
        """Test that unknown fields go to parameters dict."""
        source = """
prompt "custom_fields" {
    model: "gpt-4"
    template: "Test"
    custom_field_1: "value1"
    custom_field_2: 42
    custom_field_3: [1, 2, 3]
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.parameters["custom_field_1"] == "value1"
        assert prompt.parameters["custom_field_2"] == 42
        assert prompt.parameters["custom_field_3"] == [1, 2, 3]
    
    def test_model_params_go_to_parameters(self):
        """Test that model parameters go to parameters dict."""
        source = """
prompt "with_params" {
    model: "gpt-4"
    template: "Test"
    temperature: 0.8
    max_tokens: 1000
    top_p: 0.95
    frequency_penalty: 0.5
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.parameters["temperature"] == 0.8
        assert prompt.parameters["max_tokens"] == 1000
        assert prompt.parameters["top_p"] == 0.95
        assert prompt.parameters["frequency_penalty"] == 0.5


class TestBackwardsCompatibility:
    """Ensure backwards compatibility with existing prompt syntax."""
    
    def test_minimal_legacy_prompt(self):
        """Test minimal legacy prompt still works."""
        source = """
prompt "minimal" {
    model: "gpt-4"
    template: "Hello, world!"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.name == "minimal"
        assert prompt.model == "gpt-4"
        assert prompt.template == "Hello, world!"
    
    def test_legacy_full_prompt(self):
        """Test full legacy prompt with all old features."""
        source = """
prompt "legacy_full" {
    model: "gpt-4"
    template: "Process {{input}} and output {{output}}"
    input {
        input: string
    }
    output {
        output: string
    }
    temperature: 0.7
    max_tokens: 500
    description: "A legacy prompt"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        prompt = module.body[0]
        assert isinstance(prompt, Prompt)
        assert prompt.name == "legacy_full"
        assert prompt.model == "gpt-4"
        assert len(prompt.input_fields) > 0
        assert len(prompt.output_fields) > 0
        assert prompt.parameters["temperature"] == 0.7
        assert prompt.parameters["max_tokens"] == 500
        assert prompt.description == "A legacy prompt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
