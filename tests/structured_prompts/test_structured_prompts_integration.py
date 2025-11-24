"""Integration tests for structured prompts with mock LLMs."""

import pytest
from unittest.mock import Mock, MagicMock
from namel3ss.prompts.executor import execute_structured_prompt_sync
from namel3ss.ast import (
    Prompt, PromptArgument, OutputSchema, OutputField, OutputFieldType
)


class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.last_messages = None
    
    def supports_structured_output(self):
        return True
    
    def generate_structured(self, prompt, output_schema):
        response = self.responses[self.call_count] if self.call_count < len(self.responses) else {}
        self.call_count += 1
        self.last_messages = [{"role": "user", "content": prompt}]
        return response
    
    def generate_structured_chat(self, messages, output_schema):
        response = self.responses[self.call_count] if self.call_count < len(self.responses) else {}
        self.call_count += 1
        self.last_messages = messages
        return response


class TestExecutorIntegration:
    """Test execute_structured_prompt with mocked LLMs."""
    
    def test_successful_execution(self):
        """Test successful prompt execution with validation."""
        prompt = Prompt(
            name="classify",
            model="gpt4",
            template="Classify: {text}",
            args=[
                PromptArgument(name="text", arg_type="string"),
            ],
            output_schema=OutputSchema(fields=[
                OutputField(
                    name="category",
                    field_type=OutputFieldType(base_type="enum", enum_values=["spam", "ham"]),
                    required=True
                ),
            ])
        )
        
        llm = MockLLM(responses=[{"category": "spam"}])
        
        result = execute_structured_prompt_sync(
            prompt_def=prompt,
            llm=llm,
            args={"text": "Buy now!"},
        )
        
        assert result.output == {"category": "spam"}
        assert result.latency_ms > 0
        assert llm.call_count == 1
    
    def test_retry_on_validation_error(self):
        """Test that executor retries on validation errors."""
        prompt = Prompt(
            name="classify",
            model="gpt4",
            template="Classify: {text}",
            args=[
                PromptArgument(name="text", arg_type="string"),
            ],
            output_schema=OutputSchema(fields=[
                OutputField(
                    name="category",
                    field_type=OutputFieldType(base_type="enum", enum_values=["spam", "ham"]),
                    required=True
                ),
            ])
        )
        
        # First response invalid, second response valid
        llm = MockLLM(responses=[
            {"category": "invalid"},  # Will fail validation
            {"category": "spam"},     # Will pass validation
        ])
        
        result = execute_structured_prompt_sync(
            prompt_def=prompt,
            llm=llm,
            args={"text": "Test"},
            retry_on_validation_error=True,
            max_retries=2,
        )
        
        assert result.output == {"category": "spam"}
        assert llm.call_count == 2
        assert result.metadata.get("validation_attempts") == 2
    
    def test_max_retries_exceeded(self):
        """Test that executor stops after max retries."""
        prompt = Prompt(
            name="classify",
            model="gpt4",
            template="Classify: {text}",
            args=[
                PromptArgument(name="text", arg_type="string"),
            ],
            output_schema=OutputSchema(fields=[
                OutputField(
                    name="category",
                    field_type=OutputFieldType(base_type="enum", enum_values=["spam", "ham"]),
                    required=True
                ),
            ])
        )
        
        # All responses invalid
        llm = MockLLM(responses=[
            {"category": "invalid"},
            {"category": "invalid"},
            {"category": "invalid"},
        ])
        
        from namel3ss.prompts.validator import ValidationError
        with pytest.raises(ValidationError):
            execute_structured_prompt_sync(
                prompt_def=prompt,
                llm=llm,
                args={"text": "Test"},
                retry_on_validation_error=True,
                max_retries=2,
            )
        
        assert llm.call_count == 3  # Initial + 2 retries
    
    def test_no_retry_when_disabled(self):
        """Test that executor doesn't retry when retry is disabled."""
        prompt = Prompt(
            name="classify",
            model="gpt4",
            template="Classify: {text}",
            args=[
                PromptArgument(name="text", arg_type="string"),
            ],
            output_schema=OutputSchema(fields=[
                OutputField(
                    name="category",
                    field_type=OutputFieldType(base_type="enum", enum_values=["spam", "ham"]),
                    required=True
                ),
            ])
        )
        
        llm = MockLLM(responses=[{"category": "invalid"}])
        
        from namel3ss.prompts.validator import ValidationError
        with pytest.raises(ValidationError):
            execute_structured_prompt_sync(
                prompt_def=prompt,
                llm=llm,
                args={"text": "Test"},
                retry_on_validation_error=False,
            )
        
        assert llm.call_count == 1  # No retries


class TestComplexIntegration:
    """Test complex scenarios with multiple fields."""
    
    def test_complex_output_schema(self):
        """Test execution with complex nested output schema."""
        prompt = Prompt(
            name="analyze",
            model="gpt4",
            template="Analyze: {text}",
            args=[
                PromptArgument(name="text", arg_type="string"),
            ],
            output_schema=OutputSchema(fields=[
                OutputField(
                    name="sentiment",
                    field_type=OutputFieldType(base_type="enum", enum_values=["positive", "negative", "neutral"]),
                    required=True
                ),
                OutputField(
                    name="confidence",
                    field_type=OutputFieldType(base_type="float"),
                    required=True
                ),
                OutputField(
                    name="entities",
                    field_type=OutputFieldType(
                        base_type="list",
                        element_type=OutputFieldType(
                            base_type="object",
                            nested_fields=[
                                OutputField(name="name", field_type=OutputFieldType(base_type="string")),
                                OutputField(name="type", field_type=OutputFieldType(base_type="string")),
                            ]
                        )
                    ),
                    required=True
                ),
            ])
        )
        
        llm = MockLLM(responses=[{
            "sentiment": "positive",
            "confidence": 0.95,
            "entities": [
                {"name": "Alice", "type": "person"},
                {"name": "Paris", "type": "location"}
            ]
        }])
        
        result = execute_structured_prompt_sync(
            prompt_def=prompt,
            llm=llm,
            args={"text": "Alice loves Paris"},
        )
        
        assert result.output["sentiment"] == "positive"
        assert result.output["confidence"] == 0.95
        assert len(result.output["entities"]) == 2
        assert result.output["entities"][0]["name"] == "Alice"
    
    def test_multiple_args_with_defaults(self):
        """Test execution with multiple args including defaults."""
        prompt = Prompt(
            name="summarize",
            model="gpt4",
            template="Summarize in {max_words} words with {style} style: {text}",
            args=[
                PromptArgument(name="text", arg_type="string", required=True),
                PromptArgument(name="max_words", arg_type="int", required=False, default=50),
                PromptArgument(name="style", arg_type="string", required=False, default="concise"),
            ],
            output_schema=OutputSchema(fields=[
                OutputField(name="summary", field_type=OutputFieldType(base_type="string"), required=True),
            ])
        )
        
        llm = MockLLM(responses=[{"summary": "Short summary"}])
        
        # Test with only required arg
        result = execute_structured_prompt_sync(
            prompt_def=prompt,
            llm=llm,
            args={"text": "Long text here"},
        )
        
        assert result.output["summary"] == "Short summary"
        assert "50 words" in llm.last_messages[0]["content"]
        assert "concise style" in llm.last_messages[0]["content"]
    
    def test_prompt_without_output_schema(self):
        """Test that prompts without output_schema work (backward compat)."""
        prompt = Prompt(
            name="simple",
            model="gpt4",
            template="Generate: {topic}",
            args=[
                PromptArgument(name="topic", arg_type="string"),
            ],
            output_schema=None  # No schema
        )
        
        llm = MockLLM(responses=["Plain text response"])
        
        # Should work but return raw response (no validation)
        # Note: This tests backward compatibility
        # In practice, prompts without schemas may use different execution path


class TestErrorHandling:
    """Test error handling in executor."""
    
    def test_invalid_arg_type(self):
        """Test that invalid argument types are caught."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="{count}",
            args=[
                PromptArgument(name="count", arg_type="int"),
            ],
        )
        
        llm = MockLLM()
        
        with pytest.raises(ValueError, match="Cannot coerce"):
            execute_structured_prompt_sync(
                prompt_def=prompt,
                llm=llm,
                args={"count": "not_a_number"},
            )
    
    def test_missing_required_arg(self):
        """Test that missing required args are caught."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="{required}",
            args=[
                PromptArgument(name="required", arg_type="string", required=True),
            ],
        )
        
        llm = MockLLM()
        
        with pytest.raises(ValueError, match="Missing required argument"):
            execute_structured_prompt_sync(
                prompt_def=prompt,
                llm=llm,
                args={},
            )
    
    def test_llm_returns_invalid_json(self):
        """Test handling of invalid JSON from LLM."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            output_schema=OutputSchema(fields=[
                OutputField(name="field", field_type=OutputFieldType(base_type="string")),
            ])
        )
        
        llm = MockLLM(responses=["not json"])
        
        # Executor should handle this gracefully
        # Behavior depends on implementation - may retry or raise
