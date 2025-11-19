"""Quick integration test for structured prompts."""

from namel3ss.ast import Prompt, PromptArgument, OutputSchema, OutputField, OutputFieldType
from namel3ss.prompts.runtime import PromptProgram
from namel3ss.prompts.validator import OutputValidator

print("="*60)
print("STRUCTURED PROMPTS QUICK TEST")
print("="*60)

# Test 1: PromptProgram with arguments
print("\n[Test 1] PromptProgram with arguments")
prompt = Prompt(
    name="greet",
    model="gpt-4",
    template="Hello {name}, you are {age} years old!",
    args=[
        PromptArgument(name="name", arg_type="string", required=True),
        PromptArgument(name="age", arg_type="int", required=False, default=25),
    ]
)

program = PromptProgram(prompt)

# With both args
result1 = program.render_prompt({"name": "Alice", "age": 30})
print(f"  With both args: {result1}")
assert "Alice" in result1
assert "30" in result1

# With default
result2 = program.render_prompt({"name": "Bob"})
print(f"  With default: {result2}")
assert "Bob" in result2
assert "25" in result2

print("  ✅ PromptProgram tests passed")

# Test 2: OutputValidator with schema
print("\n[Test 2] OutputValidator with schema")
schema = OutputSchema(fields=[
    OutputField(name="sentiment", field_type=OutputFieldType(
        base_type="enum", 
        enum_values=["positive", "negative", "neutral"]
    ), required=True),
    OutputField(name="confidence", field_type=OutputFieldType(base_type="float"), required=True),
    OutputField(name="tags", field_type=OutputFieldType(
        base_type="list",
        element_type=OutputFieldType(base_type="string")
    ), required=False),
])

validator = OutputValidator(schema)

# Valid output
valid_output = {
    "sentiment": "positive",
    "confidence": 0.95,
    "tags": ["happy", "excited"]
}
result = validator.validate(valid_output)
print(f"  Valid output: valid={result.valid}, errors={len(result.errors)}")
assert result.valid

# Invalid output (bad enum)
invalid_output = {
    "sentiment": "unknown",
    "confidence": 0.5
}
result = validator.validate(invalid_output)
print(f"  Invalid output: valid={result.valid}, errors={len(result.errors)}")
assert not result.valid
assert len(result.errors) > 0
print(f"  Error: {result.errors[0].message}")

print("  ✅ OutputValidator tests passed")

# Test 3: Complex nested schema
print("\n[Test 3] Complex nested schema")
complex_schema = OutputSchema(fields=[
    OutputField(
        name="entities",
        field_type=OutputFieldType(
            base_type="list",
            element_type=OutputFieldType(
                base_type="object",
                nested_fields=[
                    OutputField(name="name", field_type=OutputFieldType(base_type="string"), required=True),
                    OutputField(name="type", field_type=OutputFieldType(
                        base_type="enum",
                        enum_values=["person", "place", "organization"]
                    ), required=True),
                ]
            )
        ),
        required=True
    )
])

complex_validator = OutputValidator(complex_schema)

complex_output = {
    "entities": [
        {"name": "Alice", "type": "person"},
        {"name": "New York", "type": "place"},
    ]
}

result = complex_validator.validate(complex_output)
print(f"  Complex valid: valid={result.valid}, errors={len(result.errors)}")
assert result.valid

print("  ✅ Complex schema tests passed")

# Test 4: JSON Schema generation
print("\n[Test 4] JSON Schema generation")
prompt_with_schema = Prompt(
    name="extract",
    model="gpt-4",
    template="Extract from: {text}",
    args=[PromptArgument(name="text", arg_type="string", required=True)],
    output_schema=schema  # Use the schema from Test 2
)
program_with_schema = PromptProgram(prompt_with_schema)
json_schema = program_with_schema.get_output_schema()
print(f"  JSON Schema keys: {list(json_schema.keys())}")
assert "type" in json_schema
assert json_schema["type"] == "object"
assert "properties" in json_schema

print("  ✅ JSON Schema generation passed")

print("\n" + "="*60)
print("✅ ALL QUICK TESTS PASSED")
print("="*60)
print(f"\nSummary:")
print(f"  - PromptProgram: argument handling, defaults, template rendering")
print(f"  - OutputValidator: type validation, enum validation, nested structures")
print(f"  - JSON Schema: generation from prompts")
print(f"\n✅ Core structured prompts functionality is working!")
