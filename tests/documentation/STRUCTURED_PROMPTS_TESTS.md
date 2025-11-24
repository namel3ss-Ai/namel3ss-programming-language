# Structured Prompts Test Suite

## Overview

Comprehensive test suite for the structured prompts feature covering parsing, validation, runtime execution, output validation, and integration scenarios.

## Test Files

### 1. `test_structured_prompts_parser.py`
Tests for DSL parsing of structured prompts.

**Coverage:**
- Argument parsing (simple types, defaults, all supported types)
- Output schema parsing (primitives, enums, lists, nested objects)
- Complex nested structures (list of objects with enums)
- Error handling (invalid types, malformed syntax)
- Backward compatibility (legacy prompts without args/schemas)

**Key Test Classes:**
- `TestPromptArgumentParsing`: Argument syntax
- `TestOutputSchemaParsing`: Schema syntax
- `TestParsingErrors`: Error cases
- `TestBackwardCompatibility`: Legacy prompt support

### 2. `test_structured_prompts_validation.py`
Tests for static validation during compilation.

**Coverage:**
- Argument validation (duplicates, invalid types, required/optional order)
- Output schema validation (field duplicates, invalid types, enum constraints)
- Template placeholder validation (undefined placeholders, unused args)
- Nested structure validation (recursive checks)

**Key Test Classes:**
- `TestArgumentValidation`: Arg checks
- `TestOutputSchemaValidation`: Schema checks
- `TestTemplatePlaceholderValidation`: Template consistency
- `TestNestedValidation`: Nested structures

### 3. `test_structured_prompts_runtime.py`
Tests for PromptProgram runtime execution.

**Coverage:**
- Argument handling (defaults, required args, unexpected args)
- Type coercion (string, int, float, bool, list, object)
- Template rendering (simple, multiline, complex)
- JSON Schema generation (primitives, enums, lists, optional fields)

**Key Test Classes:**
- `TestPromptProgramArguments`: Arg processing
- `TestTypeCoercion`: Type conversion
- `TestOutputSchemaGeneration`: Schema generation
- `TestComplexTemplates`: Template edge cases

### 4. `test_structured_prompts_validator.py`
Tests for OutputValidator runtime validation.

**Coverage:**
- Primitive type validation (string, int, float, bool)
- Enum validation (valid/invalid values)
- List validation (empty lists, invalid elements, list of objects)
- Object validation (nested objects, missing fields)
- Required vs optional fields
- Unexpected fields handling
- Nullable types
- Complex nested schemas (deep nesting, error paths)

**Key Test Classes:**
- `TestPrimitiveValidation`: Basic types
- `TestEnumValidation`: Enum constraints
- `TestListValidation`: Array handling
- `TestObjectValidation`: Nested objects
- `TestRequiredFields`: Required vs optional
- `TestUnexpectedFields`: Strict validation
- `TestNullableFields`: Null handling
- `TestComplexSchemas`: Deep nesting
- `TestValidationRaising`: Exception handling

### 5. `test_structured_prompts_integration.py`
Integration tests with mocked LLMs.

**Coverage:**
- Successful execution end-to-end
- Retry logic on validation errors
- Max retries exceeded
- Retry disabled mode
- Complex nested outputs
- Multiple args with defaults
- Error handling (invalid args, missing args, invalid JSON)

**Key Test Classes:**
- `TestExecutorIntegration`: Basic execution flow
- `TestComplexIntegration`: Complex scenarios
- `TestErrorHandling`: Error cases

## Running Tests

### Run All Structured Prompt Tests
```bash
pytest tests/test_structured_prompts_*.py -v
```

### Run Specific Test File
```bash
pytest tests/test_structured_prompts_parser.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_structured_prompts_parser.py::TestPromptArgumentParsing -v
```

### Run Specific Test
```bash
pytest tests/test_structured_prompts_parser.py::TestPromptArgumentParsing::test_parse_simple_args -v
```

### Run with Coverage
```bash
pytest tests/test_structured_prompts_*.py --cov=namel3ss.prompts --cov=namel3ss.parser.ai --cov=namel3ss.ast.ai --cov-report=html
```

## Test Statistics

- **Total Test Files**: 5
- **Estimated Test Count**: ~100+ tests
- **Coverage Areas**:
  - Parsing: ~20 tests
  - Validation: ~25 tests
  - Runtime: ~20 tests
  - Output Validation: ~30 tests
  - Integration: ~15 tests

## Test Dependencies

Required packages:
```
pytest>=7.0.0
pytest-asyncio>=0.21.0  # For async tests
pytest-cov>=4.0.0       # For coverage reports
```

## Writing New Tests

### Test Structure
```python
class TestFeature:
    """Test description."""
    
    def test_specific_behavior(self):
        """Test that specific behavior works."""
        # Arrange
        prompt = Prompt(...)
        
        # Act
        result = function_under_test(prompt)
        
        # Assert
        assert result.expected_property == expected_value
```

### Naming Conventions
- Test files: `test_structured_prompts_<component>.py`
- Test classes: `Test<Feature>`
- Test methods: `test_<specific_behavior>`

### Best Practices
1. **One assertion per test** (when possible)
2. **Clear test names** describing the scenario
3. **AAA pattern**: Arrange, Act, Assert
4. **Use fixtures** for common setup
5. **Mock external dependencies** (LLMs, HTTP calls)
6. **Test both success and failure paths**
7. **Include edge cases** (empty lists, null values, etc.)

## Continuous Integration

Tests should be run in CI/CD pipeline:

```yaml
# .github/workflows/test.yml
- name: Run Structured Prompts Tests
  run: |
    pytest tests/test_structured_prompts_*.py --cov --cov-report=xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

## Known Issues / TODOs

- [ ] Add async test variants for `execute_structured_prompt()`
- [ ] Add performance benchmarks for validation
- [ ] Add tests for chain integration (requires test chains)
- [ ] Add tests for observability metrics
- [ ] Add property-based tests with Hypothesis
- [ ] Add fuzzing tests for parser robustness

## Coverage Goals

Target coverage levels:
- **Parser**: 95%+ (critical path)
- **Validator**: 95%+ (critical path)
- **Runtime**: 90%+ (some edge cases acceptable)
- **Integration**: 80%+ (mocked, not exhaustive)
- **Overall**: 90%+

## Contributing

When adding new features to structured prompts:
1. Write tests first (TDD)
2. Ensure all tests pass
3. Add tests to appropriate test file
4. Update this README if needed
5. Run coverage report to ensure coverage goals met
