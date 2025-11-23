# Namel3ss Testing Framework Examples

This directory contains comprehensive examples of using the namel3ss testing framework in various scenarios.

## Example Applications

### 1. **basic_chatbot.ai** - Simple Chatbot
A basic chatbot with greeting and help functionality.
- **Purpose**: Demonstrates prompt testing and basic agent functionality
- **Features**: Simple prompts, greeting agent, basic conversation flow
- **Test Coverage**: Prompt assertions, agent responses, input validation

### 2. **content_analyzer.ai** - Content Analysis System  
A content analysis system with sentiment analysis and keyword extraction.
- **Purpose**: Shows complex data structure testing and multi-step workflows
- **Features**: Analysis chains, structured outputs, confidence scoring
- **Test Coverage**: JSON path assertions, nested data validation, chain testing

### 3. **research_assistant.ai** - Research Assistant
A research assistant that uses external APIs and vector search.
- **Purpose**: Demonstrates tool mocking and complex application testing  
- **Features**: HTTP API calls, vector search, multi-agent coordination
- **Test Coverage**: Tool mocks, external service simulation, full app testing

### 4. **document_processor.ai** - Document Processing Pipeline
A document processing pipeline with multiple stages.
- **Purpose**: Shows testing of complex workflows and error handling
- **Features**: Multi-step chains, error handling, file processing
- **Test Coverage**: Workflow testing, error scenarios, timeout handling

## Test Suite Examples

### Unit Testing Patterns
- **Prompt Testing**: Isolated prompt functionality
- **Agent Testing**: Single agent behavior  
- **Chain Testing**: Multi-step workflow validation
- **Error Testing**: Failure mode validation

### Integration Testing Patterns  
- **Full Application**: End-to-end application testing
- **Multi-Agent**: Agent coordination testing
- **External Services**: Tool integration testing
- **Performance**: Timing and efficiency testing

## Mock Configuration Examples

### LLM Mocking Patterns
- **Pattern Matching**: Exact and regex patterns
- **Response Variations**: Multiple responses for same pattern
- **Metadata Simulation**: Response metadata and timing
- **Fallback Handling**: Default responses for unmatched prompts

### Tool Mocking Patterns
- **HTTP APIs**: REST service simulation
- **Database Queries**: SQL result mocking
- **Vector Search**: Similarity search simulation  
- **File Operations**: File system mocking

## Running the Examples

### Quick Start
```bash
# Run all examples
namel3ss test examples/testing/

# Run specific example
namel3ss test examples/testing/basic_chatbot/

# Run with verbose output
namel3ss test examples/testing/ --verbose
```

### Individual Examples
```bash
# Basic chatbot
namel3ss test examples/testing/basic_chatbot/basic_chatbot.test.yaml

# Content analyzer  
namel3ss test examples/testing/content_analyzer/content_analyzer.test.yaml

# Research assistant
namel3ss test examples/testing/research_assistant/research_assistant.test.yaml

# Document processor
namel3ss test examples/testing/document_processor/document_processor.test.yaml
```

### Python Integration
```bash
# Run as pytest
cd examples/testing/
pytest basic_chatbot/test_basic_chatbot.py -v

# Run with coverage
pytest --cov=namel3ss.testing basic_chatbot/ -v
```

## Learning Path

### Beginner
1. **basic_chatbot** - Start here for simple prompt and agent testing
2. **content_analyzer** - Learn about structured data assertions
3. Review test patterns and mock configurations

### Intermediate  
1. **research_assistant** - Understand tool mocking and API simulation
2. **document_processor** - Learn complex workflow and error testing
3. Practice custom assertion patterns

### Advanced
1. Study all examples for comprehensive patterns
2. Create custom mock types and assertions
3. Integrate with CI/CD pipelines

## Best Practices Demonstrated

### Test Design
- **Single Responsibility**: Each test focuses on one specific behavior
- **Clear Naming**: Test names describe what is being validated
- **Comprehensive Coverage**: Tests cover success and failure scenarios
- **Realistic Data**: Test data matches production scenarios

### Mock Design  
- **Realistic Responses**: Mock responses mirror actual service behavior
- **Pattern Specificity**: Mock patterns are specific enough to be meaningful
- **Error Simulation**: Include error conditions and edge cases
- **Performance Simulation**: Include realistic response times

### Assertion Strategies
- **Multiple Aspects**: Test different aspects of the same output
- **Meaningful Messages**: Include descriptive assertion messages
- **Appropriate Granularity**: Balance thoroughness with maintainability
- **Edge Cases**: Test boundary conditions and error states

Each example includes detailed comments explaining the testing patterns and best practices demonstrated.