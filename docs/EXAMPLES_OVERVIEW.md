# Namel3ss Examples Overview

This directory contains production-ready Namel3ss (N3) applications demonstrating real-world AI programming patterns and enterprise use cases.

## 🏗️ Architecture Overview

All examples follow production patterns with:
- **Realistic domain models** and business logic
- **Professional AI agents** with specific expertise
- **Memory systems** for context and conversation state
- **Structured prompts** for consistent AI behavior
- **Production-ready configuration** with environment variables

## 📋 Available Examples

### 1. Minimal Application (`minimal.ai`)
**Purpose**: Basic N3 application structure  
**Demonstrates**: Core language constructs, LLM configuration, simple prompts

**Key Components**:
- Basic app definition
- OpenAI LLM configuration
- Simple prompt template

**Build & Run**:
```bash
namel3ss build examples/minimal.ai
namel3ss run examples/minimal.ai
```

---

### 2. Content Analyzer (`content_analyzer.ai`)
**Purpose**: AI-powered content analysis and categorization  
**Demonstrates**: Structured analysis, sentiment detection, entity extraction

**Key Components**:
- Anthropic Claude model for analysis
- Session memory for analysis context
- Structured prompt for content categorization
- Professional content analysis agent

**Use Cases**:
- Document classification
- Content moderation
- Information extraction
- Sentiment analysis

**Build & Run**:
```bash
namel3ss build examples/content_analyzer.ai
namel3ss run examples/content_analyzer.ai
```

---

### 3. Research Assistant (`research_assistant.ai`)
**Purpose**: Multi-step research workflow with information synthesis  
**Demonstrates**: Research planning, information gathering, synthesis workflows

**Key Components**:
- GPT-4o for complex reasoning
- Conversation memory for research sessions
- Research planning and synthesis prompts
- Professional research analyst agent

**Use Cases**:
- Market research
- Academic research assistance
- Information synthesis
- Report generation

**Build & Run**:
```bash
namel3ss build examples/research_assistant.ai
namel3ss run examples/research_assistant.ai
```

---

## 🔧 Configuration Requirements

### Environment Variables
All examples require appropriate API keys:

```bash
# For OpenAI-based examples
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic-based examples  
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Dependencies
Ensure your Namel3ss installation includes:
- OpenAI provider support
- Anthropic provider support
- Memory system modules
- Agent framework

## 🚀 Getting Started

1. **Choose an example** that matches your use case
2. **Set up environment variables** for the required AI providers
3. **Build the application**:
   ```bash
   namel3ss build examples/[example-name].ai
   ```
4. **Run the application**:
   ```bash
   namel3ss run examples/[example-name].ai
   ```

## 🧪 Testing Examples

To validate example functionality:

```bash
# Test compilation of all examples
for example in examples/*.ai; do
  echo "Testing $example..."
  namel3ss build "$example" || echo "❌ Build failed for $example"
done

# Run specific example tests (if available)
namel3ss test examples/content_analyzer.ai
```

## 📚 Example Patterns

### Agent Design Pattern
All examples demonstrate professional agent design:
- **Clear purpose and expertise** definition
- **Appropriate model selection** for the task
- **Memory configuration** for context retention
- **System prompts** defining professional behavior

### Memory Usage Patterns
- **Session memory**: For conversation context
- **List memory**: For structured data storage  
- **Conversation memory**: For dialogue systems

### Prompt Engineering Patterns
- **Structured templates** with clear instructions
- **Professional tone** and expert personas
- **Explicit output formatting** requirements
- **Context integration** from memory and inputs

## 🔧 Customization Guide

### Adapting Examples
1. **Change the domain**: Update app description, prompts, and agent expertise
2. **Modify AI behavior**: Adjust temperature, system prompts, and model selection
3. **Extend functionality**: Add tools, memory systems, or additional agents
4. **Scale for production**: Add error handling, monitoring, and deployment config

### Adding New Examples
1. Follow the established naming pattern
2. Include comprehensive documentation
3. Use realistic business scenarios
4. Test compilation and basic functionality
5. Update this overview document

## 📖 Documentation Standards

Each example includes:
- **Clear business purpose** and use case
- **Professional domain modeling** (no toy examples)
- **Production-ready patterns** and best practices
- **Realistic AI agent behavior** and expertise
- **Proper environment variable usage**
- **Comprehensive inline comments**

## 🎯 Best Practices

### Code Organization
- **Group related components** (models, memory, prompts, agents)
- **Use descriptive names** that reflect business purpose
- **Include inline documentation** for complex logic
- **Follow consistent formatting** and style

### AI Configuration
- **Choose appropriate models** for each task
- **Set reasonable temperature** values (0.0-0.3 for analytical tasks)
- **Define clear system prompts** with professional personas
- **Use structured output** formats where possible

### Memory Management
- **Select appropriate memory types** for each use case
- **Set reasonable size limits** to manage context
- **Use consistent scoping** (session, user, document, etc.)
- **Document memory usage** patterns

## 🤝 Contributing

To contribute new examples:
1. Ensure examples solve real business problems
2. Follow established patterns and conventions
3. Include comprehensive documentation
4. Test thoroughly before submission
5. Update this overview document

## 🔍 Troubleshooting

### Common Issues
- **API Key Errors**: Ensure environment variables are properly set
- **Model Availability**: Verify your API keys have access to specified models
- **Memory Limits**: Check memory configuration if context is lost
- **Compilation Errors**: Validate N3 syntax against current language specification

### Getting Help
- Check the main Namel3ss documentation
- Review error messages for specific syntax issues
- Validate environment variable configuration
- Test with minimal examples first

---

*Last updated: November 2025*  
*Namel3ss Version: 1.0.0*