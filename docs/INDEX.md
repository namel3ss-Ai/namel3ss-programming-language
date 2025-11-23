# Namel3ss Language Documentation

**Namel3ss (N3)** is an AI programming language that compiles declarative `.ai` source files into full-stack applications with React/Vite frontends and FastAPI backends.

This documentation provides comprehensive coverage of the language, its features, runtime behavior, and development workflow.

---

## Documentation Structure

### 1. Language Overview

- **[Getting Started](../README.md)** - Installation, quick start, basic concepts
- **[Language Version & Compatibility](spec/README.md)** - Version semantics and feature compatibility

### 2. Core Syntax & Semantics

#### Basic Constructs
- **[Applications](APPLICATIONS.md)** - App blocks, modules, and program structure
- **[Pages & UI](PAGES.md)** - Page definitions, routing, and component hierarchy
- **[Datasets](DATASETS.md)** - Data sources, transformations, and queries
- **[Types & Expressions](EXPRESSION_LANGUAGE.md)** - Type system, operators, and expression evaluation

#### Control Flow
- **[Control Flow Syntax](../CONTROL_FLOW_SYNTAX.md)** - If/else conditionals and for loops
- **[Control Flow Implementation](../CONTROL_FLOW_IMPLEMENTATION.md)** - Runtime behavior and code generation

### 3. AI Features

#### Models & Prompts
- **[LLM Providers](llm-provider-guide.md)** - Configuring OpenAI, Anthropic, and other providers
- **[Structured Prompts](STRUCTURED_PROMPTS.md)** - Typed arguments, output schemas, and validation
- **[Provider System](../PROVIDER_SYSTEM.md)** - Provider architecture and migration guide

#### Chains & Workflows
- **[Chains](CHAINS.md)** - Sequential prompt execution and data flow
- **[Agents & Graphs](../AGENT_GRAPH_GUIDE.md)** - Multi-agent systems with conditional routing

#### Memory & State
- **[Memory System](MEMORY_SYSTEM.md)** - Stateful interactions, conversation history, and scoped storage
- **[Queries & RAG](QUERIES.md)** - Knowledge retrieval and retrieval-augmented generation
- **[RAG Guide](RAG_GUIDE.md)** - End-to-end RAG implementation patterns

#### Logic & Reasoning
- **[Logic System](LOGIC.md)** - Declarative rules, facts, and backward chaining
- **[Symbolic Expressions](SYMBOLIC_EXPRESSIONS.md)** - Symbolic computation and term rewriting

#### Training & Evaluation
- **[Training & Tuning](TRAINING.md)** - Model fine-tuning and adaptation
- **[Evaluation Suites](EVAL_SUITES.md)** - Experiment frameworks and metric collection

### 4. Runtime & Code Generation

#### Backend
- **[Backend Architecture](BACKEND_ARCHITECTURE.md)** - Generated FastAPI structure and runtime
- **[Routers & Endpoints](ROUTERS.md)** - Page APIs, model endpoints, and request handling
- **[State Encoding](STATE_ENCODING.md)** - How AST is encoded into runtime registries

#### Frontend
- **[Frontend Generation](FRONTEND_GENERATION.md)** - React/Vite app structure and client library
- **[Component API](COMPONENT_API.md)** - Table, chart, form, and action components

#### Integration
- **[Frontend-Backend Contract](INTEGRATION_CONTRACT.md)** - API schemas and communication patterns
- **[Error Handling](ERROR_HANDLING.md)** - Error types, messages, and developer experience

### 5. Testing & Quality

- **[Testing Guide](TESTING.md)** - Running tests, writing new tests, and test organization
- **[Language Tests](TESTING.md#language-tests)** - Testing .ai compilation pipeline
- **[Backend Tests](TESTING.md#backend-tests)** - API endpoint validation
- **[Frontend Tests](TESTING.md#frontend-tests)** - Component and integration testing
- **[AI Feature Tests](TESTING.md#ai-tests)** - Structured prompts, chains, and validation

### 6. CLI & Tools

- **[CLI Documentation](../CLI_DOCUMENTATION.md)** - Command reference and usage
- **[CLI Quick Reference](../CLI_QUICK_REFERENCE.md)** - Common commands and workflows
- **[CLI Implementation](../CLI_IMPLEMENTATION.md)** - CLI internals and architecture

### 7. Development & Contribution

- **[Architecture Overview](ARCHITECTURE.md)** - High-level system design
- **[Parser & AST](PARSER.md)** - Parsing pipeline and AST structure
- **[Resolver](RESOLVER.md)** - Semantic analysis and symbol resolution
- **[Code Generation](CODEGEN.md)** - Backend and frontend generation process
- **[Plugin System](PLUGINS.md)** - Extending Namel3ss with plugins

---

## Document Status

### Stable Features (Production-Ready)
- ✅ Applications, pages, datasets
- ✅ Control flow (if/else, for loops)
- ✅ LLM providers (OpenAI, Anthropic, etc.)
- ✅ Structured prompts with output schemas
- ✅ Memory system with scoped storage
- ✅ Logic system with backward chaining
- ✅ Symbolic expressions
- ✅ RAG and query system
- ✅ Backend generation (FastAPI)
- ✅ Frontend generation (React/Vite)

### Experimental Features
- 🧪 Agents and graphs (syntax stable, runtime evolving)
- 🧪 Training/tuning pipelines (API subject to change)
- 🧪 Evaluation suites (under active development)

### Deprecated
- ❌ Legacy prompt syntax without `output_schema` (use structured prompts)
- ❌ Direct database connectors without datasets (use dataset abstraction)

---

## Reading Order for New Users

1. Start with **[README](../README.md)** for installation and quick start
2. Learn **[Applications](APPLICATIONS.md)** and **[Pages](PAGES.md)** for basic structure
3. Understand **[Datasets](DATASETS.md)** for data handling
4. Explore **[Structured Prompts](STRUCTURED_PROMPTS.md)** for AI integration
5. Read **[Control Flow](../CONTROL_FLOW_SYNTAX.md)** for conditionals and loops
6. Study **[Memory System](MEMORY_SYSTEM.md)** for stateful applications
7. Dive into **[Agents & Graphs](../AGENT_GRAPH_GUIDE.md)** for complex workflows
8. Review **[Testing Guide](TESTING.md)** to validate your application

---

## Contributing to Documentation

When adding or updating documentation:

1. **Match Implementation**: Ensure docs reflect actual parser/runtime behavior
2. **Include Examples**: Provide minimal, self-contained code snippets
3. **Document Errors**: Show common mistakes and exact error messages
4. **Cross-Reference**: Link related docs and corresponding test files
5. **Version Notes**: Mark experimental features and breaking changes
6. **Test Examples**: Every doc example should have a corresponding test

See [TESTING.md](TESTING.md) for how documentation examples are validated.

---

## Quick Links

- [GitHub Repository](https://github.com/SsebowaDisan/namel3ss-programming-language)
- [PyPI Package](https://pypi.org/project/namel3ss/)
- [Issue Tracker](https://github.com/SsebowaDisan/namel3ss-programming-language/issues)
- [Examples Directory](../examples/)
- [Test Suite](../tests/)

---

**Last Updated**: November 2025  
**Language Version**: See `namel3ss.lang.LANGUAGE_VERSION`  
**Documentation Version**: 1.0
