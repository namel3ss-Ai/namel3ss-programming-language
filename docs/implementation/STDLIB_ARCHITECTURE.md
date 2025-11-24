# Namel3ss Standard Library Architecture Design

## Overview

This document defines the architecture for separating the **core language** from the **standard library** in namel3ss, providing a clear distinction between intrinsic language constructs and reusable AI primitives.

## Core Language vs Standard Library

### Core Language (namel3ss/lang/)
**What the parser and type checker understand intrinsically:**

- **Syntax constructs**: `app`, `llm`, `agent`, `prompt`, `tool`, `memory`, `page`, `chain`
- **Type system**: Basic types, expressions, control flow
- **Grammar rules**: Parsing logic for language constructs
- **AST nodes**: Low-level representation of language elements

**Responsibility**: Define the fundamental building blocks and syntax rules.

### Standard Library (namel3ss/stdlib/)
**Canonical, reusable AI primitives built on top of core language:**

- **Memory policies**: Predefined behavior patterns for conversation history
- **LLM configurations**: Provider-neutral parameter definitions and constraints  
- **Tool interfaces**: Standardized contracts for HTTP, DB, and vector tools
- **AI primitives**: Common patterns and best practices

**Responsibility**: Provide production-ready, validated implementations and interfaces.

## Module Structure

```
namel3ss/
├── lang/                    # Core Language
│   ├── __init__.py
│   ├── ast/                 # Core AST nodes
│   ├── parser/              # Grammar and parsing
│   ├── types/               # Core type system
│   └── keywords.py          # Language keywords
│
├── stdlib/                  # Standard Library  
│   ├── __init__.py
│   ├── memory/              # Memory policy definitions
│   │   ├── __init__.py
│   │   ├── policies.py      # MemoryPolicy enum and specs
│   │   ├── validation.py    # Policy validation logic
│   │   └── defaults.py      # Default configurations
│   ├── llm/                 # LLM configuration standards
│   │   ├── __init__.py
│   │   ├── config.py        # Standard LLM fields and ranges
│   │   ├── validation.py    # Parameter validation
│   │   └── providers.py     # Provider-neutral interfaces
│   ├── tools/               # Tool interface definitions
│   │   ├── __init__.py
│   │   ├── base.py          # Base tool interfaces
│   │   ├── http.py          # HTTP tool standard
│   │   ├── database.py      # Database tool standard  
│   │   ├── vector.py        # Vector search tool standard
│   │   └── validation.py    # Tool validation logic
│   └── registry.py          # Standard library registry
│
├── ast/                     # Application AST (references stdlib)
├── resolver.py              # Uses stdlib for validation
├── codegen/                 # Uses stdlib interfaces
└── cli.py                   # Exposes stdlib features
```

## Design Principles

1. **Provider Neutrality**: Standard library targets interfaces, not specific vendors
2. **Backward Compatibility**: Existing .ai programs continue to work
3. **Type Safety**: Standard library definitions are enforced by type checker
4. **Discoverability**: CLI and docs expose available standard constructs
5. **Modularity**: Standard library components can be used independently

## Implementation Phases

### Phase 1: Core Infrastructure
- Create `namel3ss/stdlib/` module structure
- Define base interfaces and registry system
- Implement memory policy standardization

### Phase 2: LLM Configuration
- Standardize LLM parameter fields and constraints
- Create provider-neutral configuration interfaces
- Update existing providers to use standard definitions

### Phase 3: Tool Interfaces  
- Define HTTP, database, and vector tool standards
- Create validation logic for tool configurations
- Refactor existing tool implementations

### Phase 4: Integration
- Update type system to use stdlib definitions
- Integrate with resolver for compile-time validation
- Update codegen to use provider-neutral interfaces

### Phase 5: Testing & Documentation
- Comprehensive test coverage in tests/stdlib/
- Update CLI help and documentation
- Validate against existing codebase

## Next Steps

1. Implement core stdlib infrastructure
2. Standardize memory policies with validation
3. Create LLM configuration standards
4. Define tool interface contracts
5. Integrate with type system and resolver