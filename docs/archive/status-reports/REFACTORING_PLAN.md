# Namel3ss Production-Grade Refactoring Plan

**Date:** November 20, 2025  
**Objective:** Transform all large modules (>500 lines) into maintainable, production-ready sub-packages with comprehensive error handling, validation, and documentation.

## Executive Summary

**Total Modules Requiring Refactoring:** 42 files exceeding 500 lines  
**Total Lines to Refactor:** ~48,000 lines of code  
**Estimated Complexity:** High - this is compiler-grade infrastructure

### Critical Modules (>2000 lines - Highest Priority)

| Module | Lines | Current Issues | Priority |
|--------|-------|----------------|----------|
| `codegen/frontend.py` | 2320 | Mixed concerns: scaffolding, pages, widgets, config | **P0** |
| `codegen/backend/state.py` | 2269 | State management + validation + generation | **P0** |
| `parser/ai.py` | 2201 | Recently refactored but still monolithic | **P1** |
| `codegen/backend/core/runtime_sections/llm.py` | 2180 | LLM integration + error handling + validation | **P0** |
| `lang/grammar.py` | 2081 | Grammar + parsing + validation mixed | **P0** |

---

## Phase 1: CLI Infrastructure (Week 1)

### 1.1 CLI Module Refactoring (`cli.py` → `cli/` package)

**Current State:** 1873 lines - mixed command handling, environment setup, error formatting

**Target Structure:**
```
namel3ss/cli/
├── __init__.py           # Public API exports
├── commands/
│   ├── __init__.py
│   ├── run.py           # run command handler
│   ├── build.py         # build command handler
│   ├── watch.py         # watch command handler
│   ├── clean.py         # clean command handler
│   └── version.py       # version command handler
├── context.py           # Environment and context management
├── errors.py            # CLI-specific error handling and formatting
├── logging.py           # Logging configuration
├── validation.py        # CLI argument validation
└── utils.py             # Shared CLI utilities
```

**Key Improvements:**
- **Error Handling:** Centralized CLI error formatting with structured output
- **Validation:** Dedicated validation for CLI arguments and options
- **Docstrings:** Every command handler fully documented with examples
- **Testing:** Each command in isolation

**Estimated Effort:** 2-3 days

---

## Phase 2: Frontend Codegen (Week 1-2)

### 2.1 Frontend Module (`codegen/frontend.py` → `codegen/frontend/` package)

**Current State:** 2320 lines - React/Vite scaffolding, page generation, widget rendering

**Target Structure:**
```
namel3ss/codegen/frontend/
├── __init__.py
├── scaffold/
│   ├── __init__.py
│   ├── project.py       # Project structure creation
│   ├── config.py        # vite.config.ts, tsconfig.json generation
│   └── dependencies.py  # package.json management
├── pages/
│   ├── __init__.py
│   ├── generator.py     # Page component generation
│   ├── routing.py       # React Router setup
│   └── layout.py        # Layout components
├── components/
│   ├── __init__.py
│   ├── widgets.py       # UI widget generation
│   ├── forms.py         # Form components
│   ├── tables.py        # Table components
│   └── charts.py        # Chart components
├── assets/
│   ├── __init__.py
│   ├── styles.py        # CSS generation
│   ├── scripts.py       # JavaScript utilities
│   └── static.py        # Static asset handling
├── validation.py        # Frontend codegen validation
└── errors.py           # Frontend-specific errors
```

**Key Improvements:**
- **Separation:** Scaffold vs page generation vs component generation
- **Validation:** Input validation for all codegen operations
- **Error Context:** Rich error messages with line numbers and suggestions
- **Extensibility:** Easy to add new component types

**Estimated Effort:** 3-4 days

### 2.2 Frontend Assets (`codegen/frontend/assets.py` → integrated into above)

**Current State:** 1867 lines - static asset generation

**Action:** Merge into `codegen/frontend/assets/` sub-package (already planned above)

---

## Phase 3: Backend Codegen - Core Runtime (Week 2-3)

### 3.1 Backend State Management (`codegen/backend/state.py` → `codegen/backend/state/` package)

**Current State:** 2269 lines - state tracking, validation, generation orchestration

**Target Structure:**
```
namel3ss/codegen/backend/state/
├── __init__.py
├── tracker.py          # State tracking during codegen
├── validator.py        # State validation rules
├── context.py          # Codegen context management
├── dependencies.py     # Dependency resolution
├── errors.py          # State-specific errors
└── utils.py           # State manipulation utilities
```

**Key Improvements:**
- **Pure State Logic:** Separate state from generation
- **Validation Layer:** Centralized state validation
- **Error Recovery:** Better error messages for state inconsistencies

**Estimated Effort:** 3 days

### 3.2 LLM Runtime Sections (`codegen/backend/core/runtime_sections/llm.py`)

**Current State:** 2180 lines - LLM provider integration, prompt handling, response parsing

**Target Structure:**
```
namel3ss/codegen/backend/core/runtime_sections/llm/
├── __init__.py
├── providers/
│   ├── __init__.py
│   ├── base.py         # Provider interface
│   ├── openai.py       # OpenAI integration
│   ├── anthropic.py    # Anthropic integration
│   ├── vertex.py       # Vertex AI integration
│   └── factory.py      # Provider factory
├── prompts/
│   ├── __init__.py
│   ├── handler.py      # Prompt execution
│   ├── templates.py    # Template rendering
│   └── validation.py   # Prompt validation
├── responses/
│   ├── __init__.py
│   ├── parser.py       # Response parsing
│   ├── streaming.py    # Streaming response handling
│   └── validation.py   # Response validation
├── errors.py          # LLM-specific errors
└── config.py          # LLM configuration
```

**Key Improvements:**
- **Provider Abstraction:** Clean provider interface
- **Error Classification:** Provider errors vs prompt errors vs parsing errors
- **Retry Logic:** Centralized retry with exponential backoff
- **Validation:** Input/output validation at each layer

**Estimated Effort:** 4-5 days

### 3.3 Other Runtime Sections

| Module | Lines | Refactoring Need | Priority |
|--------|-------|------------------|----------|
| `connectors.py` | 1600 | Split into connector types | P1 |
| `models.py` | 1544 | Separate model types and validation | P1 |
| `prediction.py` | 736 | Split prediction logic and ML integration | P2 |
| `rendering.py` | 698 | Separate template rendering concerns | P2 |
| `context.py` | 676 | Context building vs context validation | P2 |
| `dataset.py` | 629 | Dataset operations vs validation | P2 |
| `training.py` | 624 | Training orchestration vs config | P2 |
| `actions.py` | 605 | Action types and execution | P2 |

**Estimated Effort:** 8-10 days total for all runtime sections

### 3.4 Frames Runtime (`codegen/backend/core/runtime/frames.py`)

**Current State:** 1759 lines - dataframe operations, transformations, aggregations

**Target Structure:**
```
namel3ss/codegen/backend/core/runtime/frames/
├── __init__.py
├── operations/
│   ├── __init__.py
│   ├── filter.py       # Filter operations
│   ├── select.py       # Select/project operations
│   ├── join.py         # Join operations
│   ├── aggregate.py    # Aggregation operations
│   └── transform.py    # Transformation operations
├── validation.py       # Frame operation validation
├── errors.py          # Frame-specific errors
└── utils.py           # Frame utilities
```

**Estimated Effort:** 2-3 days

### 3.5 Datasets Runtime (`codegen/backend/core/runtime/datasets.py`)

**Current State:** 689 lines

**Target:** Merge with frames or create focused sub-package

**Estimated Effort:** 1-2 days

### 3.6 Other Core Modules

| Module | Lines | Action | Priority |
|--------|-------|--------|----------|
| `routers.py` | 1019 | Split by endpoint type | P2 |
| `packages.py` | 655 | Package management + requirements | P2 |
| `expression_sandbox.py` | 578 | Expression evaluation + safety | P2 |
| `deploy.py` | 563 | Deployment strategies | P3 |
| `logic_engine.py` | 515 | Logic programming runtime | P3 |

**Estimated Effort:** 5-6 days total

---

## Phase 4: Parser Infrastructure (Week 3-4)

**Note:** Parser modules recently underwent Phase 4 refactoring (comprehensive docstrings, error messages, validation). However, several still exceed 500 lines and could benefit from further modularization.

### 4.1 Parser Base (`parser/base.py`)

**Current State:** 1473 lines - base parser class with all parsing primitives

**Target Structure:**
```
namel3ss/parser/base/
├── __init__.py
├── core.py            # Core ParserBase class (reduced)
├── tokens.py          # Token handling and lexing
├── indentation.py     # Indentation validation (already has IndentationInfo)
├── expressions.py     # Expression parsing helpers
├── validation.py      # Parsing validation utilities
└── errors.py         # Parser error formatting
```

**Key Considerations:**
- Recently refactored with IndentationInfo and KeywordRegistry
- May benefit from further separation of concerns
- **Decision:** Monitor - only refactor if complexity increases

**Estimated Effort:** 2-3 days (if needed)

### 4.2 AI Parser (`parser/ai.py`)

**Current State:** 2201 lines (recently refactored with comprehensive docstrings)

**Status:** **Recently refactored** - has excellent documentation and error handling  
**Decision:** **SKIP** - Already production-grade from Phase 4 refactoring

### 4.3 Other Parser Modules (All recently refactored in Phase 4)

| Module | Lines | Status | Action |
|--------|-------|--------|--------|
| `datasets.py` | 933 | Recently refactored | Monitor only |
| `symbolic.py` | 732 | Recently refactored | Monitor only |
| `insights.py` | 680 | Recently refactored | Monitor only |
| `expressions.py` | 673 | Recently refactored | Monitor only |
| `components.py` | 650 | Recently refactored | Monitor only |
| `logic.py` | 563 | Recently refactored | Monitor only |
| `frames.py` | 550 | Recently refactored | Monitor only |
| `models.py` | 518 | Recently refactored | Monitor only |

**All parser modules except base.py have comprehensive docstrings, centralized validation, and enhanced error messages from Phase 4. They are production-ready.**

---

## Phase 5: Language Infrastructure (Week 4)

### 5.1 Grammar Module (`lang/grammar.py`)

**Current State:** 2081 lines - grammar definition + validation + utilities

**Target Structure:**
```
namel3ss/lang/grammar/
├── __init__.py
├── definition.py      # Grammar rules and definitions
├── tokens.py         # Token types and patterns
├── validation.py     # Grammar validation
├── errors.py        # Grammar-specific errors
└── utils.py         # Grammar utilities
```

**Estimated Effort:** 2-3 days

### 5.2 Keywords Module (`lang/keywords.py`)

**Current State:** 539 lines (recently added KeywordRegistry)

**Status:** Recently enhanced in Phase 4  
**Decision:** Monitor - already has clean structure

---

## Phase 6: Type System & Resolution (Week 4-5)

### 6.1 Type Checker (`types/checker.py`)

**Current State:** 832 lines - type checking + inference + validation

**Target Structure:**
```
namel3ss/types/checker/
├── __init__.py
├── core.py           # Core type checking logic
├── inference.py      # Type inference
├── validation.py     # Type validation
├── errors.py        # Type error formatting
└── utils.py         # Type utilities
```

**Estimated Effort:** 2-3 days

### 6.2 Resolver (`resolver.py`)

**Current State:** 770 lines - name resolution + scope management

**Target Structure:**
```
namel3ss/resolver/
├── __init__.py
├── names.py          # Name resolution
├── scopes.py         # Scope management
├── symbols.py        # Symbol table
├── validation.py     # Resolution validation
└── errors.py        # Resolution errors
```

**Estimated Effort:** 2 days

---

## Phase 7: Runtime & Support Systems (Week 5-6)

### 7.1 React/Vite Frontend Generator (`codegen/frontend/react_vite.py`)

**Current State:** 1365 lines

**Action:** Integrate into Phase 2 frontend refactoring

### 7.2 Agents Runtime (`agents/runtime.py`)

**Current State:** 826 lines - agent execution + orchestration

**Target Structure:**
```
namel3ss/agents/runtime/
├── __init__.py
├── executor.py       # Agent execution
├── orchestration.py  # Multi-agent orchestration
├── state.py         # Agent state management
├── validation.py    # Agent validation
└── errors.py       # Agent-specific errors
```

**Estimated Effort:** 2 days

### 7.3 RAG Systems

| Module | Lines | Refactoring Need |
|--------|-------|------------------|
| `rag/rerankers.py` | 735 | Split reranker types |
| `rag/loaders.py` | 583 | Split loader types |

**Target:** Create focused sub-packages for each  
**Estimated Effort:** 2-3 days total

### 7.4 Frame Analyzer (`frames/analyzer.py`)

**Current State:** 617 lines

**Target Structure:**
```
namel3ss/frames/analyzer/
├── __init__.py
├── schema.py        # Schema analysis
├── statistics.py    # Statistical analysis
├── validation.py    # Analysis validation
└── errors.py       # Analyzer errors
```

**Estimated Effort:** 1-2 days

### 7.5 LLM Providers

| Module | Lines | Action |
|--------|-------|--------|
| `llm/vertex_llm.py` | 601 | Split into provider sub-package |
| `ml/providers/anthropic.py` | 505 | Already focused - minimal refactor |

**Estimated Effort:** 1-2 days

### 7.6 Templates Engine (`templates/engine.py`)

**Current State:** 555 lines

**Target Structure:**
```
namel3ss/templates/engine/
├── __init__.py
├── core.py          # Core template engine
├── compiler.py      # Template compilation
├── renderer.py      # Template rendering
├── validation.py    # Template validation
└── errors.py       # Template errors
```

**Estimated Effort:** 1-2 days

---

## Cross-Cutting Concerns

### Centralized Validation Framework

**Create:** `namel3ss/validation/` package

```
namel3ss/validation/
├── __init__.py
├── core.py          # Base validation framework
├── cli.py           # CLI validation helpers
├── ast.py           # AST validation helpers
├── codegen.py       # Codegen validation helpers
├── providers.py     # Provider config validation
├── schemas.py       # Schema validation
└── errors.py       # Validation-specific errors
```

**Purpose:**
- Eliminate duplicate validation logic
- Provide reusable validation decorators
- Standardize validation error messages
- Enable validation composition

### Error Handling Framework

**Enhance:** `namel3ss/errors.py`

**Add:**
- Error context (file, line, column, source snippet)
- Error codes for programmatic handling
- Structured error metadata
- Error recovery hints
- Error aggregation for batch operations

**Pattern:**
```python
class N3Error(Exception):
    """Base error with rich context."""
    def __init__(
        self,
        message: str,
        *,
        code: str,
        source_file: Optional[str] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
        hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        ...
```

---

## Implementation Strategy

### Week 1: Foundation & CLI
1. **Day 1-2:** CLI refactoring (`cli.py` → `cli/`)
2. **Day 3:** Create centralized validation framework
3. **Day 4-5:** Enhance error handling framework

### Week 2: Frontend Codegen
1. **Day 1-3:** Frontend module refactoring
2. **Day 4-5:** Frontend assets integration

### Week 3: Backend State & LLM
1. **Day 1-3:** Backend state refactoring
2. **Day 4-5:** LLM runtime sections (start)

### Week 4: Backend Runtime Completion
1. **Day 1-3:** LLM runtime sections (complete)
2. **Day 4-5:** Grammar refactoring

### Week 5: Type System & Resolution
1. **Day 1-3:** Type checker refactoring
2. **Day 4-5:** Resolver refactoring

### Week 6: Runtime Systems
1. **Day 1-2:** Agents runtime
2. **Day 3-4:** RAG systems
3. **Day 5:** Templates engine

### Week 7: Testing & Documentation
1. **Day 1-3:** Comprehensive test coverage
2. **Day 4-5:** Documentation updates

---

## Success Criteria

### Code Quality
- ✅ No Python file exceeds 500 lines
- ✅ All public APIs have comprehensive docstrings
- ✅ All modules use centralized validation
- ✅ All modules use structured error handling
- ✅ Zero `# type: ignore` comments

### Testing
- ✅ 90%+ code coverage maintained
- ✅ All refactored modules have focused unit tests
- ✅ Integration tests pass
- ✅ No regressions in existing functionality

### Documentation
- ✅ Architecture documentation updated
- ✅ API documentation generated
- ✅ Migration guide for internal changes
- ✅ Examples updated

### Performance
- ✅ No performance regressions
- ✅ Build times maintained or improved
- ✅ Memory usage maintained or improved

---

## Risk Mitigation

### High-Risk Modules
1. **Grammar Module:** Core to parsing - extensive testing required
2. **Backend State:** Central to codegen - gradual migration needed
3. **LLM Runtime:** Production critical - feature flags for rollback

### Mitigation Strategies
- Feature flags for new module structures
- Parallel implementation (old + new) during transition
- Comprehensive integration tests
- Staged rollout with monitoring
- Rollback plan for each phase

---

## Next Steps

1. **Review & Approve:** Review this plan with team
2. **Prioritize:** Confirm priority order based on business needs
3. **Start:** Begin with CLI refactoring (highest impact, lowest risk)
4. **Iterate:** Weekly reviews and adjustments
5. **Celebrate:** Each phase completion milestone

---

## Notes

- **Parser modules (Phase 4 complete):** Excellent state - comprehensive docstrings, error messages, validation. No immediate refactoring needed.
- **Focus Priority:** CLI, Frontend, Backend State, LLM Runtime
- **Defer:** Parser modularization unless complexity increases
- **Timeline:** 6-7 weeks for complete refactoring
- **Resources:** 1-2 senior engineers full-time

