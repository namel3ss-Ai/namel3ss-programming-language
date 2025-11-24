# Namel3ss Conformance Test Specification

**Version**: 1.0.0  
**Language Version**: Namel3ss Language 1.0  
**Status**: Stable

## Overview

This document defines the conformance test specification format for Namel3ss Language 1.0. The conformance test suite is the **executable specification** of the language, enabling multiple independent implementations to validate their correctness.

## Design Principles

1. **Machine-Readable**: Test descriptors are structured YAML that can be parsed by any implementation
2. **Language-Agnostic**: External implementations in any language can consume these tests
3. **Deterministic**: Tests produce repeatable, verifiable results
4. **Comprehensive**: Cover all language phases (parse, typecheck, runtime)
5. **Versioned**: Tests are tied to specific language versions
6. **Stable**: Changes only occur through the RFC process

## Test Descriptor Format

### File Structure

Each conformance test is described by a YAML file with the `.test.yaml` extension:

```yaml
# Required metadata
spec_version: "1.0.0"          # Conformance spec version
language_version: "1.0.0"       # Target language version
test_id: "parse-001"            # Unique identifier
category: "parse"               # parse | types | runtime
name: "Simple app declaration"  # Human-readable name
description: |                  # Detailed description
  Tests parsing of a minimal valid application with no agents or tools.

# Test phases to execute
phases:
  - parse                       # parse | resolve | typecheck | codegen | runtime

# Source files
sources:
  - path: "fixtures/parse/valid/simple_app.ai"
    content: |                  # Inline content (alternative to path)
      app "SimpleApp" {
      }

# Expected outcomes
expect:
  # For parse phase
  parse:
    status: "success"           # success | error
    ast:                        # Optional: expected AST structure
      type: "Module"
      body:
        - type: "App"
          name: "SimpleApp"
          agents: []
          tools: []
    
  # For typecheck phase
  typecheck:
    status: "success"
    diagnostics: []             # Expected warnings/errors
  
  # For runtime phase
  runtime:
    status: "success"
    timeout_ms: 5000
    stdout: ""                  # Expected standard output
    stderr: ""                  # Expected standard error
    exit_code: 0

# Optional: test-specific configuration
config:
  strict_ast_match: false       # Require exact AST match vs structural
  allow_extra_diagnostics: false # Allow additional warnings beyond expected
```

### Required Fields

Every test descriptor **must** include:

- `spec_version`: Version of this conformance spec format
- `language_version`: Namel3ss language version being tested
- `test_id`: Unique identifier (convention: `<category>-<number>`)
- `category`: Test category (`parse`, `types`, `runtime`)
- `name`: Short human-readable name
- `phases`: List of phases to execute

### Phase Types

1. **`parse`**: Lexing and parsing phase
   - Validates syntax correctness
   - Optionally checks AST structure
   
2. **`resolve`**: Module and symbol resolution
   - Validates imports and references
   - Checks symbol visibility

3. **`typecheck`**: Type checking and semantic analysis
   - Validates type correctness
   - Checks semantic rules

4. **`codegen`**: Code generation (IR building)
   - Validates IR structure
   - Checks code generation correctness

5. **`runtime`**: Execution behavior
   - Validates runtime semantics
   - Checks deterministic outputs

### Expected Outcomes

#### Parse Expectations

```yaml
expect:
  parse:
    status: "success" | "error"
    
    # For successful parses
    ast:
      type: "Module"
      # ... AST structure
    
    # For parse errors
    errors:
      - code: "SYNTAX_ERROR"
        message: "Unexpected token"
        location:
          file: "test.ai"
          line: 5
          column: 10
          length: 4
```

#### Typecheck Expectations

```yaml
expect:
  typecheck:
    status: "success" | "error"
    
    diagnostics:
      - severity: "error" | "warning"
        code: "TYPE_MISMATCH"
        message: "Expected string, got number"
        location:
          file: "test.ai"
          line: 12
          column: 5
```

#### Runtime Expectations

```yaml
expect:
  runtime:
    status: "success" | "error"
    timeout_ms: 5000
    
    # For successful execution
    stdout: "expected output"
    stderr: ""
    exit_code: 0
    
    # For runtime errors
    error:
      type: "RuntimeError"
      message: "Division by zero"
```

### AST Matching

AST structures can be specified at different levels of detail:

**Minimal (structural)**:
```yaml
ast:
  type: "Module"
  body:
    - type: "App"
```

**Detailed (full)**:
```yaml
ast:
  type: "Module"
  name: "main"
  path: "test.ai"
  has_explicit_app: true
  body:
    - type: "App"
      name: "TestApp"
      agents:
        - type: "AgentDefinition"
          name: "agent1"
          llm_name: "gpt-4"
```

By default, implementations should perform **structural matching** (types and structure) rather than exact matching (including source locations, etc.).

## Directory Structure

```
tests/conformance/v1/
├── SPEC.md                     # This document
├── parse/
│   ├── valid/
│   │   ├── simple_app.test.yaml
│   │   ├── app_with_agents.test.yaml
│   │   └── ...
│   └── invalid/
│       ├── missing_brace.test.yaml
│       ├── invalid_token.test.yaml
│       └── ...
├── types/
│   ├── valid/
│   │   ├── type_inference.test.yaml
│   │   └── ...
│   └── invalid/
│       ├── type_mismatch.test.yaml
│       └── ...
├── runtime/
│   ├── basic/
│   │   ├── hello_world.test.yaml
│   │   └── ...
│   └── advanced/
│       └── ...
└── fixtures/
    ├── parse/
    │   ├── valid/
    │   │   └── simple_app.ai
    │   └── invalid/
    │       └── missing_brace.ai
    ├── types/
    └── runtime/
```

## Test ID Convention

Test IDs follow the pattern: `<category>-<subcategory>-<number>`

Examples:
- `parse-valid-001`: Valid parse test #1
- `parse-invalid-042`: Invalid parse test #42
- `types-inference-003`: Type inference test #3
- `types-error-015`: Type error test #15
- `runtime-basic-001`: Basic runtime test #1
- `runtime-llm-007`: LLM interaction test #7

## Versioning

### Conformance Spec Version

The conformance test specification itself is versioned using semantic versioning:

- **Major**: Breaking changes to test descriptor format
- **Minor**: Backward-compatible additions (new fields, phase types)
- **Patch**: Clarifications, documentation fixes

Current: **1.0.0**

### Language Version

Each test targets a specific Namel3ss language version:

- **1.0.0**: Initial stable release
- **1.1.0**: Backward-compatible additions
- **2.0.0**: Breaking language changes

Tests are organized by language version:
- `tests/conformance/v1/` → Namel3ss Language 1.0.x
- `tests/conformance/v2/` → Namel3ss Language 2.0.x (future)

## Conformance Requirements

An implementation is **conformant** with Namel3ss Language 1.0 if:

1. It passes all **required** conformance tests in `tests/conformance/v1/`
2. It produces errors for all invalid test cases with matching error codes
3. It produces deterministic outputs for all runtime tests

### Test Categories

- **Required**: Must pass for conformance (majority of tests)
- **Optional**: Implementation-defined features (marked with `optional: true`)
- **Extension**: Implementation-specific extensions (not part of core conformance)

## Running Conformance Tests

### This Implementation

```bash
# Run all conformance tests
namel3ss conformance

# Run specific category
namel3ss conformance --category parse

# Run specific test
namel3ss conformance --test parse-valid-001

# Verbose output
namel3ss conformance --verbose

# CI mode (machine-readable output)
namel3ss conformance --format json
```

### External Implementations

External implementations should:

1. Parse test descriptor YAML files from `tests/conformance/v1/`
2. For each test:
   - Load source files from `fixtures/` or inline content
   - Execute specified phases (parse, typecheck, runtime)
   - Compare actual results with expected outcomes
3. Report pass/fail for each test with detailed diffs on failure
4. Exit with code 0 if all tests pass, non-zero otherwise

### Minimal Conformance Runner (Pseudocode)

```python
def run_conformance_tests(test_dir):
    results = []
    for test_file in find_tests(test_dir, "*.test.yaml"):
        descriptor = load_yaml(test_file)
        
        # Validate descriptor format
        validate_descriptor(descriptor)
        
        # Load source
        source = load_source(descriptor.sources)
        
        # Execute phases
        for phase in descriptor.phases:
            actual = execute_phase(phase, source)
            expected = descriptor.expect[phase]
            
            if not compare(actual, expected):
                results.append(TestFailure(
                    test_id=descriptor.test_id,
                    phase=phase,
                    expected=expected,
                    actual=actual
                ))
            else:
                results.append(TestSuccess(
                    test_id=descriptor.test_id,
                    phase=phase
                ))
    
    return results
```

## Extension Points

Implementations may provide extensions not covered by conformance tests:

1. **Additional diagnostics**: Extra warnings beyond required errors
2. **Optimization levels**: Different compilation strategies
3. **Runtime features**: Implementation-specific runtime behavior
4. **Tooling**: IDE support, debuggers, profilers

Extensions should:
- Not break conformance test expectations
- Be documented as implementation-specific
- Be opt-in (not change default behavior)

## Change Process

Changes to conformance tests follow the RFC process:

1. **Propose**: Submit RFC for language change
2. **Review**: Community and maintainers review
3. **Approve**: RFC approved by language steering group
4. **Implement**: Update reference implementation
5. **Test**: Add/update conformance tests
6. **Release**: New language version released

See `GOVERNANCE.md` for the full process.

## Stability Guarantees

For Namel3ss Language 1.0:

✅ **Stable** (will not change):
- Syntax and grammar
- Core semantics
- Required error codes
- Runtime behavior for conformance tests

⚠️ **May Evolve** (with minor version bump):
- Additional diagnostics (new warnings)
- Performance characteristics
- Tooling and IDE support

❌ **No Guarantee**:
- Internal implementation details
- Optimization strategies
- Error message wording (codes are stable, messages may improve)

## Reference Implementation

The reference implementation is this codebase:
- Repository: `namel3ss-programming-language`
- Branch: `main`
- Language Version: 1.0.0

External implementations should match the behavior defined by:
1. This conformance test suite
2. The language specification documents in `docs/`
3. The reference implementation's behavior on conformance tests

In case of ambiguity, the conformance test suite is authoritative.

---

**Document Status**: Stable  
**Last Updated**: November 24, 2025  
**Next Review**: With any language version change
