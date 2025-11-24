# Namel3ss Conformance Test Suite

The Namel3ss conformance test suite provides a machine-readable, implementation-agnostic specification of the Namel3ss language behavior. These tests serve as the **authoritative source of truth** for language semantics.

## Purpose

The conformance suite enables:

1. **Multiple Implementations**: External implementations can verify compliance with the language spec
2. **Regression Testing**: Prevent breaking changes to language behavior
3. **Documentation**: Executable examples of correct and incorrect usage
4. **Specification Clarity**: Resolve ambiguities in prose documentation

## Running Conformance Tests

### Using the Reference Implementation

```bash
# Run all conformance tests
namel3ss conformance

# Run tests for a specific category
namel3ss conformance --category parse
namel3ss conformance --category types
namel3ss conformance --category runtime

# Run a specific test
namel3ss conformance --test parse-valid-001

# Verbose output with details
namel3ss conformance --category parse --verbose

# JSON output for CI/automation
namel3ss conformance --format json > results.json
```

### Exit Codes

- `0`: All tests passed
- `1`: One or more tests failed
- `2`: Errors during test execution

## Test Structure

### Directory Layout

```
tests/conformance/v1/
├── parse/                  # Parsing phase tests
│   ├── valid/             # Valid programs that should parse
│   └── invalid/           # Invalid programs that should error
├── types/                  # Type checking phase tests
│   ├── valid/
│   └── invalid/
├── runtime/                # Runtime execution tests
└── fixtures/               # Actual .ai source files
    ├── parse/
    │   ├── valid/
    │   └── invalid/
    ├── types/
    │   ├── valid/
    │   └── invalid/
    └── runtime/
```

### Test Descriptor Format

Each test consists of two files:

1. **Source file** (`fixtures/category/type/example.ai`): The Namel3ss code to test
2. **Test descriptor** (`category/type/example.test.yaml`): Metadata and expectations

Example test descriptor:

```yaml
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "parse-valid-001"
category: "parse"
name: "Empty app declaration"
description: |
  Tests parsing of a minimal valid application with no declarations.
  This is the simplest possible valid Namel3ss program.

phases:
  - parse

sources:
  - path: "../../fixtures/parse/valid/empty_app.ai"

expect:
  parse:
    status: "success"
```

Example with error expectations:

```yaml
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "parse-invalid-001"
category: "parse"
name: "Missing closing brace"
description: |
  Tests that unclosed braces are detected as syntax errors.

phases:
  - parse

sources:
  - path: "../../fixtures/parse/invalid/missing_brace.ai"

expect:
  parse:
    status: "error"
```

## Test Categories

### Parse Phase (`category: parse`)

Tests that source code is correctly parsed into an AST:

- **Valid**: Programs that should parse without errors
- **Invalid**: Programs with syntax errors

**Examples:**
- `parse-valid-001`: Empty app declaration
- `parse-valid-002`: App with LLM and agent
- `parse-invalid-001`: Missing closing brace
- `parse-invalid-003`: Unquoted app name

**Coverage:** Apps, agents, LLMs, tools, prompts, memory, imports, modules, comments, strings, lists

### Types Phase (`category: types`)

Tests type checking and validation:

- **Valid**: Well-typed programs
- **Invalid**: Type errors that should be caught

**Examples:**
- Type inference for variables
- Agent llm reference validation
- Tool parameter type checking
- Import resolution

### Runtime Phase (`category: runtime`)

Tests execution behavior:

- **Valid**: Programs that execute correctly with expected output
- **Invalid**: Programs that should fail at runtime

**Examples:**
- Agent invocation and response
- Tool execution
- Memory operations
- Error handling

## Test Phases

Each test descriptor specifies one or more phases:

```yaml
phases:
  - parse       # Parse source to AST
  - resolve     # Resolve imports and references
  - typecheck   # Type check the program
  - codegen     # Generate backend code
  - runtime     # Execute and check output
```

Phases are executed in order. If a phase fails and was expected to succeed, subsequent phases are skipped.

## Expectations

### Parse Phase Expectations

```yaml
expect:
  parse:
    status: "success"  # or "error"
```

For error cases, optionally specify diagnostics:

```yaml
expect:
  parse:
    status: "error"
    diagnostics:
      - severity: "error"
        message_pattern: ".*missing.*brace.*"
```

### Type Phase Expectations

```yaml
expect:
  typecheck:
    status: "success"  # or "error"
    diagnostics:
      - severity: "error"
        message_pattern: ".*undefined.*variable.*"
        line: 5
```

### Runtime Phase Expectations

```yaml
expect:
  runtime:
    status: "success"
    output:
      stdout: "Expected output text"
      # or pattern matching
      stdout_pattern: "Hello, .+"
    exit_code: 0
```

## For External Implementers

### Getting Started

1. **Clone the test suite**:
   ```bash
   git clone https://github.com/your-org/namel3ss.git
   cd namel3ss/tests/conformance/v1
   ```

2. **Implement a test runner** for your language/platform that:
   - Reads `.test.yaml` descriptors
   - Loads source files from `fixtures/`
   - Executes tests using your implementation
   - Compares results with expectations

3. **Run the tests** and report pass rate

### Test Runner Requirements

A conforming test runner must:

- Support all descriptor fields defined in `tests/conformance/SPEC.md`
- Execute phases in order (parse → resolve → typecheck → codegen → runtime)
- Stop phase execution on first failure (unless `continue_on_error` is true)
- Report results with test IDs, pass/fail status, and any error messages

### Reporting Conformance

When claiming conformance, document:

- **Language version**: Which Namel3ss language version you target
- **Pass rate**: What % of tests pass (by category if not 100%)
- **Known deviations**: Any intentional differences from spec
- **Platform/environment**: OS, runtime version, dependencies

Example:

> **MyNamel3ss Implementation v0.5.0**
> - Target: Namel3ss Language 1.0.0
> - Parse tests: 30/30 (100%)
> - Type tests: 18/20 (90%) - missing generic type inference
> - Runtime tests: 25/30 (83%) - async execution not implemented
> - Platform: Node.js 18+, Linux/macOS/Windows

### Contributing Tests

External implementers are encouraged to contribute tests for:

- Edge cases not yet covered
- Platform-specific behavior
- Performance benchmarks
- Real-world usage patterns

Submit tests via pull request following the existing format.

## Test Maintenance

### Adding New Tests

1. Create source file in `fixtures/category/type/`
2. Create test descriptor in `category/type/`
3. Ensure test passes with reference implementation
4. Document what language feature is being tested

### Updating Tests

When the language spec changes (via RFC):

- Add new tests for new features
- Update test expectations if behavior changes
- Mark deprecated features with metadata
- Never delete tests (archive to `deprecated/` instead)

### Test Stability

Within a language major version (e.g., 1.x):

- Existing tests won't change behavior
- New tests may be added
- Tests may be marked as deprecated but not removed
- Test descriptor format is stable

## Reference Documentation

- **Test Specification**: `tests/conformance/SPEC.md` - Complete descriptor format
- **Governance**: `GOVERNANCE.md` - How language changes are made
- **Language Reference**: `docs/LANGUAGE_REFERENCE.md` - Prose description of language
- **Grammar**: `docs/GRAMMAR.md` - Formal grammar specification

## FAQ

### Q: What if a test fails in my implementation?

**A**: This indicates either:
1. A bug in your implementation (most common)
2. Ambiguity in the specification (file an issue)
3. Platform-specific difference (document as known deviation)

### Q: Can I skip certain test categories?

**A**: Yes, but document which categories you skip and why. Partial conformance is acceptable for early implementations.

### Q: How do I propose a new test?

**A**: Submit a pull request with:
- The source file (`fixtures/`)
- The test descriptor (`.test.yaml`)
- Verification that the reference implementation passes it
- Brief explanation of what it tests

### Q: Are performance tests included?

**A**: The main conformance suite focuses on correctness, not performance. Performance benchmarks are in a separate `benchmarks/` directory.

### Q: What about tests for implementation-specific features?

**A**: Conformance tests only cover the language specification. Implementation-specific features should have separate test suites.

## Current Test Coverage

**Language Version 1.0.0:**

- **Parse Phase**: 30 tests
  - Valid: 18 tests (apps, agents, LLMs, tools, prompts, memory, imports, modules)
  - Invalid: 12 tests (syntax errors, malformed declarations)
  - **Coverage**: 100% pass rate

- **Type Phase**: Coming soon

- **Runtime Phase**: Coming soon

**Total**: 30 conformance tests

---

**Conformance Suite Version**: 1.0.0  
**Language Version**: 1.0.0  
**Last Updated**: 2024-01
