# Contributing to Namel3ss

Thank you for your interest in contributing to Namel3ss! This document provides guidelines for contributing to the language specification, reference implementation, and associated tooling.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Types of Contributions](#types-of-contributions)
- [Development Process](#development-process)
- [Conformance Requirements](#conformance-requirements)
- [Language Changes and RFCs](#language-changes-and-rfcs)
- [Code Style and Standards](#code-style-and-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Basic understanding of parsers, compilers, or language design

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/SsebowaDisan/namel3ss-programming-language.git
cd namel3ss-programming-language

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify setup
pytest tests/parser/ tests/security/
namel3ss conformance
```

## Types of Contributions

### 1. Bug Reports

- Check existing issues first
- Provide minimal reproduction case
- Include system information and error messages
- For language-level bugs, specify which conformance tests should fail

### 2. Feature Requests

- Describe the use case clearly
- Explain why this belongs in the language vs. a library
- Consider proposing an RFC for language-level features

### 3. Code Contributions

- Bug fixes in the implementation
- Performance improvements
- Tooling enhancements (CLI, IDE support, etc.)
- Test coverage improvements

### 4. Language-Level Contributions

- Conformance test additions
- Grammar clarifications
- RFC proposals for new features
- Documentation improvements

### 5. Documentation

- Tutorial improvements
- API documentation
- Example code
- Translation (when supported)

## Development Process

### Branch Strategy

- `main`: Stable branch, tagged releases
- Feature branches: `feature/description`
- Bug fixes: `fix/description`
- RFCs: `rfc/nnnn-title`

### Commit Messages

Follow conventional commits:

```
type(scope): brief description

Detailed explanation if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

Examples:
```
feat(parser): add support for multi-line string literals
fix(conformance): handle semantic errors in runner
docs(governance): clarify RFC approval process
test(security): add test for capability validation
```

## Conformance Requirements

**IMPORTANT**: When making changes that affect language behavior, you **must** update or add conformance tests.

### When Conformance Tests Are Required

1. **Implementing accepted RFCs**: Add tests demonstrating the new feature
2. **Fixing language-level bugs**: Add a test that would have caught the bug
3. **Clarifying ambiguous behavior**: Add tests documenting the correct behavior
4. **Parser changes**: Update parse phase tests
5. **Type system changes**: Add type checking phase tests
6. **Runtime behavior changes**: Add runtime phase tests

### Conformance Test Guidelines

Before submitting a PR that changes language behavior:

1. **Run existing tests**: Ensure 100% pass rate maintained
   ```bash
   namel3ss conformance
   ```

2. **Add new tests** for your changes in `tests/conformance/v1/`
   - Follow the format in [tests/conformance/SPEC.md](tests/conformance/SPEC.md)
   - Use descriptive test IDs and names
   - Document what the test covers in the description field

3. **Update test counts** in [CONFORMANCE.md](CONFORMANCE.md)

4. **Link conformance tests in PR**: Mention which tests demonstrate your change

### Example: Adding a Conformance Test

```yaml
# tests/conformance/v1/parse/valid/my_feature.test.yaml
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "parse-valid-031"
category: "parse"
name: "Feature X declaration"
description: |
  Tests parsing of feature X as specified in RFC-0042.
  This ensures all implementations handle the new syntax correctly.

phases:
  - parse

sources:
  - path: "../../fixtures/parse/valid/my_feature.ai"

expect:
  parse:
    status: "success"
```

See [CONFORMANCE.md](CONFORMANCE.md) for complete guidelines.

## Language Changes and RFCs

### What Requires an RFC?

Language-level changes require an RFC (Request for Comments). See [GOVERNANCE.md](GOVERNANCE.md) for the full process.

**Requires RFC:**
- New syntax or grammar rules
- Changes to language semantics
- Breaking changes
- New keywords or operators
- Standard library additions
- Deprecations

**Does NOT require RFC:**
- Bug fixes in implementation
- Performance improvements
- Tooling enhancements
- Documentation updates
- Internal refactoring

### RFC Process Summary

1. **Draft**: Create RFC using template in `rfcs/0000-template.md`
2. **Discussion**: Submit as PR, gather feedback (2-4 weeks)
3. **FCP**: Final Comment Period (10 days)
4. **Decision**: Maintainers accept or reject
5. **Implementation**: Implement with conformance tests

See [rfcs/README.md](rfcs/README.md) for details.

### Linking RFCs to PRs

When implementing an accepted RFC:

```markdown
## Description
Implements RFC-0042: Feature X

## Changes
- Added parser support for feature X
- Updated AST nodes
- Added 3 conformance tests (parse-valid-031, parse-valid-032, parse-invalid-013)

## Conformance Tests
All tests passing (100% pass rate maintained):
- `namel3ss conformance --test parse-valid-031` ✓
- `namel3ss conformance --test parse-valid-032` ✓
- `namel3ss conformance --test parse-invalid-013` ✓

Closes #123
RFC: #456
```

## Code Style and Standards

### Python Code

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use `black` for formatting: `black namel3ss/ tests/`
- Use `mypy` for type checking: `mypy namel3ss/`

### Namel3ss Code Examples

- Use `.ai` file extension
- Follow syntax in [docs/LANGUAGE_REFERENCE.md](docs/LANGUAGE_REFERENCE.md)
- Validate examples parse correctly: `namel3ss parse example.ai`

### Documentation

- Markdown for documentation files
- Include code examples with expected output
- Keep line length under 100 characters for readability

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/parser/
pytest tests/security/
pytest tests/conformance_runner/

# Run conformance tests
namel3ss conformance

# Run with coverage
pytest --cov=namel3ss --cov-report=html
```

### Writing Tests

1. **Unit Tests**: Test individual functions/classes
   ```python
   def test_parse_app_declaration():
       """Test parsing a simple app declaration."""
       source = 'app "TestApp"'
       module = parse_module(source)
       assert len(module.body) == 1
       assert isinstance(module.body[0], App)
   ```

2. **Integration Tests**: Test component interactions
   ```python
   def test_parser_to_codegen_pipeline():
       """Test full pipeline from parsing to code generation."""
       source = 'app "TestApp" { version: "1.0" }'
       # Parse → Resolve → Typecheck → Codegen
   ```

3. **Conformance Tests**: Use YAML descriptors for language-level tests
   - See [CONFORMANCE.md](CONFORMANCE.md) for guidelines

### Test Coverage Requirements

- **New code**: Aim for 80%+ coverage
- **Bug fixes**: Add regression test
- **Language features**: Must have conformance tests
- **Security features**: Must have security tests

## Documentation

### What to Document

- **Public APIs**: All public functions, classes, methods
- **Language features**: User-facing behavior in docs/
- **RFCs**: Rationale and design decisions
- **Examples**: Working code samples

### Documentation Style

```python
def parse_module(source: str, path: str = "module.ai") -> Module:
    """Parse Namel3ss source code into an AST.
    
    Args:
        source: Source code string
        path: File path for error messages (default: "module.ai")
        
    Returns:
        Module: Parsed AST module
        
    Raises:
        N3SyntaxError: If source contains syntax errors
        
    Example:
        >>> source = 'app "MyApp"'
        >>> module = parse_module(source)
        >>> print(module.body[0].name)
        MyApp
    """
```

## Pull Request Process

### Before Submitting

1. **Run tests locally**:
   ```bash
   pytest
   namel3ss conformance
   ```

2. **Check code style**:
   ```bash
   black --check namel3ss/ tests/
   mypy namel3ss/
   ```

3. **Update documentation** if changing public APIs

4. **Add/update conformance tests** for language changes

5. **Update CHANGELOG.md** (if applicable)

### PR Checklist

- [ ] Tests pass locally
- [ ] Conformance tests pass (100% pass rate)
- [ ] New tests added for changes
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] PR description explains the change
- [ ] Related issues linked (Fixes #123)
- [ ] RFC linked if applicable

### PR Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation update
- [ ] Language-level change (requires RFC)

## Changes
- Detailed list of changes

## Conformance Tests
If this affects language behavior:
- List new/updated conformance tests
- Confirm 100% pass rate: `namel3ss conformance`

## Testing
- How was this tested?
- What edge cases were considered?

## Related Issues
Fixes #123
Implements RFC-0042
```

### Review Process

1. **Automated checks**: CI runs tests, linting, conformance
2. **Code review**: Maintainer reviews code quality, design
3. **Conformance check**: Verify language changes have conformance tests
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge for clean history

## Community

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and design discussions
- **Documentation**: Check [docs/](docs/) first

### Communication Guidelines

- Be respectful and constructive
- Search before posting
- Provide context and examples
- Link to relevant code/issues
- Follow up on your issues/PRs

### Recognition

Contributors are recognized in:
- CHANGELOG.md for releases
- GitHub contributors page
- Documentation acknowledgments

## Versioning

See [GOVERNANCE.md](GOVERNANCE.md) for versioning strategy:
- **Language Version**: Namel3ss Language 1.0.0 (semantic versioning)
- **Implementation Version**: namel3ss package version (independent)

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

## Questions?

Open a discussion on GitHub or reach out to maintainers.

---

Thank you for contributing to Namel3ss! 🚀
