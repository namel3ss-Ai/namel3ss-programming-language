# Namel3ss Language Governance

## Overview

This document defines the governance model for the Namel3ss AI-native programming language. The governance process ensures that language evolution is transparent, community-driven, and maintains stability for implementers.

## Language vs Implementation Versioning

**Important distinction:**

- **Language Version**: Semantic versioning of the language specification itself (e.g., "Namel3ss Language 1.0.0")
- **Implementation Version**: Version of this particular implementation (namel3ss Python package)

This repository maintains both:
1. The **language specification** (defined through docs/, conformance tests, and RFCs)
2. A **reference implementation** (the namel3ss Python package)

Other implementations may exist with different version numbers while targeting the same language version.

### Current Versions

- **Namel3ss Language**: 1.0.0
- **Reference Implementation (namel3ss package)**: 0.1.0 (pre-release)

## Change Process

### Types of Changes

#### 1. Language-Level Changes
Changes that affect the **language specification** and require updates to all conforming implementations:

- New syntax or grammar rules
- Changes to semantics or behavior
- New standard library features
- Breaking changes to existing features

**Process:** Requires an RFC (see RFC Process below)

#### 2. Implementation-Only Changes
Changes that only affect this implementation without changing the language:

- Performance optimizations
- Internal refactoring
- Bug fixes that bring implementation into conformance
- Tooling improvements (CLI, IDE support, etc.)
- Non-breaking additions (new CLI flags, etc.)

**Process:** Standard pull request review (no RFC needed)

## RFC Process

### What Requires an RFC?

An RFC (Request for Comments) is required for:

- New language features or syntax
- Breaking changes to existing features
- Changes to language semantics
- Deprecation of language features
- Major additions to the standard library
- Changes to the conformance test specification format

### RFC Lifecycle

#### 1. **Draft**
- Create a new RFC using the template (`rfcs/0000-template.md`)
- Number it sequentially (check existing RFCs)
- Submit as a pull request with title "RFC XXXX: [Title]"
- Label: `rfc-draft`

#### 2. **Discussion**
- Community provides feedback via PR comments
- Author iterates on the RFC based on feedback
- Duration: Minimum 2 weeks for minor changes, 4 weeks for major changes

#### 3. **Final Comment Period (FCP)**
- When consensus is reached, maintainers initiate FCP
- Label: `rfc-fcp`
- Duration: 10 days minimum
- Last chance for concerns to be raised

#### 4. **Accepted / Rejected**
- Maintainers make final decision based on community consensus
- Accepted RFCs are merged and become part of the specification
- Rejected RFCs are closed with explanation
- Labels: `rfc-accepted` or `rfc-rejected`

#### 5. **Implementation**
- Accepted RFCs must be implemented before next language version release
- Implementation tracked via issues linked to the RFC
- Conformance tests must be added/updated

### RFC Template

See `rfcs/0000-template.md` for the standard RFC structure.

## Conformance Requirements

### For Language Changes

When a language-level change is accepted:

1. **Update Documentation**
   - Modify `docs/LANGUAGE_REFERENCE.md`
   - Update grammar in `docs/GRAMMAR.md`
   - Add examples to `docs/EXAMPLES_OVERVIEW.md`

2. **Update Conformance Tests**
   - Add new tests to `tests/conformance/v1/` covering the feature
   - Update existing tests if behavior changes
   - Ensure 100% pass rate before merging

3. **Implement in Reference**
   - Update parser, AST, compiler, or runtime as needed
   - Ensure all conformance tests pass
   - Document any implementation-specific behavior

4. **Version Bump**
   - Major breaking change → increment language MAJOR version
   - New feature (backwards compatible) → increment language MINOR version
   - Clarification/bugfix → increment language PATCH version

### Conformance Test Authority

The conformance test suite (`tests/conformance/`) is the **authoritative specification** of language behavior. If documentation and conformance tests disagree, the conformance tests are correct.

External implementations should:
- Run the conformance suite regularly
- Report failures as potential bugs or spec ambiguities
- Contribute new conformance tests for edge cases

## Versioning Strategy

### Language Versioning (SemVer)

**MAJOR.MINOR.PATCH** (e.g., 1.2.3)

- **MAJOR**: Breaking changes that require code updates
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, clarifications, no behavioral changes

### Implementation Versioning

The reference implementation follows its own versioning independent of the language version, but documents which language version it targets in `pyproject.toml`:

```toml
[tool.namel3ss]
language_version = "1.0.0"
```

### Conformance Test Versioning

Conformance tests are versioned with the language:
- `tests/conformance/v1/` → Namel3ss Language 1.x.x
- `tests/conformance/v2/` → Namel3ss Language 2.x.x (future)

Each test descriptor specifies:
```yaml
spec_version: "1.0.0"  # Conformance test format version
language_version: "1.0.0"  # Language version being tested
```

## Decision-Making Authority

### Maintainers

The core maintainer team has final decision-making authority on:
- RFC acceptance/rejection
- Language version bumps
- Breaking changes

Current maintainers are listed in `MAINTAINERS.md`.

### Community Input

The community (users, implementers, contributors) provides:
- RFC proposals and feedback
- Conformance test contributions
- Bug reports and clarification requests
- Use case examples and requirements

Maintainers prioritize community consensus but may make final calls on contentious issues.

## Stability Guarantees

### For Language 1.x

Within a major version (e.g., 1.x), we guarantee:

- **Backwards compatibility**: Code that works in 1.0 will work in 1.x
- **Conformance test stability**: Tests in `v1/` won't have their behavior changed (only added to)
- **Deprecation policy**: Features can be deprecated but not removed until 2.0

### Deprecation Process

To deprecate a language feature:

1. RFC proposing deprecation
2. Mark as deprecated in docs
3. Add deprecation warnings in reference implementation
4. Remove in next MAJOR version (2.0)

Minimum deprecation period: **1 minor version** (e.g., deprecate in 1.5, remove in 2.0)

## Conformance Levels

Implementations may declare conformance levels:

- **Full Conformance**: Passes 100% of conformance suite
- **Partial Conformance**: Passes subset, with documented exceptions
- **Profile Conformance**: Implements a defined subset (e.g., "Namel3ss Core" without RLHF)

External implementations should document:
- Target language version
- Conformance level
- Known deviations or unimplemented features

## Reference Documentation

- **Language Specification**: `docs/LANGUAGE_REFERENCE.md`
- **Conformance Tests**: `tests/conformance/` (authoritative spec)
- **Conformance Guide**: `CONFORMANCE.md`
- **RFC Archive**: `rfcs/`
- **Grammar**: `docs/GRAMMAR.md`

## Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, design discussions
- **Pull Requests**: Code contributions, RFC proposals

## Amendment Process

This governance document itself can be amended via the RFC process. Proposed changes should be labeled `rfc-governance`.

---

**Last Updated**: 2024-01
**Language Version**: 1.0.0
