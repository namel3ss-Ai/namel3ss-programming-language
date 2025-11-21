# Namel3ss Versioning Policy

This document defines the semantic versioning policy for Namel3ss and the compatibility guarantees you can rely on.

## Overview

Namel3ss follows **Semantic Versioning 2.0.0** (`MAJOR.MINOR.PATCH`) but with specific rules for two distinct levels:

1. **Language Level**: The `.n3` syntax and semantics
2. **Runtime Level**: The generated backend/frontend code and behavior

## Version Format: `MAJOR.MINOR.PATCH`

### MAJOR Version (Breaking Changes)

A MAJOR version increment (e.g., `1.x.x` → `2.0.0`) indicates **breaking changes** that may require you to update your `.n3` files or generated code.

#### Language-Level Breaking Changes

- Removal of syntax constructs (e.g., deprecating a keyword)
- Changes to parsing rules that reject previously valid `.n3` files
- Semantic changes to existing constructs (e.g., `chain` execution order changes)
- Removal of previously supported features without backward compatibility

**Example**: Changing `prompt "name" { template: "..." }` to `prompt("name") { ... }` would be breaking.

#### Runtime-Level Breaking Changes

- Changes to generated API endpoints (e.g., `/api/prompts` → `/v2/prompts`)
- Changes to request/response schemas (field renames, type changes)
- Removal of generated endpoints or features
- Changes to CLI command signatures (e.g., `namel3ss build` → `namel3ss compile`)
- Changes to observability output format (if external tools depend on it)

**Example**: Changing the JSON response from `{"result": "..."}` to `{"data": "..."}` would be breaking.

### MINOR Version (Additive Changes)

A MINOR version increment (e.g., `1.1.x` → `1.2.0`) indicates **new features** that are backward compatible.

#### Language-Level Additive Changes

- New syntax constructs (e.g., adding `dataset` keyword)
- New built-in functions or operators
- New optional parameters to existing constructs
- Performance improvements with no semantic changes

**Example**: Adding `agent "name" { ... }` syntax while keeping `prompt`, `chain` unchanged.

#### Runtime-Level Additive Changes

- New API endpoints (e.g., adding `/api/agents`)
- New optional fields in request/response schemas
- New CLI commands (e.g., `namel3ss doctor`)
- New CLI flags (e.g., `--watch` mode)
- Performance improvements to generated code

**Example**: Adding a new `POST /api/agents/run` endpoint without changing existing endpoints.

### PATCH Version (Bug Fixes)

A PATCH version increment (e.g., `1.1.1` → `1.1.2`) indicates **bug fixes** with no API changes.

#### Language-Level Bug Fixes

- Fixing parser bugs that incorrectly rejected valid syntax
- Fixing semantic analysis bugs (e.g., incorrect type errors)
- Fixing code generation bugs that produced incorrect backend code
- Documentation improvements

**Example**: Fixing a bug where `chain` with nested `{{variables}}` failed to parse.

#### Runtime-Level Bug Fixes

- Fixing bugs in generated code (e.g., incorrect HTTP status codes)
- Fixing CLI bugs (e.g., incorrect error messages)
- Security patches
- Performance bug fixes

**Example**: Fixing a bug where generated FastAPI routes had incorrect CORS headers.

---

## Compatibility Guarantees

### Language Backward Compatibility

Within a MAJOR version (e.g., all `1.x.x` releases):

- **✅ GUARANTEED**: Any `.n3` file that compiled successfully in `1.0.0` will compile successfully in `1.9.0`
- **✅ GUARANTEED**: Semantic behavior of existing constructs remains the same
- **✅ GUARANTEED**: New features are opt-in (require new syntax)

**Exception**: If a construct was documented as "experimental" or "preview", it may change in MINOR versions.

### Runtime Backward Compatibility

Within a MAJOR version:

- **✅ GUARANTEED**: Generated API endpoints remain stable (same paths, same schemas)
- **✅ GUARANTEED**: CLI commands remain stable (same flags, same behavior)
- **✅ GUARANTEED**: Clients built against `1.0.0` backends will work with `1.9.0` backends

**Exception**: New optional fields may be added to responses, but existing fields won't change.

### Forward Compatibility

Forward compatibility is **NOT guaranteed**:

- A `.n3` file using features from `1.5.0` may not compile with `1.3.0`
- Generated code from `1.5.0` may not work with `1.3.0` CLI

**Best Practice**: Pin your Namel3ss version in production and test upgrades in staging.

---

## Deprecation Policy

### Deprecation Timeline

When deprecating a feature:

1. **Announce deprecation** in MINOR version release notes
2. **Add runtime warnings** for deprecated usage (if possible)
3. **Wait at least 2 MINOR versions** before removal
4. **Remove in next MAJOR version**

**Example**:
- `1.2.0`: Deprecate old `connector` syntax, add warnings
- `1.3.0`: Still works but emits warnings
- `1.4.0`: Still works but emits warnings
- `2.0.0`: Old syntax removed, must use new syntax

### Experimental Features

Features marked as **"experimental"** or **"preview"**:

- **MAY change** in MINOR versions without deprecation period
- **SHOULD be documented** with `@experimental` tags
- **SHOULD emit warnings** when used
- **MUST be marked** in release notes

**Example**: New `streaming` feature in `1.5.0` may change in `1.6.0` if marked experimental.

---

## Versioning in Practice

### Version Number in Code

The version number is defined in:

- `pyproject.toml` → `project.version`
- `namel3ss/__init__.py` → `__version__`

Both MUST be kept in sync.

### Release Process

1. **Decide version bump**:
   - Breaking change? → MAJOR
   - New feature? → MINOR
   - Bug fix? → PATCH

2. **Update version**:
   ```bash
   # In pyproject.toml and namel3ss/__init__.py
   # Example: 1.2.3 → 1.3.0 for new feature
   ```

3. **Update CHANGELOG.md**:
   ```markdown
   ## [1.3.0] - 2025-11-21
   
   ### Added
   - New `agent` syntax for multi-step workflows
   
   ### Fixed
   - Fixed parser bug with nested expressions
   ```

4. **Run full test suite**:
   ```bash
   pytest
   pytest tests/test_official_examples.py  # Must be green
   ```

5. **Tag release**:
   ```bash
   git tag -a v1.3.0 -m "Release 1.3.0"
   git push origin v1.3.0
   ```

### CI/CD Enforcement

The CI pipeline MUST:

- ✅ Run all tests on every PR
- ✅ Run `tests/test_official_examples.py` to ensure official examples build
- ✅ Block merge if any tests fail (except `@pytest.mark.xfail`)
- ✅ Run type checking with `mypy`
- ✅ Run linting with `ruff`

---

## Language vs Runtime Versioning

Namel3ss uses a **single version number** for both language and runtime, but tracks them separately in documentation.

### Language Surface Area

**Stable** (covered by backward compatibility guarantees):

- `app`, `page`, `prompt`, `chain`, `agent`, `dataset`, `tool`
- Template syntax: `{{ variable }}`, `{{ expression }}`
- Type system: `string`, `int`, `float`, `bool`, `any`, `list`, `dict`
- Control flow: `if`, `for`, `while`
- Connectors: `database`, `redis`, `vector_store`

**Experimental** (may change in MINOR versions):

- Advanced type inference
- Symbolic execution features
- RLHF training features
- Graph-based workflow syntax

### Runtime Surface Area

**Stable**:

- FastAPI backend generation
- API endpoint structure: `/api/{resource}/run`
- Request/response schemas
- CLI commands: `build`, `run`, `doctor`
- Configuration format

**Experimental**:

- Streaming responses (SSE)
- WebSocket support
- Advanced observability hooks

---

## Breaking Change Examples

### Language-Level Breaking Changes

❌ **Breaking**: Changing syntax
```n3
# 1.x.x
prompt "greeting" { template: "Hello" }

# 2.0.0 (breaking!)
prompt greeting { template: "Hello" }
```

✅ **Non-Breaking**: Adding optional parameter
```n3
# 1.x.x
prompt "greeting" { template: "Hello" }

# 1.5.0 (backward compatible)
prompt "greeting" { 
  template: "Hello"
  cache_ttl: 300  # New optional field
}
```

### Runtime-Level Breaking Changes

❌ **Breaking**: Changing endpoint path
```
# 1.x.x
POST /api/prompts/run

# 2.0.0 (breaking!)
POST /v2/prompts/execute
```

✅ **Non-Breaking**: Adding optional response field
```json
// 1.x.x response
{"result": "Hello"}

// 1.5.0 response (backward compatible)
{"result": "Hello", "metadata": {"tokens": 10}}
```

---

## Testing for Compatibility

### Automated Tests

All releases MUST pass:

1. **Parser tests**: `tests/test_parser*.py`
2. **Codegen tests**: `tests/test_codegen*.py`
3. **Official examples**: `tests/test_official_examples.py`
4. **CLI tests**: `tests/test_cli*.py`

### Manual Testing Checklist

Before MAJOR release:

- [ ] Build all official examples
- [ ] Test generated backends with existing clients
- [ ] Test CLI commands
- [ ] Review deprecation warnings
- [ ] Update migration guide

Before MINOR release:

- [ ] Build all official examples
- [ ] Test new features with backward compatibility
- [ ] Update documentation

Before PATCH release:

- [ ] Build all official examples
- [ ] Verify bug fix doesn't introduce regressions

---

## Questions and Answers

### Q: What if I need to use an experimental feature?

**A**: Pin your Namel3ss version and be prepared to update your code when the feature stabilizes:

```toml
[tool.poetry.dependencies]
namel3ss = "1.5.0"  # Pin exact version
```

### Q: How do I know if a feature is experimental?

**A**: Check the documentation. Experimental features are marked with:

```n3
// @experimental: This syntax may change in future MINOR versions
agent "my_agent" { ... }
```

### Q: Can I use `1.5.0` generated backend with `1.3.0` CLI?

**A**: No, forward compatibility is not guaranteed. Always use matching versions:

```bash
pip install namel3ss==1.5.0  # CLI version
namel3ss build app.n3         # Generates 1.5.0 backend
```

### Q: What happens if I run old generated code with new runtime?

**A**: Within the same MAJOR version, old generated code will work with new runtime. But we recommend regenerating backends after upgrading:

```bash
pip install namel3ss==1.9.0
namel3ss build app.n3  # Regenerate to get bug fixes
```

### Q: How do I migrate to a new MAJOR version?

**A**: Check `MIGRATION_GUIDE.md` for detailed instructions. Generally:

1. Read release notes for breaking changes
2. Update `.n3` syntax (if needed)
3. Regenerate backend/frontend
4. Test thoroughly in staging
5. Update clients (if API changed)

---

## Version History

| Version | Date | Type | Description |
|---------|------|------|-------------|
| 0.4.1 | 2025-11-21 | Current | Pre-1.0 (experimental) |
| 1.0.0 | TBD | Stable | First stable release |

**Note**: Versions before `1.0.0` are considered experimental and do not follow these strict guarantees.

---

## Contact

For questions about versioning policy:

- GitHub Issues: https://github.com/SsebowaDisan/namel3ss-programming-language/issues
- Discussions: https://github.com/SsebowaDisan/namel3ss-programming-language/discussions

---

**Last Updated**: November 21, 2025
**Version**: 1.0 (This Policy)
