# Namel3ss SDK - Build & Release Guide

This document describes how to build and publish the **namel3ss-sdk** package to PyPI independently from the core `namel3ss` language package.

## Package Structure

The Namel3ss monorepo contains two independently versioned Python packages:

1. **namel3ss** (core language) - Built from repo root
2. **namel3ss-sdk** (client SDK) - Built from `namel3ss_sdk/` subdirectory

Each package has its own `pyproject.toml`, versioning, and release cadence.

## SDK Version Strategy

The SDK follows **Semantic Versioning 2.0.0** independently from the core package:

- **MAJOR**: Breaking changes to SDK API (method signatures, configuration schema)
- **MINOR**: New features (new API methods, configuration options)
- **PATCH**: Bug fixes, documentation, internal improvements

**Example**: Core `namel3ss` at v0.5.0, SDK `namel3ss-sdk` at v0.2.0 is valid and expected.

## Pre-Release Checklist

Before releasing a new SDK version:

1. **Version Synchronization**:
   - [ ] `namel3ss_sdk/pyproject.toml` → `[project].version` updated
   - [ ] `namel3ss_sdk/__init__.py` → `__version__` updated
   - [ ] `namel3ss_sdk/CHANGELOG.md` → new version entry with today's date

2. **Code Quality**:
   - [ ] SDK tests pass (if present): `cd namel3ss_sdk && pytest`
   - [ ] Type checking: `mypy namel3ss_sdk`
   - [ ] Linting: `ruff check namel3ss_sdk`

3. **Packaging Validation**:
   ```bash
   cd namel3ss_sdk
   python -m build
   twine check dist/*
   ```

4. **Documentation**:
   - [ ] `README.md` reflects SDK features accurately
   - [ ] Usage examples match actual API
   - [ ] No placeholder URLs or fake endpoints

## Build Process

### Building SDK Only

```bash
# Navigate to SDK directory
cd namel3ss_sdk

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build wheel and source distribution
python -m build

# Verify package contents
twine check dist/*
```

**Expected Output**:
```
Successfully built namel3ss-sdk-<VERSION>.tar.gz and namel3ss-sdk-<VERSION>-py3-none-any.whl
```

### Building Both Packages

To release both core and SDK together (rare, but possible):

```bash
# Build core package from root
python -m build
twine check dist/*

# Build SDK package
cd namel3ss_sdk
python -m build
twine check dist/*
```

## Release Workflow

### 1. Update Version and Changelog

```bash
# Edit version in both locations
vim namel3ss_sdk/pyproject.toml    # [project].version
vim namel3ss_sdk/__init__.py        # __version__
vim namel3ss_sdk/CHANGELOG.md       # Add entry for new version

# Commit changes
git add namel3ss_sdk/pyproject.toml namel3ss_sdk/__init__.py namel3ss_sdk/CHANGELOG.md
git commit -m "Release namel3ss-sdk v<VERSION>

- Bump SDK version to <VERSION>
- Update SDK changelog
- [Brief description of changes]
"
```

### 2. Create SDK Tag

SDK tags use the format `sdk-v<VERSION>` to distinguish from core package tags:

```bash
# Create annotated tag
git tag -a sdk-v<VERSION> -m "Release namel3ss-sdk v<VERSION>"

# Example
git tag -a sdk-v0.2.0 -m "Release namel3ss-sdk v0.2.0"
```

### 3. Push Changes

```bash
git push origin main
git push origin sdk-v<VERSION>
```

### 4. Build SDK Distribution

```bash
cd namel3ss_sdk

# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build
python -m build

# Validate
twine check dist/*
ls -lh dist/
```

### 5. Publish to PyPI

**Test PyPI (recommended first)**:
```bash
# From namel3ss_sdk/ directory
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ namel3ss-sdk
```

**Production PyPI**:
```bash
# From namel3ss_sdk/ directory
twine upload dist/*
```

### 6. Verify Installation

```bash
# Fresh environment
python -m venv test_sdk_env
source test_sdk_env/bin/activate

# Install from PyPI
pip install namel3ss-sdk

# Verify version
python -c "import namel3ss_sdk; print(namel3ss_sdk.__version__)"

# Basic smoke test
python -c "from namel3ss_sdk import N3Client, N3InProcessRuntime; print('SDK imports OK')"

# Cleanup
deactivate
rm -rf test_sdk_env
```

### 7. Create GitHub Release

1. Go to https://github.com/SsebowaDisan/namel3ss-programming-language/releases
2. Click "Draft a new release"
3. Select tag `sdk-v<VERSION>`
4. Title: "Namel3ss SDK v<VERSION>"
5. Copy changelog entry from `namel3ss_sdk/CHANGELOG.md`
6. Attach `dist/namel3ss-sdk-<VERSION>*` files
7. Mark as "Pre-release" if beta, otherwise publish as latest

## CI/CD Integration

### GitHub Actions Workflow Example

Create `.github/workflows/release-sdk.yml`:

```yaml
name: Release SDK

on:
  push:
    tags:
      - 'sdk-v*'

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install build tools
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      
      - name: Build SDK
        run: |
          cd namel3ss_sdk
          python -m build
      
      - name: Check package
        run: |
          cd namel3ss_sdk
          twine check dist/*
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_SDK_TOKEN }}
        run: |
          cd namel3ss_sdk
          twine upload dist/*
      
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: namel3ss_sdk/dist/*
          body_path: namel3ss_sdk/CHANGELOG.md
```

**Setup Requirements**:
1. Add PyPI API token as `PYPI_SDK_TOKEN` secret in GitHub
2. Ensure token has permissions for `namel3ss-sdk` package

## Independent Release Cadence

The SDK can be released independently from the core package:

- **SDK releases** address client-side features, bug fixes, or integrations
- **Core releases** address language features, compiler, runtime

**Common Scenarios**:

1. **SDK patch release** (bug fix):
   ```bash
   # 0.2.0 → 0.2.1
   # Fix retry logic bug, no core changes
   cd namel3ss_sdk && python -m build && twine upload dist/*
   ```

2. **Core major release** (language breaking change):
   ```bash
   # Core: 0.5.0 → 1.0.0
   # SDK: 0.2.0 → 1.0.0 (if client API changes)
   # OR SDK: 0.2.0 → 0.3.0 (if only adding support, backward compat)
   ```

3. **SDK feature release** (new API):
   ```bash
   # 0.2.0 → 0.3.0
   # Add streaming support, no core changes needed
   cd namel3ss_sdk && python -m build && twine upload dist/*
   ```

## Version Compatibility

Document SDK compatibility with core package versions:

| SDK Version | Core Namel3ss | Notes |
|-------------|---------------|-------|
| 0.2.x       | 0.4.x - 0.5.x | Full compatibility |
| 0.1.x       | 0.3.x - 0.4.x | Initial release |

Update this table in SDK README.md when compatibility changes.

## Troubleshooting

### Build Fails

**Issue**: `ModuleNotFoundError` during build
```bash
# Solution: Ensure working directory is namel3ss_sdk/
cd namel3ss_sdk
python -m build
```

**Issue**: `Package 'namel3ss_sdk' not found`
```bash
# Solution: Check [tool.setuptools.packages.find] in pyproject.toml
# Should have: where = ["."], include = ["namel3ss_sdk*"]
```

### Twine Upload Fails

**Issue**: "File already exists"
```bash
# Solution: Version already published, increment version
vim namel3ss_sdk/pyproject.toml  # Bump version
vim namel3ss_sdk/__init__.py
```

**Issue**: "Invalid credentials"
```bash
# Solution: Check ~/.pypirc or use API token
twine upload --username __token__ --password <token> dist/*
```

### Version Mismatch

**Issue**: `import namel3ss_sdk; print(__version__)` shows old version

```bash
# Solution: Force reinstall
pip uninstall namel3ss-sdk -y
pip install namel3ss-sdk --no-cache-dir
```

## Post-Release

- [ ] Announce SDK release on project channels
- [ ] Update main README.md if SDK adds significant new capabilities
- [ ] Update SDK compatibility table in documentation
- [ ] Close related issues/PRs in GitHub
- [ ] Consider updating examples in `examples/` if SDK API changed

## Rollback

If SDK release has critical issues:

1. **Yank from PyPI** (requires maintainer access):
   ```bash
   # Does not delete, marks as unavailable
   pip install twine
   twine yank namel3ss-sdk <VERSION>
   ```

2. **Release patch version**:
   ```bash
   # 0.2.0 (broken) → 0.2.1 (fixed)
   cd namel3ss_sdk
   # Fix issue, bump version, release
   ```

3. **Delete Git tag** (if needed):
   ```bash
   git push --delete origin sdk-v<VERSION>
   git tag -d sdk-v<VERSION>
   ```

---

**Last Updated**: 2025-11-22  
**Current SDK Version**: 0.2.0  
**Current Core Version**: 0.5.0
