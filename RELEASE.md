# Release Process for Namel3ss

This document describes the steps to publish a new version of Namel3ss to PyPI.

## Pre-Release Checklist

Before creating a release, ensure:

1. **Version Synchronization**:
   - [ ] `pyproject.toml` → `[project].version` updated
   - [ ] `namel3ss/__init__.py` → fallback `__version__` updated
   - [ ] `CHANGELOG.md` → new version entry added with today's date

2. **Version Selection** (per `VERSIONING.md`):
   - **MAJOR** bump for breaking changes (language syntax, runtime APIs, CLI signatures)
   - **MINOR** bump for new backward-compatible features
   - **PATCH** bump for bug fixes only

3. **Code Quality**:
   - [ ] All tests passing: `pytest`
   - [ ] No linting errors: `ruff check .`
   - [ ] Type checking clean: `mypy namel3ss`

4. **Packaging Validation**:
   ```bash
   python -m build
   twine check dist/*
   ```
   - [ ] Build succeeds without errors
   - [ ] Twine check passes for both wheel and sdist

5. **Documentation**:
   - [ ] README.md reflects new features
   - [ ] CHANGELOG.md entry is complete and accurate
   - [ ] No placeholder or demo content in docs

## Release Steps

### 1. Final Commit

```bash
git add pyproject.toml namel3ss/__init__.py CHANGELOG.md
git commit -m "Release v<VERSION>

- Bump version to <VERSION>
- Update changelog with release notes
- Synchronize version strings across package
"
```

### 2. Create Annotated Tag

```bash
# Format: v<MAJOR>.<MINOR>.<PATCH>
git tag -a v<VERSION> -m "Release v<VERSION>"
```

**Example**:
```bash
git tag -a v0.5.0 -m "Release v0.5.0"
```

### 3. Push to GitHub

```bash
git push origin main
git push origin v<VERSION>
```

### 4. Build Distribution Packages

```bash
# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build wheel and source distribution
python -m build

# Verify contents
twine check dist/*
```

### 5. Upload to PyPI

**Test PyPI** (recommended first):
```bash
twine upload --repository testpypi dist/*
```

**Production PyPI**:
```bash
twine upload dist/*
```

### 6. Verify Installation

```bash
# Create fresh virtual environment
python -m venv test_env
source test_env/bin/activate

# Install from PyPI
pip install namel3ss

# Verify version
namel3ss --version

# Run basic smoke test
echo 'app "test".' > test.n3
namel3ss build test.n3

# Cleanup
deactivate
rm -rf test_env test.n3
```

### 7. GitHub Release

1. Go to https://github.com/SsebowaDisan/namel3ss-programming-language/releases
2. Click "Draft a new release"
3. Select the tag `v<VERSION>`
4. Copy the changelog entry from `CHANGELOG.md` as release notes
5. Attach the `dist/*.whl` and `dist/*.tar.gz` files
6. Publish release

## Post-Release

- [ ] Announce release on project channels
- [ ] Update any dependent documentation or examples
- [ ] Close milestone (if using GitHub milestones)
- [ ] Update project board or tracking system

## Rollback Procedure

If a release has critical issues:

1. **PyPI**: Cannot delete releases, but can yank them:
   ```bash
   # Requires PyPI maintainer access
   twine yank namel3ss <VERSION>
   ```

2. **GitHub**: Delete the release and tag:
   ```bash
   git push --delete origin v<VERSION>
   git tag -d v<VERSION>
   ```

3. **Immediate Fix**: Release a patch version (e.g., `0.5.0` → `0.5.1`)

## Troubleshooting

### Build Fails

- Check `pyproject.toml` syntax: `python -c "import tomli; tomli.load(open('pyproject.toml', 'rb'))"`
- Ensure all package data files are in `MANIFEST.in`
- Verify `setuptools>=68` is installed

### Twine Check Fails

- Ensure `README.md` exists and is valid markdown
- Check all URLs in `[project.urls]` are accessible
- Verify license field format matches PEP 639

### PyPI Upload Fails

- Verify credentials: `~/.pypirc` or use API token
- Check version doesn't already exist on PyPI
- Ensure package name is available (first release only)

### Version Mismatch

If `namel3ss --version` shows wrong version after release:
- Check `pyproject.toml` was included in source distribution
- Verify `_local_version()` function in `__init__.py` works
- Reinstall with `pip install --force-reinstall namel3ss`

## Versioning Philosophy

Namel3ss follows **Semantic Versioning 2.0.0** strictly:

- **MAJOR**: Breaking changes to `.n3` syntax, generated APIs, or CLI
- **MINOR**: New features, backward-compatible additions
- **PATCH**: Bug fixes, documentation, internal improvements

See `VERSIONING.md` for detailed compatibility guarantees.

---

**Last Updated**: 2025-11-22  
**Current Version**: 0.5.0
