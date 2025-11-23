#!/usr/bin/env bash
set -euo pipefail

# Release helper for namel3ss 0.5.0
# - Cleans build artefacts
# - Runs test suite
# - Builds wheel and sdist

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Cleaning previous artefacts"
rm -rf "$ROOT/build" "$ROOT/dist" "$ROOT"/namel3ss.egg-info "$ROOT"/.eggs

echo "==> Running tests"
python -m pytest

echo "==> Building wheel and sdist"
python -m build

echo "✅ Release build completed. Artefacts in dist/"
