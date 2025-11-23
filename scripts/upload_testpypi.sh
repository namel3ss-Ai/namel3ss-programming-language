#!/usr/bin/env bash
set -euo pipefail

# Upload built artefacts to TestPyPI.
# Requires environment variables:
#   TESTPYPI_USERNAME (or __token__)
#   TESTPYPI_PASSWORD (API token)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d dist ]]; then
  echo "dist/ not found. Run the build first (e.g., scripts/release_0_5_0.sh)." >&2
  exit 1
fi

if [[ -z "${TESTPYPI_USERNAME:-}" || -z "${TESTPYPI_PASSWORD:-}" ]]; then
  echo "TESTPYPI_USERNAME/TESTPYPI_PASSWORD must be set." >&2
  exit 1
fi

echo "==> Verifying distributions with twine"
python -m twine check dist/*

echo "==> Uploading to TestPyPI"
python -m twine upload \
  --repository-url https://test.pypi.org/legacy/ \
  -u "${TESTPYPI_USERNAME}" \
  -p "${TESTPYPI_PASSWORD}" \
  dist/*

echo "✅ Upload to TestPyPI complete."
