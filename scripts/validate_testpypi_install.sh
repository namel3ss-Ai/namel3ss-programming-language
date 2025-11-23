#!/usr/bin/env bash
set -euo pipefail

# Validate installation from TestPyPI and run CLI smoke tests.
# Requires env:
#   TESTPYPI_INDEX (optional override; default https://test.pypi.org/simple)
#   EXTRA_INDEX_URL (optional; default https://pypi.org/simple)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_DIR=".venv_release_0_5_0"
TESTPYPI_INDEX_URL="${TESTPYPI_INDEX:-https://test.pypi.org/simple}"
EXTRA_INDEX_URL="${EXTRA_INDEX_URL:-https://pypi.org/simple}"

echo "==> Creating clean virtualenv at ${VENV_DIR}"
python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip

echo "==> Installing namel3ss==0.5.0 from TestPyPI"
python -m pip install --index-url "${TESTPYPI_INDEX_URL}" --extra-index-url "${EXTRA_INDEX_URL}" "namel3ss==0.5.0"

echo "==> Verifying version"
python - <<'PY'
import namel3ss
assert namel3ss.__version__ == "0.5.0", namel3ss.__version__
print("namel3ss version:", namel3ss.__version__)
PY

echo "==> Running CLI smoke: build (production_app.n3)"
namel3ss build examples/production_app.n3 --build-backend -o /tmp/n3_build_prod

echo "==> Running CLI smoke: run (production_app.n3, short-lived)"
timeout 15s namel3ss run examples/production_app.n3 --host 127.0.0.1 --port 8001 || true

echo "==> Running CLI smoke: test (production_app.n3)"
namel3ss test examples/production_app.n3

echo "==> Running CLI smoke: build (agent_app.n3)"
namel3ss build examples/agent_app.n3 --build-backend -o /tmp/n3_build_agent

echo "==> Running CLI smoke: run (agent_app.n3, short-lived)"
timeout 15s namel3ss run examples/agent_app.n3 --host 127.0.0.1 --port 8002 || true

echo "==> Running CLI smoke: test (agent_app.n3)"
namel3ss test examples/agent_app.n3

echo "✅ Validation complete."
