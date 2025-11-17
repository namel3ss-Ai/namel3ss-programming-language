#!/usr/bin/env bash
# Wrapper that ensures the virtualenv is ready and executes the full test suite.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="$ROOT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  printf '==> Creating virtualenv at %s\n' "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Virtualenv activation script missing: $VENV_DIR/bin/activate" >&2
  exit 1
fi

# shellcheck disable=SC1090
. "$VENV_DIR/bin/activate"

PYTHON_BIN="${PYTHON_BIN:-python}"

printf '==> Ensuring pip is up to date\n'
$PYTHON_BIN -m pip install --upgrade pip >/dev/null

printf '==> Installing project dependencies\n'
$PYTHON_BIN -m pip install -e ".[dev]"

if [ "$#" -eq 0 ]; then
  set -- --maxfail=1 --disable-warnings --cov=namel3ss --cov-report=term-missing --cov-report=xml
fi

printf '==> Running pytest %s\n' "$*"
$PYTHON_BIN -m pytest "$@"
