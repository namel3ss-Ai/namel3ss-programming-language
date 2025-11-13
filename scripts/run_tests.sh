#!/usr/bin/env sh
# Wrapper that ensures the virtualenv is ready and executes the full test suite.

set -eu

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

printf '==> Installing project dependencies\n'
pip install -e .

if [ -f requirements-dev.txt ]; then
  pip install -r requirements-dev.txt
fi

printf '==> Running pytest %s\n' "$*"
if [ "$#" -gt 0 ]; then
  python -m pytest "$@"
else
  python -m pytest
fi
