#!/usr/bin/env sh
# Generate demo backend + frontend and print manual smoke-test instructions.

set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ -d .venv ] && [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1090
  . .venv/bin/activate
fi

BACKEND_OUT="backend_out"
SITE_OUT="site_out"

printf '==> Generating backend in %s\n' "$BACKEND_OUT"
namel3ss generate-backend examples/app.n3 "$BACKEND_OUT"

printf '==> Generating frontend in %s\n' "$SITE_OUT"
namel3ss generate-frontend examples/app.n3 "$SITE_OUT"

cat <<EOF

Smoke test ready.
1. In a new terminal run: uvicorn ${BACKEND_OUT}.main:app --reload
2. Open ${SITE_OUT}/index.html in your browser.
3. Confirm pages render, charts/tables show demo data, and widgets/toasts respond.
4. Watch the browser console if realtime is enabled—there should be no errors.

EOF
