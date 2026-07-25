#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
  elif command -v python3.9 >/dev/null 2>&1; then
    PYTHON_BIN="python3.9"
  elif command -v python3.8 >/dev/null 2>&1; then
    PYTHON_BIN="python3.8"
  else
    PYTHON_BIN="python3"
  fi
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

"${PYTHON_BIN}" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 8) or (major, minor) > (3, 10):
    raise SystemExit(
        f"Python {major}.{minor} is not recommended. "
        "Use Python 3.8, 3.9, or 3.10 for TensorFlow 2.9."
    )
PY

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-atena-baselines.txt

python scripts/check_env.py

echo ""
echo "ATENA baseline environment is ready."
echo "Activate it with:"
echo "  source ATENA_dataset/.venv/bin/activate"

