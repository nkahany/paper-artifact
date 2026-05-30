#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python scripts/check_environment.py
python -m compileall -q experiments scripts
printf "Smoke test passed. Python files compile and required files exist.
"
