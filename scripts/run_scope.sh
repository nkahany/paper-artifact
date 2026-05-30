#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/experiments/scope"
echo "Running Scope experiment..."
bash "runall_scope.sh"
