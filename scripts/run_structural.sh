#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/experiments/structural"
echo "Running Structural experiment..."
bash "runall.sh"
