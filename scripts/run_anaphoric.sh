#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/experiments/anaphoric"
echo "Running Anaphoric experiment..."
bash "run_all.sh"
