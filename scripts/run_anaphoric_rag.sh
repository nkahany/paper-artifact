#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/experiments/anaphoric_rag"
echo "Running Anaphoric Rag experiment..."
bash "run_complete_pipeline.sh"
