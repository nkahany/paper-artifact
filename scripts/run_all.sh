#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash scripts/run_anaphoric.sh
bash scripts/run_anaphoric_rag.sh
bash scripts/run_scope.sh
bash scripts/run_structural.sh
bash scripts/run_vagueness.sh
