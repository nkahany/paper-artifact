#!/usr/bin/env bash
# Reproduce all 24 experiments from the paper:
#   4 ambiguity types x 3 prompting strategies x 2 (with / without RAG)
set -euo pipefail

cd "$(dirname "$0")"

for TYPE in anaphoric structural vagueness scope; do
  for APPROACH in zero-shot one-shot few-shot; do
    echo "=== ${TYPE} / ${APPROACH} (no RAG) ==="
    python "${TYPE}.py" --approach "${APPROACH}"

    echo "=== ${TYPE} / ${APPROACH} (+ RAG) ==="
    python "${TYPE}.py" --approach "${APPROACH}" --use-rag
  done
done
