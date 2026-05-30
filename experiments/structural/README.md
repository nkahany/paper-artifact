# Structural Ambiguity

Uses `structural.py` with zero-shot, one-shot, few-shot, and optional RAG modes.

Required data:
- `Structural.csv`
- `structural_rag.csv` and `structural_test.csv` when using the pre-split data
- `srs_documents/` for RAG context

Run from this folder:

```bash
bash runall.sh
```

Expected reports from the uploaded runs are stored in `../../results/expected/structural/`.
