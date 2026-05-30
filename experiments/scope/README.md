# Scope Ambiguity

Uses `scope_complete_system.py` with zero-shot, one-shot, few-shot, and optional RAG modes.

Required data:
- `exp1b_base_dataset.csv` for base/RAG examples
- `scope_fixed.csv` for testing and evaluation
- `srs_documents/` for RAG context

Run from this folder:

```bash
bash runall_scope.sh
```

Expected outputs from the uploaded run are stored in `../../results/expected/scope/`.
