# Anaphoric Ambiguity + RAG

Uses `anaphoric_rag.py` with zero-shot, one-shot, few-shot, and RAG-enhanced prompting.

Required data:
- `ground_truths.csv`
- `srs_documents/` for retrieval context

Run from this folder:

```bash
bash run_complete_pipeline.sh
```

Expected outputs from the uploaded RAG run are stored in `../../results/expected/anaphoric_rag/`.
