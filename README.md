# Ambiguity Resolution Artifact

This repository contains the cleaned implementation, datasets, and expected outputs for ambiguity-resolution experiments over software requirements.

The updated version includes the new data folders for anaphoric ambiguity and RAG, in addition to the original scope, structural, and vagueness experiments.

## Tasks included


1. **Anaphoric ambiguity resolution**
2. **Scope ambiguity resolution**
3. **Structural ambiguity resolution**
4. **Vagueness ambiguity resolution**

The experiments evaluate zero-shot, one-shot, few-shot, and, where available, RAG-enhanced variants.


```

## Setup

Python 3.10 is recommended.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/setup_nltk_data.py
python scripts/check_environment.py
```

The full experiments use `meta-llama/Llama-3.1-8B-Instruct`. You may need Hugging Face access to that model.

## Quick smoke test

```bash
bash scripts/smoke_test.sh
```

This checks imports, required files, and Python syntax. It does not load the LLaMA model.

## Run experiments

Run one experiment:

```bash
bash scripts/run_anaphoric.sh
bash scripts/run_anaphoric_rag.sh
bash scripts/run_scope.sh
bash scripts/run_structural.sh
bash scripts/run_vagueness.sh
```

Run all experiments:

```bash
bash scripts/run_all.sh
```

Equivalent Make targets are available:

```bash
make smoke
make anaphoric
make anaphoric-rag
make scope
make structural
make vagueness
make all
```

## Expected outputs

Reference outputs from the uploaded runs are stored in:

```text
results/expected/
├── anaphoric/
├── scope/
├── structural/
└── vagueness/
```

New runs generate output inside each experiment folder. Generated logs, ChromaDB folders, and run-output directories are ignored by `.gitignore`.

## Cleaning decisions

This repository intentionally excludes generated and machine-specific files:

- `.DS_Store` and `__MACOSX`
- `__pycache__` and `.pyc` files
- ChromaDB SQLite/index files
- local virtual environments
- run logs
- local `nltk_data` cache files

