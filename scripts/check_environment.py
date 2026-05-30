#!/usr/bin/env python3
"""Lightweight dependency and data check. Does not load LLaMA weights."""
from pathlib import Path
import importlib

required_modules = [
    "pandas", "numpy", "sklearn", "tqdm", "nltk", "torch", "transformers",
    "sentence_transformers", "chromadb", "matplotlib", "seaborn",
]
missing = []
for module in required_modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append((module, str(exc)))

root = Path(__file__).resolve().parents[1]
required_files = [
    "experiments/scope/scope_complete_system.py",
    "experiments/scope/exp1b_base_dataset.csv",
    "experiments/scope/scope_fixed.csv",
    "experiments/structural/structural.py",
    "experiments/structural/Structural.csv",
    "experiments/vagueness/vagueness.py",
    "experiments/vagueness/Vaguenes_fixed_req.csv",
    "experiments/anaphoric/anaphoric.py",
    "experiments/anaphoric/ground_truths.csv",
    "experiments/anaphoric_rag/anaphoric_rag.py",
    "experiments/anaphoric_rag/ground_truths.csv",
    "experiments/ambiguity_detection/anaphoric.csv",
]
missing_files = [p for p in required_files if not (root / p).exists()]

if missing:
    print("Missing Python dependencies:")
    for name, err in missing:
        print(f"  - {name}: {err}")
else:
    print("All required Python modules import successfully.")

if missing_files:
    print("Missing expected project files:")
    for p in missing_files:
        print(f"  - {p}")
else:
    print("All expected project files are present.")

if missing or missing_files:
    raise SystemExit(1)
