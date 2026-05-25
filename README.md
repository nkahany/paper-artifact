# Software Requirements Ambiguity Resolution Using Large Language Models

This repository accompanies the paper:

> **Software Requirements Ambiguity Resolution Using Large Language Models**
> *Author information withheld during double-blind review.*

It provides a unified **Chain-of-Thought (CoT)** framework for resolving four
types of ambiguity in natural-language software requirements — **anaphoric**,
**structural**, **vagueness**, and **scope** — using **Llama 3.1-8B-Instruct**,
with optional **Retrieval-Augmented Generation (RAG)**.

---

## ✨ Highlights

- Unified prompt-engineering framework with type-specific CoT reasoning.
- Three prompting strategies: **zero-shot**, **one-shot**, **few-shot**.
- Optional **RAG** using `all-MiniLM-L6-v2` embeddings + ChromaDB.
- Evaluation with Exact Match, BLEU-1/2/3/4, Semantic Similarity, and
  manual judgements.
- 411 expert-annotated ambiguity instances spanning four ambiguity types.

---

## 📁 Repository structure

```
.
├── code/                       # Resolution scripts (one per ambiguity type)
│   ├── common.py               # Shared pipeline: prompts, RAG, evaluation
│   ├── anaphoric.py
│   ├── structural.py
│   ├── vagueness.py
│   ├── scope.py
│   ├── run_all.sh              # Reproduce all 24 experimental configurations
│   └── requirements.txt        # Python dependencies
│
├── data/                       # CSV datasets (411 ambiguity instances)
│   ├── anaphoric.csv
│   ├── structural.csv
│   ├── vagueness.csv
│   ├── scope.csv
│   └── multi_label.csv
│
├── LICENSE                     # MIT (code)
├── LICENSE-DATA                # CC BY 4.0 (data)
├── CITATION.cff
└── README.md
```

---

## 🔧 Installation

### 1. Clone and create a Python environment

```bash
git clone <repo-url> requirements-ambiguity-llm
cd requirements-ambiguity-llm
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

### 2. Authenticate with Hugging Face (for Llama 3.1)

Llama 3.1 is a gated model. Request access on the
[model page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), then:

```bash
huggingface-cli login
```

### 3. (One-time) download NLTK data used by BLEU smoothing

```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## 💻 Hardware requirements

- NVIDIA GPU with **≥ 40 GB VRAM** (e.g. A100-40GB) for Llama 3.1-8B in
  `bfloat16`.
- Runs on CPU too, but will be very slow.
- ~5 GB disk space for model weights.

---

## 📊 Dataset

The dataset contains **411 ambiguity instances** drawn from the
[PURE](https://zenodo.org/record/1414117) corpus and the multi-label
ambiguity-detection dataset of Oghabi & Kahani (2024):

| Type        | # instances |
|-------------|:-----------:|
| Anaphoric   | 114         |
| Structural  | 100         |
| Vagueness   | 101         |
| Scope       | 96          |
| **Total**   | **411**     |

Each CSV in `data/` is expected to have at least:

| Column                | Description                            |
|-----------------------|----------------------------------------|
| `Original_Requirement`| Raw, ambiguous requirement text         |
| `Fixed_Requirement`   | Expert-validated ground-truth rewrite   |

The scripts also accept a single `Description` column as a fallback when
ground truth is unavailable.

---

## 🚀 Running an experiment

Each ambiguity type has its own script. All accept the same CLI flags.

```bash
cd code

# Few-shot, no RAG (default)
python anaphoric.py --csv ../data/anaphoric.csv

# Zero-shot with RAG
python structural.py --approach zero-shot --use-rag \
                     --csv ../data/structural.csv \
                     --srs-path ../srs_documents

# Few-shot + RAG
python vagueness.py --approach few-shot --use-rag \
                    --csv ../data/vagueness.csv
```

### Reproduce all 24 configurations from the paper

```bash
bash code/run_all.sh
```

This runs **4 ambiguity types × 3 prompting strategies × 2 (with/without RAG)**.

Outputs land in `results/<ambiguity_type>/`:

- `<type>_results_<approach>[_rag].csv` — per-instance predictions and metrics
- `<type>_log_<approach>[_rag].txt`     — full run log + summary

---

## ⚙️ Model configuration

These defaults match §Experimental Setup of the paper:

| Hyperparameter         | Value                                |
|------------------------|--------------------------------------|
| Model                  | `meta-llama/Llama-3.1-8B-Instruct`   |
| Precision              | `bfloat16`                           |
| Temperature            | 0.1                                  |
| Top-p                  | 0.9                                  |
| Max new tokens         | 512                                  |
| Repetition penalty     | 1.1                                  |
| Max input (zero-shot)  | 2048 tokens                          |
| Max input (one-shot)   | 2560 tokens                          |
| Max input (few-shot)   | 3584 tokens                          |
| Embedding model        | `all-MiniLM-L6-v2`                   |
| RAG index              | ChromaDB (cosine, HNSW)              |
| Retrieved chunks       | 3 (≤ 200 tokens each)                |
| Random seed            | 42                                   |

---

## 📏 Evaluation metrics

Automatic metrics computed per instance:

- **Exact Match** (case-insensitive, whitespace-normalised)
- **BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4** (NLTK, `method1` smoothing)
- **Semantic Similarity** (cosine of `all-MiniLM-L6-v2` embeddings)

Manual evaluation (correctness, meaning preservation, naturalness) on
50-sample subsets is reported in the paper.

---

## 🔁 Reproducibility checklist

- Fixed random seed (42) for Python, NumPy, and Torch.
- All hyperparameters declared at the top of `code/common.py`.
- Per-run logs saved to `results/`.
- Self-retrieval prevented in RAG (the query is excluded from retrieved docs).

---

## 📚 Citation

Citation information is withheld during double-blind review. Once the paper is
published, please cite it as indicated on the corresponding venue page.

A `CITATION.cff` file is included for GitHub's "Cite this repository" button
and will be updated after the review period.

---

## 📜 License

- **Code** — [MIT](LICENSE)
- **Data** — [CC BY 4.0](LICENSE-DATA) (with attribution to the PURE corpus
  and the multi-label ambiguity-detection dataset cited in the paper)

---

## 🙏 Acknowledgements

- The PURE corpus (Ferrari, Spagnolo, and Gnesi, 2017).
- A prior multi-label ambiguity-detection dataset (anonymized citation in the
  paper).
- Kamath et al. (2024) for scope-ambiguity examples used in the RAG index.

---

## 📬 Contact

Contact information is withheld during double-blind review. Please use the
review system to communicate with the authors.
