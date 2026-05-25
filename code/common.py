#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for ambiguity-resolution experiments
(anaphoric, structural, vagueness, scope).

Implements the unified Chain-of-Thought framework described in:
"Software Requirements Ambiguity Resolution Using Large Language Models".
"""

import os
import re
import json
import torch
import warnings
import argparse
import pandas as pd
import numpy as np
import chromadb
from abc import ABC, abstractmethod
from datetime import datetime
from difflib import SequenceMatcher

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global defaults (can be overridden via CLI)
# ---------------------------------------------------------------------------
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SRS_DOCS_PATH = "./srs_documents"
RANDOM_SEED = 42

# Per-approach max context length (see paper §Experimental Setup)
MAX_TOKENS = {"zero-shot": 2048, "one-shot": 2560, "few-shot": 3584}

GEN_KWARGS = dict(
    temperature=0.1,
    top_p=0.9,
    max_new_tokens=512,
    repetition_penalty=1.1,
    do_sample=True,
)


# ---------------------------------------------------------------------------
# Base resolver
# ---------------------------------------------------------------------------
class BaseAmbiguityResolver(ABC):
    """
    Abstract base class that implements the shared pipeline:
    setup -> load_model -> run_analysis -> evaluation.

    Subclasses must provide:
        - ambiguity_type (str)
        - default_examples (list of {original, fixed})
        - cot_steps (list[str])         : the 5 type-specific reasoning steps
        - ambiguity_definition (str)    : type-specific definition
        - critical_rules (list[str])    : type-specific rules
    """

    ambiguity_type: str = "generic"
    default_examples: list = []
    cot_steps: list = []
    ambiguity_definition: str = ""
    critical_rules: list = []

    def __init__(self, approach_type="few-shot", use_rag=False,
                 csv_file=None, srs_path=SRS_DOCS_PATH):
        assert approach_type in ("zero-shot", "one-shot", "few-shot")
        self.approach_type = approach_type
        self.use_rag = use_rag
        self.csv_file = csv_file
        self.srs_path = srs_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

        # RAG components
        self.embedding_model = None
        self.client = None
        self.collection = None
        self.srs_requirements = []

        # Few-shot examples (loaded from CSV ground-truth if available)
        self.ground_truth_examples = list(self.default_examples)

        # Output files
        rag_suffix = "_rag" if use_rag else ""
        tag = approach_type.replace("-", "")
        out_dir = os.path.join("results", self.ambiguity_type)
        os.makedirs(out_dir, exist_ok=True)
        self.output_file = os.path.join(
            out_dir, f"{self.ambiguity_type}_results_{tag}{rag_suffix}.csv"
        )
        self.log_file = os.path.join(
            out_dir, f"{self.ambiguity_type}_log_{tag}{rag_suffix}.txt"
        )

        # Evaluation
        self.similarity_model = None
        self.smoothing = SmoothingFunction().method1

    # ---------------------------- logging ---------------------------------
    def log(self, message):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ---------------------------- setup -----------------------------------
    def setup(self):
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        banner = f"UNIFIED {self.ambiguity_type.upper()} AMBIGUITY ANALYSIS"
        print("=" * 80)
        print(f"{banner}  ({self.approach_type}{' + RAG' if self.use_rag else ''})")
        print("=" * 80)

        self.log(f"Device: {self.device}")
        if torch.cuda.is_available():
            self.log(f"GPU: {torch.cuda.get_device_name(0)}")

        self.load_ground_truth_examples()

        if self.use_rag:
            self.setup_rag_system()

        self.similarity_model = SentenceTransformer(EMBEDDING_MODEL)

    def load_model(self):
        self.log(f"Loading model: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        self.log("Model loaded.")

    # ---------------------------- examples --------------------------------
    def load_ground_truth_examples(self):
        """Load up to 3 (original, fixed) pairs from the input CSV if present."""
        if not self.csv_file or not os.path.exists(self.csv_file):
            self.log("Using hardcoded default examples.")
            return
        try:
            df = pd.read_csv(self.csv_file)
            if {"Original_Requirement", "Fixed_Requirement"}.issubset(df.columns):
                pairs = df.dropna(subset=["Original_Requirement", "Fixed_Requirement"])
                pairs = pairs[pairs["Original_Requirement"] != pairs["Fixed_Requirement"]]
                if len(pairs) >= 3:
                    sample = pairs.sample(n=3, random_state=RANDOM_SEED)
                    self.ground_truth_examples = [
                        {"original": r["Original_Requirement"],
                         "fixed": r["Fixed_Requirement"]}
                        for _, r in sample.iterrows()
                    ]
                    self.log(f"Loaded {len(self.ground_truth_examples)} examples from CSV.")
        except Exception as e:
            self.log(f"Example loading failed ({e}); using defaults.")

    # ---------------------------- RAG -------------------------------------
    def setup_rag_system(self):
        self.log("Initializing RAG system...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        persist_dir = f"./chroma_db_{self.ambiguity_type}"
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=f"{self.ambiguity_type}_patterns",
            metadata={"hnsw:space": "cosine"},
        )

        self.load_srs_documents()
        self.build_rag_index()
        self.log("RAG ready.")

    def load_srs_documents(self):
        if not os.path.isdir(self.srs_path):
            self.log(f"SRS path '{self.srs_path}' not found; skipping.")
            return
        reqs = []
        for fname in os.listdir(self.srs_path):
            fpath = os.path.join(self.srs_path, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                reqs.extend(self._extract_requirements(text, fname))
            except Exception as e:
                self.log(f"Failed to read {fname}: {e}")
        self.srs_requirements = reqs
        self.log(f"Loaded {len(reqs)} requirement snippets from SRS docs.")

    @staticmethod
    def _extract_requirements(text, source):
        out = []
        patterns = [
            r"REQ-\d+[:.]?\s*(.+?)(?=REQ-\d+|\n\n|$)",
            r"(The\s+\w+\s+shall\s+.+?\.)",
            r"\d+\.\d+\.?\d*\s+(.+?)(?=\d+\.\d+|\n\n|$)",
        ]
        for pat in patterns:
            for m in re.findall(pat, text, flags=re.DOTALL):
                s = m.strip()
                if 20 < len(s) < 400:
                    out.append({"text": s, "source": source})
        return out

    def build_rag_index(self):
        if not self.collection:
            return
        if self.collection.count() > 0:
            self.log(f"Reusing existing index ({self.collection.count()} docs).")
            return

        docs, metas, ids = [], [], []
        for i, ex in enumerate(self.ground_truth_examples):
            docs.append(f"{ex['original']} -> {ex['fixed']}")
            metas.append({"type": "example"})
            ids.append(f"ex_{i}")
        for i, req in enumerate(self.srs_requirements):
            docs.append(req["text"])
            metas.append({"type": "srs", "source": req["source"]})
            ids.append(f"srs_{i}")
        if docs:
            embs = self.embedding_model.encode(docs).tolist()
            self.collection.add(documents=docs, embeddings=embs,
                                metadatas=metas, ids=ids)
            self.log(f"Indexed {len(docs)} documents.")

    def retrieve_context(self, query, n_results=3):
        if not self.use_rag or not self.collection:
            return ""
        try:
            q_emb = self.embedding_model.encode([query]).tolist()
            res = self.collection.query(query_embeddings=q_emb, n_results=n_results)
            docs = res.get("documents", [[]])[0]
            # Limit each chunk to ~200 tokens (≈ chars/4 heuristic)
            trimmed = [d[:800] for d in docs if query.strip() not in d]
            if not trimmed:
                return ""
            return "Relevant context:\n- " + "\n- ".join(trimmed)
        except Exception as e:
            self.log(f"Retrieval error: {e}")
            return ""

    # ---------------------------- prompts ---------------------------------
    def _system_prompt(self):
        rules = "\n".join(f"- {r}" for r in self.critical_rules)
        steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(self.cot_steps))
        return (
            f"You are an expert requirements engineer specializing in "
            f"{self.ambiguity_type} ambiguity resolution.\n\n"
            f"AMBIGUITY DEFINITION:\n{self.ambiguity_definition}\n\n"
            f"CHAIN-OF-THOUGHT STEPS:\n{steps}\n\n"
            f"CRITICAL RULES:\n{rules}\n\n"
            "OUTPUT FORMAT (EXACTLY):\n"
            "Ambiguity Type: [type]\n"
            "Problem Found: [specific issue]\n"
            "Fixed Requirement: [ONLY the corrected sentence]\n"
            "Explanation: [why]\n"
        )

    def build_prompt(self, requirement):
        system = self._system_prompt()
        ctx = self.retrieve_context(requirement) if self.use_rag else ""

        # Demonstrations
        demo = ""
        if self.approach_type == "one-shot" and self.ground_truth_examples:
            ex = self.ground_truth_examples[0]
            demo = (f"\nExample:\nOriginal: {ex['original']}\n"
                    f"Fixed: {ex['fixed']}\n")
        elif self.approach_type == "few-shot" and self.ground_truth_examples:
            demo = "\nExamples:\n"
            for ex in self.ground_truth_examples[:3]:
                demo += f"Original: {ex['original']}\nFixed: {ex['fixed']}\n\n"

        user = (f"{ctx}\n{demo}\n"
                f"Requirement: {requirement}\n\n"
                "Apply the chain-of-thought steps and produce the output in "
                "the exact format specified.")
        return self._llama_chat(system, user)

    def _llama_chat(self, system, user):
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    # ---------------------------- inference -------------------------------
    @torch.no_grad()
    def generate(self, prompt):
        max_in = MAX_TOKENS[self.approach_type]
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=max_in).to(self.device)
        out = self.model.generate(
            **inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            **GEN_KWARGS,
        )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        return text.strip()

    # ---------------------------- parsing ---------------------------------
    @staticmethod
    def clean(text):
        if not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text).strip().strip('"').strip("'")

    def parse_response(self, response, original):
        result = {
            "ambiguity_type": "",
            "problem_found": "",
            "fixed_requirement": original,
            "explanation": "",
        }
        for key, field in [
            ("Ambiguity Type", "ambiguity_type"),
            ("Problem Found", "problem_found"),
            ("Fixed Requirement", "fixed_requirement"),
            ("Explanation", "explanation"),
        ]:
            m = re.search(rf"{key}\s*:\s*(.+?)(?=\n[A-Z][a-zA-Z ]+:|\Z)",
                          response, flags=re.DOTALL)
            if m:
                result[field] = self.clean(m.group(1))
        if not result["fixed_requirement"]:
            result["fixed_requirement"] = original
        return result

    # ---------------------------- metrics ---------------------------------
    def similarity(self, a, b):
        if not a or not b:
            return 0.0
        emb = self.similarity_model.encode([a, b])
        return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

    def bleu(self, pred, ref):
        if not pred or not ref:
            return [0.0] * 4
        ref_tok = [ref.lower().split()]
        pred_tok = pred.lower().split()
        return [
            sentence_bleu(ref_tok, pred_tok, weights=w,
                          smoothing_function=self.smoothing)
            for w in [(1, 0, 0, 0), (0.5, 0.5, 0, 0),
                      (0.34, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
        ]

    # ---------------------------- main loop -------------------------------
    def resolve_one(self, original, gt):
        prompt = self.build_prompt(original)
        response = self.generate(prompt)
        parsed = self.parse_response(response, original)
        fixed = parsed["fixed_requirement"]

        b1, b2, b3, b4 = self.bleu(fixed, gt) if gt else [0.0] * 4
        return {
            "Original_Requirement": original,
            "Predicted_Fix": fixed,
            "Ground_Truth": gt,
            "Ambiguity_Type": parsed["ambiguity_type"],
            "Problem_Found": parsed["problem_found"],
            "Explanation": parsed["explanation"],
            "Exact_Match": int(self.clean(fixed).lower() ==
                               self.clean(gt).lower()) if gt else 0,
            "BLEU_1": b1, "BLEU_2": b2, "BLEU_3": b3, "BLEU_4": b4,
            "Semantic_Similarity": self.similarity(fixed, gt) if gt else 0.0,
            "Raw_Response": response,
        }

    def run_analysis(self):
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(self.csv_file)
        df = pd.read_csv(self.csv_file)

        # Accept either explicit columns or fall back to "Description"
        orig_col = next((c for c in
                         ["Original_Requirement", "Description", "Requirement"]
                         if c in df.columns), None)
        gt_col = next((c for c in
                       ["Fixed_Requirement", "Ground_Truth"]
                       if c in df.columns), None)
        if orig_col is None:
            raise ValueError("CSV must contain an Original_Requirement column.")
        self.log(f"Rows: {len(df)} | orig='{orig_col}' | gt='{gt_col}'")

        rows = []
        for _, r in tqdm(df.iterrows(), total=len(df),
                         desc=f"{self.ambiguity_type}/{self.approach_type}"):
            orig = str(r[orig_col]).strip()
            gt = str(r[gt_col]).strip() if gt_col and pd.notna(r.get(gt_col)) else ""
            try:
                rows.append(self.resolve_one(orig, gt))
            except Exception as e:
                self.log(f"Failed on row: {e}")

        out_df = pd.DataFrame(rows)
        out_df.to_csv(self.output_file, index=False)
        self.log(f"Saved results: {self.output_file}")
        self.log_summary(out_df)
        return out_df

    def log_summary(self, df):
        if df.empty:
            return
        self.log("=" * 60)
        self.log("EVALUATION SUMMARY")
        self.log(f"  N                  : {len(df)}")
        self.log(f"  Exact Match        : {df['Exact_Match'].mean():.3f}")
        for k in ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4",
                  "Semantic_Similarity"]:
            self.log(f"  {k:18s} : {df[k].mean():.3f}")
        self.log("=" * 60)


# ---------------------------------------------------------------------------
# CLI helper (used by all per-type scripts)
# ---------------------------------------------------------------------------
def run_cli(resolver_cls, default_csv):
    parser = argparse.ArgumentParser(
        description=f"{resolver_cls.ambiguity_type.title()} ambiguity resolution"
    )
    parser.add_argument("--approach", choices=["zero-shot", "one-shot", "few-shot"],
                        default="few-shot")
    parser.add_argument("--use-rag", action="store_true")
    parser.add_argument("--csv", default=default_csv)
    parser.add_argument("--srs-path", default=SRS_DOCS_PATH)
    args = parser.parse_args()

    resolver = resolver_cls(approach_type=args.approach,
                            use_rag=args.use_rag,
                            csv_file=args.csv,
                            srs_path=args.srs_path)
    resolver.setup()
    resolver.load_model()
    resolver.run_analysis()
