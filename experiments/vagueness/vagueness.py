#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vagueness Resolution Script - Using Structural Code Format
EXACT COPY of structural code architecture but adapted for vagueness resolution
Uses proven parsing and prompt structure that works reliably
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import json
import os
import re
from datetime import datetime
import argparse
import sys
import random
from difflib import SequenceMatcher
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CSV_FILE = "Vaguenes_fixed_req.csv"
SRS_DOCS_PATH = "./srs_documents"

class UnifiedVaguenessResolver:
    def __init__(self, approach_type="few-shot", use_rag=False):
        self.approach_type = approach_type  # zero-shot, one-shot, few-shot
        self.use_rag = use_rag  # Whether to enhance with RAG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        # RAG components (only loaded if use_rag=True)
        self.embedding_model = None
        self.client = None
        self.collection = None
        self.srs_requirements = []
        
        # Examples from CSV ground truth
        self.ground_truth_examples = []
        
        # Output files based on approach + RAG
        rag_suffix = "_rag" if use_rag else ""
        self.output_file = f"unified_vagueness_results_{approach_type.replace('-', '')}{rag_suffix}.csv"
        self.log_file = f"unified_vagueness_log_{approach_type.replace('-', '')}{rag_suffix}.txt"
        
        # Evaluation components
        self.similarity_model = None
        self.smoothing = SmoothingFunction().method1
        
    def log_message(self, message):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(log_entry + "\n")
    
    def split_vagueness_data(self, vagueness_csv_path: str):
        """Split the CSV into 50 RAG examples and remaining test examples."""
        self.log_message(f"Splitting {vagueness_csv_path} into RAG (50) and test sets")
        
        # Load the main file
        df = pd.read_csv(vagueness_csv_path)
        self.log_message(f"Loaded {len(df)} total examples")
        
        # Clean the data
        df = df.dropna(subset=['Requirement'])
        df = df[df['Requirement'].str.strip().str.len() > 0]
        
        # Shuffle data for random split
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split: first 50 for RAG, remainder for test
        rag_data = df_shuffled[:50].copy()
        test_data = df_shuffled[50:].copy()
        
        # Save splits
        rag_data.to_csv('vagueness_rag.csv', index=False)
        test_data.to_csv('vagueness_test.csv', index=False)
        
        self.log_message(f"Created vagueness_rag.csv with {len(rag_data)} examples")
        self.log_message(f"Created vagueness_test.csv with {len(test_data)} examples")
        
        return rag_data, test_data
    
    def load_ground_truth_examples(self):
        """Load ground truth examples from vagueness_rag.csv"""
        try:
            if os.path.exists('vagueness_rag.csv'):
                df = pd.read_csv('vagueness_rag.csv')
                self.log_message(f"Loaded {len(df)} ground truth examples from vagueness_rag.csv")
                
                # Convert to examples format
                for _, row in df.iterrows():
                    original = str(row.get('Requirement', '')).strip()
                    # Try different column names for fixed requirement
                    fixed = None
                    for col in ['Vaguenes_fixed_req', 'Fixed_Requirement', 'fixed_requirement']:
                        if col in row and pd.notna(row[col]):
                            fixed = str(row[col]).strip()
                            break
                    
                    # Only include examples where we have both original and fixed AND they're different
                    if original and fixed and original != fixed:
                        self.ground_truth_examples.append({
                            'original': original,
                            'fixed': fixed
                        })
                
                self.log_message(f"Prepared {len(self.ground_truth_examples)} valid examples")
            else:
                self.log_message("vagueness_rag.csv not found, using hardcoded examples")
                self._load_hardcoded_examples()
                
        except Exception as e:
            self.log_message(f"Error loading ground truth: {e}, using hardcoded examples")
            self._load_hardcoded_examples()
    
    def _load_hardcoded_examples(self):
        """Load hardcoded vagueness examples"""
        self.ground_truth_examples = [
            {
                'original': "The system shall respond quickly to user requests.",
                'fixed': "The system shall respond to user requests within the specified performance criteria."
            },
            {
                'original': "The application shall provide sufficient security for user data.",
                'fixed': "The application shall provide security for user data meeting the defined security standards."
            },
            {
                'original': "The interface shall be user-friendly and intuitive.",
                'fixed': "The interface shall meet usability requirements as specified in the user experience guidelines."
            }
        ]
    
    def setup_rag_system(self):
        """Setup RAG system for vagueness resolution"""
        if not self.use_rag:
            return
            
        self.log_message("Setting up RAG system...")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.log_message(f"Embedding model loaded: {EMBEDDING_MODEL}")
        
        # Setup ChromaDB
        persist_dir = "./chroma_db_vagueness"
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        self.collection = self.client.get_or_create_collection(
            name="vagueness_patterns",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load SRS documents and build RAG index
        self.load_srs_documents()
        self.build_rag_index()
        
        self.log_message("RAG system setup complete")
    
    def load_srs_documents(self):
        """Load SRS documents for RAG context"""
        self.log_message("Loading SRS documents...")
        
        if not os.path.exists(SRS_DOCS_PATH):
            self.log_message(f"WARNING: SRS path {SRS_DOCS_PATH} not found")
            return
        
        requirements = []
        for filename in os.listdir(SRS_DOCS_PATH):
            if filename.endswith(('.txt', '.md', '.doc')):
                filepath = os.path.join(SRS_DOCS_PATH, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        reqs = self.extract_requirements_from_srs(content, filename)
                        requirements.extend(reqs)
                except Exception as e:
                    self.log_message(f"Error reading {filename}: {e}")
        
        self.srs_requirements = requirements
        self.log_message(f"Loaded {len(requirements)} requirements from SRS documents")
    
    def extract_requirements_from_srs(self, text: str, source: str) -> list:
        """Extract requirements from SRS text with vagueness indicators"""
        requirements = []
        patterns = [
            r'REQ-\d+[:\.]?\s*(.+?)(?=REQ-\d+|\n\n|$)',
            r'The system shall (.+?)(?=The system shall|\n\n|$)',
            r'The \w+ shall (.+?)(?=The \w+ shall|\n\n|$)',
            r'\d+\.\d+\.?\d*\s+(.+?)(?=\d+\.\d+|\n\n|$)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                req_text = match.group(1).strip()
                req_text = re.sub(r'\s+', ' ', req_text)
                
                if len(req_text) > 30 and len(req_text) < 500:
                    # Check if it has vagueness indicators
                    if self.has_vagueness_indicators(req_text):
                        requirements.append({
                            'text': req_text,
                            'source': source
                        })
        
        return requirements
    
    def has_vagueness_indicators(self, text: str) -> bool:
        """Check if text has vagueness indicators"""
        vague_patterns = [
            r'\b(sufficient|adequate|reasonable|appropriate|high|low)\b',
            r'\b(quickly|rapidly|efficiently|timely|fast|slow)\b',
            r'\b(user-friendly|robust|reliable|flexible|scalable)\b',
            r'\b(many|few|some|various|several|multiple)\b',
            r'\b(good|better|best|optimal|minimal|maximum)\b'
        ]
        
        for pattern in vague_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def build_rag_index(self):
        """Build RAG index with vagueness patterns and SRS requirements"""
        if not self.use_rag:
            return
            
        self.log_message("Building RAG index...")
        
        documents = []
        metadatas = []
        ids = []
        
        # Add ground truth examples
        for i, example in enumerate(self.ground_truth_examples):
            doc_text = f"Original: {example['original']}\nFixed: {example['fixed']}"
            documents.append(doc_text)
            metadatas.append({'type': 'vagueness_example'})
            ids.append(f"example_{i}")
        
        # Add SRS requirements
        for i, req in enumerate(self.srs_requirements):
            documents.append(req['text'])
            metadatas.append({'type': 'srs_requirement', 'source': req['source']})
            ids.append(f"srs_{i}")
        
        if documents:
            embeddings = self.embedding_model.encode(documents).tolist()
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
            self.log_message(f"Added {len(documents)} documents to RAG index")
    
    def retrieve_context(self, query, n_results=3):
        """Retrieve relevant context from RAG system"""
        if not self.use_rag or not self.collection:
            return []
        
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            context = []
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                context.append({'text': doc, 'metadata': metadata})
            return context
        except Exception as e:
            self.log_message(f"Error retrieving context: {e}")
            return []
    
    def setup(self):
        """Initialize logging and GPU setup"""
        print("=" * 80)
        print(f"UNIFIED VAGUENESS {self.approach_type.upper()} RESOLUTION ANALYSIS")
        if self.use_rag:
            print(f"WITH RAG ENHANCEMENT")
        print("=" * 80)
        
        self.log_message(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.log_message(f"GPU: {gpu_name}")
            self.log_message(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Split data first (only do this once)
        if not os.path.exists('vagueness_test.csv'):
            self.split_vagueness_data(CSV_FILE)
        else:
            self.log_message("Data splits already exist, using existing files")
        
        # Load ground truth examples from RAG split
        self.load_ground_truth_examples()
        
        # Setup RAG if enabled
        if self.use_rag:
            self.setup_rag_system()
        
        # Initialize similarity model for evaluation
        self.similarity_model = SentenceTransformer(EMBEDDING_MODEL)
    
    def load_model(self):
        """Load LLaMA model and tokenizer"""
        self.log_message("Loading LLaMA model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                use_fast=True,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.log_message(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.log_message("Model loaded successfully!")
            
        except Exception as e:
            self.log_message(f"ERROR loading model: {e}")
            sys.exit(1)
    
    def create_zero_shot_prompt(self, requirement_text):
        """Create zero-shot prompt for vagueness resolution"""
        system_prompt = """You are an expert requirements engineer. Your task is to identify and resolve vagueness in software requirements using a CONSERVATIVE approach.

Vagueness occurs when requirements use imprecise terms that lead to multiple interpretations. Common patterns:

1. Quantitative Vagueness: "sufficient", "adequate", "reasonable", "high", "low"
   - Problem: No specific metrics or thresholds
   - Conservative Fix: Add measurable criteria or reference standards without inventing specific numbers

2. Temporal Vagueness: "quickly", "timely", "efficiently", "real-time"
   - Problem: No time constraints specified
   - Conservative Fix: Add time-related criteria or standards without fabricating exact times

3. Qualitative Vagueness: "user-friendly", "robust", "flexible", "reliable"
   - Problem: Subjective terms without measurable criteria
   - Conservative Fix: Reference established standards or measurable criteria

4. Scope Vagueness: "various", "many", "some", "appropriate"
   - Problem: Unclear boundaries or extent
   - Conservative Fix: Provide structure for specification without inventing specific numbers

CONSERVATIVE RULES:
- Only add specific values if clearly derivable from context
- Focus on measurable frameworks over fabricated values
- Reference standards when possible
- Always output the COMPLETE requirement text

OUTPUT FORMAT (EXACTLY):
Vagueness Type: [type of vagueness found]
Problem Found: [which terms are vague]
Fixed Requirement: [ONLY the complete requirement sentence - nothing else]
Explanation: [what conservative approach was used]

IMPORTANT: The Fixed Requirement field should contain ONLY the requirement sentence, no extra text or formatting."""

        # Add RAG context if available
        context_text = ""
        if self.use_rag:
            context = self.retrieve_context(requirement_text)
            if context:
                context_text = "\n\nRELEVANT EXAMPLES FROM KNOWLEDGE BASE:\n"
                for i, ctx in enumerate(context[:2], 1):
                    if ctx['metadata']['type'] == 'vagueness_example':
                        context_text += f"\nExample {i}:\n{ctx['text']}\n"
                    else:
                        context_text += f"\nSimilar requirement: {ctx['text']}\n"

        user_prompt = f"""{context_text}

Requirement: {requirement_text}

Think step by step using CONSERVATIVE approach:
1. What vague terms do I see?
2. What could each vague term mean?
3. Can I derive specific values from context?
4. How should I make it measurable without fabricating information?
5. What is the complete fixed requirement?

Provide your analysis and answer in the EXACT format specified."""

        return system_prompt, user_prompt
    
    def create_one_shot_prompt(self, requirement_text):
        """Create one-shot prompt for vagueness resolution"""
        # Select best example
        example = self.ground_truth_examples[0] if self.ground_truth_examples else {
            'original': "The system shall respond quickly to user requests.",
            'fixed': "The system shall respond to user requests within the specified performance criteria."
        }
        
        system_prompt = f"""You are an expert requirements engineer. Your task is to identify and resolve vagueness in software requirements using a CONSERVATIVE approach.

Here's an example of CONSERVATIVE vagueness resolution:

Req: {example['original']}
Fixed: {example['fixed']}

CONSERVATIVE APPROACH:
- Identify vague terms (quickly, sufficient, user-friendly, etc.)
- Replace with measurable language without fabricating specific values
- Focus on frameworks and standards over invented numbers
- Output complete requirement

OUTPUT FORMAT (EXACTLY):
Vagueness Type: [type found]
Problem Found: [specific vague terms]
Fixed Requirement: [ONLY the complete requirement sentence - no extra text]
Explanation: [what conservative approach was used]

CRITICAL: The Fixed Requirement field must contain ONLY the requirement sentence."""

        # Add RAG context if available
        context_text = ""
        if self.use_rag:
            context = self.retrieve_context(requirement_text)
            if context:
                context_text = "\n\nADDITIONAL CONTEXT FROM KNOWLEDGE BASE:\n"
                for i, ctx in enumerate(context[:2], 1):
                    context_text += f"\nContext {i}: {ctx['text']}\n"

        user_prompt = f"""{context_text}

Now apply the CONSERVATIVE approach:

Req: {requirement_text}
Fixed: [You complete this]

Think step by step like the example:
1. What vague terms do I see?
2. How should I fix them conservatively?
3. What's the complete fixed requirement?

Provide your answer in the EXACT format specified."""

        return system_prompt, user_prompt
    
    def create_few_shot_prompt(self, requirement_text):
        """Create few-shot prompt with 3 examples for vagueness resolution"""
        # Select 3 examples
        selected_examples = self.ground_truth_examples[:3] if len(self.ground_truth_examples) >= 3 else self.ground_truth_examples
        
        system_prompt = """You are an expert requirements engineer. Your task is to identify and resolve vagueness in software requirements using a CONSERVATIVE approach.

Here are examples of CONSERVATIVE vagueness resolution:"""

        # Add exactly 3 examples
        for i, example in enumerate(selected_examples, 1):
            system_prompt += f"""

Example {i}:
Req: {example['original']}
Fixed: {example['fixed']}"""

        system_prompt += """

CRITICAL CONSERVATIVE RULES:
- Identify vague terms (sufficient, quickly, user-friendly, etc.)
- Replace with structured, measurable criteria
- AVOID fabricating specific numbers without contextual justification
- Focus on measurable frameworks over invented values
- Always provide the COMPLETE fixed requirement

CONSERVATIVE RESOLUTION STRATEGY:
1. Use context-derived specific values (if available)
2. Reference established standards or frameworks
3. Create structured specification requirements
4. Add measurable criteria without specific fabricated values
5. Last resort: fabricated specific values (clearly mark as such)

OUTPUT FORMAT (MUST FOLLOW EXACTLY):
Vagueness Type: [specific type of vagueness]
Problem Found: [list the vague terms found]
Fixed Requirement: [ONLY the complete requirement sentence - no extra text or formatting]
Explanation: [describe conservative approach used]

CRITICAL: The Fixed Requirement field should contain ONLY the requirement sentence, nothing else."""

        # Add RAG context if available
        context_text = ""
        if self.use_rag:
            context = self.retrieve_context(requirement_text)
            if context:
                context_text = "\n\nRELEVANT CONTEXT FROM KNOWLEDGE BASE:\n"
                for i, ctx in enumerate(context[:2], 1):
                    context_text += f"\nContext {i}: {ctx['text']}\n"

        user_prompt = f"""{context_text}

Now analyze this requirement using the CONSERVATIVE approach:

Req: {requirement_text}
Fixed: [You complete this]

Follow the Conservative Strategy:
1. Identify vague terms
2. Check if specific values can be derived from context
3. If not, use measurable frameworks
4. Output complete fixed requirement

Remember: Use the conservative approach to avoid fabricating information!
Provide your answer in the EXACT format specified above."""

        return system_prompt, user_prompt
    
    def create_llama_prompt(self, system_prompt, user_prompt):
        """Format prompt for LLaMA"""
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return formatted_prompt
    
    def clean_text(self, text):
        """Clean text by removing ** prefix and extra whitespace"""
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # Remove ** prefix that's causing exact match issues
        text = re.sub(r'^\*\*\s*', '', text)
        
        # Remove other common prefixes that might appear
        text = re.sub(r'^(Fixed|Resolved|Solution):\s*', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def parse_response(self, response_text, original_req):
        """Enhanced response parsing to extract only the fixed requirement (USING STRUCTURAL FORMAT)"""
        parsed = {
            'fixed_requirement': original_req,
            'explanation': 'No changes made',
            'vagueness_type': 'None',
            'problem_found': 'No vagueness detected'
        }
        
        try:
            # Clean up the response
            response_text = response_text.strip()
            
            # Multiple patterns to extract ONLY the fixed requirement (COPIED FROM STRUCTURAL)
            fixed_patterns = [
                # Pattern 1: Look for **Fixed Requirement:** with bold formatting
                r'\*\*Fixed Requirement:\*\*\s*([^*]+?)(?=\s*\*\*Explanation|\s*\*\*Vagueness|\s*$)',
                # Pattern 2: Regular Fixed Requirement:
                r'Fixed Requirement:\s*([^\n]+?)(?=\s*Explanation:|\s*Vagueness Type:|\s*$)',
                # Pattern 3: Look for the sentence after "Fixed Requirement:" up to next field
                r'Fixed Requirement:\s*(.+?)(?=\n\s*(?:Explanation|Vagueness|Problem):|$)',
                # Pattern 4: Extract everything between Fixed Requirement and Explanation
                r'Fixed Requirement:\s*(.+?)(?=\s*Explanation:)',
                # Pattern 5: Just grab the line after Fixed Requirement
                r'Fixed Requirement:\s*([^\n\r]+)',
            ]
            
            fixed_text = None
            
            for i, pattern in enumerate(fixed_patterns):
                matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if matches:
                    candidate = matches[0].strip()
                    candidate = self.clean_text(candidate)
                    
                    # Remove quotes if present
                    if (candidate.startswith('"') and candidate.endswith('"')) or \
                       (candidate.startswith("'") and candidate.endswith("'")):
                        candidate = candidate[1:-1]
                    
                    # Check if this looks like a complete requirement
                    if (len(candidate) > 30 and 
                        not candidate.lower().startswith(('error', 'unable', 'cannot', 'vagueness type')) and
                        '**' not in candidate and
                        'Explanation:' not in candidate and
                        'Problem Found:' not in candidate):
                        
                        fixed_text = candidate
                        break
            
            # If no pattern worked, try a more aggressive approach (COPIED FROM STRUCTURAL)
            if not fixed_text:
                sentences = re.split(r'(?<=[.])\s+', response_text)
                for sentence in sentences:
                    sentence = self.clean_text(sentence)
                    
                    if (len(sentence) > 50 and 
                        ('shall' in sentence.lower() or 'will' in sentence.lower() or 'must' in sentence.lower()) and
                        not sentence.lower().startswith(('i replaced', 'the vague', 'vagueness'))):
                        fixed_text = sentence
                        break
            
            if fixed_text and len(fixed_text) > 20:
                parsed['fixed_requirement'] = fixed_text
            
            # Extract other fields
            try:
                vagueness_match = re.search(r'(?:\*\*)?Vagueness Type(?:\*\*)?:\s*([^\n*]+)', response_text, re.IGNORECASE)
                if vagueness_match:
                    parsed['vagueness_type'] = vagueness_match.group(1).strip()
                
                problem_match = re.search(r'(?:\*\*)?Problem Found(?:\*\*)?:\s*([^\n*]+)', response_text, re.IGNORECASE)
                if problem_match:
                    parsed['problem_found'] = problem_match.group(1).strip()
                
                explanation_match = re.search(r'(?:\*\*)?Explanation(?:\*\*)?:\s*([^\n*]+)', response_text, re.IGNORECASE)
                if explanation_match:
                    parsed['explanation'] = explanation_match.group(1).strip()
            except:
                pass
                
        except Exception as e:
            self.log_message(f"Error parsing response: {e}")
        
        return parsed
    
    def calculate_text_similarity(self, text1, text2):
        """Calculate cosine similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        embeddings = self.similarity_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def calculate_bleu_score(self, predicted, reference):
        """Calculate BLEU score between predicted and reference text"""
        try:
            predicted_tokens = predicted.lower().split()
            reference_tokens = reference.lower().split()
            score = sentence_bleu([reference_tokens], predicted_tokens, smoothing_function=self.smoothing)
            return score
        except:
            return 0.0
    
    def fix_requirement(self, original_req, req_id, project_name, ground_truth):
        """Fix a single requirement using the chosen approach (COPIED FROM STRUCTURAL)"""
        try:
            # Select prompt based on approach
            if self.approach_type == "zero-shot":
                system_prompt, user_prompt = self.create_zero_shot_prompt(original_req)
                max_tokens = 350
                max_length = 2048
            elif self.approach_type == "one-shot":
                system_prompt, user_prompt = self.create_one_shot_prompt(original_req)
                max_tokens = 400
                max_length = 2560
            else:  # few-shot
                system_prompt, user_prompt = self.create_few_shot_prompt(original_req)
                max_tokens = 450
                max_length = 3072
            
            formatted_prompt = self.create_llama_prompt(system_prompt, user_prompt)
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with optimized parameters (COPIED FROM STRUCTURAL)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.02,
                    top_p=0.9,
                    top_k=30,
                    num_beams=1
                )
            
            # Decode response
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Parse response
            parsed_result = self.parse_response(response, original_req)
            
            # Determine if changes were made
            was_changed = (parsed_result['fixed_requirement'] != original_req and 
                          len(parsed_result['fixed_requirement']) > len(original_req) * 0.7)
            
            # Calculate evaluation metrics against ground truth
            predicted_fixed = self.clean_text(parsed_result['fixed_requirement'])
            gt_fixed = self.clean_text(ground_truth)
            
            similarity_score = self.calculate_text_similarity(predicted_fixed, gt_fixed)
            bleu_score = self.calculate_bleu_score(predicted_fixed, gt_fixed)
            exact_match = (predicted_fixed.lower() == gt_fixed.lower())
            
            return {
                'requirement_id': req_id,
                'project_name': project_name,
                'original_requirement': original_req,
                'predicted_fixed': predicted_fixed,
                'ground_truth_fixed': gt_fixed,
                'explanation': parsed_result['explanation'],
                'vagueness_type': parsed_result.get('vagueness_type', 'Unknown'),
                'problem_found': parsed_result.get('problem_found', 'Unknown'),
                'was_changed': was_changed,
                'similarity_score': similarity_score,
                'bleu_score': bleu_score,
                'exact_match': exact_match,
                'approach': f"{self.approach_type}{'+RAG' if self.use_rag else ''}",
                'used_rag': self.use_rag,
                'timestamp': datetime.now().isoformat(),
                'original_length': len(original_req),
                'predicted_length': len(predicted_fixed),
                'ground_truth_length': len(gt_fixed),
                'raw_response': response[:200] + "..." if len(response) > 200 else response
            }
            
        except Exception as e:
            self.log_message(f"Error fixing requirement {req_id}: {e}")
            return {
                'requirement_id': req_id,
                'project_name': project_name,
                'original_requirement': original_req,
                'predicted_fixed': original_req,
                'ground_truth_fixed': ground_truth,
                'explanation': f'Error during processing: {str(e)}',
                'vagueness_type': 'Error',
                'problem_found': 'Processing error',
                'was_changed': False,
                'similarity_score': 0.0,
                'bleu_score': 0.0,
                'exact_match': False,
                'approach': f"{self.approach_type}{'+RAG' if self.use_rag else ''}",
                'used_rag': self.use_rag,
                'timestamp': datetime.now().isoformat(),
                'original_length': len(original_req),
                'predicted_length': len(original_req),
                'ground_truth_length': len(ground_truth),
                'raw_response': f'Error: {str(e)}'
            }
    
    def run_analysis(self, csv_file):
        """Run the complete analysis (COPIED FROM STRUCTURAL)"""
        # Load test CSV
        self.log_message("Loading test CSV data...")
        try:
            df = pd.read_csv('vagueness_test.csv')
            self.log_message(f"Loaded {len(df)} requirements from vagueness_test.csv")
            
            # Verify we have the correct columns - try different column name variations
            original_col = None
            fixed_col = None
            id_col = None
            project_col = None
            
            # Find the right column names
            for col in df.columns:
                if 'requirement' in col.lower() and 'original' not in col.lower() and 'fixed' not in col.lower():
                    original_col = col
                elif 'fixed' in col.lower() or 'vaguenes_fixed' in col.lower():
                    fixed_col = col
                elif col.lower() in ['id', 'requirement_id']:
                    id_col = col
                elif 'project' in col.lower() or 'name' in col.lower():
                    project_col = col
            
            if not original_col or not fixed_col:
                # Try common variations
                if 'Requirement' in df.columns:
                    original_col = 'Requirement'
                if 'Vaguenes_fixed_req' in df.columns:
                    fixed_col = 'Vaguenes_fixed_req'
                    
            if not original_col or not fixed_col:
                raise ValueError(f"Could not find required columns. Available: {list(df.columns)}")
            
            self.log_message(f"Using columns: {original_col} -> {fixed_col}")
                    
        except Exception as e:
            self.log_message(f"ERROR loading CSV: {e}")
            return
        
        # Process requirements
        self.log_message(f"Starting UNIFIED VAGUENESS {self.approach_type} analysis...")
        results = []
        changes_made = 0
        processing_errors = 0
        exact_matches = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"UNIFIED VAGUENESS {self.approach_type}"):
            req_id = row.get(id_col, idx) if id_col else idx
            project_name = str(row.get(project_col, 'Unknown')) if project_col else 'Unknown'
            original_requirement = str(row[original_col]).strip()
            ground_truth = str(row[fixed_col]).strip()
            
            # Skip empty requirements
            if not original_requirement or original_requirement.lower() in ['nan', 'none', '']:
                continue
            
            # Skip if ground truth is missing
            if not ground_truth or ground_truth.lower() in ['nan', 'none', '']:
                continue
            
            if idx % 10 == 0:
                current_accuracy = (exact_matches / (idx + 1 - processing_errors)) if (idx + 1 - processing_errors) > 0 else 0
                self.log_message(f"Processing {idx+1}/{len(df)} - Changes: {changes_made}, Exact: {exact_matches}, Acc: {current_accuracy:.3f}, Errors: {processing_errors}")
            
            result = self.fix_requirement(original_requirement, req_id, project_name, ground_truth)
            results.append(result)
            
            if result['was_changed']:
                changes_made += 1
                # Log successful changes for monitoring
                if changes_made <= 5:  # Log first 5 successful changes
                    self.log_message(f"CHANGE #{changes_made}: {original_requirement[:50]}... -> {result['predicted_fixed'][:50]}...")
            
            if result['exact_match']:
                exact_matches += 1
            
            if 'Error' in result.get('vagueness_type', ''):
                processing_errors += 1
            
            # Save checkpoints
            if (idx + 1) % 25 == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(f"temp_unified_vagueness_{self.approach_type}_{idx+1}.csv", index=False)
        
        # Save final results
        self.log_message("Saving final results...")
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_file, index=False)
        
        # Create summary (same format as structural)
        summary_file = f"unified_vagueness_summary_{self.approach_type}_results.csv"
        summary_df = results_df[['requirement_id', 'project_name', 'original_requirement', 
                                'predicted_fixed', 'ground_truth_fixed', 'was_changed', 'exact_match', 'vagueness_type']]
        summary_df.to_csv(summary_file, index=False)
        
        # Print evaluation statistics
        self.log_evaluation_summary(results_df)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\nUNIFIED VAGUENESS {self.approach_type.upper()} COMPLETE!")
        print(f"Results: {self.output_file}")
    
    def log_evaluation_summary(self, results_df):
        """Log evaluation summary with metrics (COPIED FROM STRUCTURAL)"""
        total_processed = len(results_df)
        changes_made = len(results_df[results_df['was_changed'] == True])
        errors = len(results_df[results_df['vagueness_type'] == 'Error'])
        valid_results = results_df[results_df['vagueness_type'] != 'Error']
        
        if len(valid_results) == 0:
            self.log_message("No valid results to evaluate!")
            return
        
        # Calculate evaluation metrics
        exact_matches = len(valid_results[valid_results['exact_match'] == True])
        avg_similarity = valid_results['similarity_score'].mean()
        avg_bleu = valid_results['bleu_score'].mean()
        
        # Classification metrics (vagueness detection)
        gt_labels = (valid_results['original_requirement'] != valid_results['ground_truth_fixed']).astype(int)
        pred_labels = valid_results['was_changed'].astype(int)
        
        if len(set(gt_labels)) > 1 and len(set(pred_labels)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(gt_labels, pred_labels, average='binary', zero_division=0)
        else:
            precision = recall = f1 = 0.0
        
        accuracy = accuracy_score(gt_labels, pred_labels)
        
        exact_match_rate = exact_matches / len(valid_results) if len(valid_results) > 0 else 0
        change_rate = (changes_made / total_processed * 100) if total_processed > 0 else 0
        
        self.log_message("=" * 80)
        self.log_message(f"UNIFIED VAGUENESS {self.approach_type.upper()} EVALUATION SUMMARY")
        self.log_message("=" * 80)
        self.log_message(f"Approach: {self.approach_type}{' + RAG' if self.use_rag else ''}")
        self.log_message(f"Dataset: vagueness_test.csv")
        
        self.log_message(f"\nPROCESSING STATISTICS:")
        self.log_message(f"  Total Requirements: {total_processed}")
        self.log_message(f"  Valid Processed: {len(valid_results)}")
        self.log_message(f"  Requirements Changed: {changes_made} ({change_rate:.1f}%)")
        self.log_message(f"  Processing Errors: {errors}")
        
        self.log_message(f"\nEVALUATION METRICS (vs Ground Truth):")
        self.log_message(f"  Exact Match Score: {exact_match_rate:.4f} ({exact_matches}/{len(valid_results)})")
        self.log_message(f"  BLEU Score: {avg_bleu:.4f}")
        self.log_message(f"  Semantic Similarity: {avg_similarity:.4f}")
        self.log_message(f"  Classification Accuracy: {accuracy:.4f}")
        self.log_message(f"  Classification Precision: {precision:.4f}")
        self.log_message(f"  Classification Recall: {recall:.4f}")
        self.log_message(f"  Classification F1-Score: {f1:.4f}")
        
        self.log_message(f"\nOUTPUT FILES:")
        self.log_message(f"  Results CSV: {self.output_file}")
        self.log_message(f"  Summary CSV: unified_vagueness_summary_{self.approach_type}_results.csv")
        self.log_message(f"  Log File: {self.log_file}")
        
        self.log_message("=" * 80)


def main():
    """Main function (COPIED FROM STRUCTURAL)"""
    parser = argparse.ArgumentParser(description='Unified Vagueness Resolution')
    parser.add_argument('--approach', type=str, 
                        choices=['zero-shot', 'one-shot', 'few-shot'],
                        default='few-shot', 
                        help='Prompting approach')
    parser.add_argument('--use-rag', action='store_true',
                        help='Enable RAG enhancement')
    parser.add_argument('--csv', type=str, default='Vaguenes_fixed_req.csv',
                        help='Input CSV file')
    parser.add_argument('--srs-path', type=str, default='./srs_documents',
                        help='Path to SRS documents for RAG')
    
    args = parser.parse_args()
    
    # Update global variables
    global CSV_FILE, SRS_DOCS_PATH
    CSV_FILE = args.csv
    SRS_DOCS_PATH = args.srs_path
    
    # Validate files
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
    
    if args.use_rag and not os.path.exists(SRS_DOCS_PATH):
        print(f"Warning: SRS documents path '{SRS_DOCS_PATH}' not found!")
        print("   RAG will work with limited context.")
    
    approach_name = f"{args.approach}{' + RAG' if args.use_rag else ''}"
    print(f"Starting UNIFIED VAGUENESS {approach_name} analysis...")
    print(f"   Input: {CSV_FILE}")
    print(f"   Split: 50 for RAG, remainder for test")
    print(f"   Approach: Conservative (avoids fabricating information)")
    print(f"   Evaluation: original_requirement ? predicted_fixed vs ground_truth_fixed")
    
    # Create and run resolver
    resolver = UnifiedVaguenessResolver(approach_type=args.approach, use_rag=args.use_rag)
    resolver.setup()
    resolver.load_model()
    resolver.run_analysis('vagueness_test.csv')  # Process only the test split
    
    print(f"Complete! Check the output files.")


if __name__ == "__main__":
    main()