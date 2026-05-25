#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Structural Ambiguity Resolution Script
MATCHING the anaphoric code structure with consistent CoT prompts and evaluation
Supports: zero-shot, one-shot, few-shot (with optional RAG enhancement)
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
CSV_FILE = "Structural.csv"
SRS_DOCS_PATH = "./srs_documents"

class UnifiedStructuralResolver:
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
        
        # Examples from CSV ground truth (matching anaphoric approach)
        self.ground_truth_examples = []
        
        # Output files based on approach + RAG
        rag_suffix = "_rag" if use_rag else ""
        self.output_file = f"unified_structural_results_{approach_type.replace('-', '')}{rag_suffix}.csv"
        self.log_file = f"unified_structural_log_{approach_type.replace('-', '')}{rag_suffix}.txt"
        
        # Evaluation components (matching anaphoric structure)
        self.similarity_model = None
        self.smoothing = SmoothingFunction().method1
        
    def log_message(self, message):
        """Log messages with timestamp (matching anaphoric structure)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(log_entry + "\n")
    
    def load_ground_truth_examples(self):
        """Load ground truth examples from Structural.csv (matching anaphoric approach)"""
        try:
            if os.path.exists(CSV_FILE):
                df = pd.read_csv(CSV_FILE)
                self.log_message(f"Loaded {len(df)} ground truth examples from {CSV_FILE}")
                
                # Convert to examples format (same as anaphoric)
                for _, row in df.iterrows():
                    original = str(row.get('Original_Requirement', '')).strip()
                    fixed = str(row.get('Fixed_Requirement', '')).strip()
                    
                    # Only include examples where we have both original and fixed AND they're different
                    if original and fixed and original != fixed:
                        self.ground_truth_examples.append({
                            'original': original,
                            'fixed': fixed
                        })
                
                self.log_message(f"Prepared {len(self.ground_truth_examples)} valid examples")
            else:
                self.log_message(f"Structural CSV file {CSV_FILE} not found, using hardcoded examples")
                self._load_hardcoded_examples()
                
        except Exception as e:
            self.log_message(f"Error loading ground truth: {e}, using hardcoded examples")
            self._load_hardcoded_examples()
    
    def _load_hardcoded_examples(self):
        """Load hardcoded structural examples (matching anaphoric structure)"""
        self.ground_truth_examples = [
            {
                'original': "The system shall process user requests using the security protocols defined in the manual.",
                'fixed': "The system shall process user requests using the security protocols that are defined in the manual."
            },
            {
                'original': "The application shall validate and store user data in the database using encryption.",
                'fixed': "The application shall validate user data and store user data in the database, using encryption for both operations."
            },
            {
                'original': "The module shall send requests to servers using encryption protocols.",
                'fixed': "The module shall send requests to servers, where the module uses encryption protocols for sending."
            }
        ]
    
    def setup_rag_system(self):
        """Setup RAG system for structural ambiguity (simplified from original)"""
        if not self.use_rag:
            return
            
        self.log_message("Setting up RAG system...")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.log_message(f"Embedding model loaded: {EMBEDDING_MODEL}")
        
        # Setup ChromaDB
        persist_dir = "./chroma_db_structural"
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        self.collection = self.client.get_or_create_collection(
            name="structural_patterns",
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
        """Extract requirements from SRS text with structural indicators"""
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
                    # Check if it has structural ambiguity indicators
                    if self.has_structural_indicators(req_text):
                        requirements.append({
                            'text': req_text,
                            'source': source
                        })
        
        return requirements
    
    def has_structural_indicators(self, text: str) -> bool:
        """Check if text has structural ambiguity indicators"""
        indicators = [
            r'\b(with|using|by|in|on|for|from|to)\s+\w+',  # PP attachment
            r'\band\b|\bor\b',  # Coordination
            r'\b(that|which|who)\s+\w+',  # Relative clauses
            r'\b(using|with|by)\s+\w+.*\b(protocol|method|system|tool)'  # Modifier scope
        ]
        
        for pattern in indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def build_rag_index(self):
        """Build RAG index with structural patterns and SRS requirements"""
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
            metadatas.append({'type': 'structural_example'})
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
        """Initialize logging and GPU setup (matching anaphoric structure)"""
        print("=" * 80)
        print(f"UNIFIED STRUCTURAL {self.approach_type.upper()} AMBIGUITY ANALYSIS")
        if self.use_rag:
            print(f"WITH RAG ENHANCEMENT")
        print("=" * 80)
        
        self.log_message(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.log_message(f"GPU: {gpu_name}")
            self.log_message(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Load ground truth examples
        self.load_ground_truth_examples()
        
        # Setup RAG if enabled
        if self.use_rag:
            self.setup_rag_system()
        
        # Initialize similarity model for evaluation
        self.similarity_model = SentenceTransformer(EMBEDDING_MODEL)
    
    def load_model(self):
        """Load LLaMA model and tokenizer (matching anaphoric structure)"""
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
        """Create zero-shot prompt with Chain of Thought (matching anaphoric structure)"""
        system_prompt = """You are an expert requirements engineer. Your task is to identify and resolve structural ambiguities in software requirements.

Structural ambiguity occurs when a sentence can be parsed in multiple ways, leading to different meanings. Common patterns:

1. PP Attachment: "process data using algorithm specified in document" 
   - Does "specified" modify "algorithm" or "document"?

2. Coordination Scope: "validate and store data in database"
   - Does "in database" apply to both actions or just "store"?

3. Relative Clause Attachment: "log operations performed by users on systems"
   - Do users perform operations, or are operations performed on user systems?

4. Modifier Scope: "send requests to servers using encryption"
   - Do servers use encryption, or does the sending use encryption?

Follow this Chain of Thought process:

Step 1: Scan the requirement for structural ambiguity patterns
Step 2: For each pattern, identify what it could refer to
Step 3: Check if the reference is clear or ambiguous
Step 4: If ambiguous, rewrite to eliminate ambiguity
Step 5: Provide the complete fixed requirement

CRITICAL RULES:
- If you find structural ambiguities, assume they need fixing
- Clarify all ambiguous attachments and scope
- Always output the COMPLETE requirement text
- If no ambiguities found, output identical text

OUTPUT FORMAT (EXACTLY):
Ambiguity Type: [type of ambiguity found]
Problem Found: [which structures are ambiguous]
Fixed Requirement: [ONLY the complete requirement sentence - nothing else]
Explanation: [what you changed and why]

IMPORTANT: The Fixed Requirement field should contain ONLY the requirement sentence, no extra text or formatting."""

        # Add RAG context if available
        context_text = ""
        if self.use_rag:
            context = self.retrieve_context(requirement_text)
            if context:
                context_text = "\n\nRELEVANT EXAMPLES FROM KNOWLEDGE BASE:\n"
                for i, ctx in enumerate(context[:2], 1):
                    if ctx['metadata']['type'] == 'structural_example':
                        context_text += f"\nExample {i}:\n{ctx['text']}\n"
                    else:
                        context_text += f"\nSimilar requirement: {ctx['text']}\n"

        user_prompt = f"""{context_text}

Requirement: {requirement_text}

Think step by step:
1. What structural ambiguity patterns do I see?
2. What could each ambiguous structure refer to?
3. Are these references clear or ambiguous?
4. How should I fix each ambiguous structure?
5. What is the complete fixed requirement?

Provide your analysis and answer in the EXACT format specified."""

        return system_prompt, user_prompt
    
    def create_one_shot_prompt(self, requirement_text):
        """Create one-shot prompt with Chain of Thought (matching anaphoric structure)"""
        # Select best example
        example = self.ground_truth_examples[0] if self.ground_truth_examples else {
            'original': "The system shall process user requests using the security protocols defined in the manual.",
            'fixed': "The system shall process user requests using the security protocols that are defined in the manual."
        }
        
        system_prompt = f"""You are an expert requirements engineer. Your task is to identify and resolve structural ambiguities in software requirements.

Here's an example of how to think through this process:

Req: {example['original']}
Fixed: {example['fixed']}

Chain of Thought for the example:
1. Structural patterns found: PP attachment with "using...defined in"
2. "defined in the manual" could modify: "protocols" OR "using security protocols"
3. Ambiguity: Yes, unclear what "defined in the manual" modifies
4. Fix: Add "that are" to clearly attach to "protocols"
5. Result: Complete requirement with clear attachment

RULES:
- Always look for: PP attachment, coordination scope, relative clauses, modifier scope
- If structural ambiguities exist, they likely need fixing
- Clarify attachments and scope with additional words
- Output complete requirement

OUTPUT FORMAT (EXACTLY):
Ambiguity Type: [type found]
Problem Found: [specific structures that are ambiguous]
Fixed Requirement: [ONLY the complete requirement sentence - no extra text]
Explanation: [what was changed]

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

Now apply the same process:

Req: {requirement_text}
Fixed: [You complete this]

Think step by step like the example:
1. What structural patterns do I see?
2. What could each refer to?
3. Are they ambiguous?
4. How should I fix them?
5. What's the complete fixed requirement?

Provide your answer in the EXACT format specified."""

        return system_prompt, user_prompt
    
    def create_few_shot_prompt(self, requirement_text):
        """Create few-shot prompt with 3 examples and Chain of Thought (matching anaphoric structure)"""
        # Select 3 examples
        selected_examples = self.ground_truth_examples[:3] if len(self.ground_truth_examples) >= 3 else self.ground_truth_examples
        
        system_prompt = """You are an expert requirements engineer. Your task is to identify and resolve structural ambiguities in software requirements.

Here are examples of how to fix structural ambiguities:"""

        # Add exactly 3 examples
        for i, example in enumerate(selected_examples, 1):
            system_prompt += f"""

Example {i}:
Req: {example['original']}
Fixed: {example['fixed']}"""

        system_prompt += """

Chain of Thought Process:
1. SCAN: Look for structural patterns (PP attachment, coordination, relative clauses, modifier scope)
2. IDENTIFY: What does each ambiguous structure refer to?
3. CHECK: Is the attachment/scope clear or could it refer to multiple things?
4. FIX: If ambiguous, clarify with additional words or restructuring
5. OUTPUT: Complete requirement with all ambiguities resolved

CRITICAL RULES:
- If you see structural ambiguities, they almost always need fixing
- You MUST fix ambiguous attachments and scope
- Clarify with words like "that", "where", "both", specific noun repetition
- Always provide the COMPLETE fixed requirement
- Don't leave any structural ambiguities unfixed

OUTPUT FORMAT (MUST FOLLOW EXACTLY):
Ambiguity Type: [specific type of structural ambiguity]
Problem Found: [list the ambiguous structures]
Fixed Requirement: [ONLY the complete requirement sentence - no extra text or formatting]
Explanation: [describe what structures were clarified and how]

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

Now analyze this requirement using the same process:

Req: {requirement_text}
Fixed: [You complete this]

Follow the Chain of Thought:
1. SCAN for structural patterns
2. IDENTIFY what they refer to
3. CHECK if ambiguous
4. FIX by clarifying
5. OUTPUT complete fixed requirement

Remember: If you find structural ambiguities, you MUST fix them!
Provide your answer in the EXACT format specified above."""

        return system_prompt, user_prompt
    
    def create_llama_prompt(self, system_prompt, user_prompt):
        """Format prompt for LLaMA (matching anaphoric structure)"""
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return formatted_prompt
    
    def clean_text(self, text):
        """Clean text by removing ** prefix and extra whitespace (matching anaphoric structure)"""
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
        """Enhanced response parsing to extract only the fixed requirement (matching anaphoric structure)"""
        parsed = {
            'fixed_requirement': original_req,
            'explanation': 'No changes made',
            'ambiguity_type': 'None',
            'problem_found': 'No ambiguities detected'
        }
        
        try:
            # Clean up the response
            response_text = response_text.strip()
            
            # Multiple patterns to extract ONLY the fixed requirement
            fixed_patterns = [
                # Pattern 1: Look for **Fixed Requirement:** with bold formatting
                r'\*\*Fixed Requirement:\*\*\s*([^*]+?)(?=\s*\*\*Explanation|\s*\*\*Ambiguity|\s*$)',
                # Pattern 2: Regular Fixed Requirement:
                r'Fixed Requirement:\s*([^\n]+?)(?=\s*Explanation:|\s*Ambiguity Type:|\s*$)',
                # Pattern 3: Look for the sentence after "Fixed Requirement:" up to next field
                r'Fixed Requirement:\s*(.+?)(?=\n\s*(?:Explanation|Ambiguity|Problem):|$)',
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
                        not candidate.lower().startswith(('error', 'unable', 'cannot', 'ambiguity type')) and
                        '**' not in candidate and
                        'Explanation:' not in candidate and
                        'Problem Found:' not in candidate):
                        
                        fixed_text = candidate
                        break
            
            # If no pattern worked, try a more aggressive approach
            if not fixed_text:
                sentences = re.split(r'(?<=[.])\s+', response_text)
                for sentence in sentences:
                    sentence = self.clean_text(sentence)
                    
                    if (len(sentence) > 50 and 
                        ('shall' in sentence.lower() or 'will' in sentence.lower() or 'must' in sentence.lower()) and
                        not sentence.lower().startswith(('i replaced', 'the structure', 'ambiguity'))):
                        fixed_text = sentence
                        break
            
            if fixed_text and len(fixed_text) > 20:
                parsed['fixed_requirement'] = fixed_text
            
            # Extract other fields
            try:
                ambiguity_match = re.search(r'(?:\*\*)?Ambiguity Type(?:\*\*)?:\s*([^\n*]+)', response_text, re.IGNORECASE)
                if ambiguity_match:
                    parsed['ambiguity_type'] = ambiguity_match.group(1).strip()
                
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
        """Calculate cosine similarity between two texts (matching anaphoric structure)"""
        if not text1 or not text2:
            return 0.0
        
        embeddings = self.similarity_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def calculate_bleu_score(self, predicted, reference):
        """Calculate BLEU score between predicted and reference text (matching anaphoric structure)"""
        try:
            predicted_tokens = predicted.lower().split()
            reference_tokens = reference.lower().split()
            score = sentence_bleu([reference_tokens], predicted_tokens, smoothing_function=self.smoothing)
            return score
        except:
            return 0.0
    
    def fix_requirement(self, original_req, req_id, project_name, ground_truth):
        """Fix a single requirement using the chosen approach (matching anaphoric structure)"""
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
            
            # Generate with optimized parameters
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
                'ambiguity_type': parsed_result.get('ambiguity_type', 'Unknown'),
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
                'ambiguity_type': 'Error',
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
        """Run the complete analysis (matching anaphoric structure)"""
        # Load CSV
        self.log_message("Loading CSV data...")
        try:
            df = pd.read_csv(csv_file)
            self.log_message(f"Loaded {len(df)} requirements from {csv_file}")
            
            # Verify we have the correct columns
            required_columns = ['requirement_id', 'Name_of_Project', 'Original_Requirement', 'Fixed_Requirement']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column '{col}' in CSV")
                    
        except Exception as e:
            self.log_message(f"ERROR loading CSV: {e}")
            return
        
        # Process requirements
        self.log_message(f"Starting UNIFIED STRUCTURAL {self.approach_type} analysis...")
        results = []
        changes_made = 0
        processing_errors = 0
        exact_matches = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"UNIFIED STRUCTURAL {self.approach_type}"):
            req_id = row['requirement_id']
            project_name = str(row['Name_of_Project'])
            original_requirement = str(row['Original_Requirement']).strip()
            ground_truth = str(row['Fixed_Requirement']).strip()
            
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
            
            if 'Error' in result.get('ambiguity_type', ''):
                processing_errors += 1
            
            # Save checkpoints
            if (idx + 1) % 25 == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(f"temp_unified_structural_{self.approach_type}_{idx+1}.csv", index=False)
        
        # Save final results
        self.log_message("Saving final results...")
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_file, index=False)
        
        # Create summary (matching anaphoric structure)
        summary_file = f"unified_structural_summary_{self.approach_type}_results.csv"
        summary_df = results_df[['requirement_id', 'project_name', 'original_requirement', 
                                'predicted_fixed', 'ground_truth_fixed', 'was_changed', 'exact_match', 'ambiguity_type']]
        summary_df.to_csv(summary_file, index=False)
        
        # Print evaluation statistics (matching anaphoric structure)
        self.log_evaluation_summary(results_df)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\n?? UNIFIED STRUCTURAL {self.approach_type.upper()} COMPLETE!")
        print(f"?? Results: {self.output_file}")
    
    def log_evaluation_summary(self, results_df):
        """Log evaluation summary with metrics (matching anaphoric structure)"""
        total_processed = len(results_df)
        changes_made = len(results_df[results_df['was_changed'] == True])
        errors = len(results_df[results_df['ambiguity_type'] == 'Error'])
        valid_results = results_df[results_df['ambiguity_type'] != 'Error']
        
        if len(valid_results) == 0:
            self.log_message("No valid results to evaluate!")
            return
        
        # Calculate evaluation metrics
        exact_matches = len(valid_results[valid_results['exact_match'] == True])
        avg_similarity = valid_results['similarity_score'].mean()
        avg_bleu = valid_results['bleu_score'].mean()
        
        # Classification metrics (structural ambiguity detection)
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
        self.log_message(f"UNIFIED STRUCTURAL {self.approach_type.upper()} EVALUATION SUMMARY")
        self.log_message("=" * 80)
        self.log_message(f"Approach: {self.approach_type}{' + RAG' if self.use_rag else ''}")
        self.log_message(f"Dataset: {CSV_FILE}")
        
        self.log_message(f"\n?? PROCESSING STATISTICS:")
        self.log_message(f"  Total Requirements: {total_processed}")
        self.log_message(f"  Valid Processed: {len(valid_results)}")
        self.log_message(f"  Requirements Changed: {changes_made} ({change_rate:.1f}%)")
        self.log_message(f"  Processing Errors: {errors}")
        
        self.log_message(f"\n?? EVALUATION METRICS (vs Ground Truth):")
        self.log_message(f"  Exact Match Score: {exact_match_rate:.4f} ({exact_matches}/{len(valid_results)})")
        self.log_message(f"  BLEU Score: {avg_bleu:.4f}")
        self.log_message(f"  Semantic Similarity: {avg_similarity:.4f}")
        self.log_message(f"  Classification Accuracy: {accuracy:.4f}")
        self.log_message(f"  Classification Precision: {precision:.4f}")
        self.log_message(f"  Classification Recall: {recall:.4f}")
        self.log_message(f"  Classification F1-Score: {f1:.4f}")
        
        # Performance by ambiguity type
        if len(valid_results) > 0:
            type_performance = valid_results.groupby('ambiguity_type').agg({
                'exact_match': 'sum',
                'similarity_score': 'mean',
                'bleu_score': 'mean'
            }).round(4)
            
            if len(type_performance) > 0:
                self.log_message(f"\n?? PERFORMANCE BY AMBIGUITY TYPE:")
                for ambiguity_type, metrics in type_performance.iterrows():
                    type_count = len(valid_results[valid_results['ambiguity_type'] == ambiguity_type])
                    type_accuracy = metrics['exact_match'] / type_count if type_count > 0 else 0
                    self.log_message(f"  {ambiguity_type}:")
                    self.log_message(f"    Count: {type_count}")
                    self.log_message(f"    Exact Match: {type_accuracy:.4f}")
                    self.log_message(f"    Similarity: {metrics['similarity_score']:.4f}")
                    self.log_message(f"    BLEU: {metrics['bleu_score']:.4f}")
        
        # Show examples of good and poor performance
        if len(valid_results) > 0:
            self.log_message(f"\n? BEST PREDICTIONS (High Similarity):")
            best_results = valid_results.nlargest(3, 'similarity_score')
            for i, (_, result) in enumerate(best_results.iterrows(), 1):
                self.log_message(f"\nBest {i} (Similarity: {result['similarity_score']:.3f}):")
                self.log_message(f"  ID: {result['requirement_id']}")
                self.log_message(f"  ORIGINAL: {result['original_requirement'][:80]}...")
                self.log_message(f"  PREDICTED: {result['predicted_fixed'][:80]}...")
                self.log_message(f"  GROUND TRUTH: {result['ground_truth_fixed'][:80]}...")
                self.log_message(f"  EXACT MATCH: {result['exact_match']}")
            
            self.log_message(f"\n? WORST PREDICTIONS (Low Similarity):")
            worst_results = valid_results.nsmallest(3, 'similarity_score')
            for i, (_, result) in enumerate(worst_results.iterrows(), 1):
                self.log_message(f"\nWorst {i} (Similarity: {result['similarity_score']:.3f}):")
                self.log_message(f"  ID: {result['requirement_id']}")
                self.log_message(f"  ORIGINAL: {result['original_requirement'][:80]}...")
                self.log_message(f"  PREDICTED: {result['predicted_fixed'][:80]}...")
                self.log_message(f"  GROUND TRUTH: {result['ground_truth_fixed'][:80]}...")
            
            # Perfect matches examples
            exact_match_results = valid_results[valid_results['exact_match'] == True]
            if len(exact_match_results) > 0:
                self.log_message(f"\n?? PERFECT MATCHES (Examples):")
                for i, (_, result) in enumerate(exact_match_results.head(3).iterrows(), 1):
                    self.log_message(f"\nPerfect Match {i}:")
                    self.log_message(f"  ID: {result['requirement_id']}")
                    self.log_message(f"  TYPE: {result['ambiguity_type']}")
                    self.log_message(f"  ORIGINAL: {result['original_requirement'][:80]}...")
                    self.log_message(f"  FIXED: {result['predicted_fixed'][:80]}...")
        
        # Summary statistics
        self.log_message(f"\n?? SUMMARY STATISTICS:")
        self.log_message(f"  Average Original Length: {valid_results['original_length'].mean():.1f} chars")
        self.log_message(f"  Average Predicted Length: {valid_results['predicted_length'].mean():.1f} chars")
        self.log_message(f"  Average Ground Truth Length: {valid_results['ground_truth_length'].mean():.1f} chars")
        
        # RAG effectiveness (if RAG was used)
        if self.use_rag:
            self.log_message(f"\n?? RAG EFFECTIVENESS:")
            self.log_message(f"  RAG was used for all predictions")
            self.log_message(f"  Knowledge Base: {len(self.ground_truth_examples)} examples + {len(self.srs_requirements)} SRS requirements")
        
        self.log_message(f"\n?? OUTPUT FILES:")
        self.log_message(f"  Results CSV: {self.output_file}")
        self.log_message(f"  Summary CSV: unified_structural_summary_{self.approach_type}_results.csv")
        self.log_message(f"  Log File: {self.log_file}")
        
        self.log_message("=" * 80)


def main():
    """Main function (matching anaphoric structure)"""
    parser = argparse.ArgumentParser(description='Unified Structural Ambiguity Resolution')
    parser.add_argument('--approach', type=str, 
                        choices=['zero-shot', 'one-shot', 'few-shot'],
                        default='few-shot', 
                        help='Prompting approach')
    parser.add_argument('--use-rag', action='store_true',
                        help='Enable RAG enhancement')
    parser.add_argument('--csv', type=str, default='Structural.csv',
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
        print(f"? Error: {CSV_FILE} not found!")
        sys.exit(1)
    
    if args.use_rag and not os.path.exists(SRS_DOCS_PATH):
        print(f"?? Warning: SRS documents path '{SRS_DOCS_PATH}' not found!")
        print("   RAG will work with limited context.")
    
    approach_name = f"{args.approach}{' + RAG' if args.use_rag else ''}"
    print(f"?? Starting UNIFIED STRUCTURAL {approach_name} analysis...")
    print(f"   Input: {CSV_FILE}")
    print(f"   Evaluation: Original_Requirement ? Predicted vs Fixed_Requirement (Ground Truth)")
    if args.use_rag:
        print(f"   SRS Documents: {SRS_DOCS_PATH}")
    
    # Create and run resolver
    resolver = UnifiedStructuralResolver(approach_type=args.approach, use_rag=args.use_rag)
    resolver.setup()
    resolver.load_model()
    resolver.run_analysis(CSV_FILE)
    
    print(f"? Complete! Check the output files.")


if __name__ == "__main__":
    main()
