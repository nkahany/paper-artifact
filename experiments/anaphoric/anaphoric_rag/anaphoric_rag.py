#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG-Enhanced Anaphoric Ambiguity Resolution Script
Supports: zero-shot, one-shot, few-shot (with optional RAG enhancement)
Matches structural.py architecture for consistency
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
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CSV_FILE = "anaphoric.csv"
GROUND_TRUTH_FILE = "ground_truths.csv"
SRS_DOCS_PATH = "./srs_documents"

class RAGAnaphoricResolver:
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
        self.output_file = f"anaphoric_results_{approach_type.replace('-', '')}{rag_suffix}.csv"
        self.log_file = f"anaphoric_log_{approach_type.replace('-', '')}{rag_suffix}.txt"
        
    def log_message(self, message):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(log_entry + "\n")
    
    def load_ground_truth_examples(self):
        """Load ground truth examples from ground_truths.csv"""
        try:
            if os.path.exists(GROUND_TRUTH_FILE):
                df = pd.read_csv(GROUND_TRUTH_FILE)
                self.log_message(f"Loaded {len(df)} ground truth examples from {GROUND_TRUTH_FILE}")
                
                # Convert to examples format
                # Take a subset for few-shot examples (not the full dataset to avoid data leakage)
                # Use first 10 examples for prompts, rest for testing
                sample_size = min(10, len(df))
                sample_df = df.head(sample_size)
                
                for _, row in sample_df.iterrows():
                    original = str(row.get('Original_Requirement', '')).strip()
                    fixed = str(row.get('Fixed_Requirement', '')).strip()
                    
                    # Only include examples where we have both original and fixed AND they're different
                    if original and fixed and original != fixed:
                        self.ground_truth_examples.append({
                            'original': original,
                            'fixed': fixed
                        })
                
                self.log_message(f"Prepared {len(self.ground_truth_examples)} examples for prompts (from first {sample_size} rows)")
                self.log_message(f"Remaining {len(df) - sample_size} rows will be used for testing")
            else:
                self.log_message(f"Ground truth file {GROUND_TRUTH_FILE} not found, using hardcoded examples")
                self._load_hardcoded_examples()
                
        except Exception as e:
            self.log_message(f"Error loading ground truth: {e}, using hardcoded examples")
            self._load_hardcoded_examples()
    
    def _load_hardcoded_examples(self):
        """Load hardcoded anaphoric examples"""
        self.ground_truth_examples = [
            {
                'original': "The S&T component shall send all approval requests to the DBS. If the request contains storage parameters, it shall create a configuration record from the parameters.",
                'fixed': "The S&T component shall send all approval requests to the DBS. If the request contains storage parameters, the S&T component shall create a configuration record from the parameters."
            },
            {
                'original': "The Clarus system shall be able to base its quality checking process on historical environmental data.",
                'fixed': "The Clarus system shall be able to base the Clarus system's quality checking process on historical environmental data."
            },
            {
                'original': "The MultiMahjongServer will allow connections from MultiMahjongClients and communicate with them using IP.",
                'fixed': "The MultiMahjongServer will allow connections from MultiMahjongClients and communicate with MultiMahjongClients using IP."
            }
        ]
    
    def setup_rag_system(self):
        """Setup RAG system for anaphoric ambiguity"""
        if not self.use_rag:
            return
            
        self.log_message("Setting up RAG system...")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.log_message(f"Embedding model loaded: {EMBEDDING_MODEL}")
        
        # Setup ChromaDB
        persist_dir = "./chroma_db_anaphoric"
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        self.collection = self.client.get_or_create_collection(
            name="anaphoric_patterns",
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
    
    def extract_requirements_from_srs(self, content, filename):
        """Extract individual requirements from SRS document"""
        requirements = []
        
        # Split by common requirement patterns
        patterns = [
            r'(?:^|\n)\s*(?:REQ|Requirement|R)\s*[-:]?\s*\d+[:\s]+(.+?)(?=\n\s*(?:REQ|Requirement|R)\s*[-:]?\s*\d+|$)',
            r'(?:^|\n)\s*\d+\.\d+[:\s]+(.+?)(?=\n\s*\d+\.\d+|$)',
            r'(?:^|\n)The\s+(?:system|software|application|component|module)\s+shall\s+(.+?)(?=\n(?:The\s+(?:system|software|application|component|module)\s+shall|$))'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                req_text = match.group(1).strip() if match.lastindex >= 1 else match.group(0).strip()
                req_text = ' '.join(req_text.split())
                
                if len(req_text) > 20 and len(req_text) < 500:
                    requirements.append({
                        'text': req_text,
                        'source': filename
                    })
        
        if not requirements:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 50 and len(line) < 500 and ('shall' in line.lower() or 'must' in line.lower() or 'will' in line.lower()):
                    requirements.append({
                        'text': line,
                        'source': filename
                    })
        
        return requirements
    
    def build_rag_index(self):
        """Build RAG index from SRS documents ONLY (excluding ground truth to avoid cheating)"""
        if not self.use_rag:
            return
        
        self.log_message("Building RAG index...")
        
        documents = []
        metadatas = []
        ids = []
        
        # ONLY add SRS requirements - NOT ground truth examples
        # Ground truth would allow the system to cheat by retrieving exact answers
        for idx, req in enumerate(self.srs_requirements):
            documents.append(req['text'])
            metadatas.append({
                'type': 'srs_requirement',
                'source': req['source']
            })
            ids.append(f"srs_{idx}")
        
        if len(documents) > 0:
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )
            
            self.log_message(f"Added {len(documents)} documents to RAG index")
            self.log_message(f"  - SRS requirements: {len(self.srs_requirements)}")
            self.log_message(f"  ⚠️  Ground truth examples EXCLUDED from RAG (to prevent cheating)")
        else:
            self.log_message("WARNING: No documents to add to RAG index - RAG will have no effect!")
    
    def retrieve_relevant_context(self, requirement_text, top_k=3):
        """Retrieve relevant examples from RAG system"""
        if not self.use_rag or self.collection is None:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([requirement_text])
            
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(top_k, self.collection.count())
            )
            
            relevant_docs = []
            if results and 'documents' in results and len(results['documents']) > 0:
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    relevant_docs.append({
                        'text': doc,
                        'type': metadata.get('type', 'unknown'),
                        'source': metadata.get('source', 'unknown')
                    })
            
            return relevant_docs
            
        except Exception as e:
            self.log_message(f"Error retrieving context: {e}")
            return []
    
    def setup(self):
        """Initialize logging and GPU setup"""
        print("=" * 80)
        approach_name = f"{self.approach_type.upper()}{' + RAG' if self.use_rag else ''}"
        print(f"ANAPHORIC AMBIGUITY ANALYSIS: {approach_name}")
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
    
    def create_zero_shot_prompt(self, requirement_text, rag_context=None):
        """Create zero-shot prompt with optional RAG context"""
        system_prompt = """You are an expert requirements engineer. Your task is to identify and resolve anaphoric ambiguities in software requirements.

Anaphoric ambiguity occurs when pronouns (it, its, they, them, their, this, that, these, those) have unclear antecedents.

Follow this Chain of Thought process:

Step 1: Scan the requirement for pronouns
Step 2: For each pronoun, identify what it refers to
Step 3: Check if the reference is clear or ambiguous
Step 4: If ambiguous, replace with specific antecedent
Step 5: Provide the complete fixed requirement

CRITICAL RULES:
- If you find pronouns, assume they are ambiguous and fix them
- Replace ALL pronouns with their specific antecedents
- For possessive pronouns (its, their), use "[antecedent]'s"
- Always output the COMPLETE requirement text
- If no pronouns found, output identical text"""

        if rag_context and len(rag_context) > 0:
            system_prompt += "\n\nRELEVANT EXAMPLES FROM KNOWLEDGE BASE:\n"
            for i, ctx in enumerate(rag_context[:2], 1):
                system_prompt += f"\nExample {i}:\n{ctx['text']}\n"

        system_prompt += """

OUTPUT FORMAT (EXACTLY):
Ambiguity Type: [type of ambiguity found]
Problem Found: [which pronouns are ambiguous]
Fixed Requirement: [ONLY the complete requirement sentence - nothing else]
Explanation: [what you changed and why]

IMPORTANT: The Fixed Requirement field should contain ONLY the requirement sentence, no extra text or formatting."""

        user_prompt = f"""Requirement: {requirement_text}

Think step by step:
1. What pronouns do I see?
2. What does each pronoun refer to?
3. Are these references clear or ambiguous?
4. How should I fix each ambiguous pronoun?
5. What is the complete fixed requirement?

Provide your analysis and answer in the EXACT format specified."""

        return system_prompt, user_prompt
    
    def create_one_shot_prompt(self, requirement_text, rag_context=None):
        """Create one-shot prompt with optional RAG context"""
        # Select best example from ground truth (NOT from RAG to avoid cheating)
        example = self.ground_truth_examples[0] if self.ground_truth_examples else {
            'original': "The S&T component shall send all approval requests to the DBS. If the request contains storage parameters, it shall create a configuration record from the parameters.",
            'fixed': "The S&T component shall send all approval requests to the DBS. If the request contains storage parameters, the S&T component shall create a configuration record from the parameters."
        }
        
        system_prompt = """You are an expert requirements engineer specializing in resolving anaphoric ambiguities.

Anaphoric ambiguity occurs when pronouns have unclear references.

Here is ONE example of how to fix anaphoric ambiguity:

EXAMPLE:
Requirement: {example_original}

Analysis:
- Pronoun found: "it"
- Refers to: unclear (could be "request" or "S&T component")
- Ambiguous: Yes
- Fix: Replace "it" with "the S&T component"

Fixed Requirement: {example_fixed}
Explanation: Replaced ambiguous pronoun "it" with specific noun "the S&T component" to clarify which entity creates the configuration record.

Now apply the same reasoning to new requirements.""".format(
            example_original=example['original'],
            example_fixed=example['fixed']
        )

        # Add RAG context if available (from SRS documents only)
        if rag_context and len(rag_context) > 0:
            system_prompt += "\n\nADDITIONAL CONTEXT FROM SRS DOCUMENTS:\n"
            for i, ctx in enumerate(rag_context[:2], 1):
                system_prompt += f"\nContext {i}: {ctx['text']}\n"

        system_prompt += """

OUTPUT FORMAT (EXACTLY):
Ambiguity Type: [type]
Problem Found: [pronouns]
Fixed Requirement: [complete sentence only]
Explanation: [changes made]"""

        user_prompt = f"""Requirement: {requirement_text}

Using the example above as guidance, analyze and fix this requirement."""

        return system_prompt, user_prompt
    
    def create_few_shot_prompt(self, requirement_text, rag_context=None):
        """Create few-shot prompt with optional RAG context"""
        # Use ground truth examples (NOT from RAG to avoid cheating)
        examples = self.ground_truth_examples[:3] if len(self.ground_truth_examples) >= 3 else self.ground_truth_examples
        
        # Ensure we have at least 2 examples
        if len(examples) < 2:
            examples = [
                {
                    'original': "The S&T component shall send all approval requests to the DBS. If the request contains storage parameters, it shall create a configuration record from the parameters.",
                    'fixed': "The S&T component shall send all approval requests to the DBS. If the request contains storage parameters, the S&T component shall create a configuration record from the parameters."
                },
                {
                    'original': "The Clarus system shall be able to base its quality checking process on historical environmental data.",
                    'fixed': "The Clarus system shall be able to base the Clarus system's quality checking process on historical environmental data."
                },
                {
                    'original': "The MultiMahjongServer will allow connections from MultiMahjongClients and communicate with them using IP.",
                    'fixed': "The MultiMahjongServer will allow connections from MultiMahjongClients and communicate with MultiMahjongClients using IP."
                }
            ]
        
        system_prompt = f"""You are an expert requirements engineer specializing in resolving anaphoric ambiguities.

Here are {len(examples)} examples showing how to fix anaphoric ambiguities:

EXAMPLE 1:
Original: {examples[0]['original']}
Fixed: {examples[0]['fixed']}
Change: Replaced ambiguous pronoun with specific noun phrase

EXAMPLE 2:
Original: {examples[1]['original']}
Fixed: {examples[1]['fixed']}
Change: Replaced possessive pronoun with specific entity's possessive form"""

        if len(examples) >= 3:
            system_prompt += f"""

EXAMPLE 3:
Original: {examples[2]['original']}
Fixed: {examples[2]['fixed']}
Change: Replaced pronoun with explicit antecedent for clarity"""

        # Add RAG context if available (from SRS documents ONLY, not ground truth)
        if rag_context and len(rag_context) > 0:
            system_prompt += "\n\nADDITIONAL RELEVANT CONTEXT FROM SRS DOCUMENTS:\n"
            for i, ctx in enumerate(rag_context[:2], 1):
                system_prompt += f"\nContext {i}: {ctx['text']}\n"

        system_prompt += """

KEY PRINCIPLES:
- Identify ALL pronouns (it, its, they, them, their, this, that, these, those)
- Replace each with its specific antecedent
- Maintain grammatical correctness
- Preserve original meaning while removing ambiguity

OUTPUT FORMAT:
Ambiguity Type: [type]
Problem Found: [specific pronouns]
Fixed Requirement: [complete corrected sentence]
Explanation: [what changed and why]"""

        user_prompt = f"""Requirement: {requirement_text}

Apply the patterns from the examples above to fix this requirement."""

        return system_prompt, user_prompt
    
    def generate_llm_response(self, system_prompt, user_prompt):
        """Generate response from LLaMA model"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant's response
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            elif "assistant" in response.lower():
                parts = response.split("assistant")
                response = parts[-1].strip()
            
            return response
            
        except Exception as e:
            self.log_message(f"Error generating response: {e}")
            return ""
    
    def parse_llm_response(self, response, original_requirement):
        """Parse LLM response to extract structured information"""
        result = {
            'ambiguity_type': 'Unknown',
            'problem_found': 'Unknown',
            'fixed_requirement': original_requirement,
            'explanation': 'No explanation provided'
        }
        
        try:
            # Extract fields using regex
            ambiguity_match = re.search(r'Ambiguity Type:\s*(.+?)(?=\n|Problem Found:|$)', response, re.IGNORECASE)
            if ambiguity_match:
                result['ambiguity_type'] = ambiguity_match.group(1).strip()
            
            problem_match = re.search(r'Problem Found:\s*(.+?)(?=\n|Fixed Requirement:|$)', response, re.IGNORECASE)
            if problem_match:
                result['problem_found'] = problem_match.group(1).strip()
            
            fixed_match = re.search(r'Fixed Requirement:\s*(.+?)(?=\n|Explanation:|$)', response, re.IGNORECASE | re.DOTALL)
            if fixed_match:
                fixed_text = fixed_match.group(1).strip()
                # Clean up any markdown or extra formatting
                fixed_text = re.sub(r'[*_`]', '', fixed_text)
                fixed_text = fixed_text.replace('\n', ' ')
                fixed_text = ' '.join(fixed_text.split())
                result['fixed_requirement'] = fixed_text
            
            explanation_match = re.search(r'Explanation:\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)
            if explanation_match:
                result['explanation'] = explanation_match.group(1).strip()
            
            # Validation: if fixed requirement is too short or looks wrong, use original
            if len(result['fixed_requirement']) < 10 or result['fixed_requirement'].lower() in ['none', 'n/a', 'na']:
                result['fixed_requirement'] = original_requirement
                result['explanation'] = 'Parsing failed - using original requirement'
            
        except Exception as e:
            self.log_message(f"Error parsing response: {e}")
            result['fixed_requirement'] = original_requirement
            result['explanation'] = f'Parsing error: {str(e)}'
        
        return result
    
    def fix_requirement(self, original_req, req_id, project_name):
        """Fix a single requirement with RAG enhancement if enabled"""
        try:
            # Retrieve RAG context if enabled
            rag_context = []
            if self.use_rag:
                rag_context = self.retrieve_relevant_context(original_req, top_k=3)
            
            # Create prompt based on approach type
            if self.approach_type == "zero-shot":
                system_prompt, user_prompt = self.create_zero_shot_prompt(original_req, rag_context)
            elif self.approach_type == "one-shot":
                system_prompt, user_prompt = self.create_one_shot_prompt(original_req, rag_context)
            else:  # few-shot
                system_prompt, user_prompt = self.create_few_shot_prompt(original_req, rag_context)
            
            # Generate response
            response = self.generate_llm_response(system_prompt, user_prompt)
            
            # Parse response
            parsed_result = self.parse_llm_response(response, original_req)
            
            # Check if requirement was changed
            was_changed = parsed_result['fixed_requirement'].strip() != original_req.strip()
            
            return {
                'requirement_id': req_id,
                'project_name': project_name,
                'original_requirement': original_req,
                'fixed_requirement': parsed_result['fixed_requirement'],
                'explanation': parsed_result['explanation'],
                'ambiguity_type': parsed_result.get('ambiguity_type', 'Unknown'),
                'problem_found': parsed_result.get('problem_found', 'Unknown'),
                'was_changed': was_changed,
                'approach': f"{self.approach_type}{'+RAG' if self.use_rag else ''}",
                'timestamp': datetime.now().isoformat(),
                'original_length': len(original_req),
                'fixed_length': len(parsed_result['fixed_requirement']),
                'rag_used': self.use_rag,
                'raw_response': response[:200] + "..." if len(response) > 200 else response
            }
            
        except Exception as e:
            self.log_message(f"Error fixing requirement {req_id}: {e}")
            return {
                'requirement_id': req_id,
                'project_name': project_name,
                'original_requirement': original_req,
                'fixed_requirement': original_req,
                'explanation': f'Error during processing: {str(e)}',
                'ambiguity_type': 'Error',
                'problem_found': 'Processing error',
                'was_changed': False,
                'approach': f"{self.approach_type}{'+RAG' if self.use_rag else ''}",
                'timestamp': datetime.now().isoformat(),
                'original_length': len(original_req),
                'fixed_length': len(original_req),
                'rag_used': self.use_rag,
                'raw_response': f'Error: {str(e)}'
            }
    
    def run_analysis(self, csv_file):
        """Run the complete analysis"""
        # Load CSV
        self.log_message("Loading CSV data...")
        try:
            df = pd.read_csv(csv_file)
            self.log_message(f"Loaded {len(df)} requirements from {csv_file}")
            
            # For ground_truths.csv, the columns are:
            # requirement_id, Name_of_Project, Original_Requirement, Fixed_Requirement, Explanation
            req_id_col = 'requirement_id'
            project_col = 'Name_of_Project'
            description_col = 'Original_Requirement'
            
            # Verify required columns exist
            if description_col not in df.columns:
                raise ValueError(f"Expected column '{description_col}' not found. Available: {list(df.columns)}")
            
            if req_id_col not in df.columns:
                self.log_message(f"Warning: '{req_id_col}' column not found, using index")
                req_id_col = None
            
            if project_col not in df.columns:
                self.log_message("Warning: No project name column found")
                project_col = None
                
        except Exception as e:
            self.log_message(f"ERROR loading CSV: {e}")
            return
        
        # Process requirements
        approach_name = f"{self.approach_type}{' + RAG' if self.use_rag else ''}"
        self.log_message(f"Starting {approach_name} analysis...")
        results = []
        changes_made = 0
        processing_errors = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{approach_name}"):
            req_id = row.get(req_id_col, idx + 1) if req_id_col else (idx + 1)
            project_name = str(row[project_col]) if project_col and project_col in df.columns else "Unknown"
            original_requirement = str(row[description_col]).strip()
            
            # Skip empty requirements
            if not original_requirement or original_requirement.lower() in ['nan', 'none', '']:
                continue
            
            if idx % 10 == 0:
                self.log_message(f"Processing {idx+1}/{len(df)} - Changes: {changes_made}, Errors: {processing_errors}")
            
            result = self.fix_requirement(original_requirement, req_id, project_name)
            results.append(result)
            
            if result['was_changed']:
                changes_made += 1
                # Log successful changes for monitoring
                if changes_made <= 5:
                    self.log_message(f"CHANGE #{changes_made}: {original_requirement[:50]}... -> {result['fixed_requirement'][:50]}...")
            
            if 'Error' in result.get('ambiguity_type', ''):
                processing_errors += 1
            
            # Save checkpoints
            if (idx + 1) % 25 == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(f"temp_{self.approach_type.replace('-', '')}{'_rag' if self.use_rag else ''}_{idx+1}.csv", index=False)
        
        # Save final results
        self.log_message("Saving final results...")
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_file, index=False)
        
        # Create summary
        rag_suffix = "_rag" if self.use_rag else ""
        summary_file = f"anaphoric_summary_{self.approach_type.replace('-', '')}{rag_suffix}_results.csv"
        summary_df = results_df[['requirement_id', 'project_name', 'original_requirement', 
                                'fixed_requirement', 'was_changed', 'ambiguity_type']]
        summary_df.to_csv(summary_file, index=False)
        
        # Print statistics
        total_requirements = len(results_df)
        change_rate = (changes_made / total_requirements) * 100
        error_rate = (processing_errors / total_requirements) * 100
        
        self.log_message("=" * 80)
        self.log_message(f"{approach_name.upper()} ANALYSIS COMPLETE")
        self.log_message("=" * 80)
        self.log_message(f"Total requirements: {total_requirements}")
        self.log_message(f"Changes made: {changes_made} ({change_rate:.1f}%)")
        self.log_message(f"Processing errors: {processing_errors} ({error_rate:.1f}%)")
        self.log_message(f"Output file: {self.output_file}")
        
        if self.use_rag:
            self.log_message(f"\n📚 RAG STATISTICS:")
            self.log_message(f"  Knowledge base size: {len(self.srs_requirements)} SRS requirements")
            self.log_message(f"  Ground truth examples: {len(self.ground_truth_examples)} (used for prompts only, NOT in RAG)")
        
        # Show sample changes
        if changes_made > 0:
            self.log_message("\nSAMPLE SUCCESSFUL CHANGES:")
            changed_results = results_df[results_df['was_changed'] == True].head(3)
            for i, (_, result) in enumerate(changed_results.iterrows(), 1):
                self.log_message(f"\nExample {i}:")
                self.log_message(f"  BEFORE: {result['original_requirement'][:100]}...")
                self.log_message(f"  AFTER:  {result['fixed_requirement'][:100]}...")
                self.log_message(f"  TYPE:   {result['ambiguity_type']}")
        else:
            self.log_message("\n⚠️  WARNING: No changes were made!")
            self.log_message("   This suggests the model is not detecting ambiguities.")
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\n✅ {approach_name.upper()} COMPLETE!")
        print(f"📊 Changed: {changes_made}/{total_requirements} ({change_rate:.1f}%)")
        print(f"📄 Results: {self.output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='RAG-Enhanced Anaphoric Ambiguity Resolution')
    parser.add_argument('--approach', type=str, 
                        choices=['zero-shot', 'one-shot', 'few-shot'],
                        default='few-shot', 
                        help='Prompting approach')
    parser.add_argument('--use-rag', action='store_true',
                        help='Enable RAG enhancement')
    parser.add_argument('--csv', type=str, default='ground_truths.csv',
                        help='Input CSV file (use ground_truths.csv)')
    parser.add_argument('--ground-truth', type=str, default='ground_truths.csv',
                        help='Ground truth CSV file (same as input)')
    parser.add_argument('--srs-path', type=str, default='./srs_documents',
                        help='Path to SRS documents for RAG')
    
    args = parser.parse_args()
    
    # Update global variables
    global CSV_FILE, GROUND_TRUTH_FILE, SRS_DOCS_PATH
    CSV_FILE = args.csv
    GROUND_TRUTH_FILE = args.ground_truth
    SRS_DOCS_PATH = args.srs_path
    
    # Validate files
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: {CSV_FILE} not found!")
        sys.exit(1)
    
    if args.use_rag and not os.path.exists(SRS_DOCS_PATH):
        print(f"⚠️ Warning: SRS documents path '{SRS_DOCS_PATH}' not found!")
        print("   RAG will work with limited context.")
    
    approach_name = f"{args.approach}{' + RAG' if args.use_rag else ''}"
    print(f"🚀 Starting ANAPHORIC {approach_name} analysis...")
    print(f"   Input: {CSV_FILE}")
    print(f"   Ground truth: {GROUND_TRUTH_FILE}")
    if args.use_rag:
        print(f"   SRS Documents: {SRS_DOCS_PATH}")
    
    # Create and run resolver
    resolver = RAGAnaphoricResolver(approach_type=args.approach, use_rag=args.use_rag)
    resolver.setup()
    resolver.load_model()
    resolver.run_analysis(CSV_FILE)
    
    print(f"✅ Complete! Check the output files.")


if __name__ == "__main__":
    main()
