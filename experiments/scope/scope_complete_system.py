#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Scope Ambiguity Resolution System
Uses exp1b_base_dataset.csv for RAG, SRS documents for context, scope_fixed.csv for testing
Supports: zero-shot, one-shot, few-shot with optional RAG enhancement
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
from collections import defaultdict
warnings.filterwarnings('ignore')
import time



# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BASE_DATASET_FILE = "exp1b_base_dataset.csv"
SCOPE_FIXED_FILE = "scope_fixed.csv"
SRS_DOCS_PATH = "./srs_documents"

class ScopeAmbiguityResolver:
    def __init__(self, approach_type="few-shot", use_rag=False):
        self.approach_type = approach_type
        self.use_rag = use_rag
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        # RAG components
        self.embedding_model = None
        self.client = None
        self.collection = None
        
        # Data storage
        self.base_dataset_examples = []  # For RAG
        self.scope_test_examples = []    # For testing
        self.srs_requirements = []       # SRS context
        self.scope_patterns = defaultdict(list)
        
        # Output files
        rag_suffix = "_rag" if use_rag else ""
        self.output_file = f"scope_results_{approach_type.replace('-', '')}{rag_suffix}.csv"
        self.log_file = f"scope_log_{approach_type.replace('-', '')}{rag_suffix}.txt"
        
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
    
    # Add this global variable to track execution
    EXECUTION_START_TIME = time.time()
    
    def log_execution_time(message):
        """Log execution time from start"""
        elapsed = time.time() - EXECUTION_START_TIME
        print(f"[{elapsed:.1f}s] {message}")

    def load_datasets(self):
      """Load all datasets: base for RAG, scope for testing"""
      self.log_message("Loading datasets...")
      
      # Load exp1b_base_dataset.csv for RAG
      try:
          # Try different encodings with error handling
          encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'windows-1252']
          base_df = None
          
          for encoding in encodings_to_try:
              try:
                  base_df = pd.read_csv(BASE_DATASET_FILE, encoding=encoding)
                  self.log_message(f"Loaded {len(base_df)} examples from {BASE_DATASET_FILE} using {encoding} encoding")
                  break
              except (UnicodeDecodeError, Exception) as e:
                  self.log_message(f"Failed to read {BASE_DATASET_FILE} with {encoding}: {e}")
                  continue
          
          if base_df is None:
              # If all encodings fail, use utf-8 with error replacement
              base_df = pd.read_csv(BASE_DATASET_FILE, encoding='utf-8', errors='replace')
              self.log_message(f"Loaded {len(base_df)} examples from {BASE_DATASET_FILE} with error replacement")
          
          # Clean the data to remove any problematic characters
          for col in base_df.select_dtypes(include=['object']).columns:
              base_df[col] = base_df[col].apply(self.clean_text)
    
          # Process base dataset into structured examples
          sentence_groups = base_df.groupby('sentence')
          for sentence, group in sentence_groups:
              interpretations = []
              
              for _, row in group.iterrows():
                  interpretations.append({
                      'option_a': str(row['Option A']) if pd.notna(row['Option A']) else '',
                      'option_b': str(row['Option B']) if pd.notna(row['Option B']) else '',
                      'gold_answer': str(row['gold_ans']) if pd.notna(row['gold_ans']) else '',
                      'scope_label': str(row['gold_scope_label']) if pd.notna(row['gold_scope_label']) else '',
                      'op1': str(row['OP1']) if pd.notna(row['OP1']) else '',
                      'op1_type': str(row['OP1_type']) if pd.notna(row['OP1_type']) else '',
                      'op2': str(row['OP2']) if pd.notna(row['OP2']) else '',
                      'op2_type': str(row['OP2_type']) if pd.notna(row['OP2_type']) else ''
                  })
              
              # Get correct interpretation
              correct_interp = None
              for interp in interpretations:
                  if interp['gold_answer'] == 'A':
                      correct_interp = interp['option_a']
                  elif interp['gold_answer'] == 'B':
                      correct_interp = interp['option_b']
                  if correct_interp:
                      break
              
              if correct_interp:
                  self.base_dataset_examples.append({
                      'sentence': str(sentence),
                      'correct_interpretation': str(correct_interp),
                      'interpretations': interpretations,
                      'pattern_type': self.classify_scope_pattern(str(sentence)),
                      'operators': interpretations[0] if interpretations else {}
                  })
                  
      except Exception as e:
          self.log_message(f"Error loading {BASE_DATASET_FILE}: {e}")
      
      # Load scope_fixed.csv for testing
      try:
          # Try different encodings
          for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
              try:
                  scope_df = pd.read_csv(SCOPE_FIXED_FILE, encoding=encoding)
                  self.log_message(f"Loaded {len(scope_df)} examples from {SCOPE_FIXED_FILE} using {encoding} encoding")
                  break
              except UnicodeDecodeError:
                  continue
          else:
              # If all encodings fail, use utf-8 with error handling
              scope_df = pd.read_csv(SCOPE_FIXED_FILE, encoding='utf-8', errors='replace')
              self.log_message(f"Loaded {len(scope_df)} examples from {SCOPE_FIXED_FILE} with error replacement")
          
          for _, row in scope_df.iterrows():
              self.scope_test_examples.append({
                  'description': str(row['Description']) if pd.notna(row['Description']) else '',
                  'fixed_requirement': str(row['Fixed_Requirement']) if pd.notna(row['Fixed_Requirement']) else '',
                  'scope_ambiguity': int(row['Scope_ambiguity']) if pd.notna(row['Scope_ambiguity']) else 0,
                  'source_of_fix': str(row['Source_of_Fix']) if pd.notna(row['Source_of_Fix']) else '',
                  'explanation_of_fix': str(row['Explanation_of_Fix']) if pd.notna(row['Explanation_of_Fix']) else ''
              })
              
      except Exception as e:
          self.log_message(f"Error loading {SCOPE_FIXED_FILE}: {e}")
      
      self.log_message(f"Dataset loading complete: {len(self.base_dataset_examples)} RAG examples, {len(self.scope_test_examples)} test examples")
    
    def classify_scope_pattern(self, sentence):
        """Classify the type of scope ambiguity"""
        sentence_lower = sentence.lower()
        
        if re.search(r'\b(a|an)\s+\w+.*\d+', sentence_lower):
            return 'quantifier_scope'
        elif re.search(r'\bnot\s+\w+.*\b(all|every)', sentence_lower):
            return 'negation_scope'
        elif re.search(r'\b(must|can|should).*\b(every|all)', sentence_lower):
            return 'modal_scope'
        elif re.search(r'\b(every|each).*\band\b', sentence_lower):
            return 'coordination_scope'
        elif re.search(r'\b(only|just|even)', sentence_lower):
            return 'focus_scope'
        else:
            return 'quantifier_scope'
    
    def load_srs_documents(self):
        """Load SRS documents for enhanced context"""
        if not self.use_rag or not os.path.exists(SRS_DOCS_PATH):
            return
            
        self.log_message("Loading SRS documents...")
        
        for filename in os.listdir(SRS_DOCS_PATH):
            if filename.endswith('.txt'):
                filepath = os.path.join(SRS_DOCS_PATH, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        requirements = self.extract_srs_requirements(content, filename)
                        self.srs_requirements.extend(requirements)
                except Exception as e:
                    self.log_message(f"Error reading {filepath}: {e}")
        
        self.log_message(f"Loaded {len(self.srs_requirements)} SRS requirements")
    
    def extract_srs_requirements(self, text, source):
        """Extract requirements from SRS documents"""
        requirements = []
        
        # Patterns for requirement extraction
        patterns = [
            r'REQ-[A-Z0-9-]+[:\s]*(.+?)(?=REQ-[A-Z0-9-]+|\n\n|$)',
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
                    if self.has_scope_indicators(req_text):
                        requirements.append({
                            'text': req_text,
                            'source': source,
                            'pattern_type': self.classify_scope_pattern(req_text)
                        })
        
        return requirements
    
    def has_scope_indicators(self, text):
        """Check if text has scope ambiguity indicators"""
        indicators = [
            r'\b(a|an|the)\s+\w+\s+\w+\s+\d+',
            r'\b(every|each|all|some|many|few)\s+\w+',
            r'\b(not|never|rarely)\s+\w+',
            r'\b(must|shall|can|may|should)\s+\w+',
            r'\b(only|just|even)\s+\w+'
        ]
        
        for pattern in indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def setup_rag(self):
        """Setup RAG system with base dataset and SRS documents"""
        if not self.use_rag:
            return
            
        self.log_message("Setting up RAG system...")
        
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self.client = chromadb.Client()
            self.collection = self.client.create_collection("scope_rag_knowledge")
            
            self.build_rag_index()
            
        except Exception as e:
            self.log_message(f"RAG setup failed: {e}. Continuing without RAG.")
            self.use_rag = False
    
    def build_rag_index(self):
        """Build comprehensive RAG index"""
        documents = []
        metadatas = []
        ids = []
        
        # Add base dataset examples
        for i, example in enumerate(self.base_dataset_examples):
            # Main example document
            doc_text = f"""SCOPE AMBIGUITY EXAMPLE

Pattern Type: {example['pattern_type']}
Original: {example['sentence']}
Correct Interpretation: {example['correct_interpretation']}

Explanation: This demonstrates {example['pattern_type']} where the scope of operators affects meaning. The preferred interpretation follows standard scope resolution principles.

Operators: {example['operators'].get('op1', '')} ({example['operators'].get('op1_type', '')}), {example['operators'].get('op2', '')} ({example['operators'].get('op2_type', '')})"""

            documents.append(doc_text)
            metadatas.append({'type': 'base_example', 'pattern_type': example['pattern_type']})
            ids.append(f"base_{i}")
            
            # Alternative interpretations document
            alt_interpretations = []
            for interp in example['interpretations']:
                alt_interpretations.extend([interp['option_a'], interp['option_b']])
            
            if len(set(alt_interpretations)) > 1:
                alt_doc = f"""ALTERNATIVE INTERPRETATIONS

For: {example['sentence']}
Pattern: {example['pattern_type']}

Possible interpretations:
{chr(10).join([f"- {interp}" for interp in set(alt_interpretations)])}

Correct choice: {example['correct_interpretation']}"""

                documents.append(alt_doc)
                metadatas.append({'type': 'alternatives', 'pattern_type': example['pattern_type']})
                ids.append(f"alt_{i}")
        
        # Add SRS requirements
        for i, req in enumerate(self.srs_requirements):
            srs_doc = f"""SRS REQUIREMENT CONTEXT

Pattern Type: {req['pattern_type']}
Requirement: {req['text']}
Source Document: {req['source']}

Context: This requirement demonstrates scope ambiguity patterns in software requirements. Understanding the domain context helps resolve ambiguous scope relationships."""

            documents.append(srs_doc)
            metadatas.append({'type': 'srs_context', 'pattern_type': req['pattern_type'], 'source': req['source']})
            ids.append(f"srs_{i}")
        
        # Add pattern-specific guidance
        pattern_guides = self.create_pattern_guidance()
        for pattern_type, guide_text in pattern_guides.items():
            documents.append(guide_text)
            metadatas.append({'type': 'guidance', 'pattern_type': pattern_type})
            ids.append(f"guide_{pattern_type}")
        
        # Encode and store
        if documents:
            embeddings = self.embedding_model.encode(documents, show_progress_bar=True).tolist()
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            
            self.log_message(f"Built RAG index with {len(documents)} documents")
    def create_pattern_guidance(self):
      """Create pattern-specific guidance documents"""
      
      return {
        'quantifier_scope': """QUANTIFIER SCOPE RESOLUTION GUIDE

Key patterns:
- Indefinite article + number: "A chef prepares seven dishes"
- Universal + existential: "Every student read some books"

Resolution strategy:
1. Identify quantifiers and their scopes
2. Consider surface (exists x, for all y) vs inverse (for all y, exists x) readings
3. Apply pragmatic preference for surface reading
4. Use context clues from domain

Example: "A programmer debugs three applications" - Surface reading: one programmer, all three applications""",

        'negation_scope': """NEGATION SCOPE RESOLUTION GUIDE

Key patterns:
- Negation + universal: "Not all users can access"
- Negation + existential: "No user should..."

Resolution strategy:
1. Identify negation operator scope
2. Distinguish narrow (not exists x such that P(x)) vs wide (exists x such that not P(x)) scope
3. Apply principle of charity
4. Consider logical equivalences

Example: "Not all requests are processed" - Narrow scope: some requests are not processed""",

        'modal_scope': """MODAL SCOPE RESOLUTION GUIDE

Key patterns:
- Modal + quantifier: "Someone must solve every problem"
- Quantifier + modal: "Every problem must be solved"

Resolution strategy:
1. Identify modal operators (must, can, should)
2. Determine scope over quantifiers
3. Consider deontic vs epistemic readings
4. Apply contextual constraints

Example: "Every user must authenticate" - Universal obligation: all users have authentication requirement""",

        'coordination_scope': """COORDINATION SCOPE RESOLUTION GUIDE

Key patterns:
- Quantifier + coordination: "Every user can read and write"
- Coordination + quantifier: "Read and write every file"

Resolution strategy:
1. Identify coordinated elements
2. Determine distributive vs collective reading
3. Consider scope of quantifiers over coordination
4. Apply default distributive interpretation

Example: "Each developer can read and write code" - Distributive: each has both abilities""",

        'focus_scope': """FOCUS SCOPE RESOLUTION GUIDE

Key patterns:
- Focus particles: "Only managers can approve"
- Contrastive focus: "EVEN students can access"

Resolution strategy:
1. Identify focus particles and alternatives
2. Determine focus domain and scope
3. Consider presuppositions and implicatures
4. Apply exclusivity or additivity

Example: "Only authorized users can access" - Exclusive: no unauthorized users can access"""
    }
    
    def retrieve_rag_context(self, query_sentence, top_k=5):
        """Retrieve relevant context from RAG system"""
        if not self.use_rag or not self.collection:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query_sentence]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            
            context = []
            for i, doc in enumerate(results['documents'][0]):
                context.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            return context
            
        except Exception as e:
            self.log_message(f"RAG retrieval failed: {e}")
            return []
    
    # Replace the load_model method in ScopeAmbiguityResolver class

    def load_model(self):
        """Load the language model with strict error handling"""
        self.log_message(f"Loading model: {MODEL_NAME}")
        
        try:
            # Step 1: Load tokenizer
            self.log_message("Step 1: Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.log_message(f"? Tokenizer loaded successfully! Vocab size: {len(self.tokenizer)}")
            
            # Step 2: Load model  
            self.log_message("Step 2: Loading model...")
            self.log_message(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto", 
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.log_message(f"? Model loaded successfully!")
            
            # Step 3: CRITICAL - Test model inference
            self.log_message("Step 3: Testing model inference...")
            test_input = "Hello, how are you?"
            inputs = self.tokenizer(test_input, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            self.log_message(f"? Model test successful: '{test_input}' -> '{response}'")
            
            # If we get here, everything is working
            return True
            
        except Exception as e:
            self.log_message(f"? CRITICAL ERROR: Model loading failed: {e}")
            self.log_message(f"Error type: {type(e).__name__}")
            
            # Set flags to None so we know it failed
            self.model = None
            self.tokenizer = None
            
            # FORCE stop execution - don't continue silently
            raise RuntimeError(f"Model loading failed: {e}. Cannot continue without working model.")

    def generate_response(self, system_prompt, user_prompt, max_tokens=512):
      """Generate model response with strict validation"""
        
        # CRITICAL CHECK - Don't proceed if model failed to load
        
      if self.model is None or self.tokenizer is None:
        error_msg = "Cannot generate response: Model or tokenizer failed to load"
        self.log_message(f"? {error_msg}")
        raise RuntimeError(error_msg)
      
      try:
        self.log_message("?? Generating model response...")
        start_time = time.time()
          
        messages = [
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt}
          ]
          
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(self.device)
          
        self.log_message(f"Input tokens: {inputs.shape[-1]}")
        
        # This should take significant time for large models
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        self.log_message(f"? Response generated in {generation_time:.2f}s: '{response[:100]}...'")
        
        # If generation was too fast, something's wrong
        if generation_time < 1.0 and max_tokens > 100:
            self.log_message(f"?? WARNING: Generation was very fast ({generation_time:.2f}s) - this may indicate an issue")
        
        return response.strip()
          
      except Exception as e:
          error_msg = f"Generation failed: {e}"
          self.log_message(f"? {error_msg}")
          raise RuntimeError(error_msg)
    
    def create_prompts(self, sentence):
        """Create prompts based on approach type"""
        # Get RAG context if enabled
        rag_context = ""
        if self.use_rag:
            context = self.retrieve_rag_context(sentence)
            if context:
                rag_context = "\n\nRELEVANT CONTEXT FROM KNOWLEDGE BASE:\n"
                for ctx in context[:3]:
                    if ctx['metadata']['type'] == 'base_example':
                        rag_context += f"\nScope Example: {ctx['text'][:300]}...\n"
                    elif ctx['metadata']['type'] == 'guidance':
                        rag_context += f"\nResolution Strategy: {ctx['text'][:300]}...\n"
                    elif ctx['metadata']['type'] == 'srs_context':
                        rag_context += f"\nSRS Context: {ctx['text'][:200]}...\n"
        
        # Base system prompt
        system_prompt = """You are an expert in requirements engineering and scope ambiguity resolution. Your task is to identify and resolve scope ambiguities in software requirements.

Scope ambiguity occurs when quantifiers, operators, or modifiers can have different scopes, leading to different interpretations:

1. QUANTIFIER SCOPE: "Each system shall maintain equipment list"
   - Distributive: Each system has its own separate list
   - Collective: One shared list for all systems

2. UNIQUENESS SCOPE: "Each component shall have unique ID"  
   - Local uniqueness: Unique within each component type
   - Global uniqueness: Unique across entire system

3. NEGATION SCOPE: "Not all users can access system"
   - Narrow scope: Some users cannot access (?)
   - Wide scope: No users can access (?)

4. MODAL SCOPE: "System must validate each input"
   - Universal obligation: Every input must be validated
   - Existential requirement: System has validation capability

5. COORDINATION SCOPE: "Users can read and write data"
   - Distributive: Each user has both abilities
   - Collective: Users collectively have abilities

RESOLUTION STRATEGY:
1. Identify scope-bearing elements (quantifiers, modals, negation, coordination)
2. Determine possible scope interpretations
3. Choose the interpretation that best serves the system requirements
4. Make scope relationships explicit through precise language

OUTPUT FORMAT (EXACTLY):
Scope Elements: [quantifiers, modals, negation found]
Ambiguity Type: [quantifier/uniqueness/negation/modal/coordination scope]
Problem Found: [describe the specific ambiguity]
Fixed Requirement: [ONLY the complete disambiguated requirement]
Explanation: [justify the chosen interpretation]

CRITICAL: The Fixed Requirement field should contain ONLY the requirement sentence, no extra text."""
        
        # Add examples based on approach
        if self.approach_type == "few-shot":
            system_prompt += """

EXAMPLES FROM REQUIREMENTS:

Example 1:
Original: The EMC shall be able to maintain an equipment list for each station.
Fixed: The EMC shall maintain a separate equipment list for each individual station.
Type: Quantifier-Quantifier scope ambiguity
Explanation: 'each station' clarified as distributive - separate individual lists per station

Example 2:
Original: Each QChS shall produce a unique quality flag value.
Fixed: Each QChS shall produce a quality flag value that is unique across the entire system.
Type: Uniqueness scope ambiguity  
Explanation: Uniqueness scope clarified as global across entire system rather than local

Example 3:
Original: The system shall process all user requests efficiently.
Fixed: The system shall process each user request with efficient response time.
Type: Quantifier scope with performance constraint
Explanation: Clarified distributive processing with individual efficiency requirements"""
            
        elif self.approach_type == "one-shot":
            system_prompt += """

EXAMPLE FROM REQUIREMENTS:
Original: The EMC shall be able to maintain an equipment list for each station.
Fixed: The EMC shall maintain a separate equipment list for each individual station.
Type: Quantifier-Quantifier scope ambiguity
Explanation: Clarified 'each station' as distributive relationship - separate lists rather than one shared list"""
        
        # Add RAG context
        system_prompt += rag_context
        
        # User prompt
        user_prompt = f"""Analyze this software requirement for scope ambiguity:

Requirement: {sentence}

Follow the resolution process:
1. Identify scope-bearing elements (quantifiers, modals, negation, coordination)
2. Determine what scope interpretations are possible
3. Assess which interpretation best serves the system requirements
4. Choose the most precise and unambiguous interpretation
5. Rewrite the requirement to make the scope explicit

Focus on requirements engineering principles:
- Clarity and precision in system behavior
- Unambiguous specification of system responsibilities  
- Clear boundaries and constraints
- Testable and verifiable requirements

Provide your answer in the EXACT format specified."""
        
        return system_prompt, user_prompt
    
    
    
    def parse_response(self, response):
        """Parse model response for requirements format"""
        try:
            patterns = {
                'scope_elements': r'Scope Elements:\s*(.+?)(?=\n\w+:|$)',
                'ambiguity_type': r'Ambiguity Type:\s*(.+?)(?=\n\w+:|$)',
                'problem_found': r'Problem Found:\s*(.+?)(?=\n\w+:|$)',
                'fixed_requirement': r'Fixed Requirement:\s*(.+?)(?=\n\w+:|$)',
                'explanation': r'Explanation:\s*(.+?)(?=\n\w+:|$)',
            }
            
            parsed = {}
            for field, pattern in patterns.items():
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                parsed[field] = match.group(1).strip() if match else ""
            
            # Clean fixed requirement
            if parsed['fixed_requirement']:
                parsed['fixed_requirement'] = re.sub(r'^["\']+|["\']+$', '', parsed['fixed_requirement'])
                parsed['fixed_requirement'] = re.sub(r'\s+', ' ', parsed['fixed_requirement']).strip()
            
            return parsed
            
        except Exception as e:
            return {
                'scope_elements': 'Parse Error',
                'ambiguity_type': 'Parse Error',
                'problem_found': f'Error: {e}',
                'fixed_requirement': '',
                'explanation': 'Parse failed'
            }
    
    def calculate_metrics(self, predicted, ground_truth):
        """Calculate evaluation metrics"""
        # Exact match
        exact_match = predicted.lower().strip() == ground_truth.lower().strip()
        
        # Semantic similarity
        try:
            if not self.similarity_model:
                self.similarity_model = SentenceTransformer(EMBEDDING_MODEL)
            pred_emb = self.similarity_model.encode([predicted])
            gt_emb = self.similarity_model.encode([ground_truth])
            similarity = cosine_similarity(pred_emb, gt_emb)[0][0]
        except:
            similarity = SequenceMatcher(None, predicted.lower(), ground_truth.lower()).ratio()
        
        # BLEU score
        try:
            pred_tokens = predicted.lower().split()
            gt_tokens = ground_truth.lower().split()
            if len(pred_tokens) > 0 and len(gt_tokens) > 0:
                bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=self.smoothing)
            else:
                bleu = 0.0
        except:
            bleu = 0.0
        
        return exact_match, float(similarity), float(bleu)
    
    def create_test_dataset(self):
        """Create test dataset from scope_fixed.csv"""
        self.log_message("Creating test dataset from scope_fixed.csv...")
        
        test_data = []
        for i, example in enumerate(self.scope_test_examples):
            test_data.append({
                'requirement_id': f"SCOPE_{i+1:03d}",
                'project_name': example['source_of_fix'].replace('.pdf', '').replace(' - ', '_') if example['source_of_fix'] else 'ScopeTest',
                'original_requirement': example['description'],
                'ground_truth_fixed': example['fixed_requirement'],
                'has_scope_ambiguity': example['scope_ambiguity'],
                'source_of_fix': example['source_of_fix'],
                'explanation_of_fix': example['explanation_of_fix']
            })
        
        test_df = pd.DataFrame(test_data)
        test_df.to_csv("scope_test.csv", index=False)
        self.log_message(f"Created scope_test.csv with {len(test_df)} examples from scope_fixed.csv")
        
        return test_df
    
    def run_scope_resolution(self):
        """Run the complete scope resolution process"""
        self.log_message("=" * 80)
        self.log_message(f"STARTING SCOPE AMBIGUITY RESOLUTION - {self.approach_type.upper()}")
        if self.use_rag:
            self.log_message("RAG ENHANCEMENT: ENABLED")
        self.log_message("=" * 80)
        
        # Load datasets and setup
        self.load_datasets()
        self.load_srs_documents()
        
        if self.use_rag:
            self.setup_rag()
        
        # Create test dataset
        test_df = self.create_test_dataset()
        
        # Load model
        self.load_model()
        
        # Process each test example
        results = []
        changes_made = 0
        exact_matches = 0
        processing_errors = 0
        
        self.log_message(f"Processing {len(test_df)} test examples...")
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
            original = row['original_requirement']
            ground_truth = row['ground_truth_fixed']
            req_id = row['requirement_id']
            
            try:
                # Create prompts
                system_prompt, user_prompt = self.create_prompts(original)
                
                # Generate response
                response = self.generate_response(system_prompt, user_prompt)
                
                # Parse response
                parsed = self.parse_response(response)
                
                # Get predicted fix
                predicted_fixed = parsed['fixed_requirement'] if parsed['fixed_requirement'] else original
                
                # Calculate metrics
                exact_match, similarity, bleu = self.calculate_metrics(predicted_fixed, ground_truth)
                was_changed = predicted_fixed.lower() != original.lower()
                
                # Store result
                result = {
                    'requirement_id': req_id,
                    'project_name': test_df.loc[idx, 'project_name'],
                    'original_requirement': original,
                    'predicted_fixed': predicted_fixed,
                    'ground_truth_fixed': ground_truth,
                    'was_changed': was_changed,
                    'exact_match': exact_match,
                    'scope_elements': parsed['scope_elements'],
                    'ambiguity_type': parsed['ambiguity_type'],
                    'problem_found': parsed['problem_found'],
                    'explanation': parsed['explanation'],
                    'similarity_score': similarity,
                    'bleu_score': bleu,
                    'source_of_fix': test_df.loc[idx, 'source_of_fix'],
                    'ground_truth_explanation': test_df.loc[idx, 'explanation_of_fix']
                }
                
                results.append(result)
                
                if was_changed:
                    changes_made += 1
                if exact_match:
                    exact_matches += 1
                
                # Progress logging
                if (idx + 1) % 10 == 0:
                    acc = exact_matches / (idx + 1 - processing_errors) if (idx + 1 - processing_errors) > 0 else 0
                    self.log_message(f"Progress {idx+1}/{len(test_df)} - Changes: {changes_made}, Exact: {exact_matches}, Acc: {acc:.3f}")
                
            except Exception as e:
                processing_errors += 1
                self.log_message(f"Error processing {req_id}: {e}")
                results.append({
                    'requirement_id': req_id,
                    'project_name': test_df.loc[idx, 'project_name'] if 'project_name' in test_df.columns else 'ScopeTest',
                    'original_requirement': original,
                    'predicted_fixed': original,
                    'ground_truth_fixed': ground_truth,
                    'was_changed': False,
                    'exact_match': False,
                    'scope_elements': 'Error',
                    'ambiguity_type': 'Error',
                    'problem_found': f'Error: {e}',
                    'explanation': 'Processing failed',
                    'similarity_score': 0.0,
                    'bleu_score': 0.0,
                    'source_of_fix': test_df.loc[idx, 'source_of_fix'] if 'source_of_fix' in test_df.columns else '',
                    'ground_truth_explanation': test_df.loc[idx, 'explanation_of_fix'] if 'explanation_of_fix' in test_df.columns else ''
                })
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_file, index=False)
        
        # Log summary
        self.log_evaluation_summary(results_df)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.log_message(f"Scope resolution completed! Results saved to {self.output_file}")
        return results_df
    
    def log_evaluation_summary(self, results_df):
        """Log evaluation summary"""
        valid_results = results_df[results_df['scope_elements'] != 'Error']
        
        if len(valid_results) == 0:
            self.log_message("No valid results to evaluate!")
            return
        
        # Calculate metrics
        exact_matches = len(valid_results[valid_results['exact_match'] == True])
        changes_made = len(valid_results[valid_results['was_changed'] == True])
        avg_similarity = valid_results['similarity_score'].mean()
        avg_bleu = valid_results['bleu_score'].mean()
        
        # Classification metrics
        gt_labels = (valid_results['original_requirement'] != valid_results['ground_truth_fixed']).astype(int)
        pred_labels = valid_results['was_changed'].astype(int)
        
        accuracy = accuracy_score(gt_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(gt_labels, pred_labels, average='binary', zero_division=0)
        
        # Log results
        self.log_message("=" * 80)
        self.log_message(f"SCOPE AMBIGUITY RESOLUTION EVALUATION - {self.approach_type.upper()}")
        self.log_message("=" * 80)
        
        self.log_message(f"?? PROCESSING STATISTICS:")
        self.log_message(f"  Total Examples: {len(results_df)}")
        self.log_message(f"  Valid Processed: {len(valid_results)}")
        self.log_message(f"  Changes Made: {changes_made}")
        
        self.log_message(f"?? ACCURACY METRICS:")
        self.log_message(f"  Exact Match Rate: {exact_matches/len(valid_results):.3f} ({exact_matches}/{len(valid_results)})")
        self.log_message(f"  Average Similarity: {avg_similarity:.3f}")
        self.log_message(f"  Average BLEU: {avg_bleu:.3f}")
        
        self.log_message(f"?? DETECTION METRICS:")
        self.log_message(f"  Detection Accuracy: {accuracy:.3f}")
        self.log_message(f"  Precision: {precision:.3f}")
        self.log_message(f"  Recall: {recall:.3f}")
        self.log_message(f"  F1-Score: {f1:.3f}")
        
        self.log_message("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Scope Ambiguity Resolution System')
    parser.add_argument('--approach', choices=['zero-shot', 'one-shot', 'few-shot'], 
                       default='few-shot', help='Prompting approach')
    parser.add_argument('--use-rag', action='store_true', 
                       help='Enable RAG enhancement')
    
    args = parser.parse_args()
    
    print(f"?? Starting Scope Ambiguity Resolution")
    print(f"?? Approach: {args.approach}")
    print(f"?? RAG: {'Enabled' if args.use_rag else 'Disabled'}")
    
    resolver = ScopeAmbiguityResolver(approach_type=args.approach, use_rag=args.use_rag)
    results = resolver.run_scope_resolution()
    
    print(f"? Process completed!")
        

if __name__ == "__main__":
    main()