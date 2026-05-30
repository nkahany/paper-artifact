#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New Chain-of-Thought Anaphoric Ambiguity Resolution Script
Fixed prompt templates with aggressive pronoun detection
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import os
import re
from datetime import datetime
import argparse
import sys
import random

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CSV_FILE = "anaphoric.csv"
GROUND_TRUTH_FILE = "ground_truths.csv"

class NewCOTAnaphoricResolver:
    def __init__(self, approach_type="few-shot"):
        self.approach_type = approach_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.ground_truth_examples = []
        
        # Output files based on approach
        self.output_file = f"new_cot_fixed_requirements_{approach_type.replace('-', '')}.csv"
        self.log_file = f"new_cot_analysis_log_{approach_type.replace('-', '')}.txt"
        
    def log_message(self, message):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(log_entry + "\n")
    
    def load_ground_truth_examples(self):
        """Load and prepare ground truth examples"""
        try:
            if os.path.exists(GROUND_TRUTH_FILE):
                df = pd.read_csv(GROUND_TRUTH_FILE)
                self.log_message(f"Loaded {len(df)} ground truth examples")
                
                # Convert to examples format
                for _, row in df.iterrows():
                    example = {
                        'original': str(row.get('Original_Requirement', '')).strip(),
                        'fixed': str(row.get('Fixed_Requirement', '')).strip()
                    }
                    # Only include examples where we have both original and fixed
                    if example['original'] and example['fixed'] and example['original'] != example['fixed']:
                        self.ground_truth_examples.append(example)
                
                self.log_message(f"Prepared {len(self.ground_truth_examples)} valid examples")
            else:
                self.log_message(f"Ground truth file {GROUND_TRUTH_FILE} not found, using hardcoded examples")
                self._load_hardcoded_examples()
                
        except Exception as e:
            self.log_message(f"Error loading ground truth: {e}, using hardcoded examples")
            self._load_hardcoded_examples()
    
    def _load_hardcoded_examples(self):
        """Load hardcoded examples"""
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
    
    def setup(self):
        """Initialize logging and GPU setup"""
        print("=" * 80)
        print(f"NEW COT {self.approach_type.upper()} ANAPHORIC AMBIGUITY ANALYSIS")
        print("=" * 80)
        
        self.log_message(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.log_message(f"GPU: {gpu_name}")
            self.log_message(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Load ground truth examples
        self.load_ground_truth_examples()
    
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
        """Create zero-shot prompt with Chain of Thought"""
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
- If no pronouns found, output identical text

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
    
    def create_one_shot_prompt(self, requirement_text):
        """Create one-shot prompt with Chain of Thought"""
        # Select best example
        example = self.ground_truth_examples[0] if self.ground_truth_examples else {
            'original': "The S&T component shall send all approval requests to the DBS. If the request contains storage parameters, it shall create a configuration record from the parameters.",
            'fixed': "The S&T component shall send all approval requests to the DBS. If the request contains storage parameters, the S&T component shall create a configuration record from the parameters."
        }
        
        system_prompt = f"""You are an expert requirements engineer. Your task is to identify and resolve anaphoric ambiguities in software requirements.

Here's an example of how to think through this process:

Req: {example['original']}
Fixed: {example['fixed']}

Chain of Thought for the example:
1. Pronouns found: "it" in second sentence
2. "it" refers to: could be "The S&T component", "approval requests", "the DBS", or "the request"
3. Ambiguity: Yes, "it" is unclear
4. Fix: Replace "it" with "the S&T component" (the logical subject)
5. Result: Complete requirement with "it" replaced

RULES:
- Always look for pronouns: it, its, they, them, their, this, that, these, those
- If pronouns exist, they are likely ambiguous - fix them
- Replace with specific nouns/phrases
- Output complete requirement

OUTPUT FORMAT (EXACTLY):
Ambiguity Type: [type found]
Problem Found: [specific pronouns that are ambiguous]
Fixed Requirement: [ONLY the complete requirement sentence - no extra text]
Explanation: [what was changed]

CRITICAL: The Fixed Requirement field must contain ONLY the requirement sentence."""

        user_prompt = f"""Now apply the same process:

Req: {requirement_text}
Fixed: [You complete this]

Think step by step like the example:
1. What pronouns do I see?
2. What could each refer to?
3. Are they ambiguous?
4. How should I fix them?
5. What's the complete fixed requirement?

Provide your answer in the EXACT format specified."""

        return system_prompt, user_prompt
    
    def create_few_shot_prompt(self, requirement_text):
        """Create few-shot prompt with 3 examples and Chain of Thought"""
        # Select 3 examples
        selected_examples = self.ground_truth_examples[:3] if len(self.ground_truth_examples) >= 3 else self.ground_truth_examples
        
        system_prompt = """You are an expert requirements engineer. Your task is to identify and resolve anaphoric ambiguities in software requirements.

Here are examples of how to fix ambiguous pronouns:"""

        # Add exactly 3 examples
        for i, example in enumerate(selected_examples, 1):
            system_prompt += f"""

Example {i}:
Req: {example['original']}
Fixed: {example['fixed']}"""

        system_prompt += """

Chain of Thought Process:
1. SCAN: Look for pronouns (it, its, they, them, their, this, that, these, those)
2. IDENTIFY: What does each pronoun refer to?
3. CHECK: Is the reference clear or could it refer to multiple things?
4. FIX: If ambiguous, replace pronoun with specific antecedent
5. OUTPUT: Complete requirement with all pronouns resolved

CRITICAL RULES:
- If you see pronouns, they are almost always ambiguous in requirements
- You MUST fix ambiguous pronouns by replacing them
- For possessive pronouns (its, their), use "[antecedent]'s" format
- Always provide the COMPLETE fixed requirement
- Don't leave any pronouns unfixed

OUTPUT FORMAT (MUST FOLLOW EXACTLY):
Ambiguity Type: [specific type of pronoun ambiguity]
Problem Found: [list the ambiguous pronouns]
Fixed Requirement: [ONLY the complete requirement sentence - no extra text or formatting]
Explanation: [describe what pronouns were replaced with what]

CRITICAL: The Fixed Requirement field should contain ONLY the requirement sentence, nothing else."""

        user_prompt = f"""Now analyze this requirement using the same process:

Req: {requirement_text}
Fixed: [You complete this]

Follow the Chain of Thought:
1. SCAN for pronouns
2. IDENTIFY what they refer to
3. CHECK if ambiguous
4. FIX by replacing
5. OUTPUT complete fixed requirement

Remember: If you find pronouns, you MUST fix them!
Provide your answer in the EXACT format specified above."""

        return system_prompt, user_prompt
    
    def create_llama_prompt(self, system_prompt, user_prompt):
        """Format prompt for LLaMA"""
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return formatted_prompt
    
    def parse_response(self, response_text, original_req):
        """Enhanced response parsing to extract only the fixed requirement"""
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
                    # Take the first match
                    candidate = matches[0].strip()
                    
                    # Clean the candidate
                    candidate = re.sub(r'\*\*', '', candidate)  # Remove bold markers
                    candidate = re.sub(r'\n+', ' ', candidate)  # Replace newlines with spaces
                    candidate = re.sub(r'\s+', ' ', candidate)  # Normalize whitespace
                    candidate = candidate.strip()
                    
                    # Remove quotes if present
                    if (candidate.startswith('"') and candidate.endswith('"')) or \
                       (candidate.startswith("'") and candidate.endswith("'")):
                        candidate = candidate[1:-1]
                    
                    # Check if this looks like a complete requirement (not a fragment)
                    if (len(candidate) > 30 and 
                        not candidate.lower().startswith(('error', 'unable', 'cannot', 'ambiguity type')) and
                        '**' not in candidate and
                        'Explanation:' not in candidate and
                        'Problem Found:' not in candidate):
                        
                        fixed_text = candidate
                        self.log_message(f"DEBUG: Extracted using pattern {i+1}: {fixed_text[:50]}...")
                        break
            
            # If no pattern worked, try a more aggressive approach
            if not fixed_text:
                # Look for sentences that seem like requirements
                sentences = re.split(r'(?<=[.])\s+', response_text)
                for sentence in sentences:
                    sentence = sentence.strip()
                    # Clean up
                    sentence = re.sub(r'\*\*[^*]*\*\*:?', '', sentence)  # Remove bold text
                    sentence = re.sub(r'(Ambiguity Type|Problem Found|Explanation):', '', sentence)
                    sentence = re.sub(r'\s+', ' ', sentence).strip()
                    
                    # Check if this looks like a requirement
                    if (len(sentence) > 50 and 
                        ('shall' in sentence.lower() or 'will' in sentence.lower() or 'must' in sentence.lower()) and
                        not sentence.lower().startswith(('i replaced', 'the pronoun', 'ambiguity'))):
                        fixed_text = sentence
                        self.log_message(f"DEBUG: Found requirement sentence: {fixed_text[:50]}...")
                        break
            
            # If we found a valid fixed text, use it
            if fixed_text and len(fixed_text) > 20:
                parsed['fixed_requirement'] = fixed_text
            else:
                # Log the problem for debugging
                self.log_message(f"WARNING: Could not extract fixed requirement from: {response_text[:100]}...")
            
            # Try to extract other fields (but these are less critical)
            try:
                # Extract ambiguity type
                ambiguity_match = re.search(r'(?:\*\*)?Ambiguity Type(?:\*\*)?:\s*([^\n*]+)', response_text, re.IGNORECASE)
                if ambiguity_match:
                    parsed['ambiguity_type'] = ambiguity_match.group(1).strip()
                
                # Extract problem found
                problem_match = re.search(r'(?:\*\*)?Problem Found(?:\*\*)?:\s*([^\n*]+)', response_text, re.IGNORECASE)
                if problem_match:
                    parsed['problem_found'] = problem_match.group(1).strip()
                
                # Extract explanation
                explanation_match = re.search(r'(?:\*\*)?Explanation(?:\*\*)?:\s*([^\n*]+)', response_text, re.IGNORECASE)
                if explanation_match:
                    parsed['explanation'] = explanation_match.group(1).strip()
            except:
                pass  # Don't let field extraction errors break the main parsing
                
        except Exception as e:
            self.log_message(f"Error parsing response: {e}")
            self.log_message(f"Raw response: {response_text[:200]}...")
        
        return parsed
    
    def fix_requirement(self, original_req, req_id, project_name):
        """Fix a single requirement using the chosen approach"""
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
            
            # Generate with optimized parameters for better pronoun detection
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,  # Lower temperature for more deterministic output
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
            
            return {
                'requirement_id': req_id,
                'project_name': project_name,
                'original_requirement': original_req,
                'fixed_requirement': parsed_result['fixed_requirement'],
                'explanation': parsed_result['explanation'],
                'ambiguity_type': parsed_result.get('ambiguity_type', 'Unknown'),
                'problem_found': parsed_result.get('problem_found', 'Unknown'),
                'was_changed': was_changed,
                'approach': self.approach_type,
                'timestamp': datetime.now().isoformat(),
                'original_length': len(original_req),
                'fixed_length': len(parsed_result['fixed_requirement']),
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
                'approach': self.approach_type,
                'timestamp': datetime.now().isoformat(),
                'original_length': len(original_req),
                'fixed_length': len(original_req),
                'raw_response': f'Error: {str(e)}'
            }
    
    def run_analysis(self, csv_file):
        """Run the complete analysis"""
        # Load CSV
        self.log_message("Loading CSV data...")
        try:
            df = pd.read_csv(csv_file)
            self.log_message(f"Loaded {len(df)} requirements from {csv_file}")
            
            # Find description column
            description_col = None
            project_col = None
            
            for col in df.columns:
                if 'description' in col.lower() or 'requirement' in col.lower():
                    description_col = col
                if 'project' in col.lower() or 'name' in col.lower():
                    project_col = col
            
            if not description_col:
                raise ValueError(f"Could not find description column in: {list(df.columns)}")
            if not project_col:
                self.log_message("Warning: No project name column found")
                project_col = None
                
        except Exception as e:
            self.log_message(f"ERROR loading CSV: {e}")
            return
        
        # Process requirements
        self.log_message(f"Starting NEW COT {self.approach_type} analysis...")
        results = []
        changes_made = 0
        processing_errors = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"NEW COT {self.approach_type}"):
            req_id = row.get('', idx + 1)
            project_name = str(row[project_col]) if project_col else "Unknown"
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
                if changes_made <= 5:  # Log first 5 successful changes
                    self.log_message(f"CHANGE #{changes_made}: {original_requirement[:50]}... -> {result['fixed_requirement'][:50]}...")
            
            if 'Error' in result.get('ambiguity_type', ''):
                processing_errors += 1
            
            # Save checkpoints
            if (idx + 1) % 25 == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(f"temp_new_cot_{self.approach_type}_{idx+1}.csv", index=False)
        
        # Save final results
        self.log_message("Saving final results...")
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_file, index=False)
        
        # Create summary
        summary_file = f"new_cot_summary_{self.approach_type}_results.csv"
        summary_df = results_df[['requirement_id', 'project_name', 'original_requirement', 
                                'fixed_requirement', 'was_changed', 'ambiguity_type']]
        summary_df.to_csv(summary_file, index=False)
        
        # Print statistics
        total_requirements = len(results_df)
        change_rate = (changes_made / total_requirements) * 100
        error_rate = (processing_errors / total_requirements) * 100
        
        self.log_message("=" * 80)
        self.log_message(f"NEW COT {self.approach_type.upper()} ANALYSIS COMPLETE")
        self.log_message("=" * 80)
        self.log_message(f"Total requirements: {total_requirements}")
        self.log_message(f"Changes made: {changes_made} ({change_rate:.1f}%)")
        self.log_message(f"Processing errors: {processing_errors} ({error_rate:.1f}%)")
        self.log_message(f"Output file: {self.output_file}")
        
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
            self.log_message("\n??  WARNING: No changes were made!")
            self.log_message("   This suggests the model is not detecting ambiguities.")
            self.log_message("   Check the raw_response column in the output for debugging.")
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\n?? NEW COT {self.approach_type.upper()} COMPLETE!")
        print(f"?? Changed: {changes_made}/{total_requirements} ({change_rate:.1f}%)")
        print(f"?? Results: {self.output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='New COT Anaphoric Ambiguity Resolution')
    parser.add_argument('--approach', type=str, 
                        choices=['zero-shot', 'one-shot', 'few-shot'],
                        default='few-shot', 
                        help='Prompting approach')
    parser.add_argument('--csv', type=str, default='anaphoric.csv',
                        help='Input CSV file')
    parser.add_argument('--ground-truth', type=str, default='ground_truths.csv',
                        help='Ground truth CSV file')
    
    args = parser.parse_args()
    
    # Update global variables
    global CSV_FILE, GROUND_TRUTH_FILE
    CSV_FILE = args.csv
    GROUND_TRUTH_FILE = args.ground_truth
    
    # Validate files
    if not os.path.exists(CSV_FILE):
        print(f"? Error: {CSV_FILE} not found!")
        sys.exit(1)
    
    print(f"?? Starting NEW COT {args.approach} analysis...")
    print(f"   Input: {CSV_FILE}")
    print(f"   Ground truth: {GROUND_TRUTH_FILE}")
    
    # Create and run resolver
    resolver = NewCOTAnaphoricResolver(approach_type=args.approach)
    resolver.setup()
    resolver.load_model()
    resolver.run_analysis(CSV_FILE)
    
    print(f"? Complete! Check the output files.")


if __name__ == "__main__":
    main()