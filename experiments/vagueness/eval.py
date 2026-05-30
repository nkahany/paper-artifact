#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vagueness Resolution Evaluation Script
EXACT COPY of structural evaluation but adapted for vagueness
Uses the proven evaluation structure that works
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse
import os
import sys
from datetime import datetime
import warnings
import re
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class UnifiedVaguenessEvaluator:
    def __init__(self, similarity_model='all-MiniLM-L6-v2'):
        """Initialize evaluator with sentence transformer model"""
        print("Initializing Unified Vagueness Resolution Evaluator...")
        self.similarity_model = SentenceTransformer(similarity_model)
        self.smoothing = SmoothingFunction().method1
        print(f"Loaded similarity model: {similarity_model}")
    
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
    
    def load_data(self, results_file, ground_truth_file):
        """Load results and ground truth data"""
        try:
            # Load results
            print(f"Loading results from: {results_file}")
            results_df = pd.read_csv(results_file)
            print(f"   Found {len(results_df)} result entries")
            
            # Clean the predicted fixed requirements
            if 'predicted_fixed' in results_df.columns:
                results_df['predicted_fixed'] = results_df['predicted_fixed'].apply(self.clean_text)
                print("   Cleaned predicted_fixed column (removed ** prefixes)")
            elif 'fixed_requirement' in results_df.columns:
                results_df['predicted_fixed'] = results_df['fixed_requirement'].apply(self.clean_text)
                print("   Cleaned fixed_requirement column (removed ** prefixes)")
            
            # Load ground truth (vagueness_test.csv)
            print(f"Loading ground truth from: {ground_truth_file}")
            gt_df = pd.read_csv(ground_truth_file)
            print(f"   Found {len(gt_df)} ground truth entries")
            
            # Clean ground truth - try different column name variations
            fixed_col = None
            original_col = None
            
            for col in gt_df.columns:
                if 'fixed' in col.lower() or 'vaguenes_fixed' in col.lower():
                    fixed_col = col
                elif 'requirement' in col.lower() and 'fixed' not in col.lower():
                    original_col = col
            
            if fixed_col and fixed_col in gt_df.columns:
                gt_df['Fixed_Requirement'] = gt_df[fixed_col].apply(self.clean_text)
            if original_col and original_col in gt_df.columns:
                gt_df['Original_Requirement'] = gt_df[original_col].apply(self.clean_text)
            
            return results_df, gt_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def calculate_text_similarity(self, text1, text2):
        """Calculate cosine similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        embeddings = self.similarity_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def align_data(self, results_df, gt_df):
        """Align results with ground truth based on original requirements"""
        print("Aligning results with ground truth...")
        
        aligned_data = []
        
        for _, gt_row in gt_df.iterrows():
            gt_original = self.clean_text(gt_row.get('Original_Requirement', ''))
            gt_fixed = self.clean_text(gt_row.get('Fixed_Requirement', ''))
            gt_req_id = gt_row.get('requirement_id', gt_row.get('ID', 'unknown'))
            
            # Find matching result by requirement_id first, then by similarity
            best_match = None
            best_similarity = 0
            
            # Try exact ID match first
            id_col = 'requirement_id' if 'requirement_id' in results_df.columns else 'ID'
            if id_col in results_df.columns:
                id_matches = results_df[results_df[id_col] == gt_req_id]
                if len(id_matches) > 0:
                    best_match = id_matches.iloc[0]
                    best_similarity = 1.0
            
            if best_match is None:
                # Fall back to similarity matching
                for _, result_row in results_df.iterrows():
                    result_original = self.clean_text(result_row.get('original_requirement', ''))
                    
                    if gt_original and result_original:
                        if gt_original.lower() == result_original.lower():
                            similarity = 1.0
                        else:
                            similarity = self.calculate_text_similarity(gt_original, result_original)
                        
                        if similarity > best_similarity and similarity > 0.7:
                            best_similarity = similarity
                            best_match = result_row
            
            if best_match is not None:
                aligned_data.append({
                    'requirement_id': gt_req_id,
                    'gt_original': gt_original,
                    'gt_fixed': gt_fixed,
                    'pred_original': self.clean_text(best_match.get('original_requirement', '')),
                    'pred_fixed': self.clean_text(best_match.get('predicted_fixed', 
                                                  best_match.get('fixed_requirement', ''))),
                    'pred_explanation': str(best_match.get('explanation', '')).strip(),
                    'was_changed': best_match.get('was_changed', False),
                    'approach': best_match.get('approach', 'unknown'),
                    'vagueness_type': best_match.get('vagueness_type', 'unknown'),
                    'alignment_similarity': best_similarity
                })
        
        aligned_df = pd.DataFrame(aligned_data)
        print(f"Successfully aligned {len(aligned_df)} entries")
        
        if len(aligned_df) > 0:
            print(f"   Average alignment similarity: {aligned_df['alignment_similarity'].mean():.3f}")
        
        return aligned_df
    
    def calculate_exact_match(self, aligned_df):
        """Calculate exact match accuracy with cleaned text"""
        print("Calculating Exact Match scores...")
        
        exact_matches = 0
        total = len(aligned_df)
        
        exact_match_details = []
        
        for _, row in aligned_df.iterrows():
            gt_fixed = row['gt_fixed'].strip().lower()
            pred_fixed = row['pred_fixed'].strip().lower()
            
            is_exact_match = gt_fixed == pred_fixed
            if is_exact_match:
                exact_matches += 1
            
            exact_match_details.append({
                'requirement_id': row['requirement_id'],
                'exact_match': is_exact_match,
                'gt_fixed': row['gt_fixed'],
                'pred_fixed': row['pred_fixed'],
                'gt_length': len(gt_fixed),
                'pred_length': len(pred_fixed)
            })
        
        exact_match_score = exact_matches / total if total > 0 else 0
        
        results = {
            'exact_match_score': exact_match_score,
            'exact_matches': exact_matches,
            'total': total,
            'details': exact_match_details
        }
        
        print(f"   Exact Match: {exact_match_score:.4f} ({exact_matches}/{total})")
        return results
    
    def calculate_bleu_scores(self, aligned_df):
        """Calculate BLEU scores with proper tokenization"""
        print("Calculating BLEU scores...")
        
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_3_scores = []
        bleu_4_scores = []
        
        bleu_details = []
        
        for _, row in aligned_df.iterrows():
            gt_fixed = row['gt_fixed'].strip()
            pred_fixed = row['pred_fixed'].strip()
            
            if not gt_fixed or not pred_fixed:
                bleu_1 = bleu_2 = bleu_3 = bleu_4 = 0.0
            else:
                # Tokenize
                gt_tokens = nltk.word_tokenize(gt_fixed.lower())
                pred_tokens = nltk.word_tokenize(pred_fixed.lower())
                
                # Calculate BLEU scores with smoothing
                bleu_1 = sentence_bleu([gt_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smoothing)
                bleu_2 = sentence_bleu([gt_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing)
                bleu_3 = sentence_bleu([gt_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothing)
                bleu_4 = sentence_bleu([gt_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing)
            
            bleu_1_scores.append(bleu_1)
            bleu_2_scores.append(bleu_2)
            bleu_3_scores.append(bleu_3)
            bleu_4_scores.append(bleu_4)
            
            bleu_details.append({
                'requirement_id': row['requirement_id'],
                'bleu_1': bleu_1,
                'bleu_2': bleu_2,
                'bleu_3': bleu_3,
                'bleu_4': bleu_4
            })
        
        results = {
            'bleu_1': np.mean(bleu_1_scores),
            'bleu_2': np.mean(bleu_2_scores),
            'bleu_3': np.mean(bleu_3_scores),
            'bleu_4': np.mean(bleu_4_scores),
            'details': bleu_details
        }
        
        print(f"   BLEU-1: {results['bleu_1']:.4f}")
        print(f"   BLEU-2: {results['bleu_2']:.4f}")
        print(f"   BLEU-3: {results['bleu_3']:.4f}")
        print(f"   BLEU-4: {results['bleu_4']:.4f}")
        
        return results
    
    def calculate_semantic_similarity(self, aligned_df):
        """Calculate semantic similarity using sentence transformers"""
        print("Calculating Semantic Similarity scores...")
        
        similarities = []
        similarity_details = []
        
        for _, row in aligned_df.iterrows():
            gt_fixed = row['gt_fixed'].strip()
            pred_fixed = row['pred_fixed'].strip()
            
            if gt_fixed and pred_fixed:
                similarity = self.calculate_text_similarity(gt_fixed, pred_fixed)
                similarities.append(similarity)
                
                similarity_details.append({
                    'requirement_id': row['requirement_id'],
                    'semantic_similarity': similarity,
                    'gt_fixed': gt_fixed,
                    'pred_fixed': pred_fixed
                })
            else:
                similarities.append(0.0)
                similarity_details.append({
                    'requirement_id': row['requirement_id'],
                    'semantic_similarity': 0.0,
                    'gt_fixed': gt_fixed,
                    'pred_fixed': pred_fixed
                })
        
        avg_similarity = np.mean(similarities)
        
        results = {
            'avg_semantic_similarity': avg_similarity,
            'similarities': similarities,
            'details': similarity_details
        }
        
        print(f"   Semantic Similarity: {avg_similarity:.4f}")
        return results
    
    def calculate_classification_metrics(self, aligned_df):
        """Calculate precision, recall, F1 for vagueness detection"""
        print("Calculating Classification Metrics...")
        
        gt_labels = []
        pred_labels = []
        
        classification_details = []
        
        for _, row in aligned_df.iterrows():
            # Ground truth label: if original != fixed, then it's vague
            gt_is_vague = row['gt_original'].strip() != row['gt_fixed'].strip()
            
            # Prediction label: from was_changed field
            pred_is_vague = bool(row['was_changed'])
            
            gt_labels.append(int(gt_is_vague))
            pred_labels.append(int(pred_is_vague))
            
            classification_details.append({
                'requirement_id': row['requirement_id'],
                'gt_is_vague': gt_is_vague,
                'pred_is_vague': pred_is_vague,
                'correct_prediction': gt_is_vague == pred_is_vague
            })
        
        # Calculate metrics
        if len(set(gt_labels)) > 1 and len(set(pred_labels)) > 1:
            precision, recall, f1, support = precision_recall_fscore_support(
                gt_labels, pred_labels, average='binary', zero_division=0
            )
        else:
            # Handle edge case where all predictions are the same
            precision = recall = f1 = 0.0
        
        accuracy = accuracy_score(gt_labels, pred_labels)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'details': classification_details,
            'gt_vague_count': sum(gt_labels),
            'pred_vague_count': sum(pred_labels)
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   GT Vague: {sum(gt_labels)}/{len(gt_labels)}")
        print(f"   Pred Vague: {sum(pred_labels)}/{len(pred_labels)}")
        
        return results
    
    def create_detailed_report(self, all_results, approach, output_dir):
        """Create detailed evaluation report"""
        report_file = os.path.join(output_dir, f"vagueness_evaluation_report_{approach}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"UNIFIED VAGUENESS RESOLUTION EVALUATION REPORT\n")
            f.write(f"Approach: {approach.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS:\n")
            f.write(f"Exact Match Score: {all_results['exact_match']['exact_match_score']:.4f}\n")
            f.write(f"BLEU-1 Score: {all_results['bleu']['bleu_1']:.4f}\n")
            f.write(f"BLEU-2 Score: {all_results['bleu']['bleu_2']:.4f}\n")
            f.write(f"BLEU-3 Score: {all_results['bleu']['bleu_3']:.4f}\n")
            f.write(f"BLEU-4 Score: {all_results['bleu']['bleu_4']:.4f}\n")
            f.write(f"Semantic Similarity: {all_results['similarity']['avg_semantic_similarity']:.4f}\n")
            f.write(f"Classification Accuracy: {all_results['classification']['accuracy']:.4f}\n")
            f.write(f"Classification Precision: {all_results['classification']['precision']:.4f}\n")
            f.write(f"Classification Recall: {all_results['classification']['recall']:.4f}\n")
            f.write(f"Classification F1-Score: {all_results['classification']['f1']:.4f}\n\n")
            
            # Detection stats
            f.write("DETECTION STATISTICS:\n")
            f.write(f"Ground Truth Vague: {all_results['classification']['gt_vague_count']}\n")
            f.write(f"Predicted Vague: {all_results['classification']['pred_vague_count']}\n")
            f.write(f"Total Requirements: {all_results['exact_match']['total']}\n\n")
            
            # Sample comparisons
            f.write("SAMPLE EXACT MATCHES:\n")
            exact_matches = [d for d in all_results['exact_match']['details'] if d['exact_match']][:5]
            for i, detail in enumerate(exact_matches, 1):
                f.write(f"\nMatch {i} (ID: {detail['requirement_id']}):\n")
                f.write(f"  Ground Truth: {detail['gt_fixed'][:100]}...\n")
                f.write(f"  Prediction:   {detail['pred_fixed'][:100]}...\n")
            
            f.write("\nSAMPLE EXACT MISMATCHES:\n")
            exact_mismatches = [d for d in all_results['exact_match']['details'] if not d['exact_match']][:5]
            for i, detail in enumerate(exact_mismatches, 1):
                f.write(f"\nMismatch {i} (ID: {detail['requirement_id']}):\n")
                f.write(f"  Ground Truth: {detail['gt_fixed'][:100]}...\n")
                f.write(f"  Prediction:   {detail['pred_fixed'][:100]}...\n")
            
            f.write(f"\nReport generated: {datetime.now().isoformat()}\n")
        
        print(f"Detailed report saved: {report_file}")
        return report_file
    
    def save_detailed_results(self, aligned_df, all_results, approach, output_dir):
        """Save detailed results to CSV"""
        detailed_df = aligned_df.copy()
        
        # Add all metric details
        for metric_name, metric_data in all_results.items():
            if metric_name == 'exact_match':
                exact_match_dict = {d['requirement_id']: d['exact_match'] for d in metric_data['details']}
                detailed_df['exact_match'] = detailed_df['requirement_id'].map(exact_match_dict)
            elif metric_name == 'bleu':
                bleu_dict = {d['requirement_id']: d for d in metric_data['details']}
                for bleu_type in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']:
                    detailed_df[bleu_type] = detailed_df['requirement_id'].map(
                        lambda x: bleu_dict.get(x, {}).get(bleu_type, 0)
                    )
            elif metric_name == 'similarity':
                sim_dict = {d['requirement_id']: d['semantic_similarity'] for d in metric_data['details']}
                detailed_df['semantic_similarity'] = detailed_df['requirement_id'].map(sim_dict)
            elif metric_name == 'classification':
                class_dict = {d['requirement_id']: d for d in metric_data['details']}
                for field in ['gt_is_vague', 'pred_is_vague', 'correct_prediction']:
                    detailed_df[field] = detailed_df['requirement_id'].map(
                        lambda x: class_dict.get(x, {}).get(field, False)
                    )
        
        # Save to CSV
        detailed_file = os.path.join(output_dir, f"detailed_vagueness_evaluation_{approach}.csv")
        detailed_df.to_csv(detailed_file, index=False)
        print(f"Detailed results saved: {detailed_file}")
        
        return detailed_file
    
    def evaluate_approach(self, results_file, ground_truth_file, approach, output_dir):
        """Evaluate a single approach"""
        print(f"\nEvaluating VAGUENESS {approach.upper()} approach")
        print("=" * 60)
        
        # Load data
        results_df, gt_df = self.load_data(results_file, ground_truth_file)
        if results_df is None or gt_df is None:
            print(f"Failed to load data for {approach}")
            return None
        
        # Align data
        aligned_df = self.align_data(results_df, gt_df)
        if len(aligned_df) == 0:
            print(f"No alignable data found for {approach}")
            return None
        
        # Calculate all metrics
        exact_match_results = self.calculate_exact_match(aligned_df)
        bleu_results = self.calculate_bleu_scores(aligned_df)
        similarity_results = self.calculate_semantic_similarity(aligned_df)
        classification_results = self.calculate_classification_metrics(aligned_df)
        
        # Combine results
        all_results = {
            'approach': approach,
            'exact_match': exact_match_results,
            'bleu': bleu_results,
            'similarity': similarity_results,
            'classification': classification_results,
            'aligned_count': len(aligned_df)
        }
        
        # Create detailed reports
        self.create_detailed_report(all_results, approach, output_dir)
        self.save_detailed_results(aligned_df, all_results, approach, output_dir)
        
        return all_results
    
    def create_comparison_report(self, all_approach_results, output_dir):
        """Create comparison report across all approaches"""
        report_file = os.path.join(output_dir, "vagueness_comparison_report.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("UNIFIED VAGUENESS RESOLUTION - APPROACH COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            
            # Create comparison table
            approaches = list(all_approach_results.keys())
            metrics = [
                ('Exact Match', 'exact_match', 'exact_match_score'),
                ('BLEU-1', 'bleu', 'bleu_1'),
                ('BLEU-2', 'bleu', 'bleu_2'),
                ('BLEU-4', 'bleu', 'bleu_4'),
                ('Semantic Sim', 'similarity', 'avg_semantic_similarity'),
                ('Accuracy', 'classification', 'accuracy'),
                ('Precision', 'classification', 'precision'),
                ('Recall', 'classification', 'recall'),
                ('F1-Score', 'classification', 'f1')
            ]
            
            f.write("METRIC COMPARISON TABLE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Metric':<15}")
            for approach in approaches:
                f.write(f"{approach.capitalize():<12}")
            f.write("Best\n")
            f.write("-" * 80 + "\n")
            
            for metric_name, category, key in metrics:
                f.write(f"{metric_name:<15}")
                scores = {}
                for approach in approaches:
                    if approach in all_approach_results:
                        score = all_approach_results[approach][category][key]
                        scores[approach] = score
                        f.write(f"{score:<12.4f}")
                    else:
                        f.write(f"{'N/A':<12}")
                
                # Find best score
                if scores:
                    best_approach = max(scores, key=scores.get)
                    f.write(f"{best_approach.capitalize()}\n")
                else:
                    f.write("N/A\n")
            
            f.write("-" * 80 + "\n\n")
            
            # Summary
            f.write("SUMMARY:\n")
            if scores:
                f1_scores = {app: all_approach_results[app]['classification']['f1'] for app in approaches if app in all_approach_results}
                if f1_scores:
                    best_overall = max(f1_scores, key=f1_scores.get)
                    f.write(f"Best Overall (F1): {best_overall.upper()} ({f1_scores[best_overall]:.4f})\n")
                
                exact_match_scores = {app: all_approach_results[app]['exact_match']['exact_match_score'] for app in approaches if app in all_approach_results}
                if exact_match_scores:
                    best_exact = max(exact_match_scores, key=exact_match_scores.get)
                    f.write(f"Best Exact Match: {best_exact.upper()} ({exact_match_scores[best_exact]:.4f})\n")
                
                bleu_scores = {app: all_approach_results[app]['bleu']['bleu_4'] for app in approaches if app in all_approach_results}
                if bleu_scores:
                    best_bleu = max(bleu_scores, key=bleu_scores.get)
                    f.write(f"Best BLEU-4: {best_bleu.upper()} ({bleu_scores[best_bleu]:.4f})\n")
            
            f.write(f"\nGenerated: {datetime.now().isoformat()}\n")
        
        print(f"Comparison report saved: {report_file}")
        return report_file


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Unified Vagueness Resolution Evaluation')
    parser.add_argument('--ground-truth', type=str, default='vagueness_test.csv',
                        help='Path to ground truth CSV file (vagueness_test.csv)')
    parser.add_argument('--results-dir', type=str, default='.',
                        help='Directory containing result files')
    parser.add_argument('--output-dir', type=str, default='vagueness_evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--approaches', nargs='+', 
                        default=['zeroshot', 'oneshot', 'fewshot'],
                        help='Approaches to evaluate')
    parser.add_argument('--prefix', type=str, default='unified_vagueness_results_',
                        help='Prefix for result filenames')
    parser.add_argument('--rag-variants', action='store_true',
                        help='Also evaluate RAG variants')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = UnifiedVaguenessEvaluator()
    
    # Build list of approaches to evaluate
    approaches_to_evaluate = []
    
    # Add base approaches
    for approach in args.approaches:
        approaches_to_evaluate.append((approach, f"{args.prefix}{approach}.csv"))
    
    # Add RAG variants if requested
    if args.rag_variants:
        for approach in args.approaches:
            rag_file = f"{args.prefix}{approach}_rag.csv"
            approaches_to_evaluate.append((f"{approach}_rag", rag_file))
    
    # Evaluate each approach
    all_results = {}
    
    for approach_name, filename in approaches_to_evaluate:
        results_file = os.path.join(args.results_dir, filename)
        
        if os.path.exists(results_file):
            results = evaluator.evaluate_approach(
                results_file, args.ground_truth, approach_name, args.output_dir
            )
            if results:
                all_results[approach_name] = results
        else:
            print(f"Results file not found: {results_file}")
    
    # Create comparison report
    if len(all_results) > 1:
        evaluator.create_comparison_report(all_results, args.output_dir)
    
    # Summary
    print(f"\nVagueness evaluation complete!")
    print(f"Results saved in: {args.output_dir}")
    print(f"Evaluated {len(all_results)} approaches")
    
    # Quick summary
    if all_results:
        print(f"\nQuick Summary:")
        for approach, results in all_results.items():
            exact_match = results['exact_match']['exact_match_score']
            f1 = results['classification']['f1']
            bleu4 = results['bleu']['bleu_4']
            semantic_sim = results['similarity']['avg_semantic_similarity']
            print(f"   {approach.upper()}: Exact={exact_match:.3f}, F1={f1:.3f}, BLEU-4={bleu4:.3f}, Sim={semantic_sim:.3f}")


if __name__ == "__main__":
    main()