#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Script for Anaphoric RAG Results
Evaluates all 6 configurations against ground truth
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse
import os
from datetime import datetime

class AnaphoricEvaluator:
    def __init__(self, ground_truth_file, output_dir="eval"):
        self.ground_truth_file = ground_truth_file
        self.output_dir = output_dir
        self.ground_truth_df = None
        self.similarity_model = None
        self.smoothing = SmoothingFunction().method1
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📊 Anaphoric RAG Evaluator initialized")
        print(f"   Output directory: {output_dir}")
    
    def load_ground_truth(self):
        """Load ground truth data"""
        print(f"\n📖 Loading ground truth from {self.ground_truth_file}...")
        self.ground_truth_df = pd.read_csv(self.ground_truth_file)
        print(f"   Loaded {len(self.ground_truth_df)} ground truth entries")
        
        # Create lookup dictionary
        self.gt_lookup = {}
        for _, row in self.ground_truth_df.iterrows():
            req_id = row['requirement_id']
            self.gt_lookup[req_id] = {
                'original': str(row['Original_Requirement']).strip(),
                'fixed': str(row['Fixed_Requirement']).strip(),
                'explanation': str(row.get('Explanation', '')).strip()
            }
        
        print(f"   Created lookup for {len(self.gt_lookup)} requirements")
    
    def load_similarity_model(self):
        """Load sentence transformer model"""
        print(f"\n🤖 Loading similarity model...")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"   ✓ Model loaded")
    
    def compute_bleu(self, predicted, reference):
        """Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores (standard cumulative)"""
        try:
            pred_tokens = predicted.lower().split()
            ref_tokens = reference.lower().split()
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                return {
                    'bleu_1': 0.0,
                    'bleu_2': 0.0,
                    'bleu_3': 0.0,
                    'bleu_4': 0.0
                }
            
            # Standard cumulative BLEU scores
            # BLEU-1: unigram only
            bleu_1 = sentence_bleu([ref_tokens], pred_tokens, 
                                   weights=(1.0, 0, 0, 0), 
                                   smoothing_function=self.smoothing)
            
            # BLEU-2: unigram + bigram (cumulative)
            bleu_2 = sentence_bleu([ref_tokens], pred_tokens, 
                                   weights=(0.5, 0.5, 0, 0), 
                                   smoothing_function=self.smoothing)
            
            # BLEU-3: unigram + bigram + trigram (cumulative)
            bleu_3 = sentence_bleu([ref_tokens], pred_tokens, 
                                   weights=(1.0/3, 1.0/3, 1.0/3, 0), 
                                   smoothing_function=self.smoothing)
            
            # BLEU-4: all 4-grams (cumulative, standard BLEU)
            bleu_4 = sentence_bleu([ref_tokens], pred_tokens, 
                                   weights=(0.25, 0.25, 0.25, 0.25), 
                                   smoothing_function=self.smoothing)
            
            return {
                'bleu_1': float(bleu_1),
                'bleu_2': float(bleu_2),
                'bleu_3': float(bleu_3),
                'bleu_4': float(bleu_4)
            }
        except:
            return {
                'bleu_1': 0.0,
                'bleu_2': 0.0,
                'bleu_3': 0.0,
                'bleu_4': 0.0
            }
    
    def compute_semantic_similarity(self, text1, text2):
        """Compute semantic similarity"""
        try:
            embeddings = self.similarity_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def evaluate_results(self, results_file, config_name):
        """Evaluate a single results file"""
        print(f"\n{'='*80}")
        print(f"📊 Evaluating: {config_name}")
        print(f"{'='*80}")
        
        if not os.path.exists(results_file):
            print(f"❌ File not found: {results_file}")
            return None
        
        # Load results
        results_df = pd.read_csv(results_file)
        print(f"Loaded {len(results_df)} results")
        
        # Prepare evaluation data
        eval_data = []
        
        for _, row in results_df.iterrows():
            req_id = row['requirement_id']
            
            # Get ground truth
            if req_id not in self.gt_lookup:
                continue
            
            gt = self.gt_lookup[req_id]
            predicted = str(row['fixed_requirement']).strip()
            
            # Compute metrics
            exact_match = predicted == gt['fixed']
            bleu_scores = self.compute_bleu(predicted, gt['fixed'])
            similarity = self.compute_semantic_similarity(predicted, gt['fixed'])
            
            eval_data.append({
                'requirement_id': req_id,
                'original': row['original_requirement'],
                'predicted': predicted,
                'ground_truth': gt['fixed'],
                'exact_match': exact_match,
                'bleu_1': bleu_scores['bleu_1'],
                'bleu_2': bleu_scores['bleu_2'],
                'bleu_3': bleu_scores['bleu_3'],
                'bleu_4': bleu_scores['bleu_4'],
                'similarity_score': similarity,
                'was_changed': row.get('was_changed', False),
                'ambiguity_type': row.get('ambiguity_type', 'Unknown')
            })
        
        eval_df = pd.DataFrame(eval_data)
        
        # Compute aggregate metrics
        metrics = {
            'config': config_name,
            'total_evaluated': len(eval_df),
            'exact_match_count': int(eval_df['exact_match'].sum()),
            'exact_match_rate': float(eval_df['exact_match'].mean()),
            'avg_bleu_1': float(eval_df['bleu_1'].mean()),
            'std_bleu_1': float(eval_df['bleu_1'].std()),
            'avg_bleu_2': float(eval_df['bleu_2'].mean()),
            'std_bleu_2': float(eval_df['bleu_2'].std()),
            'avg_bleu_3': float(eval_df['bleu_3'].mean()),
            'std_bleu_3': float(eval_df['bleu_3'].std()),
            'avg_bleu_4': float(eval_df['bleu_4'].mean()),
            'std_bleu_4': float(eval_df['bleu_4'].std()),
            'avg_similarity': float(eval_df['similarity_score'].mean()),
            'std_similarity': float(eval_df['similarity_score'].std()),
            'change_rate': float(eval_df['was_changed'].mean())
        }
        
        # Print summary
        print(f"\n📈 Results Summary:")
        print(f"   Total evaluated: {metrics['total_evaluated']}")
        print(f"   Exact matches: {metrics['exact_match_count']} ({metrics['exact_match_rate']:.1%})")
        print(f"   BLEU-1: {metrics['avg_bleu_1']:.4f} (±{metrics['std_bleu_1']:.4f})")
        print(f"   BLEU-2: {metrics['avg_bleu_2']:.4f} (±{metrics['std_bleu_2']:.4f})")
        print(f"   BLEU-3: {metrics['avg_bleu_3']:.4f} (±{metrics['std_bleu_3']:.4f})")
        print(f"   BLEU-4: {metrics['avg_bleu_4']:.4f} (±{metrics['std_bleu_4']:.4f})")
        print(f"   Avg Similarity: {metrics['avg_similarity']:.4f} (±{metrics['std_similarity']:.4f})")
        print(f"   Change rate: {metrics['change_rate']:.1%}")
        
        # Save detailed results
        output_file = os.path.join(self.output_dir, f"detailed_eval_{config_name.replace(' ', '_').replace('+', 'plus')}.csv")
        eval_df.to_csv(output_file, index=False)
        print(f"\n💾 Saved detailed results to: {output_file}")
        
        return metrics, eval_df
    
    def compare_all_configurations(self, metrics_list):
        """Compare all configurations"""
        print(f"\n{'='*80}")
        print(f"📊 COMPREHENSIVE COMPARISON")
        print(f"{'='*80}")
        
        comparison_df = pd.DataFrame(metrics_list)
        
        # Sort by similarity score
        comparison_df = comparison_df.sort_values('avg_similarity', ascending=False)
        
        print("\nRankings by Average Semantic Similarity:")
        print("-" * 100)
        print(f"{'Rank':<5} {'Configuration':<25} {'Similarity':<12} {'BLEU-1':<10} {'BLEU-2':<10} {'BLEU-3':<10} {'BLEU-4':<10} {'Exact Match':<12}")
        print("-" * 100)
        for i, row in enumerate(comparison_df.itertuples(), 1):
            print(f"{i:<5} {row.config:<25} {row.avg_similarity:.4f}      {row.avg_bleu_1:.4f}    {row.avg_bleu_2:.4f}    {row.avg_bleu_3:.4f}    {row.avg_bleu_4:.4f}    {row.exact_match_rate:.1%}")
        
        # Compare RAG vs non-RAG
        print("\n" + "="*80)
        print("RAG IMPACT ANALYSIS")
        print("="*80)
        
        for approach in ['zero-shot', 'one-shot', 'few-shot']:
            base_config = f"{approach}"
            rag_config = f"{approach} + RAG"
            
            base_metrics = comparison_df[comparison_df['config'] == base_config]
            rag_metrics = comparison_df[comparison_df['config'] == rag_config]
            
            if not base_metrics.empty and not rag_metrics.empty:
                base = base_metrics.iloc[0]
                rag = rag_metrics.iloc[0]
                
                sim_diff = rag['avg_similarity'] - base['avg_similarity']
                bleu1_diff = rag['avg_bleu_1'] - base['avg_bleu_1']
                bleu2_diff = rag['avg_bleu_2'] - base['avg_bleu_2']
                bleu3_diff = rag['avg_bleu_3'] - base['avg_bleu_3']
                bleu4_diff = rag['avg_bleu_4'] - base['avg_bleu_4']
                em_diff = rag['exact_match_rate'] - base['exact_match_rate']
                
                print(f"\n{approach.upper()}:")
                print(f"  Similarity: {base['avg_similarity']:.4f} → {rag['avg_similarity']:.4f} (Δ{sim_diff:+.4f})")
                print(f"  BLEU-1:     {base['avg_bleu_1']:.4f} → {rag['avg_bleu_1']:.4f} (Δ{bleu1_diff:+.4f})")
                print(f"  BLEU-2:     {base['avg_bleu_2']:.4f} → {rag['avg_bleu_2']:.4f} (Δ{bleu2_diff:+.4f})")
                print(f"  BLEU-3:     {base['avg_bleu_3']:.4f} → {rag['avg_bleu_3']:.4f} (Δ{bleu3_diff:+.4f})")
                print(f"  BLEU-4:     {base['avg_bleu_4']:.4f} → {rag['avg_bleu_4']:.4f} (Δ{bleu4_diff:+.4f})")
                print(f"  Exact Match: {base['exact_match_rate']:.1%} → {rag['exact_match_rate']:.1%} (Δ{em_diff:+.1%})")
                
                if sim_diff > 0.01:
                    print(f"  ✅ RAG improves performance")
                elif sim_diff < -0.01:
                    print(f"  ⚠️  RAG decreases performance")
                else:
                    print(f"  ➡️  RAG has minimal impact")
        
        # Save comparison
        comparison_file = os.path.join(self.output_dir, "comparison_all_configs.csv")
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\n💾 Saved comparison to: {comparison_file}")
        
        # Generate report
        report_file = os.path.join(self.output_dir, "evaluation_report.txt")
        with open(report_file, 'w') as f:
            f.write("ANAPHORIC AMBIGUITY - RAG EVALUATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OVERALL RANKINGS (by Semantic Similarity):\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Rank':<5} {'Configuration':<25} {'Similarity':<12} {'BLEU-1':<10} {'BLEU-2':<10} {'BLEU-3':<10} {'BLEU-4':<10} {'EM':<10}\n")
            f.write("-" * 80 + "\n")
            for i, row in enumerate(comparison_df.itertuples(), 1):
                f.write(f"{i:<5} {row.config:<25} {row.avg_similarity:.4f}      {row.avg_bleu_1:.4f}    {row.avg_bleu_2:.4f}    {row.avg_bleu_3:.4f}    {row.avg_bleu_4:.4f}    {row.exact_match_rate:.1%}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILED METRICS:\n")
            f.write("="*80 + "\n\n")
            
            for _, row in comparison_df.iterrows():
                f.write(f"{row['config']}\n")
                f.write(f"  Evaluated: {row['total_evaluated']}\n")
                f.write(f"  Exact Match: {row['exact_match_count']} ({row['exact_match_rate']:.1%})\n")
                f.write(f"  BLEU-1: {row['avg_bleu_1']:.4f} ± {row['std_bleu_1']:.4f}\n")
                f.write(f"  BLEU-2: {row['avg_bleu_2']:.4f} ± {row['std_bleu_2']:.4f}\n")
                f.write(f"  BLEU-3: {row['avg_bleu_3']:.4f} ± {row['std_bleu_3']:.4f}\n")
                f.write(f"  BLEU-4: {row['avg_bleu_4']:.4f} ± {row['std_bleu_4']:.4f}\n")
                f.write(f"  Similarity: {row['avg_similarity']:.4f} ± {row['std_similarity']:.4f}\n")
                f.write(f"  Change Rate: {row['change_rate']:.1%}\n\n")
            
            # Add RAG impact section
            f.write("\n" + "="*80 + "\n")
            f.write("RAG IMPACT ANALYSIS:\n")
            f.write("="*80 + "\n\n")
            
            for approach in ['zero-shot', 'one-shot', 'few-shot']:
                base_config = f"{approach}"
                rag_config = f"{approach} + RAG"
                
                base_metrics = comparison_df[comparison_df['config'] == base_config]
                rag_metrics = comparison_df[comparison_df['config'] == rag_config]
                
                if not base_metrics.empty and not rag_metrics.empty:
                    base = base_metrics.iloc[0]
                    rag = rag_metrics.iloc[0]
                    
                    f.write(f"{approach.upper()}:\n")
                    f.write(f"  Similarity: {base['avg_similarity']:.4f} → {rag['avg_similarity']:.4f} (Δ{rag['avg_similarity'] - base['avg_similarity']:+.4f})\n")
                    f.write(f"  BLEU-1: {base['avg_bleu_1']:.4f} → {rag['avg_bleu_1']:.4f} (Δ{rag['avg_bleu_1'] - base['avg_bleu_1']:+.4f})\n")
                    f.write(f"  BLEU-2: {base['avg_bleu_2']:.4f} → {rag['avg_bleu_2']:.4f} (Δ{rag['avg_bleu_2'] - base['avg_bleu_2']:+.4f})\n")
                    f.write(f"  BLEU-3: {base['avg_bleu_3']:.4f} → {rag['avg_bleu_3']:.4f} (Δ{rag['avg_bleu_3'] - base['avg_bleu_3']:+.4f})\n")
                    f.write(f"  BLEU-4: {base['avg_bleu_4']:.4f} → {rag['avg_bleu_4']:.4f} (Δ{rag['avg_bleu_4'] - base['avg_bleu_4']:+.4f})\n")
                    f.write(f"  Exact Match: {base['exact_match_rate']:.1%} → {rag['exact_match_rate']:.1%} (Δ{rag['exact_match_rate'] - base['exact_match_rate']:+.1%})\n\n")
        
        print(f"📄 Saved report to: {report_file}")
        
        return comparison_df


def main():
    parser = argparse.ArgumentParser(description='Evaluate Anaphoric RAG Results')
    parser.add_argument('--ground-truth', type=str, default='ground_truths.csv',
                        help='Ground truth CSV file')
    parser.add_argument('--results-dir', type=str, default='.',
                        help='Directory containing result files')
    parser.add_argument('--output-dir', type=str, default='eval',
                        help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    print("🚀 Starting Anaphoric RAG Evaluation")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = AnaphoricEvaluator(args.ground_truth, args.output_dir)
    evaluator.load_ground_truth()
    evaluator.load_similarity_model()
    
    # Define all configurations to evaluate
    configurations = [
        ('anaphoric_results_zeroshot.csv', 'zero-shot'),
        ('anaphoric_results_zeroshot_rag.csv', 'zero-shot + RAG'),
        ('anaphoric_results_oneshot.csv', 'one-shot'),
        ('anaphoric_results_oneshot_rag.csv', 'one-shot + RAG'),
        ('anaphoric_results_fewshot.csv', 'few-shot'),
        ('anaphoric_results_fewshot_rag.csv', 'few-shot + RAG'),
    ]
    
    # Evaluate each configuration
    all_metrics = []
    
    for results_file, config_name in configurations:
        full_path = os.path.join(args.results_dir, results_file)
        
        if os.path.exists(full_path):
            metrics, eval_df = evaluator.evaluate_results(full_path, config_name)
            if metrics:
                all_metrics.append(metrics)
        else:
            print(f"\n⚠️  Skipping {config_name} - file not found: {full_path}")
    
    # Compare all configurations
    if all_metrics:
        print(f"\n{'='*80}")
        print(f"Evaluated {len(all_metrics)} configurations")
        evaluator.compare_all_configurations(all_metrics)
        print(f"\n✅ Evaluation complete!")
    else:
        print(f"\n❌ No results to evaluate!")


if __name__ == "__main__":
    main()
