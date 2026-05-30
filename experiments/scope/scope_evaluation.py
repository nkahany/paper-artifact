#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scope Ambiguity Evaluation Script
Comprehensive evaluation of different approaches and RAG enhancement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import re
import json
from datetime import datetime
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class ScopeEvaluator:
    def __init__(self):
        self.results_data = {}
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.evaluation_metrics = {}
    
    def load_results(self, results_files):
        """Load results from different approaches"""
        print("📊 Loading evaluation results...")
        
        for approach, filepath in results_files.items():
            try:
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    self.results_data[approach] = df
                    print(f"✅ {approach}: {len(df)} results")
                else:
                    print(f"⚠️ File not found: {filepath}")
            except Exception as e:
                print(f"❌ Failed to load {approach}: {e}")
        
        return len(self.results_data)
    
    def evaluate_scope_detection(self, approach_name, results_df):
        """Evaluate scope ambiguity detection"""
        print(f"🔍 Evaluating scope detection for {approach_name}...")
        
        # Ground truth: sentence actually changed
        gt_has_ambiguity = (results_df['original_requirement'] != results_df['ground_truth_fixed']).astype(int)
        
        # Prediction: model detected and changed
        pred_detected_ambiguity = results_df['was_changed'].astype(int)
        
        # Metrics
        accuracy = accuracy_score(gt_has_ambiguity, pred_detected_ambiguity)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt_has_ambiguity, pred_detected_ambiguity, average='binary', zero_division=0
        )
        
        # Detailed classification report
        class_report = classification_report(
            gt_has_ambiguity, pred_detected_ambiguity,
            target_names=['No Ambiguity', 'Has Ambiguity'],
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'total_cases': len(results_df),
            'ambiguous_cases': gt_has_ambiguity.sum(),
            'detected_cases': pred_detected_ambiguity.sum()
        }
    
    def evaluate_resolution_quality(self, approach_name, results_df):
        """Evaluate resolution quality"""
        print(f"🎯 Evaluating resolution quality for {approach_name}...")
        
        # Filter ambiguous cases
        ambiguous_cases = results_df[results_df['original_requirement'] != results_df['ground_truth_fixed']]
        
        if len(ambiguous_cases) == 0:
            return {'total_ambiguous_cases': 0}
        
        # Exact matches
        exact_matches = (ambiguous_cases['predicted_fixed'] == ambiguous_cases['ground_truth_fixed']).sum()
        exact_match_rate = exact_matches / len(ambiguous_cases)
        
        # Semantic similarity
        if 'similarity_score' in ambiguous_cases.columns:
            avg_similarity = ambiguous_cases['similarity_score'].mean()
            similarity_std = ambiguous_cases['similarity_score'].std()
        else:
            similarities = []
            for _, row in ambiguous_cases.iterrows():
                sim = self.calculate_semantic_similarity(row['predicted_fixed'], row['ground_truth_fixed'])
                similarities.append(sim)
            avg_similarity = np.mean(similarities)
            similarity_std = np.std(similarities)
        
        # BLEU scores
        avg_bleu = ambiguous_cases.get('bleu_score', pd.Series([0])).mean()
        bleu_std = ambiguous_cases.get('bleu_score', pd.Series([0])).std()
        
        return {
            'total_ambiguous_cases': len(ambiguous_cases),
            'exact_matches': exact_matches,
            'exact_match_rate': exact_match_rate,
            'avg_semantic_similarity': avg_similarity,
            'similarity_std': similarity_std,
            'avg_bleu_score': avg_bleu,
            'bleu_std': bleu_std
        }
    
    def analyze_scope_patterns(self, approach_name, results_df):
        """Analyze performance by scope pattern"""
        print(f"🔬 Analyzing scope patterns for {approach_name}...")
        
        pattern_performance = defaultdict(list)
        
        for _, row in results_df.iterrows():
            sentence = row['original_requirement']
            pattern_type = self.classify_scope_pattern(sentence)
            
            was_correct = row.get('exact_match', False)
            was_changed = row.get('was_changed', False)
            similarity = row.get('similarity_score', 0.0)
            
            pattern_performance[pattern_type].append({
                'correct': was_correct,
                'changed': was_changed,
                'similarity': similarity,
                'sentence': sentence
            })
        
        # Aggregate statistics
        pattern_stats = {}
        for pattern, cases in pattern_performance.items():
            if len(cases) > 0:
                pattern_stats[pattern] = {
                    'total_cases': len(cases),
                    'accuracy': sum([c['correct'] for c in cases]) / len(cases),
                    'change_rate': sum([c['changed'] for c in cases]) / len(cases),
                    'avg_similarity': np.mean([c['similarity'] for c in cases]),
                    'example_sentences': [c['sentence'] for c in cases[:3]]
                }
        
        return pattern_stats
    
    def classify_scope_pattern(self, sentence):
        """Classify scope pattern type"""
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
            return 'other_scope'
    
    def calculate_semantic_similarity(self, pred, truth):
        """Calculate semantic similarity"""
        try:
            embeddings = self.embedding_model.encode([pred, truth])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def compare_approaches(self):
        """Compare all approaches"""
        print("📈 Comparing all approaches...")
        
        comparison_results = {}
        
        for approach_name, results_df in self.results_data.items():
            if len(results_df) == 0:
                continue
            
            print(f"\n🔄 Evaluating {approach_name}...")
            
            # Detection evaluation
            detection_metrics = self.evaluate_scope_detection(approach_name, results_df)
            
            # Resolution evaluation
            resolution_metrics = self.evaluate_resolution_quality(approach_name, results_df)
            
            # Pattern analysis
            pattern_analysis = self.analyze_scope_patterns(approach_name, results_df)
            
            # Interpretation analysis
            interpretation_analysis = self.analyze_interpretations(approach_name, results_df)
            
            comparison_results[approach_name] = {
                'detection': detection_metrics,
                'resolution': resolution_metrics,
                'patterns': pattern_analysis,
                'interpretations': interpretation_analysis,
                'total_processed': len(results_df)
            }
        
        self.evaluation_metrics = comparison_results
        return comparison_results
    
    def analyze_interpretations(self, approach_name, results_df):
        """Analyze interpretation types chosen"""
        if 'interpretation' not in results_df.columns:
            return {}
        
        interpretation_counts = results_df['interpretation'].value_counts()
        
        # Accuracy by interpretation type
        interp_accuracy = {}
        for interp_type in interpretation_counts.index:
            subset = results_df[results_df['interpretation'] == interp_type]
            if len(subset) > 0:
                accuracy = subset['exact_match'].mean()
                interp_accuracy[interp_type] = {
                    'count': len(subset),
                    'accuracy': accuracy
                }
        
        return {
            'interpretation_distribution': interpretation_counts.to_dict(),
            'interpretation_accuracy': interp_accuracy
        }
    
    def generate_comprehensive_report(self, output_file="scope_evaluation_report.txt"):
        """Generate comprehensive evaluation report"""
        print(f"📝 Generating evaluation report...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("SCOPE AMBIGUITY RESOLUTION EVALUATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            
            if self.evaluation_metrics:
                # Find best approach
                best_approach = None
                best_f1 = 0
                
                for approach, metrics in self.evaluation_metrics.items():
                    f1 = metrics['detection'].get('f1_score', 0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_approach = approach
                
                f.write(f"Best Overall Approach: {best_approach} (F1: {best_f1:.3f})\n")
                
                # RAG effectiveness
                rag_approaches = [a for a in self.evaluation_metrics.keys() if 'rag' in a.lower()]
                non_rag = [a for a in self.evaluation_metrics.keys() if 'rag' not in a.lower()]
                
                if rag_approaches and non_rag:
                    rag_avg_f1 = np.mean([self.evaluation_metrics[a]['detection'].get('f1_score', 0) for a in rag_approaches])
                    non_rag_avg_f1 = np.mean([self.evaluation_metrics[a]['detection'].get('f1_score', 0) for a in non_rag])
                    
                    f.write(f"RAG Enhancement Effect: {rag_avg_f1:.3f} vs {non_rag_avg_f1:.3f} (+{rag_avg_f1-non_rag_avg_f1:.3f})\n")
                
                f.write("\n")
            
            # Detailed Results
            f.write("DETAILED RESULTS BY APPROACH\n")
            f.write("-" * 35 + "\n\n")
            
            # Comparison table
            if self.evaluation_metrics:
                f.write(f"{'Approach':<20} {'Det F1':<8} {'Exact':<8} {'Similarity':<10} {'Cases':<8}\n")
                f.write("-" * 54 + "\n")
                
                for approach, metrics in self.evaluation_metrics.items():
                    det_f1 = metrics['detection'].get('f1_score', 0.0)
                    exact = metrics['resolution'].get('exact_match_rate', 0.0)
                    sim = metrics['resolution'].get('avg_semantic_similarity', 0.0)
                    cases = metrics.get('total_processed', 0)
                    
                    f.write(f"{approach:<20} {det_f1:<8.3f} {exact:<8.3f} {sim:<10.3f} {cases:<8}\n")
                f.write("\n")
            
            # Individual approach details
            for approach, metrics in self.evaluation_metrics.items():
                f.write(f"APPROACH: {approach.upper()}\n")
                f.write("-" * 30 + "\n")
                
                # Detection metrics
                detection = metrics['detection']
                f.write("Detection Performance:\n")
                f.write(f"  Accuracy: {detection.get('accuracy', 0):.3f}\n")
                f.write(f"  Precision: {detection.get('precision', 0):.3f}\n")
                f.write(f"  Recall: {detection.get('recall', 0):.3f}\n")
                f.write(f"  F1-Score: {detection.get('f1_score', 0):.3f}\n")
                f.write(f"  Total Cases: {detection.get('total_cases', 0)}\n")
                f.write(f"  Ambiguous Cases: {detection.get('ambiguous_cases', 0)}\n")
                f.write(f"  Detected Cases: {detection.get('detected_cases', 0)}\n\n")
                
                # Resolution metrics
                resolution = metrics['resolution']
                f.write("Resolution Quality:\n")
                f.write(f"  Exact Match Rate: {resolution.get('exact_match_rate', 0):.3f}\n")
                f.write(f"  Semantic Similarity: {resolution.get('avg_semantic_similarity', 0):.3f}\n")
                f.write(f"  BLEU Score: {resolution.get('avg_bleu_score', 0):.3f}\n")
                f.write(f"  Ambiguous Cases Processed: {resolution.get('total_ambiguous_cases', 0)}\n\n")
                
                # Pattern analysis
                patterns = metrics.get('patterns', {})
                f.write("Pattern-Specific Performance:\n")
                for pattern, stats in patterns.items():
                    f.write(f"  {pattern}:\n")
                    f.write(f"    Cases: {stats['total_cases']}\n")
                    f.write(f"    Accuracy: {stats['accuracy']:.3f}\n")
                    f.write(f"    Change Rate: {stats['change_rate']:.3f}\n")
                    f.write(f"    Avg Similarity: {stats['avg_similarity']:.3f}\n")
                
                # Interpretation analysis
                interpretations = metrics.get('interpretations', {})
                if 'interpretation_distribution' in interpretations:
                    f.write("\nInterpretation Distribution:\n")
                    for interp, count in interpretations['interpretation_distribution'].items():
                        f.write(f"  {interp}: {count}\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"✅ Report saved to {output_file}")
        return output_file
    
    def create_visualizations(self, output_dir="scope_evaluation_plots"):
        """Create evaluation visualizations"""
        print("📊 Creating visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.evaluation_metrics:
            print("⚠️ No data to visualize")
            return
        
        # Overall performance comparison
        approaches = list(self.evaluation_metrics.keys())
        detection_f1 = [self.evaluation_metrics[a]['detection'].get('f1_score', 0) for a in approaches]
        exact_match = [self.evaluation_metrics[a]['resolution'].get('exact_match_rate', 0) for a in approaches]
        similarity = [self.evaluation_metrics[a]['resolution'].get('avg_semantic_similarity', 0) for a in approaches]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Detection F1
        bars1 = ax1.bar(range(len(approaches)), detection_f1, color='skyblue')
        ax1.set_title('Scope Detection F1-Score')
        ax1.set_ylabel('F1-Score')
        ax1.set_xticks(range(len(approaches)))
        ax1.set_xticklabels(approaches, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars1, detection_f1):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Exact match
        bars2 = ax2.bar(range(len(approaches)), exact_match, color='lightgreen')
        ax2.set_title('Exact Match Rate')
        ax2.set_ylabel('Exact Match Rate')
        ax2.set_xticks(range(len(approaches)))
        ax2.set_xticklabels(approaches, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        
        for bar, val in zip(bars2, exact_match):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Similarity
        bars3 = ax3.bar(range(len(approaches)), similarity, color='salmon')
        ax3.set_title('Semantic Similarity')
        ax3.set_ylabel('Similarity Score')
        ax3.set_xticks(range(len(approaches)))
        ax3.set_xticklabels(approaches, rotation=45, ha='right')
        ax3.set_ylim(0, 1)
        
        for bar, val in zip(bars3, similarity):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Pattern-specific heatmap
        self.create_pattern_heatmap(output_dir)
        
        # RAG comparison if applicable
        self.create_rag_comparison(output_dir)
        
        print(f"✅ Visualizations saved to {output_dir}/")
        return output_dir
    
    def create_pattern_heatmap(self, output_dir):
        """Create pattern-specific performance heatmap"""
        # Collect all patterns
        all_patterns = set()
        for metrics in self.evaluation_metrics.values():
            all_patterns.update(metrics.get('patterns', {}).keys())
        
        if not all_patterns:
            return
        
        approaches = list(self.evaluation_metrics.keys())
        pattern_matrix = []
        
        for approach in approaches:
            row = []
            for pattern in sorted(all_patterns):
                accuracy = self.evaluation_metrics[approach].get('patterns', {}).get(pattern, {}).get('accuracy', 0)
                row.append(accuracy)
            pattern_matrix.append(row)
        
        if pattern_matrix:
            plt.figure(figsize=(12, 8))
            sns.heatmap(pattern_matrix,
                       xticklabels=sorted(all_patterns),
                       yticklabels=approaches,
                       annot=True,
                       fmt='.3f',
                       cmap='RdYlBu_r',
                       center=0.5,
                       cbar_kws={'label': 'Accuracy'})
            plt.title('Pattern-Specific Accuracy by Approach')
            plt.xlabel('Scope Pattern Type')
            plt.ylabel('Approach')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/pattern_accuracy_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_rag_comparison(self, output_dir):
        """Create RAG vs non-RAG comparison"""
        rag_approaches = {k: v for k, v in self.evaluation_metrics.items() if 'rag' in k.lower()}
        non_rag_approaches = {k: v for k, v in self.evaluation_metrics.items() if 'rag' not in k.lower()}
        
        if not (rag_approaches and non_rag_approaches):
            return
        
        # Extract base approach names
        base_approaches = set()
        for approach in rag_approaches.keys():
            base = approach.replace('_rag', '').replace('-rag', '')
            base_approaches.add(base)
        
        # Compare RAG vs non-RAG for same base approach
        comparison_data = []
        for base in base_approaches:
            rag_key = None
            non_rag_key = None
            
            for k in rag_approaches.keys():
                if base in k:
                    rag_key = k
                    break
            
            for k in non_rag_approaches.keys():
                if base in k:
                    non_rag_key = k
                    break
            
            if rag_key and non_rag_key:
                rag_f1 = rag_approaches[rag_key]['detection'].get('f1_score', 0)
                non_rag_f1 = non_rag_approaches[non_rag_key]['detection'].get('f1_score', 0)
                
                rag_exact = rag_approaches[rag_key]['resolution'].get('exact_match_rate', 0)
                non_rag_exact = non_rag_approaches[non_rag_key]['resolution'].get('exact_match_rate', 0)
                
                comparison_data.append({
                    'approach': base,
                    'rag_f1': rag_f1,
                    'non_rag_f1': non_rag_f1,
                    'rag_exact': rag_exact,
                    'non_rag_exact': non_rag_exact
                })
        
        if comparison_data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            approaches = [d['approach'] for d in comparison_data]
            x = np.arange(len(approaches))
            width = 0.35
            
            # F1 comparison
            rag_f1_vals = [d['rag_f1'] for d in comparison_data]
            non_rag_f1_vals = [d['non_rag_f1'] for d in comparison_data]
            
            ax1.bar(x - width/2, non_rag_f1_vals, width, label='Without RAG', color='lightcoral')
            ax1.bar(x + width/2, rag_f1_vals, width, label='With RAG', color='lightblue')
            ax1.set_title('Detection F1-Score: RAG vs Non-RAG')
            ax1.set_ylabel('F1-Score')
            ax1.set_xticks(x)
            ax1.set_xticklabels(approaches)
            ax1.legend()
            ax1.set_ylim(0, 1)
            
            # Exact match comparison
            rag_exact_vals = [d['rag_exact'] for d in comparison_data]
            non_rag_exact_vals = [d['non_rag_exact'] for d in comparison_data]
            
            ax2.bar(x - width/2, non_rag_exact_vals, width, label='Without RAG', color='lightcoral')
            ax2.bar(x + width/2, rag_exact_vals, width, label='With RAG', color='lightblue')
            ax2.set_title('Exact Match Rate: RAG vs Non-RAG')
            ax2.set_ylabel('Exact Match Rate')
            ax2.set_xticks(x)
            ax2.set_xticklabels(approaches)
            ax2.legend()
            ax2.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/rag_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_comprehensive_evaluation(self, results_files):
        """Run complete evaluation pipeline"""
        print("🚀 Starting comprehensive scope evaluation...")
        print("=" * 60)
        
        # Load results
        num_loaded = self.load_results(results_files)
        if num_loaded == 0:
            print("❌ No results loaded")
            return
        
        # Compare approaches
        comparison_results = self.compare_approaches()
        
        # Generate report
        report_file = self.generate_comprehensive_report()
        
        # Create visualizations
        plot_dir = self.create_visualizations()
        
        # Save detailed results
        self.save_detailed_results()
        
        print("\n🎉 Evaluation completed!")
        print(f"📄 Report: {report_file}")
        print(f"📊 Plots: {plot_dir}")
        print(f"💾 Data: scope_evaluation_results.json")
        
        return comparison_results
    
    def save_detailed_results(self, filename="scope_evaluation_results.json"):
        """Save detailed results to JSON"""
        json_results = {}
        
        for approach, metrics in self.evaluation_metrics.items():
            json_results[approach] = {
                'detection': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                             for k, v in metrics['detection'].items() if k != 'classification_report'},
                'resolution': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                              for k, v in metrics['resolution'].items()},
                'patterns': {
                    pattern: {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                             for k, v in stats.items() if k != 'example_sentences'}
                    for pattern, stats in metrics.get('patterns', {}).items()
                },
                'total_processed': int(metrics.get('total_processed', 0))
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'approaches_evaluated': len(self.evaluation_metrics),
                    'evaluator_version': '1.0'
                },
                'results': json_results
            }, f, indent=2)
        
        print(f"✅ Detailed results saved to {filename}")


def main():
    """Main evaluation function"""
    # Define expected result files
    results_files = {
        'zero_shot': 'scope_results_zeroshot.csv',
        'zero_shot_rag': 'scope_results_zeroshot_rag.csv',
        'one_shot': 'scope_results_oneshot.csv',
        'one_shot_rag': 'scope_results_oneshot_rag.csv',
        'few_shot': 'scope_results_fewshot.csv',
        'few_shot_rag': 'scope_results_fewshot_rag.csv'
    }
    
    print("🎯 Scope Ambiguity Evaluation Suite")
    print("=" * 50)
    
    evaluator = ScopeEvaluator()
    results = evaluator.run_comprehensive_evaluation(results_files)
    
    if results:
        print("\n📊 EVALUATION SUMMARY:")
        print("-" * 30)
        for approach, metrics in results.items():
            det_f1 = metrics['detection'].get('f1_score', 0)
            exact_rate = metrics['resolution'].get('exact_match_rate', 0)
            similarity = metrics['resolution'].get('avg_semantic_similarity', 0)
            print(f"{approach}:")
            print(f"  Detection F1: {det_f1:.3f}")
            print(f"  Exact Match: {exact_rate:.3f}")
            print(f"  Similarity: {similarity:.3f}")
        
        print(f"\n✅ Comprehensive evaluation completed!")
    else:
        print("❌ No results to evaluate")


if __name__ == "__main__":
    main()
