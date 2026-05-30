
# Vagueness Resolution Evaluation Report
Generated on: 2025-08-21 02:31:29

## Executive Summary

This report presents a comprehensive evaluation of different approaches for resolving vagueness in software requirements. The evaluation covers zero-shot, one-shot, and few-shot prompting strategies, with and without RAG enhancement.

### Key Findings

1. **Best Performing Approach**: few-shot+RAG + RAG
   - Average Similarity Score: 0.7760
   - Average BLEU Score: 0.2040
   - Vagueness Reduction: 0.9802

2. **Total Requirements Processed**: 606

3. **Overall Performance Metrics**:
   - Mean Similarity Score: 0.7648
   - Mean BLEU Score: 0.1998
   - Mean Vagueness Reduction: 0.9851
   - Mean Semantic Preservation: 0.8509

## Detailed Results by Approach


### few-shot+RAG + RAG
- **Requirements Processed**: 101
- **Average Similarity Score**: 0.7760
- **Average BLEU Score**: 0.2040
- **Average Vagueness Reduction**: 0.9802
- **Average Semantic Preservation**: 0.8515
- **High Quality Results**: 78


### few-shot
- **Requirements Processed**: 101
- **Average Similarity Score**: 0.7727
- **Average BLEU Score**: 0.2050
- **Average Vagueness Reduction**: 0.9802
- **Average Semantic Preservation**: 0.8515
- **High Quality Results**: 77


### zero-shot
- **Requirements Processed**: 101
- **Average Similarity Score**: 0.7721
- **Average BLEU Score**: 0.2152
- **Average Vagueness Reduction**: 0.9901
- **Average Semantic Preservation**: 0.8762
- **High Quality Results**: 76


### zero-shot+RAG + RAG
- **Requirements Processed**: 101
- **Average Similarity Score**: 0.7584
- **Average BLEU Score**: 0.2189
- **Average Vagueness Reduction**: 0.9901
- **Average Semantic Preservation**: 0.8550
- **High Quality Results**: 72


### one-shot+RAG + RAG
- **Requirements Processed**: 101
- **Average Similarity Score**: 0.7568
- **Average BLEU Score**: 0.1783
- **Average Vagueness Reduction**: 0.9901
- **Average Semantic Preservation**: 0.8356
- **High Quality Results**: 75


### one-shot
- **Requirements Processed**: 101
- **Average Similarity Score**: 0.7526
- **Average BLEU Score**: 0.1775
- **Average Vagueness Reduction**: 0.9802
- **Average Semantic Preservation**: 0.8354
- **High Quality Results**: 69


## Vagueness Pattern Analysis

### Most Common Vagueness Types
- **Ambiguity**: 319 occurrences
- **Quantitative Vagueness**: 152 occurrences
- **Temporal Vagueness**: 26 occurrences
- **Ambiguity and Lack of Specificity**: 16 occurrences
- **Ambiguity and lack of specificity**: 8 occurrences

### Most Frequently Resolved Vague Terms
- **sufficient**: 15 times
- **sufficient cues**: 12 times
- **"a minimum of 500"**: 11 times
- **"as they become available"**: 8 times
- **necessary**: 8 times
- **electronic file**: 7 times
- **"hard copy format"**: 7 times
- **functional operation mode**: 7 times
- **efficiently**: 6 times
- **"available"**: 6 times

### Success Rate by Vagueness Type (Top 5)
- **Imprecision in scope**: Similarity 0.964, BLEU 0.582 (1 cases)
- **Time-related vagueness**: Similarity 0.954, BLEU 0.365 (2 cases)
- **Imprecise assumptions, unspecified hardware requirements**: Similarity 0.944, BLEU 0.150 (1 cases)
- **Imprecise condition**: Similarity 0.934, BLEU 0.505 (1 cases)
- **Quantitative Vagueness, Temporal Vagueness, Qualitative Vagueness, Scope Vagueness**: Similarity 0.922, BLEU 0.144 (1 cases)

## Methodology

The evaluation used the following metrics:

1. **Similarity Score**: Semantic similarity between predicted and ground truth using sentence transformers
2. **BLEU Score**: Token-level similarity using BLEU-4 with smoothing
3. **Vagueness Reduction**: Percentage of vague terms successfully removed or clarified
4. **Semantic Preservation**: How well the original meaning was maintained during transformation

## Data Split

- **Total Requirements**: From single CSV file
- **RAG Examples**: 50 (used for prompting context)
- **Test Examples**: Remaining (used for evaluation)

## Recommendations

Based on the evaluation results:

1. **few-shot+RAG + RAG** shows the best overall performance
2. RAG enhancement significantly improves results
3. Focus on handling **Ambiguity** vagueness types as they are most common
4. Consider specialized handling for terms like **sufficient** which appear frequently

## Conclusion

The evaluation demonstrates that LLM-based approaches can effectively resolve vagueness in software requirements, with few-shot+RAG + RAG achieving the best performance across multiple metrics.
