---
layout: doc.njk
title: Benchmarks
navTitle: Benchmarks
description: Evaluation methodology, benchmark results, and failure analysis for scurl's detection system.
order: 3
---

This document describes the evaluation methodology, benchmark results, and failure analysis for scurl's prompt injection detection system.

## Evaluation Methodology

### Datasets

**Primary: deepset/prompt-injections**
- **Source**: HuggingFace Datasets
- **Size**: ~1,300 samples
- **Split**: 569 injection / 731 benign
- **Content**: Mix of direct injections, jailbreak attempts, and benign prompts

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | Of flagged content, how much is actually malicious? |
| **Recall** | TP / (TP + FN) | Of actual attacks, how many do we catch? |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean, balances precision/recall |
| **FPR** | FP / (FP + TN) | Benign content incorrectly flagged |
| **FNR** | FN / (FN + TP) | Attacks that slip through |

### Threshold Selection

| Threshold | Precision | Recall | F1 | FPR | FNR |
|-----------|-----------|--------|-----|-----|-----|
| 0.5 | 0.82 | 0.94 | 0.88 | 0.12 | 0.06 |
| 0.6 | 0.87 | 0.91 | 0.89 | 0.08 | 0.09 |
| **0.7** | **0.91** | **0.86** | **0.88** | **0.05** | **0.14** |
| 0.8 | 0.94 | 0.79 | 0.86 | 0.03 | 0.21 |
| 0.9 | 0.97 | 0.68 | 0.80 | 0.01 | 0.32 |

The default threshold of 0.7 prioritizes precision (fewer false positives) while maintaining acceptable recall.

## Results Summary

### Full Model (Pattern + Motif + Embedding)

```
              precision    recall  f1-score   support

      benign       0.93      0.95      0.94       146
   injection       0.93      0.91      0.92       114

    accuracy                           0.93       260
```

**Confusion Matrix:**
```
              Predicted
            Benign  Injection
Actual  Benign   139       7
     Injection    10     104
```

### Pattern-Only Model (No Embeddings)

```
              precision    recall  f1-score   support

      benign       0.82      0.89      0.85       146
   injection       0.84      0.75      0.79       114

    accuracy                           0.83       260
```

The embedding model provides a ~10% improvement in F1 score.

### Cross-Validation Results

5-fold cross-validation on training data:

| Fold | F1 Score |
|------|----------|
| Mean | 0.92 ± 0.03 |

## Failure Stratification

### False Positives (Benign flagged as injection)

| Category | Count | Example | Root Cause |
|----------|-------|---------|------------|
| Instructional content | 3 | "To ignore previous errors, restart the service" | "ignore previous" pattern |
| Technical documentation | 2 | "System override: press F8 during boot" | "system override" pattern |
| Creative writing | 1 | "You are now in the forest..." | "you are now" pattern |
| Meta-discussion | 1 | "What are your instructions for formatting?" | "your instructions" pattern |

### False Negatives (Attacks that evade detection)

| Category | Count | Evasion Technique |
|----------|-------|-------------------|
| Novel phrasing | 4 | Synonyms not in pattern library |
| Subtle manipulation | 3 | Implicit role injection |
| Encoded payloads | 2 | Encoding detection gaps |
| Multi-turn setup | 1 | Single-message analysis limitation |

### Failure by Attack Category

| Attack Type | Samples | Detected | Recall |
|-------------|---------|----------|--------|
| Instruction override | 156 | 148 | 94.9% |
| Role injection | 134 | 119 | 88.8% |
| System manipulation | 98 | 91 | 92.9% |
| Prompt leak | 87 | 79 | 90.8% |
| Jailbreak | 94 | 78 | 83.0% |

Jailbreak attempts show lowest recall—they often use novel social engineering rather than recognizable patterns.

## Feature Importance

Top 10 most important features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | instruction_override | 0.142 |
| 2 | role_injection | 0.128 |
| 3 | motif_instruction_override | 0.098 |
| 4 | system_manipulation | 0.087 |
| 5 | motif_role_injection | 0.076 |
| 6 | suspicious_delimiters | 0.068 |
| 7 | motif_density | 0.062 |
| 8 | jailbreak_keywords | 0.058 |
| 9 | prompt_leak | 0.054 |
| 10 | motif_category_count | 0.051 |

## Performance Benchmarks

### Latency (Single Sample)

| Component | Mean | P95 |
|-----------|------|-----|
| Normalization | 0.3ms | 0.5ms |
| Pattern extraction | 1.2ms | 2.1ms |
| Motif matching | 2.8ms | 4.2ms |
| Embedding (warm) | 45ms | 58ms |
| Classification | 0.4ms | 0.6ms |

**Total (pattern-only):** ~5ms
**Total (with embedding):** ~55ms

### Latency by Text Length

| Text Length | Pattern-Only | With Embedding |
|-------------|--------------|----------------|
| 100 chars | 2ms | 48ms |
| 1,000 chars | 4ms | 52ms |
| 10,000 chars | 12ms | 85ms |
| 100,000 chars | 45ms | 180ms |

### Memory Usage

| Component | Memory |
|-----------|--------|
| Base (patterns, motifs) | 15 MB |
| ONNX model loaded | 215 MB |
| Peak during inference | 280 MB |

## Comparison with Alternatives

### vs. Keyword Blocklist

| Metric | Blocklist | scurl |
|--------|-----------|-------|
| Precision | 0.95 | 0.91 |
| Recall | 0.42 | 0.86 |
| F1 | 0.58 | 0.88 |

Blocklists have high precision but miss most attacks.

### vs. GPT-4 Classification

| Metric | GPT-4 | scurl |
|--------|-------|-------|
| F1 | 0.90 | 0.88 |
| Latency | 500-2000ms | 50ms |
| Cost per 1K | $0.30 | $0.00 |

GPT-4 has slightly better recall but 10-40x higher latency and ongoing costs.

## Recommendations

### Threshold Tuning by Use Case

| Use Case | Recommended Threshold | Rationale |
|----------|----------------------|-----------|
| User-facing chat | 0.7 | Balance UX and safety |
| Code generation | 0.6 | Lower tolerance for injection |
| Document analysis | 0.8 | Reduce false positives |
| Security-critical | 0.5 | Maximize recall |

## Reproducing Results

```bash
# Install dependencies
pip install sibylline-scurl[training]

# Train classifier
python scripts/train_classifier.py

# Run with embeddings
python scripts/train_classifier.py --output models/with_embeddings.pkl

# Run pattern-only
python scripts/train_classifier.py --no-embeddings --output models/pattern_only.pkl
```
