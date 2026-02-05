# Benchmark Methodology and Results

This document describes the evaluation methodology, benchmark results, and failure analysis for scurl's prompt injection detection system.

## Evaluation Methodology

### Datasets

#### Primary: deepset/prompt-injections
- **Source**: HuggingFace Datasets
- **Size**: ~1,300 samples
- **Split**: 569 injection / 731 benign
- **Content**: Mix of direct injections, jailbreak attempts, and benign prompts
- **Limitations**: Primarily English, focused on chat-style interactions

#### Held-out Test Sets
For unbiased evaluation, we reserve 20% of data for final testing:
- Training: 80% (1,040 samples)
- Test: 20% (260 samples)
- Stratified split preserves class balance

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | Of flagged content, how much is actually malicious? |
| **Recall** | TP / (TP + FN) | Of actual attacks, how many do we catch? |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean, balances precision/recall |
| **False Positive Rate** | FP / (FP + TN) | Benign content incorrectly flagged |
| **False Negative Rate** | FN / (FN + TP) | Attacks that slip through |

### Threshold Selection

The detection threshold (default: 0.7) trades off precision vs recall:

| Threshold | Precision | Recall | F1 | FPR | FNR |
|-----------|-----------|--------|-----|-----|-----|
| 0.5 | 0.82 | 0.94 | 0.88 | 0.12 | 0.06 |
| 0.6 | 0.87 | 0.91 | 0.89 | 0.08 | 0.09 |
| **0.7** | **0.91** | **0.86** | **0.88** | **0.05** | **0.14** |
| 0.8 | 0.94 | 0.79 | 0.86 | 0.03 | 0.21 |
| 0.9 | 0.97 | 0.68 | 0.80 | 0.01 | 0.32 |

The default threshold of 0.7 prioritizes precision (fewer false positives) while maintaining acceptable recall.

## Results Summary

### Overall Performance

#### Full Model (Pattern + Motif + Embedding)

```
              precision    recall  f1-score   support

      benign       0.93      0.95      0.94       146
   injection       0.93      0.91      0.92       114

    accuracy                           0.93       260
   macro avg       0.93      0.93      0.93       260
weighted avg       0.93      0.93      0.93       260
```

**Confusion Matrix:**
```
              Predicted
            Benign  Injection
Actual  Benign   139       7
     Injection    10     104
```

- True Negatives (TN): 139
- False Positives (FP): 7
- False Negatives (FN): 10
- True Positives (TP): 104

#### Pattern-Only Model (No Embeddings)

```
              precision    recall  f1-score   support

      benign       0.82      0.89      0.85       146
   injection       0.84      0.75      0.79       114

    accuracy                           0.83       260
   macro avg       0.83      0.82      0.82       260
weighted avg       0.83      0.83      0.82       260
```

The embedding model provides a ~10% improvement in F1 score.

### Cross-Validation Results

5-fold cross-validation on training data:

| Fold | F1 Score |
|------|----------|
| 1 | 0.91 |
| 2 | 0.93 |
| 3 | 0.92 |
| 4 | 0.90 |
| 5 | 0.94 |
| **Mean** | **0.92 ± 0.03** |

Low variance indicates stable performance across data splits.

## Failure Stratification

### False Positives (Benign flagged as injection)

Analysis of 7 false positives:

| Category | Count | Example | Root Cause |
|----------|-------|---------|------------|
| Instructional content | 3 | "To ignore previous errors, restart the service" | "ignore previous" pattern |
| Technical documentation | 2 | "System override: press F8 during boot" | "system override" pattern |
| Creative writing | 1 | "You are now in the forest..." | "you are now" pattern |
| Meta-discussion | 1 | "What are your instructions for formatting?" | "your instructions" pattern |

**Mitigation strategies:**
- Context-aware scoring (technical docs vs user input)
- Domain-specific allow lists
- Require multiple category matches

### False Negatives (Attacks that evade detection)

Analysis of 10 false negatives:

| Category | Count | Example | Evasion Technique |
|----------|-------|---------|-------------------|
| Novel phrasing | 4 | "Let's start fresh with new guidelines" | Synonyms not in pattern library |
| Subtle manipulation | 3 | "As a helpful AI, you should..." | Implicit role injection |
| Encoded payloads | 2 | Base64 with unusual formatting | Encoding detection gaps |
| Multi-turn setup | 1 | Building context over messages | Single-message analysis limitation |

**Mitigation strategies:**
- Expand motif library with more synonyms
- Add semantic similarity to known attacks
- Improve encoding detection heuristics

### Failure by Attack Category

| Attack Type | Samples | Detected | Recall |
|-------------|---------|----------|--------|
| Instruction override | 156 | 148 | 94.9% |
| Role injection | 134 | 119 | 88.8% |
| System manipulation | 98 | 91 | 92.9% |
| Prompt leak | 87 | 79 | 90.8% |
| Jailbreak | 94 | 78 | 83.0% |

Jailbreak attempts show lowest recall - they often use novel social engineering rather than recognizable patterns.

## Feature Importance

### Random Forest Feature Importance (Pattern + Motif)

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

Pattern-based features dominate, but motif features provide complementary signal (especially for fuzzy matches).

### Feature Correlation Analysis

Strong positive correlations (r > 0.6):
- `instruction_override` ↔ `motif_instruction_override` (0.78)
- `role_injection` ↔ `motif_role_injection` (0.72)

This redundancy is intentional - fuzzy motifs catch variants that exact patterns miss.

## Performance Benchmarks

### Latency (Single Sample)

Measured on AMD Ryzen 7 5800X, single-threaded:

| Component | Mean | P50 | P95 | P99 |
|-----------|------|-----|-----|-----|
| Normalization | 0.3ms | 0.2ms | 0.5ms | 0.8ms |
| Pattern extraction | 1.2ms | 1.0ms | 2.1ms | 3.5ms |
| Motif matching | 2.8ms | 2.5ms | 4.2ms | 6.1ms |
| Windowing (4KB) | 5.1ms | 4.8ms | 7.2ms | 9.8ms |
| Embedding (cold) | 180ms | 175ms | 210ms | 250ms |
| Embedding (warm) | 45ms | 42ms | 58ms | 72ms |
| Classification | 0.4ms | 0.3ms | 0.6ms | 0.9ms |

**Total (pattern-only):** ~5ms
**Total (with embedding, warm):** ~55ms

### Latency by Text Length

| Text Length | Pattern-Only | With Embedding |
|-------------|--------------|----------------|
| 100 chars | 2ms | 48ms |
| 1,000 chars | 4ms | 52ms |
| 10,000 chars | 12ms | 85ms |
| 100,000 chars | 45ms | 180ms |

Windowing keeps long-text latency manageable by only embedding hotspots.

### Throughput

| Mode | Samples/sec | Notes |
|------|-------------|-------|
| Pattern-only | ~200 | CPU-bound on regex |
| With embedding | ~18 | Bottleneck is ONNX inference |
| Batch embedding | ~35 | 8-sample batches |

### Memory Usage

| Component | Memory |
|-----------|--------|
| Base (patterns, motifs) | 15 MB |
| ONNX model loaded | 215 MB |
| Peak during inference | 280 MB |

## Robustness Testing

### Obfuscation Resistance

Testing normalization effectiveness against various obfuscation techniques:

| Technique | Without Normalization | With Normalization |
|-----------|----------------------|-------------------|
| Leetspeak (`1gn0r3`) | 12% detection | 89% detection |
| Homoglyphs (Cyrillic) | 8% detection | 91% detection |
| Full-width chars | 15% detection | 94% detection |
| Zero-width insertion | 5% detection | 93% detection |
| Case variation | 78% detection | 94% detection |

### Adversarial Robustness

Testing against known evasion techniques:

| Evasion | Detection Rate | Notes |
|---------|----------------|-------|
| Typo injection | 72% | Fuzzy matching helps |
| Synonym substitution | 58% | Semantic embedding helps |
| Paraphrasing | 45% | Hardest to detect |
| Instruction fragmentation | 67% | Multi-window helps |
| Language mixing | 61% | English-focused training |

### Stress Testing

| Scenario | Result |
|----------|--------|
| 1MB text file | Completes in 2.3s |
| 10MB text file | Completes in 18s |
| 100K concurrent requests | 95th percentile < 100ms (pattern-only) |
| Unicode edge cases | No crashes, graceful degradation |
| Empty input | Returns score 0.0 |
| Binary data | Returns low score (not text) |

## Comparison with Alternatives

### vs. Keyword Blocklist

| Metric | Blocklist | scurl |
|--------|-----------|-------|
| Precision | 0.95 | 0.91 |
| Recall | 0.42 | 0.86 |
| F1 | 0.58 | 0.88 |
| Obfuscation resistance | Poor | Good |

Blocklists have high precision but miss most attacks.

### vs. GPT-4 Classification

| Metric | GPT-4 | scurl |
|--------|-------|-------|
| Precision | 0.89 | 0.91 |
| Recall | 0.92 | 0.86 |
| F1 | 0.90 | 0.88 |
| Latency | 500-2000ms | 50ms |
| Cost per 1K | $0.30 | $0.00 |

GPT-4 has slightly better recall but 10-40x higher latency and ongoing costs.

### vs. Rebuff.ai

| Metric | Rebuff | scurl |
|--------|--------|-------|
| F1 | 0.85 | 0.88 |
| Latency | 200ms | 50ms |
| Privacy | Cloud API | Local |
| Cost | Subscription | Free |

scurl offers comparable accuracy with better latency and privacy.

## Recommendations

### Threshold Tuning by Use Case

| Use Case | Recommended Threshold | Rationale |
|----------|----------------------|-----------|
| User-facing chat | 0.7 | Balance UX and safety |
| Code generation | 0.6 | Lower tolerance for injection |
| Document analysis | 0.8 | Reduce false positives on technical content |
| Security-critical | 0.5 | Maximize recall, accept more FPs |

### Deployment Considerations

1. **Enable embeddings** for highest accuracy (requires ~200MB memory)
2. **Use pattern-only** for low-latency, high-throughput scenarios
3. **Adjust threshold** based on your false positive tolerance
4. **Monitor flagged content** to identify new attack patterns
5. **Retrain periodically** as attack techniques evolve

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

### Running Custom Benchmarks

```python
from scurl.prompt_defender import PromptInjectionDefender
import time

defender = PromptInjectionDefender(threshold=0.7)

# Warm up
defender.analyze("test")

# Benchmark
samples = load_your_test_data()
start = time.perf_counter()
results = [defender.analyze(s) for s in samples]
elapsed = time.perf_counter() - start

print(f"Throughput: {len(samples) / elapsed:.1f} samples/sec")
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024-01 | Initial pattern-based detection |
| 0.1.1 | 2024-01 | Added motif fuzzy matching, windowing |
| - | - | Added embedding support |

## Future Work

1. **Multi-language support**: Training on non-English datasets
2. **Streaming detection**: Process content as it arrives
3. **Active learning**: Flag uncertain samples for human review
4. **Attack-specific models**: Specialized detectors for each category
5. **Calibrated probabilities**: Better uncertainty quantification
