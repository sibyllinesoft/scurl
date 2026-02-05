# Prompt Injection Detection Algorithm

This document describes the multi-stage detection algorithm used by scurl's prompt-defender middleware to identify and mitigate prompt injection attacks in web content.

## Overview

The detection pipeline combines multiple complementary techniques:

1. **Text Normalization** - Defeats obfuscation attempts
2. **Pattern-Based Feature Extraction** - Regex patterns for known attack signatures
3. **Motif-Based Fuzzy Matching** - DNA-inspired fuzzy fragment detection
4. **Sliding Window Analysis** - Density-based hotspot localization
5. **Semantic Embeddings** - Captures meaning beyond surface patterns
6. **ML Classification** - Combines all signals for final prediction

This layered approach provides defense-in-depth: each layer catches attacks that might slip through others.

## Stage 1: Text Normalization

Before any detection, text undergoes normalization to defeat common obfuscation techniques.

### Transformations Applied

| Transformation | Purpose | Example |
|---------------|---------|---------|
| NFKC Normalization | Converts Unicode variants to canonical form | `ｉｇｎｏｒｅ` → `ignore` |
| Homoglyph Resolution | Maps visually similar characters | `іgnоrе` (Cyrillic) → `ignore` |
| Zero-Width Removal | Strips invisible characters | `ig​no​re` → `ignore` |
| BiDi Override Removal | Removes text direction manipulation | Prevents right-to-left tricks |
| Whitespace Normalization | Collapses whitespace sequences | Multiple spaces → single space |
| Lowercase | Case-insensitive matching | `IGNORE` → `ignore` |

### Implementation

```python
class TextNormalizer:
    def normalize(self, text: str) -> str:
        # 1. Unicode NFKC normalization
        text = unicodedata.normalize('NFKC', text)

        # 2. Homoglyph/confusable resolution (optional)
        if self._confusables:
            text = self._resolve_confusables(text)

        # 3. Remove zero-width and BiDi characters
        text = self._strip_invisible(text)

        # 4. Normalize whitespace
        text = ' '.join(text.split())

        # 5. Lowercase
        return text.lower()
```

### Confusable Characters

When the `confusable-homoglyphs` library is available, we resolve characters that look identical but have different Unicode code points:

- Cyrillic `а` (U+0430) → Latin `a` (U+0061)
- Greek `ο` (U+03BF) → Latin `o` (U+006F)
- Full-width `Ａ` (U+FF21) → Latin `A` (U+0041)

This defeats attacks like: `ιgnοrе ρrеvιοus ιnstructιοns` (mixed scripts)

## Stage 2: Pattern-Based Feature Extraction

We extract 7 pattern categories plus 5 text statistics, producing a 12-dimensional feature vector.

### Pattern Categories

Each category contains 8-15 regex patterns. Match counts are normalized by text length (matches per 1000 characters, capped at 1.0).

#### 1. Instruction Override
Patterns that attempt to cancel or replace previous instructions.

```
ignore (all )?(previous|prior|above) (instructions|prompts|rules)
disregard (all )?(previous|prior|earlier)
forget (everything )?(above|before|prior)
new instructions:
actual instructions:
```

#### 2. Role Injection
Patterns that attempt to redefine the AI's identity or purpose.

```
you are now
from now on,? (you|your)
act as (if )?(you )?(are |were )?
pretend (to be|you are|that you)
your new (role|goal|purpose|instruction)
```

#### 3. System Manipulation
Patterns targeting system-level access or safety bypasses.

```
(admin|developer|god|sudo|root) mode
system (override|prompt|instruction)
unlock (all )?(restrictions|capabilities)
disable (all )?(safety|security)?(filters|guards)
bypass (all )?(restrictions|filters|safety)
```

#### 4. Prompt Leak
Patterns attempting to extract system prompts or instructions.

```
reveal (your )?(system )?prompt
show (me )?(your )?(system )?(prompt|instructions)
what (are|is) (your )?(system )?prompt
print (your )?(system )?prompt
(initial|original|hidden|secret) prompt
```

#### 5. Jailbreak Keywords
Known jailbreak terminology and techniques.

```
\bDAN\b  (Do Anything Now)
\bjailbreak(ed|ing)?\b
do anything now
(evil|dark|unrestricted) (mode|assistant|ai)
hypothetically speaking
for educational purposes only
```

#### 6. Encoding Markers
Indicators of encoded or obfuscated payloads.

```
base64:
decode (this|the following)
\\x[0-9a-fA-F]{2}  (hex escapes)
&#x?[0-9a-fA-F]+;  (HTML entities)
%[0-9a-fA-F]{2}   (URL encoding)
```

#### 7. Suspicious Delimiters
Fake message boundaries or role markers.

```
\[\s*system\s*\]
\[\s*instruction[s]?\s*\]
<\|?\s*(system|instruction|user|assistant)\s*\|?>
###\s*(system|instruction)
```

### Text Statistics

| Feature | Calculation | Rationale |
|---------|-------------|-----------|
| `text_length` | len / 10000, capped at 1.0 | Longer text may dilute attack signals |
| `special_char_ratio` | special / total | High ratio may indicate obfuscation |
| `caps_ratio` | uppercase / alphabetic | Shouting patterns, emphasis |
| `newline_density` | newlines / total | Structured injection attempts |
| `avg_word_length` | mean(word_lengths) / 20 | Unusual encoding detection |

## Stage 3: Motif-Based Fuzzy Matching

Inspired by DNA sequence analysis, we treat attack patterns as "motifs" - short, fuzzy fragments that can tolerate mutations.

### Motivation

Traditional regex is brittle against:
- Typos: `ignor previus instructions`
- Insertions: `ig.no" re pre-vi-ous`
- Character substitutions: `1gnore prev10us`

Fuzzy matching using edit distance (Levenshtein) catches these variants.

### Motif Library

60+ short phrases across 6 categories:

```python
MOTIF_LIBRARY = {
    'instruction_override': [
        'ignore previous',
        'ignore all',
        'disregard above',
        'forget everything',
        'new instructions',
        ...
    ],
    'role_injection': [
        'you are now',
        'from now on',
        'act as',
        'pretend to be',
        ...
    ],
    ...
}
```

### Fuzzy Matching Algorithm

Using RapidFuzz's `partial_ratio` for substring-aware fuzzy matching:

```python
def find_matches(text, window_size=50, step=25):
    matches = []
    for pos in range(0, len(text), step):
        window = text[pos:pos + window_size]
        for motif, category in flat_motifs:
            score = fuzz.partial_ratio(window, motif)
            if score >= threshold:  # default: 75
                matches.append(MotifMatch(
                    motif=motif,
                    category=category,
                    position=pos,
                    score=score,
                ))
    return matches
```

### Motif Features

9 features extracted from motif matching:

| Feature | Description |
|---------|-------------|
| `motif_density` | Matches per 1000 chars |
| `motif_instruction_override` | Max score in category |
| `motif_role_injection` | Max score in category |
| `motif_system_manipulation` | Max score in category |
| `motif_prompt_leak` | Max score in category |
| `motif_jailbreak` | Max score in category |
| `motif_delimiters` | Max score in category |
| `motif_max_score` | Highest score across all |
| `motif_category_count` | Number of categories with matches |

## Stage 4: Sliding Window Analysis

For long documents, we use a two-phase approach to efficiently locate injection hotspots.

### Phase 1: Coarse Scan

Scan the full text with large overlapping windows (~4096 chars, 50% overlap).

```
Document: [=================================================]
Windows:  [====W1====]
              [====W2====]
                  [====W3====]
                      [====W4====]
```

Each window receives a suspiciousness score based on motif density and category coverage.

### Phase 2: Fine Drill-Down

Windows exceeding the threshold (default: 0.3) become hotspots. These regions are re-scanned with smaller windows (~512 chars) for precise localization.

```
Hotspot:  [==========SUSPICIOUS REGION==========]
Fine:     [==F1==]
            [==F2==]
              [==F3==]  ← Highest score, likely injection location
                [==F4==]
```

### Hotspot Clustering

Adjacent suspicious windows are merged using a DBSCAN-inspired 1D clustering:

```python
def detect_hotspots(windows):
    suspicious = [w for w in windows if w.score >= threshold]

    clusters = []
    current = [suspicious[0]]

    for window in suspicious[1:]:
        if window.start - current[-1].end <= step_size:
            # Adjacent - merge
            current.append(window)
        else:
            # Gap - finalize cluster, start new
            clusters.append(create_hotspot(current))
            current = [window]

    return clusters
```

### Batching for Embedding

Nearby hotspots are batched (up to 1024 chars) to minimize embedding API calls:

```
Hotspots: [H1]    [H2]  [H3]        [H4]
          ↓         ↓                ↓
Batches:  [==Batch1==]           [Batch2]
          (H1+H2+H3 merged)      (H4 alone, too far)
```

## Stage 5: Semantic Embeddings

Pattern matching alone misses novel attacks. Semantic embeddings capture meaning.

### Model: EmbeddingGemma-300M

- **Architecture**: Gemma-based encoder (300M parameters)
- **Output**: 768-dimensional embeddings
- **Matryoshka**: Truncate to 128 dims with minimal accuracy loss
- **Runtime**: ONNX for CPU inference without PyTorch

### Embedding Strategy

For short texts (< 4096 chars):
```python
embedding = embedder.embed_chunks(text, chunk_size=400, overlap=50)
```

For long texts with windowing:
```python
# Only embed identified hotspot regions
embeddings = []
for start, end in hotspot_regions:
    embeddings.append(embedder.embed(text[start:end]))

# Max-pool to capture strongest signal
final_embedding = np.max(embeddings, axis=0)
```

### Chunking Strategy

Long texts are split into overlapping word-based chunks:

```
Text:     [word1 word2 word3 word4 word5 word6 word7 word8 ...]
Chunk 1:  [word1 word2 word3 word4]
Chunk 2:        [word3 word4 word5 word6]  (overlap)
Chunk 3:              [word5 word6 word7 word8]
```

Chunk embeddings are max-pooled to preserve the strongest semantic signals.

## Stage 6: ML Classification

All features are combined and fed to a Random Forest classifier.

### Feature Vector

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Pattern features | 12 | Regex matches + text stats |
| Motif features | 9 | Fuzzy matching signals |
| Embedding | 128 | Semantic representation |
| **Total** | **149** | |

In pattern-only mode (no embeddings): 21 dimensions.

### Classifier Architecture

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # Handle imbalanced data
)
```

### Training Data

Trained on the `deepset/prompt-injections` dataset:
- ~600 prompt injection samples
- ~700 benign samples
- 5-fold cross-validation

### Probability Calibration

The classifier outputs a probability score (0.0 to 1.0). The threshold (default: 0.7) determines the flagging decision:

```python
score = classifier.predict_proba(features)
flagged = score >= threshold
```

Higher threshold = fewer false positives, more false negatives.

## Mitigation Actions

When injection is detected, several mitigation modes are available:

### Warn Mode
```xml
<pi p="0.85" t="override,role">
Original content unchanged, but wrapped in warning tag
</pi>
```

### Redact Mode
```xml
<pi p="0.85" t="override,role">
Please ████████████████████████████. You are now ████████████.
</pi>
```
Matched patterns replaced with Unicode block characters (█).

### Datamark Mode (Spotlighting)
```xml
<pi p="0.85" t="override,role">
Please\ue000ignore\ue000all\ue000previous\ue000instructions.
</pi>
```
Words delimited with Unicode Private Use Area character, following Microsoft's spotlighting defense.

### Metadata Mode
```json
{
  "content": "Original text...",
  "injection_analysis": {
    "score": 0.8523,
    "threshold": 0.7,
    "flagged": true,
    "pattern_features": {...},
    "matched_patterns": {...},
    "matched_spans": [[45, 89], [102, 145]],
    "mode": "ml"
  }
}
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Normalization | O(n) | < 1ms |
| Pattern extraction | O(n × p) | < 5ms |
| Motif matching | O(n × m × w) | < 10ms |
| Embedding (per chunk) | O(1) | ~50ms |
| Classification | O(1) | < 1ms |

Where n = text length, p = pattern count, m = motif count, w = window operations.

### Memory Usage

| Component | Memory |
|-----------|--------|
| Pattern regexes | ~50 KB |
| Motif library | ~10 KB |
| ONNX model | ~200 MB (loaded on demand) |
| Tokenizer | ~5 MB |

### Fallback Behavior

If embedding dependencies unavailable:
1. Motif features provide fuzzy signal
2. Pattern classifier uses 21-dim feature vector
3. Accuracy degrades ~15-20% but remains functional

## Architecture Diagram

```
                         ┌─────────────────┐
                         │   Input Text    │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  Normalization  │
                         │  (NFKC, etc.)   │
                         └────────┬────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
   ┌────────▼────────┐   ┌───────▼───────┐   ┌────────▼────────┐
   │ Pattern Extract │   │ Motif Matcher │   │ Window Analyzer │
   │   (12 features) │   │  (9 features) │   │   (hotspots)    │
   └────────┬────────┘   └───────┬───────┘   └────────┬────────┘
            │                     │                    │
            │                     │           ┌───────▼───────┐
            │                     │           │   Embedder    │
            │                     │           │ (128 features)│
            │                     │           └───────┬───────┘
            │                     │                    │
            └─────────────────────┼────────────────────┘
                                  │
                         ┌────────▼────────┐
                         │  Concatenate    │
                         │  (149 features) │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │ Random Forest   │
                         │  Classifier     │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  Score & Flag   │
                         └────────┬────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
              ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
              │   Warn    │ │  Redact   │ │ Datamark  │
              └───────────┘ └───────────┘ └───────────┘
```

## References

1. Greshake et al. "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection" (2023)
2. Microsoft. "Spotlighting: Defending against prompt injection" (2024)
3. Perez & Ribeiro. "Ignore This Title and HackAPrompt" (2023)
4. Lee et al. "Matryoshka Representation Learning" (2022)
