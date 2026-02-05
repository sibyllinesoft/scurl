---
layout: doc.njk
title: Detection Algorithm
navTitle: Algorithm
description: Technical details of scurl's multi-stage prompt injection detection pipeline.
order: 2
---

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
```

#### 4. Prompt Leak
Patterns attempting to extract system prompts or instructions.

```
reveal (your )?(system )?prompt
show (me )?(your )?(system )?(prompt|instructions)
what (are|is) (your )?(system )?prompt
print (your )?(system )?prompt
```

#### 5. Jailbreak Keywords
Known jailbreak terminology and techniques.

```
\bDAN\b  (Do Anything Now)
\bjailbreak(ed|ing)?\b
(evil|dark|unrestricted) (mode|assistant|ai)
hypothetically speaking
```

#### 6. Encoding Markers
Indicators of encoded or obfuscated payloads.

```
base64:
decode (this|the following)
\\x[0-9a-fA-F]{2}  (hex escapes)
&#x?[0-9a-fA-F]+;  (HTML entities)
```

#### 7. Suspicious Delimiters
Fake message boundaries or role markers.

```
\[\s*system\s*\]
\[\s*instruction[s]?\s*\]
<\|?\s*(system|instruction|user|assistant)\s*\|?>
```

## Stage 3: Motif-Based Fuzzy Matching

Inspired by DNA sequence analysis, we treat attack patterns as "motifs"—short, fuzzy fragments that can tolerate mutations.

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
        ...
    ],
    'role_injection': [
        'you are now',
        'from now on',
        'act as',
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

## Stage 4: Sliding Window Analysis

For long documents, we use a two-phase approach to efficiently locate injection hotspots.

### Phase 1: Coarse Scan

Scan the full text with large overlapping windows (~4096 chars, 50% overlap).

```
Document: [=================================================]
Windows:  [====W1====]
              [====W2====]
                  [====W3====]
```

Each window receives a suspiciousness score based on motif density and category coverage.

### Phase 2: Fine Drill-Down

Windows exceeding the threshold (default: 0.3) become hotspots. These regions are re-scanned with smaller windows (~512 chars) for precise localization.

### Hotspot Clustering

Adjacent suspicious windows are merged using a DBSCAN-inspired 1D clustering:

```python
def detect_hotspots(windows):
    suspicious = [w for w in windows if w.score >= threshold]

    clusters = []
    current = [suspicious[0]]

    for window in suspicious[1:]:
        if window.start - current[-1].end <= step_size:
            current.append(window)  # Adjacent - merge
        else:
            clusters.append(create_hotspot(current))
            current = [window]  # Gap - start new

    return clusters
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
    class_weight='balanced',
)
```

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

### Datamark Mode (Spotlighting)
```xml
<pi p="0.85" t="override,role">
Please\ue000ignore\ue000all\ue000previous\ue000instructions.
</pi>
```

## Architecture Diagram

```
                         ┌─────────────────┐
                         │   Input Text    │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  Normalization  │
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
                         │  Random Forest  │
                         │   Classifier    │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  Score & Flag   │
                         └─────────────────┘
```
