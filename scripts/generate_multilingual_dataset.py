#!/usr/bin/env python3
"""
Generate multilingual prompt injection training dataset.

This script combines benign text samples with injection templates to create
a labeled dataset for training the prompt injection classifier.

Usage:
    python scripts/generate_multilingual_dataset.py

Output:
    data/multilingual_training/dataset.json - Combined training dataset
"""

import json
import random
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "multilingual_training"
BENIGN_FILE = DATA_DIR / "benign_samples.json"
INJECTION_FILE = DATA_DIR / "injection_templates.json"
OUTPUT_FILE = DATA_DIR / "dataset.json"


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def insert_injection(benign_text: str, injection: str, position: str = "random") -> str:
    """Insert an injection into benign text at various positions."""
    if position == "start":
        return f"{injection} {benign_text}"
    elif position == "end":
        return f"{benign_text} {injection}"
    elif position == "middle":
        words = benign_text.split()
        mid = len(words) // 2
        return " ".join(words[:mid]) + f" {injection} " + " ".join(words[mid:])
    else:  # random
        pos = random.choice(["start", "middle", "end"])
        return insert_injection(benign_text, injection, pos)


def generate_dataset(balance_ratio: float = 0.5):
    """Generate the combined dataset.

    Args:
        balance_ratio: Target ratio of benign samples (0.5 = balanced).
    """
    random.seed(42)

    benign_samples = load_json(BENIGN_FILE)
    injection_templates = load_json(INJECTION_FILE)

    dataset = []

    for lang, texts in benign_samples.items():
        print(f"\nProcessing {lang}...")

        # Add all benign samples as label=0
        benign_count = 0
        for text in texts:
            dataset.append({
                "text": text,
                "label": 0,
                "language": lang,
                "category": "benign"
            })
            benign_count += 1

        print(f"  Added {benign_count} benign samples")

        # Get injection templates for this language
        if lang not in injection_templates:
            print(f"  Warning: No injection templates for {lang}")
            continue

        lang_injections = injection_templates[lang]
        injected_count = 0

        # For each injection category, create samples
        for category, injections in lang_injections.items():
            # Use a subset of benign texts for each injection
            sample_texts = random.sample(texts, min(5, len(texts)))

            for benign_text in sample_texts:
                # Pick a random injection from this category
                injection = random.choice(injections)

                # Insert at random position
                injected_text = insert_injection(benign_text, injection)

                dataset.append({
                    "text": injected_text,
                    "label": 1,
                    "language": lang,
                    "category": category
                })
                injected_count += 1

        print(f"  Added {injected_count} injection samples")

    # Also add pure injection samples (injection only, no benign context)
    print("\nAdding pure injection samples...")
    pure_count = 0
    for lang, categories in injection_templates.items():
        for category, injections in categories.items():
            for injection in injections:
                dataset.append({
                    "text": injection,
                    "label": 1,
                    "language": lang,
                    "category": f"pure_{category}"
                })
                pure_count += 1
    print(f"  Added {pure_count} pure injection samples")

    # Shuffle the dataset
    random.shuffle(dataset)

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Print statistics
    print(f"\n{'='*50}")
    print("Dataset Statistics:")
    print(f"{'='*50}")

    total = len(dataset)
    benign = sum(1 for d in dataset if d["label"] == 0)
    malicious = sum(1 for d in dataset if d["label"] == 1)

    print(f"Total samples: {total}")
    print(f"Benign (label=0): {benign} ({100*benign/total:.1f}%)")
    print(f"Malicious (label=1): {malicious} ({100*malicious/total:.1f}%)")

    print("\nBy language:")
    for lang in benign_samples.keys():
        lang_samples = [d for d in dataset if d["language"] == lang]
        lang_benign = sum(1 for d in lang_samples if d["label"] == 0)
        lang_mal = sum(1 for d in lang_samples if d["label"] == 1)
        print(f"  {lang}: {len(lang_samples)} total ({lang_benign} benign, {lang_mal} malicious)")

    print("\nBy injection category:")
    categories = set(d["category"] for d in dataset if d["label"] == 1)
    for cat in sorted(categories):
        count = sum(1 for d in dataset if d["category"] == cat)
        print(f"  {cat}: {count}")

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_dataset()
