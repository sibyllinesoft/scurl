#!/usr/bin/env python3
"""
Train prompt injection classifier on public datasets.

This script:
1. Downloads training data from HuggingFace
2. Extracts pattern features and embeddings for each sample
3. Trains a Random Forest classifier
4. Evaluates with cross-validation
5. Saves the model for use by scurl

Usage:
    python scripts/train_classifier.py

Requirements:
    pip install datasets scikit-learn

Optional (for embeddings):
    pip install onnxruntime tokenizers huggingface-hub
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scurl.prompt_defender.normalizer import TextNormalizer
from scurl.prompt_defender.patterns import PatternExtractor
from scurl.prompt_defender.motifs import MotifFeatureExtractor


def load_deepset_dataset():
    """Load the deepset/prompt-injections dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library required. Install with: pip install datasets")
        sys.exit(1)

    print("Loading deepset/prompt-injections dataset...")
    dataset = load_dataset("deepset/prompt-injections", split="train")
    return dataset


def load_embedder(use_embeddings: bool):
    """Load the embedding model if available."""
    if not use_embeddings:
        return None

    try:
        from scurl.prompt_defender.embedder import EmbeddingGemmaONNX
        print("Loading EmbeddingGemma model (this may download ~200MB on first run)...")
        embedder = EmbeddingGemmaONNX()
        # Trigger lazy load
        embedder._ensure_loaded()
        print(f"Embedder loaded: {embedder.embedding_dim} dimensions")
        return embedder
    except ImportError as e:
        print(f"Warning: Could not load embedder ({e})")
        print("Training with pattern features only.")
        return None
    except Exception as e:
        print(f"Warning: Embedder initialization failed ({e})")
        print("Training with pattern features only.")
        return None


def extract_features(
    texts: list,
    labels: list,
    normalizer: TextNormalizer,
    pattern_extractor: PatternExtractor,
    motif_extractor: MotifFeatureExtractor = None,
    embedder=None,
    show_progress: bool = True,
):
    """Extract features for all samples."""
    features = []
    valid_labels = []
    skipped = 0

    # Initialize motif extractor if not provided
    if motif_extractor is None:
        motif_extractor = MotifFeatureExtractor()

    total = len(texts)
    for i, (text, label) in enumerate(zip(texts, labels)):
        if show_progress and i % 100 == 0:
            print(f"  Processing {i}/{total}...", end="\r")

        try:
            # Normalize text
            normalized = normalizer.normalize(str(text))

            # Extract pattern features
            pattern_feat = pattern_extractor.extract(normalized)
            pattern_array = np.array(pattern_feat.to_array(), dtype=np.float32)

            # Extract motif features
            motif_array = np.array(motif_extractor.extract(normalized), dtype=np.float32)

            if embedder is not None:
                # Generate embedding
                embedding = embedder.embed(normalized)
                combined = np.concatenate([pattern_array, motif_array, embedding])
            else:
                combined = np.concatenate([pattern_array, motif_array])

            features.append(combined)
            valid_labels.append(label)

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"\nWarning: Skipping sample due to error: {e}")

    if show_progress:
        print(f"  Processed {total}/{total} samples" + " " * 20)

    if skipped > 0:
        print(f"  Skipped {skipped} samples due to errors")

    return np.array(features), np.array(valid_labels)


def train_classifier(X: np.ndarray, y: np.ndarray, n_estimators: int = 100):
    """Train Random Forest classifier with cross-validation."""
    print(f"\nTraining Random Forest with {n_estimators} estimators...")
    print(f"  Feature dimensions: {X.shape[1]}")
    print(f"  Training samples: {len(y)}")
    print(f"  Class distribution: {np.bincount(y)}")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="f1")
    print(f"  F1 scores: {cv_scores}")
    print(f"  Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Train/test split for detailed evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nClassification Report (on 20% held-out test set):")
    print(classification_report(y_test, y_pred, target_names=["benign", "injection"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    # Final training on full dataset
    print("\nTraining final model on full dataset...")
    clf.fit(X, y)

    return clf


def save_model(clf, output_path: Path):
    """Save trained model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Train prompt injection classifier"
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Train with pattern features only (faster, smaller model)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/scurl/prompt_defender/models/prompt_injection_rf.pkl"),
        help="Output path for trained model",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in Random Forest",
    )
    args = parser.parse_args()

    # Load dataset
    dataset = load_deepset_dataset()
    print(f"  Loaded {len(dataset)} samples")

    # Initialize components
    print("\nInitializing components...")
    normalizer = TextNormalizer(use_confusables=True)
    pattern_extractor = PatternExtractor()
    motif_extractor = MotifFeatureExtractor()
    embedder = load_embedder(use_embeddings=not args.no_embeddings)

    # Extract features
    print("\nExtracting features...")
    X, y = extract_features(
        dataset["text"],
        dataset["label"],
        normalizer,
        pattern_extractor,
        motif_extractor,
        embedder,
    )

    # Train classifier
    clf = train_classifier(X, y, n_estimators=args.n_estimators)

    # Save model
    save_model(clf, args.output)

    # Print feature importances for pattern and motif features
    if embedder is None:
        from scurl.prompt_defender.patterns import PatternFeatures
        feature_names = PatternFeatures.feature_names() + MotifFeatureExtractor.FEATURE_NAMES
        importances = clf.feature_importances_
        print("\nFeature Importances:")
        for name, imp in sorted(
            zip(feature_names, importances), key=lambda x: -x[1]
        ):
            print(f"  {name}: {imp:.4f}")


if __name__ == "__main__":
    main()
