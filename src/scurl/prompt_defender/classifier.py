"""Classifier for prompt injection detection."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .patterns import PatternFeatures


class InjectionClassifier:
    """Random Forest classifier for prompt injection detection.

    Expects feature vectors combining pattern features and embeddings.
    Lazy-loads model on first use.
    """

    MODEL_FILENAME = "prompt_injection_rf.pkl"

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize classifier.

        Args:
            model_path: Path to pickled model file. If not provided,
                       looks in package data and default cache locations.
        """
        self._model_path = model_path
        self._model = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model on first use."""
        if self._model is not None:
            return

        # Try explicit path first
        if self._model_path and self._model_path.exists():
            with open(self._model_path, 'rb') as f:
                self._model = pickle.load(f)
            return

        # Try package data
        try:
            import importlib.resources as resources
            try:
                # Python 3.9+
                files = resources.files('scurl.prompt_defender.models')
                model_file = files.joinpath(self.MODEL_FILENAME)
                if model_file.is_file():
                    with model_file.open('rb') as f:
                        self._model = pickle.load(f)
                    return
            except (AttributeError, TypeError):
                # Python 3.8 fallback
                with resources.open_binary(
                    'scurl.prompt_defender.models',
                    self.MODEL_FILENAME
                ) as f:
                    self._model = pickle.load(f)
                return
        except (FileNotFoundError, ModuleNotFoundError):
            pass

        # Try cache directory
        from .embedder import EmbeddingGemmaONNX
        cache_path = EmbeddingGemmaONNX._default_model_dir() / self.MODEL_FILENAME
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                self._model = pickle.load(f)
            return

        raise RuntimeError(
            f"Classifier model not found. Expected at:\n"
            f"  - {self._model_path or 'Not specified'}\n"
            f"  - Package data: scurl.prompt_defender.models/{self.MODEL_FILENAME}\n"
            f"  - Cache: {cache_path}\n\n"
            f"Please run the training script or download a pre-trained model."
        )

    def predict_proba(self, features: np.ndarray) -> float:
        """Predict probability of prompt injection.

        Args:
            features: Feature vector of shape (n_features,) or (1, n_features).

        Returns:
            Probability of injection (0.0 to 1.0).
        """
        self._ensure_loaded()

        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get probability of positive class (injection = 1)
        proba = self._model.predict_proba(features)[0, 1]
        return float(proba)

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> bool:
        """Predict whether input is prompt injection.

        Args:
            features: Feature vector.
            threshold: Classification threshold.

        Returns:
            True if predicted as injection.
        """
        return self.predict_proba(features) >= threshold

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def n_features(self) -> Optional[int]:
        """Return expected number of features, or None if not loaded."""
        if self._model is None:
            return None
        return self._model.n_features_in_


class PatternOnlyClassifier:
    """Simple threshold-based classifier using only pattern features.

    Useful as a fallback when the full model isn't available,
    or for fast-path detection of obvious injections.
    """

    # Weights for each pattern category
    WEIGHTS = {
        'instruction_override': 3.0,
        'role_injection': 2.5,
        'system_manipulation': 3.0,
        'prompt_leak': 2.0,
        'jailbreak_keywords': 2.5,
        'encoding_markers': 1.0,
        'suspicious_delimiters': 1.5,
    }

    def __init__(self, threshold: float = 0.3):
        """Initialize classifier.

        Args:
            threshold: Score threshold for detection.
        """
        self.threshold = threshold

    def predict_proba(self, pattern_features: 'PatternFeatures') -> float:
        """Calculate weighted score from pattern features.

        Args:
            pattern_features: PatternFeatures dataclass instance.

        Returns:
            Weighted score (higher = more likely injection).
        """

        score = 0.0
        score += pattern_features.instruction_override * self.WEIGHTS['instruction_override']
        score += pattern_features.role_injection * self.WEIGHTS['role_injection']
        score += pattern_features.system_manipulation * self.WEIGHTS['system_manipulation']
        score += pattern_features.prompt_leak * self.WEIGHTS['prompt_leak']
        score += pattern_features.jailbreak_keywords * self.WEIGHTS['jailbreak_keywords']
        score += pattern_features.encoding_markers * self.WEIGHTS['encoding_markers']
        score += pattern_features.suspicious_delimiters * self.WEIGHTS['suspicious_delimiters']

        # Normalize to roughly [0, 1] range
        max_possible = sum(self.WEIGHTS.values())
        return min(score / max_possible, 1.0)

    def predict(self, pattern_features: 'PatternFeatures') -> bool:
        """Predict whether input is prompt injection."""
        return self.predict_proba(pattern_features) >= self.threshold
