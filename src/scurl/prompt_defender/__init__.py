"""Prompt injection detection middleware for scurl."""

from .middleware import PromptInjectionDefender
from .normalizer import TextNormalizer
from .patterns import PatternExtractor, PatternFeatures
from .motifs import MotifMatcher, MotifFeatureExtractor, MotifSignal, HAS_RAPIDFUZZ
from .windowing import SlidingWindowAnalyzer, AdaptiveWindowAnalyzer

__all__ = [
    "PromptInjectionDefender",
    "TextNormalizer",
    "PatternExtractor",
    "PatternFeatures",
    "MotifMatcher",
    "MotifFeatureExtractor",
    "MotifSignal",
    "SlidingWindowAnalyzer",
    "AdaptiveWindowAnalyzer",
    "HAS_RAPIDFUZZ",
]
