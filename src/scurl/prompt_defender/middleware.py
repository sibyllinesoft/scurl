"""Prompt injection detection middleware for scurl."""

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import json

from ..middleware import ResponseMiddleware, ResponseContext, ResponseMiddlewareResult
from .normalizer import TextNormalizer
from .patterns import PatternExtractor, PATTERN_CATEGORIES
from .classifier import InjectionClassifier, PatternOnlyClassifier
from .motifs import MotifMatcher, MotifFeatureExtractor
from .windowing import AdaptiveWindowAnalyzer


# Unicode block character for redaction
REDACT_CHAR = '█'


@dataclass
class InjectionAnalysis:
    """Results of prompt injection analysis."""

    score: float
    """Probability of prompt injection (0.0 to 1.0)."""

    threshold: float
    """Detection threshold used."""

    flagged: bool
    """Whether the score exceeds the threshold."""

    pattern_features: Dict[str, float]
    """Individual pattern category scores."""

    matched_patterns: Dict[str, List[str]]
    """Actual pattern matches by category (for debugging)."""

    matched_spans: List[Tuple[int, int]]
    """Character spans (start, end) of all matched patterns in original text."""

    action_taken: str
    """Action taken: 'none', 'warn', 'redact', 'datamark', 'metadata', or 'silent'."""


class PromptInjectionDefender(ResponseMiddleware):
    """Detect and flag prompt injection attempts in web content.

    This middleware analyzes text content for potential prompt injection
    attacks using a combination of:
    1. Text normalization to defeat obfuscation
    2. Pattern-based feature extraction
    3. Semantic embeddings (optional, requires model)
    4. ML classifier for final prediction

    Can operate in pattern-only mode if embeddings/classifier aren't available.
    """

    # Valid action modes
    ACTIONS = {'warn', 'redact', 'datamark', 'metadata', 'silent'}

    def __init__(
        self,
        threshold: float = 0.3,
        action: str = "warn",
        use_embeddings: bool = True,
        use_windowing: bool = True,
        lazy_load: bool = True,
    ):
        """Initialize the defender.

        Args:
            threshold: Detection threshold (0.0 to 1.0). Higher = fewer false positives.
            action: Action when injection detected:
                   - "warn": Wrap in XML tag, content unchanged
                   - "redact": Wrap in XML tag, mask matched patterns with █
                   - "datamark": Wrap in XML tag, insert ^ between words (spotlighting)
                   - "metadata": Return JSON with full analysis
                   - "silent": Pass through unchanged
            use_embeddings: Whether to use embedding-based detection.
                           Falls back to pattern-only if False or unavailable.
            use_windowing: Whether to use sliding window analysis for long texts.
                          Provides more precise hotspot detection.
            lazy_load: Whether to lazy-load heavy components (embedder, classifier).
        """
        self.threshold = threshold
        self.action = action if action in self.ACTIONS else "warn"
        self._use_embeddings = use_embeddings
        self._use_windowing = use_windowing

        # Always-available components
        self._normalizer = TextNormalizer()
        self._pattern_extractor = PatternExtractor()
        self._pattern_classifier = PatternOnlyClassifier(threshold=0.3)

        # Motif-based detection
        self._motif_matcher = MotifMatcher(threshold=75)
        self._motif_extractor = MotifFeatureExtractor(threshold=75)

        # Windowing analyzer for long texts
        self._window_analyzer = AdaptiveWindowAnalyzer(
            coarse_window=4096,
            coarse_step=2048,
            fine_window=512,
            fine_step=256,
            hotspot_threshold=0.3,
            batch_max_size=1024,
            batch_gap_tolerance=256,
        )

        # Compiled patterns for span detection (case-insensitive on original text)
        self._span_patterns = self._compile_span_patterns()

        # Heavy components (lazy-loaded)
        self._embedder = None
        self._classifier = None
        self._load_failed = False

        if not lazy_load:
            self._ensure_heavy_components()

    def _compile_span_patterns(self) -> List[re.Pattern]:
        """Compile all patterns for span detection."""
        patterns = []
        for category_patterns in PATTERN_CATEGORIES.values():
            for pattern_str in category_patterns:
                try:
                    patterns.append(re.compile(pattern_str, re.IGNORECASE))
                except re.error:
                    pass  # Skip invalid patterns
        return patterns

    @property
    def name(self) -> str:
        return "PromptInjectionDefender"

    @property
    def slug(self) -> str:
        return "prompt-defender"

    def should_process(self, context: ResponseContext) -> bool:
        """Process text content types."""
        content_type = context.content_type or ""
        return any(
            t in content_type.lower()
            for t in ["text/markdown", "text/plain", "text/html"]
        )

    def _ensure_heavy_components(self) -> bool:
        """Try to load embedder and classifier. Returns True if successful."""
        if self._load_failed:
            return False

        if self._embedder is not None and self._classifier is not None:
            return True

        try:
            from .embedder import EmbeddingGemmaONNX
            self._embedder = EmbeddingGemmaONNX()
            self._classifier = InjectionClassifier()
            return True
        except (ImportError, RuntimeError):
            # Missing dependencies or model - fall back to pattern-only
            self._load_failed = True
            return False

    def _find_pattern_spans(self, text: str) -> List[Tuple[int, int]]:
        """Find all pattern match spans in original text."""
        spans: Set[Tuple[int, int]] = set()

        # Regex-based pattern matches
        for pattern in self._span_patterns:
            for match in pattern.finditer(text):
                spans.add((match.start(), match.end()))

        # Motif-based matches (fuzzy)
        motif_spans = self._motif_matcher.get_match_positions(text)
        for start, end in motif_spans:
            # Clamp to text bounds
            spans.add((max(0, start), min(len(text), end)))

        # Merge overlapping spans
        if not spans:
            return []

        sorted_spans = sorted(spans)
        merged = [sorted_spans[0]]

        for start, end in sorted_spans[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                # Overlapping - extend
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        return merged

    def analyze(self, text: str, include_matches: bool = False) -> InjectionAnalysis:
        """Analyze text for prompt injection.

        Args:
            text: Text content to analyze.
            include_matches: Whether to include actual pattern matches
                            (useful for debugging but slower).

        Returns:
            InjectionAnalysis with detection results.
        """
        import numpy as np  # Lazy import - optional dependency

        # Normalize text to defeat obfuscation
        normalized = self._normalizer.normalize(text)

        # Extract pattern features
        pattern_features = self._pattern_extractor.extract(normalized)

        # Extract motif features
        motif_features = self._motif_extractor.extract(normalized)

        # Early exit: if no suspicious patterns detected, skip expensive embedding
        # This provides ~100x speedup for benign content
        pattern_score = self._pattern_classifier.predict_proba(pattern_features)
        motif_signal = self._motif_matcher.compute_signal(normalized)
        has_suspicious_patterns = (
            pattern_features.instruction_override > 0 or
            pattern_features.role_injection > 0 or
            pattern_features.system_manipulation > 0 or
            pattern_features.prompt_leak > 0 or
            pattern_features.jailbreak_keywords > 0 or
            pattern_features.encoding_markers > 0 or
            pattern_features.suspicious_delimiters > 0 or
            motif_signal.density > 0
        )

        if not has_suspicious_patterns and pattern_score < 0.1:
            # Fast path: no suspicious signals, return low score without embedding
            return InjectionAnalysis(
                score=pattern_score,
                threshold=self.threshold,
                flagged=False,
                pattern_features={
                    'instruction_override': pattern_features.instruction_override,
                    'role_injection': pattern_features.role_injection,
                    'system_manipulation': pattern_features.system_manipulation,
                    'prompt_leak': pattern_features.prompt_leak,
                    'jailbreak_keywords': pattern_features.jailbreak_keywords,
                    'encoding_markers': pattern_features.encoding_markers,
                    'suspicious_delimiters': pattern_features.suspicious_delimiters,
                },
                matched_patterns={},
                matched_spans=[],
                action_taken="none",
            )

        # For long texts, use windowing to find hotspots
        hotspot_regions = []
        if self._use_windowing and len(text) > 4096:
            analysis_result = self._window_analyzer.analyze(normalized)
            hotspot_regions = analysis_result.batched_regions

        # Try to use full ML pipeline
        if self._use_embeddings and self._ensure_heavy_components():
            # Smart embedding: only embed windows around detected patterns
            pattern_spans = self._find_pattern_spans(normalized) if has_suspicious_patterns else []

            if pattern_spans and len(normalized) > 1024:
                # Build windows around detected patterns (512 chars each, merge if close)
                WINDOW_SIZE = 512
                MERGE_GAP = 256
                windows = []

                for start, end in pattern_spans:
                    # Center window on the pattern
                    center = (start + end) // 2
                    win_start = max(0, center - WINDOW_SIZE // 2)
                    win_end = min(len(normalized), center + WINDOW_SIZE // 2)

                    # Merge with previous window if close
                    if windows and win_start - windows[-1][1] < MERGE_GAP:
                        windows[-1] = (windows[-1][0], win_end)
                    else:
                        windows.append((win_start, win_end))

                # Embed only the pattern windows
                region_embeddings = []
                for start, end in windows:
                    region_text = normalized[start:end]
                    if len(region_text) > 20:  # Skip tiny windows
                        region_embeddings.append(self._embedder.embed(region_text))

                # Max-pool across windows
                if region_embeddings:
                    embedding = np.max(region_embeddings, axis=0)
                else:
                    embedding = self._embedder.embed(normalized[:1024])
            elif hotspot_regions:
                # Long text hotspots from windowing analysis
                region_embeddings = []
                for start, end in hotspot_regions:
                    region_text = normalized[start:end]
                    region_embeddings.append(self._embedder.embed(region_text))

                # Max-pool across regions
                if region_embeddings:
                    embedding = np.max(region_embeddings, axis=0)
                else:
                    embedding = self._embedder.embed(normalized[:4096])
            else:
                # Short text or no patterns: embed full text (but limit size)
                embedding = self._embedder.embed(normalized[:2048])

            # Combine features: pattern + motif + embedding
            pattern_array = np.array(pattern_features.to_array(), dtype=np.float32)
            motif_array = np.array(motif_features, dtype=np.float32)
            features = np.concatenate([pattern_array, motif_array, embedding])

            # Classify
            score = self._classifier.predict_proba(features)
        else:
            # Pattern-only fallback (includes motif features via density)
            motif_signal = self._motif_matcher.compute_signal(normalized)
            motif_boost = min(motif_signal.density / 10, 0.3)  # Up to 0.3 boost
            base_score = self._pattern_classifier.predict_proba(pattern_features)
            score = min(base_score + motif_boost, 1.0)

        # Determine if flagged
        flagged = score >= self.threshold

        # Get pattern matches if requested
        matched_patterns = {}
        if include_matches:
            matched_patterns = self._pattern_extractor.get_matches(normalized)

        # Find spans for redaction (on original text, case-insensitive)
        matched_spans = self._find_pattern_spans(text) if flagged else []

        # Determine action
        action_taken = self.action if flagged else "none"

        return InjectionAnalysis(
            score=score,
            threshold=self.threshold,
            flagged=flagged,
            pattern_features={
                'instruction_override': pattern_features.instruction_override,
                'role_injection': pattern_features.role_injection,
                'system_manipulation': pattern_features.system_manipulation,
                'prompt_leak': pattern_features.prompt_leak,
                'jailbreak_keywords': pattern_features.jailbreak_keywords,
                'encoding_markers': pattern_features.encoding_markers,
                'suspicious_delimiters': pattern_features.suspicious_delimiters,
            },
            matched_patterns=matched_patterns,
            matched_spans=matched_spans,
            action_taken=action_taken,
        )

    def _redact_spans(self, text: str, spans: List[Tuple[int, int]]) -> str:
        """Replace matched spans with redaction characters."""
        if not spans:
            return text

        result = []
        last_end = 0

        for start, end in spans:
            # Add text before this span
            result.append(text[last_end:start])
            # Add redaction (same length as original)
            result.append(REDACT_CHAR * (end - start))
            last_end = end

        # Add remaining text
        result.append(text[last_end:])

        return ''.join(result)

    def _datamark_text(self, text: str) -> str:
        """Apply datamarking (spotlighting) - insert ^ between words."""
        # Use Unicode Private Use Area character as recommended by Microsoft
        # U+E000 is guaranteed not to appear in normal text
        marker = '\ue000'
        # Replace whitespace sequences with the marker
        return re.sub(r'\s+', marker, text)

    def _format_signal_types(self, features: Dict[str, float]) -> str:
        """Format active signal types as comma-separated short codes."""
        type_map = {
            'instruction_override': 'override',
            'role_injection': 'role',
            'system_manipulation': 'system',
            'prompt_leak': 'leak',
            'jailbreak_keywords': 'jailbreak',
            'encoding_markers': 'encoding',
            'suspicious_delimiters': 'delimiters',
        }
        active = [
            type_map[k] for k, v in features.items()
            if v > 0 and k in type_map
        ]
        return ','.join(active) if active else 'semantic'

    def process(self, context: ResponseContext) -> ResponseMiddlewareResult:
        """Process response, detecting and flagging injections."""
        text = context.body.decode('utf-8', errors='replace')

        # Analyze for injection
        analysis = self.analyze(text, include_matches=(self.action == "metadata"))

        # If not flagged, pass through unchanged
        if not analysis.flagged:
            return ResponseMiddlewareResult(body=context.body)

        # Handle based on action mode
        if self.action == "silent":
            return ResponseMiddlewareResult(body=context.body)

        if self.action == "metadata":
            output = {
                "content": text,
                "injection_analysis": {
                    "score": round(analysis.score, 4),
                    "threshold": analysis.threshold,
                    "flagged": analysis.flagged,
                    "pattern_features": {
                        k: round(v, 4) for k, v in analysis.pattern_features.items()
                    },
                    "matched_patterns": analysis.matched_patterns,
                    "matched_spans": analysis.matched_spans,
                    "mode": "ml" if self._embedder else "pattern-only",
                },
            }
            return ResponseMiddlewareResult(
                body=json.dumps(output, indent=2).encode('utf-8'),
                content_type="application/json",
            )

        # Format signal types for XML attribute
        signal_types = self._format_signal_types(analysis.pattern_features)

        # Process content based on action mode
        if self.action == "redact":
            processed_text = self._redact_spans(text, analysis.matched_spans)
        elif self.action == "datamark":
            processed_text = self._datamark_text(text)
        else:  # warn
            processed_text = text

        # Wrap in XML tag
        score_str = f"{analysis.score:.2f}"
        body = f'<suspected-prompt-injection p="{score_str}" t="{signal_types}">\n{processed_text}\n</suspected-prompt-injection>'

        return ResponseMiddlewareResult(body=body.encode('utf-8'))

    @property
    def mode(self) -> str:
        """Return current detection mode."""
        if self._embedder is not None:
            return "ml"
        return "pattern-only"
