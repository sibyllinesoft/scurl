"""Motif-based fuzzy pattern matching for prompt injection detection.

Uses short, fuzzy "motifs" (like DNA sequence analysis) that capture
attack patterns even with obfuscation like leetspeak, typos, or inserted
characters.

Motifs are intentionally shorter and looser than full regex patterns,
designed to be combined as features for ML classification rather than
used as direct detection rules.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import re

# Try to import rapidfuzz for fuzzy matching
try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


# Motif categories - shorter, fuzzier fragments than full patterns
# These are designed to catch partial matches and obfuscated variants
MOTIF_LIBRARY: Dict[str, List[str]] = {
    'instruction_override': [
        'ignore previous',
        'ignore all',
        'disregard above',
        'forget everything',
        'new instructions',
        'actual instructions',
        'real instructions',
        'override previous',
        'stop following',
        'do not follow',
    ],
    'role_injection': [
        'you are now',
        'from now on',
        'act as',
        'pretend to be',
        'your new role',
        'imagine you are',
        'roleplay as',
        'switch to mode',
        'behave like',
        'assume the role',
    ],
    'system_manipulation': [
        'admin mode',
        'developer mode',
        'god mode',
        'sudo mode',
        'debug mode',
        'system override',
        'unlock restrictions',
        'disable safety',
        'bypass filters',
        'enable unrestricted',
        'remove limitations',
        'turn off filters',
    ],
    'prompt_leak': [
        'reveal your prompt',
        'show me your instructions',
        'what are your rules',
        'print your prompt',
        'tell me your instructions',
        'initial prompt',
        'hidden instructions',
        'secret prompt',
        'beginning of prompt',
    ],
    'jailbreak': [
        'do anything now',
        'jailbreak',
        'evil mode',
        'unrestricted mode',
        'unfiltered mode',
        'no ethical guidelines',
        'hypothetically speaking',
        'educational purposes only',
        'fictional scenario',
        'chaos mode',
        'no censorship',
        'bypass safeguards',
    ],
    'delimiters': [
        '[system]',
        '[instruction]',
        '[admin]',
        '[assistant]',
        '<|system|>',
        '<|im_start|>',
        '### system',
        '*** override',
        '=== system',
        '``` system',
    ],
}


@dataclass
class MotifMatch:
    """A fuzzy motif match in text."""
    motif: str
    category: str
    position: int  # Start position in text (of the matched substring)
    length: int    # Length of matched region in text
    score: float   # Fuzzy match score (0-100)


@dataclass
class MotifSignal:
    """Aggregated motif signals for a text region."""
    matches: List[MotifMatch]
    density: float  # Matches per 1000 chars
    category_scores: Dict[str, float]  # Max score per category

    @property
    def total_score(self) -> float:
        """Sum of all category max scores."""
        return sum(self.category_scores.values())


class MotifMatcher:
    """Fuzzy motif matching using RapidFuzz.

    Falls back to simple substring matching if RapidFuzz unavailable.
    """

    def __init__(
        self,
        threshold: int = 75,
        motif_library: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize matcher.

        Args:
            threshold: Minimum fuzzy match score (0-100). Lower = more matches.
            motif_library: Custom motif dictionary, or use defaults.
        """
        self.threshold = threshold
        self.motifs = motif_library or MOTIF_LIBRARY

        # Build flat list for efficient matching
        self._flat_motifs: List[Tuple[str, str]] = []  # (motif, category)
        for category, motif_list in self.motifs.items():
            for motif in motif_list:
                self._flat_motifs.append((motif.lower(), category))

    def find_matches(
        self,
        text: str,
        window_size: int = 50,
        step: int = 25,
    ) -> List[MotifMatch]:
        """Find all motif matches in text using sliding window.

        Args:
            text: Text to search.
            window_size: Size of sliding window for fuzzy matching.
            step: Step size between windows.

        Returns:
            List of MotifMatch objects sorted by position.
        """
        if not text:
            return []

        text_lower = text.lower()
        matches: List[MotifMatch] = []
        seen_positions: Set[Tuple[int, str]] = set()  # Dedupe by (pos, motif)

        # Slide window across text
        for pos in range(0, len(text_lower), step):
            window = text_lower[pos:pos + window_size]
            if len(window) < 10:  # Skip tiny windows
                continue

            # Check each motif against this window
            for motif, category in self._flat_motifs:
                score = self._match_score(window, motif)

                if score >= self.threshold:
                    # Find the precise location of the best match within the window
                    match_start, match_len = self._locate_match(window, motif)
                    abs_start = pos + match_start
                    key = (abs_start, motif)
                    if key not in seen_positions:
                        seen_positions.add(key)
                        matches.append(MotifMatch(
                            motif=motif,
                            category=category,
                            position=abs_start,
                            length=match_len,
                            score=score,
                        ))

        # Sort by position
        matches.sort(key=lambda m: m.position)
        return matches

    def _match_score(self, text: str, motif: str) -> float:
        """Calculate fuzzy match score between text and motif."""
        if HAS_RAPIDFUZZ:
            # Use partial_ratio for substring-like matching
            return fuzz.partial_ratio(text, motif)
        else:
            # Fallback: simple substring check
            return 100.0 if motif in text else 0.0

    def _locate_match(self, window: str, motif: str) -> Tuple[int, int]:
        """Find the precise start offset and length of a motif match within a window.

        Returns:
            (offset_within_window, match_length) tuple.
        """
        # Exact substring match first
        idx = window.find(motif)
        if idx >= 0:
            return (idx, len(motif))

        if HAS_RAPIDFUZZ:
            # Use partial_ratio_alignment to find the best matching substring
            try:
                alignment = fuzz.partial_ratio_alignment(window, motif)
                if alignment is not None:
                    # alignment gives (src_start, src_end, dest_start, dest_end)
                    # src refers to the first argument (window)
                    return (alignment.src_start, alignment.src_end - alignment.src_start)
            except Exception:
                pass

        # Fallback: center a motif-length region in the window
        center = len(window) // 2
        half = len(motif) // 2
        start = max(0, center - half)
        return (start, len(motif))

    def compute_signal(
        self,
        text: str,
        window_size: int = 50,
        step: int = 25,
    ) -> MotifSignal:
        """Compute aggregated motif signal for text.

        Args:
            text: Text to analyze.
            window_size: Sliding window size for matching.
            step: Step between windows.

        Returns:
            MotifSignal with aggregated scores and density.
        """
        matches = self.find_matches(text, window_size, step)

        # Calculate density (matches per 1000 chars)
        text_len = max(len(text), 1)
        density = len(matches) * 1000 / text_len

        # Max score per category
        category_scores: Dict[str, float] = {}
        for match in matches:
            current = category_scores.get(match.category, 0)
            category_scores[match.category] = max(current, match.score)

        return MotifSignal(
            matches=matches,
            density=density,
            category_scores=category_scores,
        )

    def get_match_positions(self, text: str) -> List[Tuple[int, int]]:
        """Get all matched span positions for highlighting/redaction.

        Returns:
            List of (start, end) tuples tightly covering matched regions.
        """
        matches = self.find_matches(text)

        if not matches:
            return []

        # Use the precise match boundaries
        spans = [(m.position, m.position + m.length) for m in matches]
        spans.sort()

        # Merge only truly overlapping spans (no gap tolerance)
        merged = [spans[0]]
        for start, end in spans[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        return merged


class MotifFeatureExtractor:
    """Extract motif-based features for ML classification.

    Produces a fixed-size feature vector from motif matching results,
    suitable for input to a classifier alongside embeddings.
    """

    FEATURE_NAMES = [
        'motif_density',
        'motif_instruction_override',
        'motif_role_injection',
        'motif_system_manipulation',
        'motif_prompt_leak',
        'motif_jailbreak',
        'motif_delimiters',
        'motif_max_score',
        'motif_category_count',
    ]

    def __init__(self, threshold: int = 75):
        self.matcher = MotifMatcher(threshold=threshold)

    def extract(self, text: str) -> List[float]:
        """Extract motif features from text.

        Returns:
            List of 9 normalized features (all 0-1 range).
        """
        signal = self.matcher.compute_signal(text)

        # Normalize density (cap at 1.0)
        density = min(signal.density / 10, 1.0)

        # Category scores (already 0-100, normalize to 0-1)
        cat_scores = {
            'instruction_override': signal.category_scores.get('instruction_override', 0) / 100,
            'role_injection': signal.category_scores.get('role_injection', 0) / 100,
            'system_manipulation': signal.category_scores.get('system_manipulation', 0) / 100,
            'prompt_leak': signal.category_scores.get('prompt_leak', 0) / 100,
            'jailbreak': signal.category_scores.get('jailbreak', 0) / 100,
            'delimiters': signal.category_scores.get('delimiters', 0) / 100,
        }

        # Max score across all categories
        max_score = max(cat_scores.values()) if cat_scores else 0

        # Number of categories with matches (0-6, normalize to 0-1)
        category_count = sum(1 for v in cat_scores.values() if v > 0) / 6

        return [
            density,
            cat_scores['instruction_override'],
            cat_scores['role_injection'],
            cat_scores['system_manipulation'],
            cat_scores['prompt_leak'],
            cat_scores['jailbreak'],
            cat_scores['delimiters'],
            max_score,
            category_count,
        ]
