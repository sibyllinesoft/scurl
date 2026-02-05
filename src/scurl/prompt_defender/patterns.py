"""Pattern-based feature extraction for prompt injection detection."""

import re
from dataclasses import dataclass, fields
from typing import Dict, List

# Pattern categories for detecting different types of prompt injection
PATTERN_CATEGORIES: Dict[str, List[str]] = {
    'instruction_override': [
        r'ignore\s+(all\s+)?(previous|prior|above|earlier|preceding)\s+'
        r'(instructions?|prompts?|rules?|guidelines?|directions?|commands?)',
        r'disregard\s+(all\s+)?(previous|prior|earlier|above|preceding)',
        r'forget\s+(everything\s+)?(above|before|prior|previous|earlier)',
        r'do\s+not\s+follow\s+(the\s+)?(previous|above|prior|earlier)',
        r'override\s+(all\s+)?(previous|prior|earlier)',
        r'stop\s+following\s+(your\s+)?(previous|original|initial)',
        r'new\s+instructions?\s*[:=]',
        r'actual\s+instructions?\s*[:=]',
        r'real\s+instructions?\s*[:=]',
        r'updated?\s+instructions?\s*[:=]',
    ],
    'role_injection': [
        r'you\s+are\s+now\s+',
        r'from\s+now\s+on\s*[,:]?\s*(you|your)',
        r'act\s+as\s+(if\s+)?(you\s+)?(are\s+|were\s+)?',
        r'pretend\s+(to\s+be|you\s+are|you\'re|that\s+you)',
        r'your\s+new\s+(role|goal|purpose|instruction|directive|objective)',
        r'imagine\s+(that\s+)?you\s+(are|were)',
        r'roleplay\s+as',
        r'switch\s+to\s+.{0,30}\s+mode',
        r'you\s+must\s+now\s+(act|be|become)',
        r'for\s+the\s+rest\s+of\s+this\s+(conversation|session|chat)',
        r'behave\s+(like|as)\s+(a|an)',
        r'assume\s+the\s+(role|identity|persona)',
        r'you\s+will\s+(now\s+)?(respond|act|behave)',
    ],
    'system_manipulation': [
        r'(admin|administrator|developer|god|sudo|root|maintenance|debug)\s+mode',
        r'system\s+(override|prompt|instruction|message|command)',
        r'unlock\s+(all\s+)?(restrictions?|capabilities?|features?|access)',
        r'disable\s+(all\s+)?(safety|security|content\s+)?(filters?|guards?|restrictions?|limits?)',
        r'bypass\s+(all\s+)?(restrictions?|filters?|safety|security|limits?)',
        r'enable\s+(unrestricted|unlimited|full)\s+(mode|access)',
        r'remove\s+(all\s+)?(limitations?|restrictions?|filters?)',
        r'without\s+(any\s+)?(restrictions?|limitations?|filters?)',
        r'turn\s+off\s+(safety|security|content)?\s*(filters?|checks?|restrictions?)',
        r'deactivate\s+(safety|security|content)\s+(filters?|checks?)',
        r'override\s+safety',
        r'ignore\s+(safety|content)\s+(guidelines?|policies?|filters?)',
    ],
    'prompt_leak': [
        r'reveal\s+(your\s+)?(system\s+)?(prompt|instructions?|directives?)',
        r'show\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instructions?|rules?|directives?)',
        r'what\s+(are|is|were)\s+(your\s+)?(system\s+)?(prompt|instructions?|rules?|directives?)',
        r'(print|display|output|echo|write|repeat)\s+(your\s+)?(system\s+)?'
        r'(prompt|instructions?|directives?)',
        r'tell\s+me\s+(your|the)\s+(system\s+)?(prompt|instructions?)',
        r'(initial|original|hidden|secret|base)\s+(prompt|instructions?|directives?)',
        r'what\s+were\s+you\s+(told|instructed|programmed)',
        r'(copy|paste|quote)\s+(your\s+)?(system\s+)?(prompt|instructions?)',
        r'(beginning|start)\s+of\s+(your\s+)?(prompt|instructions?)',
    ],
    'jailbreak_keywords': [
        r'\bDAN\b',  # "Do Anything Now"
        r'\bjailbreak(ed|ing)?\b',
        r'do\s+anything\s+now',
        r'(evil|dark|shadow|unrestricted|unfiltered)\s+(mode|assistant|ai|version)',
        r'(no|without|ignore)\s+(ethical|moral)\s+(guidelines?|restrictions?|limits?)',
        r'hypothetically\s+(speaking|if)',
        r'for\s+(educational|research|academic)\s+purposes?\s+only',
        r'in\s+a\s+fictional\s+(scenario|world|context)',
        r'(opposite|anti)\s+day',
        r'chaos\s+mode',
        r'maximum\s+freedom',
        r'no\s+censorship',
        r'uncensored\s+(mode|response|version)',
        r'(bypass|skip|avoid)\s+(all\s+)?safeguards?',
    ],
    'encoding_markers': [
        r'base64\s*[:=]',
        r'decode\s+(this|the\s+following|below)',
        r'encoded\s+(message|instruction|prompt)',
        r'\\x[0-9a-fA-F]{2}',  # Hex escapes
        r'&#x?[0-9a-fA-F]+;',  # HTML entities
        r'%[0-9a-fA-F]{2}',    # URL encoding
        r'\\u[0-9a-fA-F]{4}',  # Unicode escapes
        r'rot13',
        r'caesar\s+cipher',
    ],
    'suspicious_delimiters': [
        r'\[\s*system\s*\]',
        r'\[\s*instruction[s]?\s*\]',
        r'\[\s*admin\s*\]',
        r'\[\s*assistant\s*\]',
        r'\[\s*user\s*\]',
        r'<\|?\s*(system|instruction|user|assistant|im_start|im_end)\s*\|?>',
        r'###\s*(system|instruction|new\s+task)',
        r'\*\*\*\s*(override|system|admin)',
        r'={3,}\s*(system|instruction|override)',
        r'```\s*(system|instruction|override)',
        r'---\s*(system|instruction|begin)',
    ],
}


@dataclass
class PatternFeatures:
    """Features extracted from pattern matching.

    All pattern features are normalized by text length (matches per 1000 chars),
    capped at 1.0 to prevent outliers from dominating.

    Text statistics are also normalized to [0, 1] range where possible.
    """
    # Pattern match densities (matches per 1000 chars, capped at 1.0)
    instruction_override: float
    role_injection: float
    system_manipulation: float
    prompt_leak: float
    jailbreak_keywords: float
    encoding_markers: float
    suspicious_delimiters: float

    # Text statistics
    text_length: float        # Length / 10000, capped at 1.0
    special_char_ratio: float # Special chars / total chars
    caps_ratio: float         # Uppercase / alphabetic chars
    newline_density: float    # Newlines / total chars
    avg_word_length: float    # Average word length / 20, capped at 1.0

    def to_array(self) -> List[float]:
        """Convert to list of floats for classifier input."""
        return [getattr(self, f.name) for f in fields(self)]

    @classmethod
    def feature_names(cls) -> List[str]:
        """Return list of feature names in order."""
        return [f.name for f in fields(cls)]


class PatternExtractor:
    """Extract pattern-based features from normalized text."""

    def __init__(self):
        """Initialize with compiled regex patterns."""
        self._compiled: Dict[str, List[re.Pattern]] = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in PATTERN_CATEGORIES.items()
        }

    def extract(self, normalized_text: str) -> PatternFeatures:
        """Extract all features from normalized text.

        Args:
            normalized_text: Text that has been passed through TextNormalizer.

        Returns:
            PatternFeatures dataclass with all extracted features.
        """
        text_len = max(len(normalized_text), 1)  # Avoid division by zero

        def count_matches(category: str) -> float:
            """Count pattern matches, normalized by text length."""
            total = sum(
                len(pattern.findall(normalized_text))
                for pattern in self._compiled[category]
            )
            # Normalize: matches per 1000 chars, capped at 1.0
            return min(total * 1000 / text_len, 1.0)

        # Text statistics
        words = normalized_text.split()
        special_chars = sum(
            1 for c in normalized_text
            if not c.isalnum() and not c.isspace()
        )
        caps = sum(1 for c in normalized_text if c.isupper())
        alpha_chars = sum(1 for c in normalized_text if c.isalpha())
        newlines = normalized_text.count('\n')

        return PatternFeatures(
            # Pattern densities
            instruction_override=count_matches('instruction_override'),
            role_injection=count_matches('role_injection'),
            system_manipulation=count_matches('system_manipulation'),
            prompt_leak=count_matches('prompt_leak'),
            jailbreak_keywords=count_matches('jailbreak_keywords'),
            encoding_markers=count_matches('encoding_markers'),
            suspicious_delimiters=count_matches('suspicious_delimiters'),
            # Text statistics
            text_length=min(text_len / 10000, 1.0),
            special_char_ratio=special_chars / text_len,
            caps_ratio=caps / alpha_chars if alpha_chars > 0 else 0.0,
            newline_density=newlines / text_len,
            avg_word_length=min(
                (sum(len(w) for w in words) / len(words) / 20) if words else 0.0,
                1.0
            ),
        )

    def has_any_match(self, normalized_text: str) -> bool:
        """Quick check if any pattern matches.

        Useful for fast-path rejection of obviously clean text.
        """
        for patterns in self._compiled.values():
            for pattern in patterns:
                if pattern.search(normalized_text):
                    return True
        return False

    def get_matches(self, normalized_text: str) -> Dict[str, List[str]]:
        """Get all pattern matches by category.

        Useful for debugging and explaining detections.

        Returns:
            Dict mapping category names to lists of matched strings.
        """
        matches: Dict[str, List[str]] = {}
        for category, patterns in self._compiled.items():
            category_matches = []
            for pattern in patterns:
                category_matches.extend(pattern.findall(normalized_text))
            if category_matches:
                matches[category] = category_matches
        return matches
