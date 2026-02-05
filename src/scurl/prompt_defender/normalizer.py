"""Text normalization to defeat obfuscation techniques."""

import re
import unicodedata
from typing import Optional

# Zero-width and invisible characters that can be used to hide content
ZERO_WIDTH_CHARS = frozenset([
    '\u200b',  # Zero-width space
    '\u200c',  # Zero-width non-joiner
    '\u200d',  # Zero-width joiner
    '\u2060',  # Word joiner
    '\u2061',  # Function application
    '\u2062',  # Invisible times
    '\u2063',  # Invisible separator
    '\u2064',  # Invisible plus
    '\ufeff',  # Byte order mark / zero-width no-break space
    '\u00ad',  # Soft hyphen
    '\u034f',  # Combining grapheme joiner
    '\u061c',  # Arabic letter mark
    '\u115f',  # Hangul choseong filler
    '\u1160',  # Hangul jungseong filler
    '\u17b4',  # Khmer vowel inherent aq
    '\u17b5',  # Khmer vowel inherent aa
    '\u180e',  # Mongolian vowel separator
    '\u3164',  # Hangul filler
    '\uffa0',  # Halfwidth hangul filler
])

# Bidirectional override characters that can manipulate text display
BIDI_CHARS = frozenset([
    '\u202a',  # Left-to-right embedding
    '\u202b',  # Right-to-left embedding
    '\u202c',  # Pop directional formatting
    '\u202d',  # Left-to-right override
    '\u202e',  # Right-to-left override
    '\u2066',  # Left-to-right isolate
    '\u2067',  # Right-to-left isolate
    '\u2068',  # First strong isolate
    '\u2069',  # Pop directional isolate
])


class TextNormalizer:
    """Normalize text to defeat obfuscation techniques.

    Applies multiple normalization passes:
    1. Unicode NFKC normalization (handles fullwidth chars, compatibility forms)
    2. Zero-width character stripping
    3. Bidirectional override removal
    4. Confusable/homoglyph normalization (optional, requires confusable-homoglyphs)
    5. Whitespace normalization
    """

    def __init__(self, use_confusables: bool = True):
        """Initialize normalizer.

        Args:
            use_confusables: Whether to normalize confusable characters.
                            Requires confusable-homoglyphs package.
        """
        self._use_confusables = use_confusables
        self._confusables_available: Optional[bool] = None

    def normalize(self, text: str) -> str:
        """Apply full normalization pipeline.

        Args:
            text: Input text to normalize.

        Returns:
            Normalized lowercase text with obfuscation removed.
        """
        text = self._normalize_unicode(text)
        text = self._strip_invisible(text)
        text = self._normalize_confusables(text)
        text = self._normalize_whitespace(text)
        return text.lower()

    def _normalize_unicode(self, text: str) -> str:
        """Apply NFKC normalization.

        Converts:
        - Fullwidth characters: ｆｕｌｌｗｉｄｔｈ → fullwidth
        - Compatibility characters: ﬁ → fi
        - Composed forms: é (decomposed) → é (composed)
        """
        return unicodedata.normalize('NFKC', text)

    def _strip_invisible(self, text: str) -> str:
        """Remove zero-width and bidirectional override characters."""
        chars_to_remove = ZERO_WIDTH_CHARS | BIDI_CHARS
        return ''.join(c for c in text if c not in chars_to_remove)

    def _normalize_confusables(self, text: str) -> str:
        """Convert confusable/homoglyph characters to ASCII equivalents.

        Examples:
        - Cyrillic а → Latin a
        - Greek ο → Latin o
        - ι → i (Greek iota to Latin i)
        """
        if not self._use_confusables:
            return text

        # Lazy check for confusables library
        if self._confusables_available is None:
            try:
                from confusable_homoglyphs import confusables
                self._confusables_available = True
            except ImportError:
                self._confusables_available = False

        if not self._confusables_available:
            return text

        from confusable_homoglyphs import confusables

        result = []
        for char in text:
            # Check if character is confusable with Latin
            conf = confusables.is_confusable(char, preferred_aliases=['latin', 'common'])
            if conf:
                # Get the Latin equivalent
                for item in conf:
                    if item.get('character'):
                        result.append(item['character'])
                        break
                else:
                    result.append(char)
            else:
                result.append(char)

        return ''.join(result)

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace to single space and strip."""
        return re.sub(r'\s+', ' ', text).strip()


def normalize_for_matching(text: str) -> str:
    """Convenience function for quick normalization.

    Creates a TextNormalizer instance and normalizes the text.
    For repeated use, prefer creating a TextNormalizer instance.
    """
    return TextNormalizer().normalize(text)
