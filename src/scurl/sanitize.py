"""Text sanitization for pasted content.

Strips HTML tags, normalizes whitespace, and converts to clean plaintext or markdown.
No external dependencies — uses only the standard library.
"""

import html
import re


def sanitize_text(text: str) -> str:
    """Sanitize pasted text/HTML into clean readable text.

    - Strips HTML tags if present
    - Decodes HTML entities
    - Normalizes whitespace
    - Preserves paragraph structure
    """
    if not text or not text.strip():
        return ""

    # Check if input looks like HTML
    if _looks_like_html(text):
        return _strip_html(text)

    # Plain text — just normalize whitespace
    return _normalize_whitespace(text)


def _looks_like_html(text: str) -> bool:
    """Heuristic: does this text contain HTML tags?"""
    return bool(re.search(r"<[a-zA-Z][^>]*>", text))


def _strip_html(html_text: str) -> str:
    """Strip HTML to clean text, preserving block structure."""
    text = html_text

    # Remove script and style blocks entirely
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Convert block elements to newlines
    block_tags = r"</?(p|div|br|hr|h[1-6]|li|tr|blockquote|pre|table|ul|ol|dl|dt|dd|section|article|header|footer|nav|main|aside|figure|figcaption)[^>]*>"
    text = re.sub(block_tags, "\n", text, flags=re.IGNORECASE)

    # Strip remaining tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode HTML entities
    text = html.unescape(text)

    return _normalize_whitespace(text)


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse runs of whitespace within lines (but not newlines)
    lines = text.split("\n")
    lines = [" ".join(line.split()) for line in lines]

    # Collapse 3+ blank lines to 2 (one blank line between paragraphs)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
