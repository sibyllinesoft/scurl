"""Response middleware implementations including ReadabilityExtractor."""

from .middleware import (
    ResponseMiddleware,
    ResponseContext,
    ResponseMiddlewareResult,
)


class ReadabilityExtractor(ResponseMiddleware):
    """Extract readable content from HTML and convert to markdown.

    Uses readability-lxml for content extraction and html2text for
    markdown conversion.
    """

    def __init__(
        self,
        include_links: bool = True,
        include_images: bool = True,
        include_tables: bool = True,
        body_width: int = 0,
        use_readability: bool = False,
    ):
        self._include_links = include_links
        self._include_images = include_images
        self._include_tables = include_tables
        self._body_width = body_width
        self._use_readability = use_readability

    @property
    def name(self) -> str:
        return "ReadabilityExtractor"

    @property
    def slug(self) -> str:
        return "readability"

    def should_process(self, context: ResponseContext) -> bool:
        """Only process HTML content."""
        content_type = context.content_type or ""
        return "text/html" in content_type.lower()

    def _extract_with_readability(self, html: str, url: str) -> str | None:
        """Extract content using readability + html2text."""
        try:
            from readability import Document
            import html2text

            # Extract main content with readability
            doc = Document(html, url=url)
            clean_html = doc.summary()

            if not clean_html or clean_html == "<html><body></body></html>":
                return None

            # Convert to markdown with html2text
            h = html2text.HTML2Text()
            h.ignore_links = not self._include_links
            h.ignore_images = not self._include_images
            h.ignore_tables = not self._include_tables
            h.body_width = self._body_width
            h.unicode_snob = True
            h.skip_internal_links = True

            # Get title and prepend if available
            title = doc.title()
            markdown = h.handle(clean_html)

            # Add title as h1 if not already in content
            if title and not markdown.strip().startswith("# "):
                markdown = f"# {title}\n\n{markdown}"

            return markdown.strip()

        except ImportError:
            return None
        except Exception:
            return None

    def _extract_with_html2text_direct(self, html: str) -> str | None:
        """Convert HTML directly to markdown with html2text.

        This preserves links but doesn't remove boilerplate.
        Used as fallback when readability can't extract content.
        """
        try:
            import html2text

            h = html2text.HTML2Text()
            h.ignore_links = not self._include_links
            h.ignore_images = not self._include_images
            h.ignore_tables = not self._include_tables
            h.body_width = self._body_width
            h.unicode_snob = True
            h.skip_internal_links = True
            h.ignore_emphasis = False

            result = h.handle(html).strip()
            return result if result else None

        except ImportError:
            return None
        except Exception:
            return None

    def process(self, context: ResponseContext) -> ResponseMiddlewareResult:
        """Extract markdown from HTML.

        By default uses html2text directly for full page content.
        With use_readability=True, tries readability article extraction first.
        """
        html = context.body.decode("utf-8", errors="replace")
        url = context.url or ""

        result = None
        if self._use_readability:
            # Try readability + html2text (best for article-like content)
            result = self._extract_with_readability(html, url)

        # Use html2text direct (full page, preserves all content)
        if not result:
            result = self._extract_with_html2text_direct(html)

        if result:
            return ResponseMiddlewareResult(
                body=result.encode("utf-8"),
                content_type="text/markdown",
            )

        return ResponseMiddlewareResult(body=context.body)


class JsonPrettifier(ResponseMiddleware):
    """Pretty-print JSON responses."""

    def __init__(self, indent: int = 2):
        self._indent = indent

    @property
    def name(self) -> str:
        return "JsonPrettifier"

    @property
    def slug(self) -> str:
        return "json-prettify"

    def should_process(self, context: ResponseContext) -> bool:
        """Only process JSON content."""
        content_type = context.content_type or ""
        return "application/json" in content_type.lower() or "text/json" in content_type.lower()

    def process(self, context: ResponseContext) -> ResponseMiddlewareResult:
        """Pretty-print JSON."""
        import json

        try:
            text = context.body.decode("utf-8", errors="replace")
            data = json.loads(text)
            pretty = json.dumps(data, indent=self._indent, ensure_ascii=False)
            return ResponseMiddlewareResult(
                body=pretty.encode("utf-8"),
                content_type=context.content_type,
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            return ResponseMiddlewareResult(body=context.body)
