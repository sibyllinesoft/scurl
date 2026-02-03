"""Response middleware implementations including TrafilaturaExtractor."""

from .middleware import (
    ResponseMiddleware,
    ResponseContext,
    ResponseMiddlewareResult,
)


class TrafilaturaExtractor(ResponseMiddleware):
    """Extract readable content from HTML using trafilatura."""

    def __init__(
        self,
        include_links: bool = True,
        include_images: bool = True,
        include_tables: bool = True,
    ):
        self._include_links = include_links
        self._include_images = include_images
        self._include_tables = include_tables

    @property
    def name(self) -> str:
        return "TrafilaturaExtractor"

    @property
    def slug(self) -> str:
        return "trafilatura"

    def should_process(self, context: ResponseContext) -> bool:
        """Only process HTML content."""
        content_type = context.content_type or ""
        return "text/html" in content_type.lower()

    def process(self, context: ResponseContext) -> ResponseMiddlewareResult:
        """Extract markdown from HTML."""
        try:
            import trafilatura

            html = context.body.decode("utf-8", errors="replace")
            result = trafilatura.extract(
                html,
                url=context.url,
                output_format="markdown",
                include_links=self._include_links,
                include_images=self._include_images,
                include_tables=self._include_tables,
            )
            if result:
                return ResponseMiddlewareResult(
                    body=result.encode("utf-8"),
                    content_type="text/markdown",
                )
        except ImportError:
            pass  # trafilatura not installed, return original
        except Exception:
            pass  # extraction failed, return original

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
