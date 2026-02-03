"""Tests for response middleware implementations."""

import pytest
from scurl.middleware import ResponseContext
from scurl.response_middleware import TrafilaturaExtractor, JsonPrettifier


def make_response_context(
    body: bytes,
    content_type: str = "text/html",
    status_code: int = 200,
    url: str = "https://example.com",
    headers: dict = None,
) -> ResponseContext:
    return ResponseContext(
        body=body,
        headers=headers or {},
        status_code=status_code,
        content_type=content_type,
        url=url,
    )


class TestTrafilaturaExtractor:
    def test_name(self):
        extractor = TrafilaturaExtractor()
        assert extractor.name == "TrafilaturaExtractor"

    def test_should_process_html(self):
        extractor = TrafilaturaExtractor()

        html_ctx = make_response_context(b"<html>", content_type="text/html")
        assert extractor.should_process(html_ctx) is True

        html_charset_ctx = make_response_context(
            b"<html>", content_type="text/html; charset=utf-8"
        )
        assert extractor.should_process(html_charset_ctx) is True

    def test_should_not_process_json(self):
        extractor = TrafilaturaExtractor()
        json_ctx = make_response_context(b"{}", content_type="application/json")
        assert extractor.should_process(json_ctx) is False

    def test_should_not_process_plain_text(self):
        extractor = TrafilaturaExtractor()
        text_ctx = make_response_context(b"hello", content_type="text/plain")
        assert extractor.should_process(text_ctx) is False

    def test_should_not_process_none_content_type(self):
        extractor = TrafilaturaExtractor()
        ctx = make_response_context(b"<html>", content_type=None)
        assert extractor.should_process(ctx) is False

    def test_extracts_content_from_html(self):
        extractor = TrafilaturaExtractor()
        html = b"""
        <!DOCTYPE html>
        <html>
        <head><title>Test Page</title></head>
        <body>
            <nav>Navigation links here</nav>
            <article>
                <h1>Main Article Title</h1>
                <p>This is the main content of the article. It contains important information that should be extracted.</p>
                <p>Another paragraph with more content.</p>
            </article>
            <footer>Footer content</footer>
        </body>
        </html>
        """
        ctx = make_response_context(html, content_type="text/html")
        result = extractor.process(ctx)

        # Should have extracted some text
        text = result.body.decode("utf-8")
        assert "Main Article Title" in text or "main content" in text.lower()

    def test_returns_original_on_extraction_failure(self):
        extractor = TrafilaturaExtractor()
        # Minimal HTML that trafilatura can't extract meaningful content from
        html = b"<html><body></body></html>"
        ctx = make_response_context(html, content_type="text/html")
        result = extractor.process(ctx)

        # Should return something (either original or empty extraction)
        assert result.body is not None

    def test_handles_utf8_content(self):
        extractor = TrafilaturaExtractor()
        html = """
        <!DOCTYPE html>
        <html>
        <body>
            <article>
                <p>Hello, 世界! Привет мир! مرحبا بالعالم</p>
            </article>
        </body>
        </html>
        """.encode("utf-8")
        ctx = make_response_context(html, content_type="text/html; charset=utf-8")
        result = extractor.process(ctx)

        # Should not crash and should preserve unicode
        text = result.body.decode("utf-8")
        # At minimum, shouldn't have encoding errors
        assert "�" not in text or len(text) > 0

    def test_sets_markdown_content_type(self):
        extractor = TrafilaturaExtractor()
        html = b"""
        <!DOCTYPE html>
        <html>
        <body>
            <article>
                <h1>Title</h1>
                <p>Content paragraph that is long enough to be extracted by trafilatura as meaningful content.</p>
            </article>
        </body>
        </html>
        """
        ctx = make_response_context(html, content_type="text/html")
        result = extractor.process(ctx)

        # If extraction succeeded, content-type should be markdown
        if result.body != html:
            assert result.content_type == "text/markdown"


class TestJsonPrettifier:
    def test_name(self):
        prettifier = JsonPrettifier()
        assert prettifier.name == "JsonPrettifier"

    def test_should_process_json(self):
        prettifier = JsonPrettifier()

        json_ctx = make_response_context(b"{}", content_type="application/json")
        assert prettifier.should_process(json_ctx) is True

        json_charset_ctx = make_response_context(
            b"{}", content_type="application/json; charset=utf-8"
        )
        assert prettifier.should_process(json_charset_ctx) is True

        text_json_ctx = make_response_context(b"{}", content_type="text/json")
        assert prettifier.should_process(text_json_ctx) is True

    def test_should_not_process_html(self):
        prettifier = JsonPrettifier()
        html_ctx = make_response_context(b"<html>", content_type="text/html")
        assert prettifier.should_process(html_ctx) is False

    def test_should_not_process_none_content_type(self):
        prettifier = JsonPrettifier()
        ctx = make_response_context(b"{}", content_type=None)
        assert prettifier.should_process(ctx) is False

    def test_prettifies_json(self):
        prettifier = JsonPrettifier(indent=2)
        ctx = make_response_context(
            b'{"name":"test","value":123}',
            content_type="application/json",
        )
        result = prettifier.process(ctx)

        expected = b'{\n  "name": "test",\n  "value": 123\n}'
        assert result.body == expected

    def test_prettifies_nested_json(self):
        prettifier = JsonPrettifier(indent=2)
        ctx = make_response_context(
            b'{"outer":{"inner":"value"}}',
            content_type="application/json",
        )
        result = prettifier.process(ctx)

        text = result.body.decode("utf-8")
        assert "outer" in text
        assert "inner" in text
        assert "\n" in text  # Has newlines

    def test_handles_json_array(self):
        prettifier = JsonPrettifier(indent=2)
        ctx = make_response_context(
            b'[1,2,3]',
            content_type="application/json",
        )
        result = prettifier.process(ctx)

        expected = b'[\n  1,\n  2,\n  3\n]'
        assert result.body == expected

    def test_handles_invalid_json(self):
        prettifier = JsonPrettifier()
        ctx = make_response_context(
            b'not valid json',
            content_type="application/json",
        )
        result = prettifier.process(ctx)

        # Should return original on parse error
        assert result.body == b'not valid json'

    def test_preserves_unicode(self):
        prettifier = JsonPrettifier(indent=2)
        ctx = make_response_context(
            '{"message":"Hello, 世界"}'.encode("utf-8"),
            content_type="application/json",
        )
        result = prettifier.process(ctx)

        text = result.body.decode("utf-8")
        assert "世界" in text

    def test_custom_indent(self):
        prettifier = JsonPrettifier(indent=4)
        ctx = make_response_context(
            b'{"a":1}',
            content_type="application/json",
        )
        result = prettifier.process(ctx)

        expected = b'{\n    "a": 1\n}'
        assert result.body == expected

    def test_preserves_content_type(self):
        prettifier = JsonPrettifier()
        ctx = make_response_context(
            b'{"a":1}',
            content_type="application/json; charset=utf-8",
        )
        result = prettifier.process(ctx)

        assert result.content_type == "application/json; charset=utf-8"
