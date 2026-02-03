"""Tests for middleware base classes."""

import pytest
from scurl.middleware import (
    RequestAction,
    RequestContext,
    RequestMiddlewareResult,
    RequestMiddleware,
    RequestMiddlewareChain,
    ResponseContext,
    ResponseMiddlewareResult,
    ResponseMiddleware,
    ResponseMiddlewareChain,
)


class PassThroughRequestMiddleware(RequestMiddleware):
    @property
    def name(self) -> str:
        return "PassThrough"

    @property
    def slug(self) -> str:
        return "pass-through"

    def process(self, context: RequestContext) -> RequestMiddlewareResult:
        return RequestMiddlewareResult.pass_through()


class BlockingRequestMiddleware(RequestMiddleware):
    def __init__(self, reason: str = "Blocked"):
        self._reason = reason

    @property
    def name(self) -> str:
        return "Blocker"

    @property
    def slug(self) -> str:
        return "blocker"

    def process(self, context: RequestContext) -> RequestMiddlewareResult:
        return RequestMiddlewareResult.block(self._reason)


class ModifyingRequestMiddleware(RequestMiddleware):
    def __init__(self, new_url: str):
        self._new_url = new_url

    @property
    def name(self) -> str:
        return "Modifier"

    @property
    def slug(self) -> str:
        return "modifier"

    def process(self, context: RequestContext) -> RequestMiddlewareResult:
        return RequestMiddlewareResult.modify(context.clone(url=self._new_url))


class UppercaseResponseMiddleware(ResponseMiddleware):
    @property
    def name(self) -> str:
        return "Uppercase"

    @property
    def slug(self) -> str:
        return "uppercase"

    def should_process(self, context: ResponseContext) -> bool:
        return True

    def process(self, context: ResponseContext) -> ResponseMiddlewareResult:
        return ResponseMiddlewareResult(
            body=context.body.upper(),
            content_type=context.content_type,
        )


class HtmlOnlyResponseMiddleware(ResponseMiddleware):
    @property
    def name(self) -> str:
        return "HtmlOnly"

    @property
    def slug(self) -> str:
        return "html-only"

    def should_process(self, context: ResponseContext) -> bool:
        return context.content_type and "text/html" in context.content_type

    def process(self, context: ResponseContext) -> ResponseMiddlewareResult:
        return ResponseMiddlewareResult(
            body=b"processed",
            content_type="text/plain",
        )


class TestRequestContext:
    def test_clone_preserves_values(self):
        ctx = RequestContext(
            url="https://example.com",
            method="GET",
            headers={"Accept": "text/html"},
            body=None,
            curl_args=["https://example.com"],
        )
        cloned = ctx.clone()

        assert cloned.url == ctx.url
        assert cloned.method == ctx.method
        assert cloned.headers == ctx.headers
        assert cloned.body == ctx.body
        assert cloned.curl_args == ctx.curl_args
        # Should be copies, not same objects
        assert cloned.headers is not ctx.headers
        assert cloned.curl_args is not ctx.curl_args

    def test_clone_with_overrides(self):
        ctx = RequestContext(
            url="https://example.com",
            method="GET",
            headers={},
            body=None,
            curl_args=[],
        )
        cloned = ctx.clone(url="https://other.com", method="POST")

        assert cloned.url == "https://other.com"
        assert cloned.method == "POST"
        assert ctx.url == "https://example.com"
        assert ctx.method == "GET"


class TestRequestMiddlewareChain:
    def test_empty_chain_passes_through(self):
        chain = RequestMiddlewareChain()
        ctx = RequestContext(
            url="https://example.com",
            method="GET",
            headers={},
            body=None,
            curl_args=[],
        )
        result = chain.execute(ctx)

        assert result.action == RequestAction.PASS

    def test_pass_through_continues_chain(self):
        chain = RequestMiddlewareChain([
            PassThroughRequestMiddleware(),
            PassThroughRequestMiddleware(),
        ])
        ctx = RequestContext(
            url="https://example.com",
            method="GET",
            headers={},
            body=None,
            curl_args=[],
        )
        result = chain.execute(ctx)

        assert result.action == RequestAction.PASS

    def test_block_stops_chain(self):
        chain = RequestMiddlewareChain([
            BlockingRequestMiddleware("First block"),
            BlockingRequestMiddleware("Second block"),
        ])
        ctx = RequestContext(
            url="https://example.com",
            method="GET",
            headers={},
            body=None,
            curl_args=[],
        )
        result = chain.execute(ctx)

        assert result.action == RequestAction.BLOCK
        assert result.reason == "First block"

    def test_modify_updates_context(self):
        chain = RequestMiddlewareChain([
            ModifyingRequestMiddleware("https://modified.com"),
        ])
        ctx = RequestContext(
            url="https://example.com",
            method="GET",
            headers={},
            body=None,
            curl_args=[],
        )
        result = chain.execute(ctx)

        assert result.action == RequestAction.PASS
        assert result.context.url == "https://modified.com"

    def test_modify_then_block(self):
        chain = RequestMiddlewareChain([
            ModifyingRequestMiddleware("https://modified.com"),
            BlockingRequestMiddleware("Blocked after modify"),
        ])
        ctx = RequestContext(
            url="https://example.com",
            method="GET",
            headers={},
            body=None,
            curl_args=[],
        )
        result = chain.execute(ctx)

        assert result.action == RequestAction.BLOCK


class TestResponseContext:
    def test_body_text_decodes_utf8(self):
        ctx = ResponseContext(
            body="Hello, 世界".encode("utf-8"),
            headers={},
            status_code=200,
            content_type="text/plain",
            url="https://example.com",
        )
        assert ctx.body_text == "Hello, 世界"

    def test_body_text_handles_invalid_utf8(self):
        ctx = ResponseContext(
            body=b"\xff\xfe",
            headers={},
            status_code=200,
            content_type="text/plain",
            url="https://example.com",
        )
        # Should not raise, uses replacement chars
        assert "�" in ctx.body_text


class TestResponseMiddlewareChain:
    def test_empty_chain_returns_original(self):
        chain = ResponseMiddlewareChain()
        ctx = ResponseContext(
            body=b"original",
            headers={},
            status_code=200,
            content_type="text/plain",
            url="https://example.com",
        )
        result = chain.execute(ctx)

        assert result.body == b"original"

    def test_middleware_transforms_body(self):
        chain = ResponseMiddlewareChain([UppercaseResponseMiddleware()])
        ctx = ResponseContext(
            body=b"hello",
            headers={},
            status_code=200,
            content_type="text/plain",
            url="https://example.com",
        )
        result = chain.execute(ctx)

        assert result.body == b"HELLO"

    def test_should_process_filters(self):
        chain = ResponseMiddlewareChain([HtmlOnlyResponseMiddleware()])

        # HTML should be processed
        html_ctx = ResponseContext(
            body=b"<html>",
            headers={},
            status_code=200,
            content_type="text/html",
            url="https://example.com",
        )
        result = chain.execute(html_ctx)
        assert result.body == b"processed"

        # JSON should not be processed
        json_ctx = ResponseContext(
            body=b"{}",
            headers={},
            status_code=200,
            content_type="application/json",
            url="https://example.com",
        )
        result = chain.execute(json_ctx)
        assert result.body == b"{}"

    def test_chain_order_matters(self):
        chain = ResponseMiddlewareChain([
            UppercaseResponseMiddleware(),
            UppercaseResponseMiddleware(),  # Second uppercase does nothing
        ])
        ctx = ResponseContext(
            body=b"hello",
            headers={},
            status_code=200,
            content_type="text/plain",
            url="https://example.com",
        )
        result = chain.execute(ctx)

        assert result.body == b"HELLO"
