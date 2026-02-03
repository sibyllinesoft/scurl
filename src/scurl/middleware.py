"""Base middleware classes for request and response processing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RequestAction(Enum):
    """Action to take after request middleware processing."""
    PASS = "pass"
    MODIFY = "modify"
    BLOCK = "block"


@dataclass
class RequestContext:
    """Context for an outgoing request."""
    url: str
    method: str
    headers: dict[str, str]
    body: Optional[str]
    curl_args: list[str]

    def clone(self, **kwargs) -> "RequestContext":
        """Create a copy with optional field overrides."""
        return RequestContext(
            url=kwargs.get("url", self.url),
            method=kwargs.get("method", self.method),
            headers=kwargs.get("headers", self.headers.copy()),
            body=kwargs.get("body", self.body),
            curl_args=kwargs.get("curl_args", self.curl_args.copy()),
        )


@dataclass
class RequestMiddlewareResult:
    """Result from request middleware processing."""
    action: RequestAction
    context: Optional[RequestContext] = None
    reason: Optional[str] = None

    @classmethod
    def pass_through(cls, context: Optional[RequestContext] = None) -> "RequestMiddlewareResult":
        return cls(action=RequestAction.PASS, context=context)

    @classmethod
    def modify(cls, context: RequestContext) -> "RequestMiddlewareResult":
        return cls(action=RequestAction.MODIFY, context=context)

    @classmethod
    def block(cls, reason: str) -> "RequestMiddlewareResult":
        return cls(action=RequestAction.BLOCK, reason=reason)


class RequestMiddleware(ABC):
    """Base class for request middleware."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable middleware name."""
        pass

    @property
    @abstractmethod
    def slug(self) -> str:
        """Canonical slug for --skip/--enable flags (e.g., 'secret-defender')."""
        pass

    @abstractmethod
    def process(self, context: RequestContext) -> RequestMiddlewareResult:
        """Process request, returning action to take."""
        pass


@dataclass
class ResponseContext:
    """Context for a response from curl."""
    body: bytes
    headers: dict[str, str]
    status_code: int
    content_type: Optional[str]
    url: str

    @property
    def body_text(self) -> str:
        """Decode body as UTF-8."""
        return self.body.decode("utf-8", errors="replace")


@dataclass
class ResponseMiddlewareResult:
    """Result from response middleware processing."""
    body: bytes
    content_type: Optional[str] = None


class ResponseMiddleware(ABC):
    """Base class for response middleware."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable middleware name."""
        pass

    @property
    @abstractmethod
    def slug(self) -> str:
        """Canonical slug for --skip/--enable flags (e.g., 'trafilatura')."""
        pass

    @abstractmethod
    def process(self, context: ResponseContext) -> ResponseMiddlewareResult:
        """Transform response, returning new body."""
        pass

    @abstractmethod
    def should_process(self, context: ResponseContext) -> bool:
        """Return True if this middleware should handle this response."""
        pass


class RequestMiddlewareChain:
    """Chain of request middleware executed in order."""

    def __init__(self, middlewares: Optional[list[RequestMiddleware]] = None):
        self.middlewares = middlewares or []

    def add(self, middleware: RequestMiddleware) -> None:
        self.middlewares.append(middleware)

    def execute(self, context: RequestContext) -> RequestMiddlewareResult:
        """Execute middleware chain. First BLOCK stops the chain."""
        current_context = context
        for mw in self.middlewares:
            result = mw.process(current_context)
            if result.action == RequestAction.BLOCK:
                return result
            if result.action == RequestAction.MODIFY and result.context:
                current_context = result.context
        return RequestMiddlewareResult.pass_through(current_context)


class ResponseMiddlewareChain:
    """Chain of response middleware executed in order."""

    def __init__(self, middlewares: Optional[list[ResponseMiddleware]] = None):
        self.middlewares = middlewares or []

    def add(self, middleware: ResponseMiddleware) -> None:
        self.middlewares.append(middleware)

    def execute(self, context: ResponseContext) -> ResponseMiddlewareResult:
        """Execute middleware chain, transforming response through each."""
        body = context.body
        content_type = context.content_type

        for mw in self.middlewares:
            ctx = ResponseContext(
                body=body,
                headers=context.headers,
                status_code=context.status_code,
                content_type=content_type,
                url=context.url,
            )
            if mw.should_process(ctx):
                result = mw.process(ctx)
                body = result.body
                content_type = result.content_type or content_type

        return ResponseMiddlewareResult(body=body, content_type=content_type)
