"""Request middleware implementations including SecretDefender."""

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, parse_qs

from .middleware import (
    RequestMiddleware,
    RequestContext,
    RequestMiddlewareResult,
)


@dataclass
class SecretPattern:
    """Pattern for detecting secrets."""
    name: str
    pattern: re.Pattern
    description: str


# High-confidence secret patterns
SECRET_PATTERNS: list[SecretPattern] = [
    # AWS
    SecretPattern(
        name="AWS Access Key ID",
        pattern=re.compile(r"AKIA[0-9A-Z]{16}"),
        description="AWS Access Key ID",
    ),
    SecretPattern(
        name="AWS Secret Key",
        pattern=re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),
        description="Potential AWS Secret Access Key (40 char base64)",
    ),
    # GitHub
    SecretPattern(
        name="GitHub PAT (classic)",
        pattern=re.compile(r"ghp_[A-Za-z0-9]{36}"),
        description="GitHub Personal Access Token (classic)",
    ),
    SecretPattern(
        name="GitHub PAT (fine-grained)",
        pattern=re.compile(r"github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59}"),
        description="GitHub Personal Access Token (fine-grained)",
    ),
    SecretPattern(
        name="GitHub OAuth",
        pattern=re.compile(r"gho_[A-Za-z0-9]{36}"),
        description="GitHub OAuth Access Token",
    ),
    SecretPattern(
        name="GitHub App Token",
        pattern=re.compile(r"ghs_[A-Za-z0-9]{36}"),
        description="GitHub App Installation Token",
    ),
    SecretPattern(
        name="GitHub Refresh Token",
        pattern=re.compile(r"ghr_[A-Za-z0-9]{36}"),
        description="GitHub Refresh Token",
    ),
    # GitLab
    SecretPattern(
        name="GitLab PAT",
        pattern=re.compile(r"glpat-[A-Za-z0-9\-]{20}"),
        description="GitLab Personal Access Token",
    ),
    # npm
    SecretPattern(
        name="npm Token",
        pattern=re.compile(r"npm_[A-Za-z0-9]{36}"),
        description="npm Access Token",
    ),
    # PyPI
    SecretPattern(
        name="PyPI Token",
        pattern=re.compile(r"pypi-[A-Za-z0-9\-_]{50,}"),
        description="PyPI API Token",
    ),
    # Slack
    SecretPattern(
        name="Slack Bot Token",
        pattern=re.compile(r"xoxb-[0-9]{10,}-[0-9]{10,}-[A-Za-z0-9]{24}"),
        description="Slack Bot Token",
    ),
    SecretPattern(
        name="Slack User Token",
        pattern=re.compile(r"xoxp-[0-9]{10,}-[0-9]{10,}-[A-Za-z0-9]{24}"),
        description="Slack User Token",
    ),
    # Stripe
    SecretPattern(
        name="Stripe Live Key",
        pattern=re.compile(r"sk_live_[A-Za-z0-9]{24,}"),
        description="Stripe Live Secret Key",
    ),
    SecretPattern(
        name="Stripe Test Key",
        pattern=re.compile(r"sk_test_[A-Za-z0-9]{24,}"),
        description="Stripe Test Secret Key",
    ),
    # Google
    SecretPattern(
        name="Google API Key",
        pattern=re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
        description="Google API Key",
    ),
    # Twilio
    SecretPattern(
        name="Twilio API Key",
        pattern=re.compile(r"SK[0-9a-fA-F]{32}"),
        description="Twilio API Key",
    ),
    # SendGrid
    SecretPattern(
        name="SendGrid API Key",
        pattern=re.compile(r"SG\.[A-Za-z0-9\-_]{22}\.[A-Za-z0-9\-_]{43}"),
        description="SendGrid API Key",
    ),
    # DigitalOcean
    SecretPattern(
        name="DigitalOcean PAT",
        pattern=re.compile(r"dop_v1_[a-f0-9]{64}"),
        description="DigitalOcean Personal Access Token",
    ),
    # Doppler
    SecretPattern(
        name="Doppler Token",
        pattern=re.compile(r"dp\.pt\.[A-Za-z0-9]{43}"),
        description="Doppler Personal Token",
    ),
    # Discord
    SecretPattern(
        name="Discord Bot Token",
        pattern=re.compile(r"[MN][A-Za-z\d]{23,}\.[\w-]{6}\.[\w-]{27}"),
        description="Discord Bot Token",
    ),
    # Generic patterns
    SecretPattern(
        name="Private Key",
        pattern=re.compile(r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
        description="Private Key (PEM format)",
    ),
    SecretPattern(
        name="Password in URL",
        pattern=re.compile(r"://[^:]+:[^@]+@"),
        description="Password embedded in URL",
    ),
]


class SecretDefender(RequestMiddleware):
    """Middleware that detects and blocks requests containing secrets."""

    def __init__(
        self,
        patterns: Optional[list[SecretPattern]] = None,
        allow_list: Optional[list[str]] = None,
    ):
        self._patterns = patterns or SECRET_PATTERNS
        self._allow_list = allow_list or []

    @property
    def name(self) -> str:
        return "SecretDefender"

    @property
    def slug(self) -> str:
        return "secret-defender"

    def _is_allowed_domain(self, url: str) -> bool:
        """Check if URL domain is in allow list."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            for allowed in self._allow_list:
                if domain == allowed.lower() or domain.endswith("." + allowed.lower()):
                    return True
        except Exception:
            pass
        return False

    def _scan_text(self, text: str) -> Optional[SecretPattern]:
        """Scan text for secret patterns. Returns first match."""
        for pattern in self._patterns:
            if pattern.pattern.search(text):
                return pattern
        return None

    def _scan_url(self, url: str) -> Optional[SecretPattern]:
        """Scan URL including query parameters."""
        # Check full URL
        match = self._scan_text(url)
        if match:
            return match

        # Parse and check query params individually
        try:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            for values in params.values():
                for value in values:
                    match = self._scan_text(value)
                    if match:
                        return match
        except Exception:
            pass
        return None

    def _scan_headers(self, headers: dict[str, str]) -> Optional[SecretPattern]:
        """Scan request headers for secrets."""
        for name, value in headers.items():
            # Skip Authorization header - that's expected to have tokens
            if name.lower() == "authorization":
                continue
            match = self._scan_text(value)
            if match:
                return match
        return None

    def process(self, context: RequestContext) -> RequestMiddlewareResult:
        """Scan request for secrets and block if found."""
        # Skip allowed domains
        if self._is_allowed_domain(context.url):
            return RequestMiddlewareResult.pass_through()

        # Scan URL
        match = self._scan_url(context.url)
        if match:
            return RequestMiddlewareResult.block(
                f"Blocked: {match.name} detected in URL. "
                f"Use --enable secret-defender to override."
            )

        # Scan headers (except Authorization)
        match = self._scan_headers(context.headers)
        if match:
            return RequestMiddlewareResult.block(
                f"Blocked: {match.name} detected in headers. "
                f"Use --enable secret-defender to override."
            )

        # Scan body
        if context.body:
            match = self._scan_text(context.body)
            if match:
                return RequestMiddlewareResult.block(
                    f"Blocked: {match.name} detected in request body. "
                    f"Use --enable secret-defender to override."
                )

        # Scan curl args for any embedded secrets
        for arg in context.curl_args:
            match = self._scan_text(arg)
            if match:
                return RequestMiddlewareResult.block(
                    f"Blocked: {match.name} detected in curl arguments. "
                    f"Use --enable secret-defender to override."
                )

        return RequestMiddlewareResult.pass_through()
