"""Tests for SecretDefender middleware."""

from scurl.middleware import RequestAction, RequestContext
from scurl.request_middleware import SecretDefender


def make_context(
    url: str = "https://example.com",
    method: str = "GET",
    headers: dict = None,
    body: str = None,
    curl_args: list = None,
) -> RequestContext:
    return RequestContext(
        url=url,
        method=method,
        headers=headers or {},
        body=body,
        curl_args=curl_args or [url],
    )


class TestSecretPatterns:
    """Test that secret patterns detect known formats."""

    def test_aws_access_key_id(self):
        defender = SecretDefender()
        ctx = make_context(url="https://example.com?key=AKIAIOSFODNN7EXAMPLE")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "AWS Access Key ID" in result.reason

    def test_github_pat_classic(self):
        defender = SecretDefender()
        # ghp_ + 36 alphanumeric chars
        token = "ghp_" + "a" * 36
        ctx = make_context(url=f"https://api.github.com?token={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "GitHub PAT (classic)" in result.reason

    def test_github_pat_fine_grained(self):
        defender = SecretDefender()
        # github_pat_ + 22 chars + _ + 59 chars
        token = "github_pat_" + "a" * 22 + "_" + "b" * 59
        ctx = make_context(url=f"https://api.github.com?token={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "GitHub PAT (fine-grained)" in result.reason

    def test_github_oauth(self):
        defender = SecretDefender()
        token = "gho_" + "a" * 36
        ctx = make_context(url=f"https://example.com?token={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "GitHub OAuth" in result.reason

    def test_github_app_token(self):
        defender = SecretDefender()
        token = "ghs_" + "a" * 36
        ctx = make_context(url=f"https://example.com?token={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "GitHub App Token" in result.reason

    def test_gitlab_pat(self):
        defender = SecretDefender()
        token = "glpat-" + "a" * 20
        ctx = make_context(url=f"https://gitlab.com?token={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "GitLab PAT" in result.reason

    def test_npm_token(self):
        defender = SecretDefender()
        token = "npm_" + "a" * 36
        ctx = make_context(url=f"https://registry.npmjs.org?token={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "npm Token" in result.reason

    def test_pypi_token(self):
        defender = SecretDefender()
        token = "pypi-" + "a" * 50
        ctx = make_context(url=f"https://pypi.org?token={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "PyPI Token" in result.reason

    def test_slack_bot_token(self):
        defender = SecretDefender()
        token = "xoxb-1234567890-1234567890-" + "a" * 24
        ctx = make_context(url=f"https://slack.com?token={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "Slack Bot Token" in result.reason

    def test_stripe_live_key(self):
        defender = SecretDefender()
        token = "sk_live_" + "a" * 24
        ctx = make_context(url=f"https://api.stripe.com?key={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "Stripe Live Key" in result.reason

    def test_stripe_test_key(self):
        defender = SecretDefender()
        token = "sk_test_" + "a" * 24
        ctx = make_context(url=f"https://api.stripe.com?key={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "Stripe Test Key" in result.reason

    def test_google_api_key(self):
        defender = SecretDefender()
        token = "AIza" + "a" * 35
        ctx = make_context(url=f"https://maps.googleapis.com?key={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "Google API Key" in result.reason

    def test_sendgrid_api_key(self):
        defender = SecretDefender()
        token = "SG." + "a" * 22 + "." + "b" * 43
        ctx = make_context(url=f"https://api.sendgrid.com?key={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "SendGrid API Key" in result.reason

    def test_digitalocean_pat(self):
        defender = SecretDefender()
        token = "dop_v1_" + "a" * 64
        ctx = make_context(url=f"https://api.digitalocean.com?token={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "DigitalOcean PAT" in result.reason

    def test_private_key_pem(self):
        defender = SecretDefender()
        ctx = make_context(body="-----BEGIN RSA PRIVATE KEY-----\nMIIE...")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "Private Key" in result.reason

    def test_password_in_url(self):
        defender = SecretDefender()
        ctx = make_context(url="https://user:password123@example.com/api")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK
        assert "Password in URL" in result.reason


class TestSecretDefenderBehavior:
    """Test SecretDefender behavior and edge cases."""

    def test_clean_url_passes(self):
        defender = SecretDefender()
        ctx = make_context(url="https://example.com/page?foo=bar")
        result = defender.process(ctx)

        assert result.action == RequestAction.PASS

    def test_authorization_header_allowed(self):
        """Authorization header is expected to contain tokens."""
        defender = SecretDefender()
        ctx = make_context(
            url="https://api.example.com",
            headers={"Authorization": "Bearer ghp_" + "a" * 36},
        )
        result = defender.process(ctx)

        assert result.action == RequestAction.PASS

    def test_other_headers_scanned(self):
        """Non-Authorization headers should be scanned."""
        defender = SecretDefender()
        ctx = make_context(
            url="https://api.example.com",
            headers={"X-Custom-Token": "ghp_" + "a" * 36},
        )
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK

    def test_body_scanned(self):
        defender = SecretDefender()
        ctx = make_context(
            url="https://api.example.com",
            body='{"token": "ghp_' + "a" * 36 + '"}',
        )
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK

    def test_curl_args_scanned(self):
        defender = SecretDefender()
        token = "ghp_" + "a" * 36
        ctx = make_context(
            url="https://api.example.com",
            curl_args=["https://api.example.com", "-H", f"X-Token: {token}"],
        )
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK

    def test_allow_list_bypasses_check(self):
        defender = SecretDefender(allow_list=["example.com"])
        token = "ghp_" + "a" * 36
        ctx = make_context(url=f"https://example.com?token={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.PASS

    def test_allow_list_subdomain(self):
        defender = SecretDefender(allow_list=["example.com"])
        token = "ghp_" + "a" * 36
        ctx = make_context(url=f"https://api.example.com?token={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.PASS

    def test_allow_list_different_domain_blocked(self):
        defender = SecretDefender(allow_list=["example.com"])
        token = "ghp_" + "a" * 36
        ctx = make_context(url=f"https://other.com?token={token}")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK

    def test_query_param_encoding_detected(self):
        """Secrets in query params should be detected."""
        defender = SecretDefender()
        token = "AKIAIOSFODNN7EXAMPLE"
        ctx = make_context(url=f"https://example.com/api?access_key={token}&other=value")
        result = defender.process(ctx)

        assert result.action == RequestAction.BLOCK

    def test_custom_patterns(self):
        """Can use custom patterns instead of defaults."""
        import re
        from scurl.request_middleware import SecretPattern

        custom_pattern = SecretPattern(
            name="Custom Secret",
            pattern=re.compile(r"CUSTOM_[A-Z]{10}"),
            description="Custom secret format",
        )
        defender = SecretDefender(patterns=[custom_pattern])

        # Default pattern should not match
        ctx = make_context(url="https://example.com?key=AKIAIOSFODNN7EXAMPLE")
        result = defender.process(ctx)
        assert result.action == RequestAction.PASS

        # Custom pattern should match
        ctx = make_context(url="https://example.com?key=CUSTOM_ABCDEFGHIJ")
        result = defender.process(ctx)
        assert result.action == RequestAction.BLOCK
        assert "Custom Secret" in result.reason


class TestSecretDefenderFalsePositives:
    """Test that common non-secrets don't trigger false positives."""

    def test_short_base64_allowed(self):
        """Short base64 strings that look like secrets should pass."""
        defender = SecretDefender()
        # Not long enough to be a secret
        ctx = make_context(url="https://example.com?data=abc123")
        result = defender.process(ctx)

        assert result.action == RequestAction.PASS

    def test_uuid_allowed(self):
        """UUIDs should not trigger Heroku pattern."""
        defender = SecretDefender()
        ctx = make_context(url="https://example.com?id=550e8400-e29b-41d4-a716-446655440000")
        result = defender.process(ctx)

        assert result.action == RequestAction.PASS

    def test_normal_url_params_allowed(self):
        defender = SecretDefender()
        ctx = make_context(
            url="https://example.com/search?q=hello+world&page=1&sort=date"
        )
        result = defender.process(ctx)

        assert result.action == RequestAction.PASS
