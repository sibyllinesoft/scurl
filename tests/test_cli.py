"""Tests for CLI entry point."""

import pytest
from scurl.cli import extract_scurl_flags, run, ScurlFlags


class TestExtractScurlFlags:
    def test_no_flags(self):
        flags, remaining = extract_scurl_flags(["https://example.com"])

        assert flags.raw is False
        assert flags.disable == set()
        assert flags.enable == set()
        assert flags.list_middleware is False
        assert remaining == ["https://example.com"]

    def test_raw_flag(self):
        flags, remaining = extract_scurl_flags(["--raw", "https://example.com"])

        assert flags.raw is True
        assert remaining == ["https://example.com"]

    def test_disable_single(self):
        flags, remaining = extract_scurl_flags(["--disable", "readability", "https://example.com"])

        assert "readability" in flags.disable
        assert remaining == ["https://example.com"]

    def test_disable_multiple(self):
        flags, remaining = extract_scurl_flags([
            "--disable", "readability",
            "--disable", "secret-defender",
            "https://example.com"
        ])

        assert flags.disable == {"readability", "secret-defender"}
        assert remaining == ["https://example.com"]

    def test_enable_single(self):
        flags, remaining = extract_scurl_flags(["--enable", "secret-defender", "https://example.com"])

        assert "secret-defender" in flags.enable
        assert remaining == ["https://example.com"]

    def test_enable_multiple(self):
        flags, remaining = extract_scurl_flags([
            "--enable", "secret-defender",
            "--enable", "other-middleware",
            "https://example.com"
        ])

        assert flags.enable == {"secret-defender", "other-middleware"}
        assert remaining == ["https://example.com"]

    def test_list_middleware_flag(self):
        flags, remaining = extract_scurl_flags(["--list-middleware"])

        assert flags.list_middleware is True
        assert remaining == []

    def test_help_flag_first(self):
        flags, remaining = extract_scurl_flags(["--help"])

        assert flags.help is True
        assert remaining == []

    def test_help_flag_h_first(self):
        flags, remaining = extract_scurl_flags(["-h"])

        assert flags.help is True
        assert remaining == []

    def test_help_flag_not_first(self):
        """--help after other args should be passed to curl."""
        flags, remaining = extract_scurl_flags(["https://example.com", "--help"])

        assert flags.help is False
        assert "--help" in remaining

    def test_mixed_flags(self):
        flags, remaining = extract_scurl_flags([
            "--raw",
            "-H", "Accept: text/html",
            "--disable", "readability",
            "-v",
            "https://example.com",
        ])

        assert flags.raw is True
        assert "readability" in flags.disable
        assert remaining == ["-H", "Accept: text/html", "-v", "https://example.com"]

    def test_curl_flags_preserved(self):
        flags, remaining = extract_scurl_flags([
            "-H", "Accept: application/json",
            "-X", "POST",
            "https://example.com",
        ])

        assert remaining == ["-H", "Accept: application/json", "-X", "POST", "https://example.com"]

    def test_disable_without_value_passed_through(self):
        """--disable at end without value should be passed to curl."""
        flags, remaining = extract_scurl_flags(["https://example.com", "--disable"])

        assert flags.disable == set()
        assert "--disable" in remaining


class TestRunHelp:
    def test_help_returns_zero(self, capsys):
        result = run(["--help"])

        assert result == 0
        captured = capsys.readouterr()
        assert "scurl" in captured.out
        assert "--raw" in captured.out
        assert "--disable" in captured.out
        assert "--enable" in captured.out

    def test_list_middleware_returns_zero(self, capsys):
        result = run(["--list-middleware"])

        assert result == 0
        captured = capsys.readouterr()
        assert "secret-defender" in captured.out
        assert "readability" in captured.out


class TestRunErrors:
    def test_no_url_returns_error(self, capsys):
        result = run([])

        assert result == 1
        captured = capsys.readouterr()
        assert "no URL specified" in captured.err

    def test_only_flags_no_url(self, capsys):
        result = run(["--raw"])

        assert result == 1
        captured = capsys.readouterr()
        assert "no URL specified" in captured.err


class TestSecretDefenderIntegration:
    def test_blocks_github_token_in_url(self, capsys):
        token = "ghp_" + "a" * 36
        result = run([f"https://example.com?token={token}"])

        assert result == 1
        captured = capsys.readouterr()
        assert "Blocked" in captured.err
        assert "GitHub PAT" in captured.err

    def test_enable_bypasses_block(self, capsys, mocker):
        """--enable secret-defender should bypass SecretDefender."""
        # Mock execute_curl to avoid actual network call
        mock_result = mocker.MagicMock()
        mock_result.return_code = 0
        mock_result.body = b"response"
        mock_result.headers = {}
        mock_result.status_code = 200
        mock_result.content_type = "text/plain"
        mock_result.final_url = "https://example.com"
        mock_result.stderr = ""

        mocker.patch("scurl.cli.execute_curl", return_value=mock_result)

        token = "ghp_" + "a" * 36
        result = run(["--enable", "secret-defender", f"https://example.com?token={token}"])

        assert result == 0

    def test_disable_disables_defender(self, capsys, mocker):
        """--disable secret-defender should disable SecretDefender entirely."""
        mock_result = mocker.MagicMock()
        mock_result.return_code = 0
        mock_result.body = b"response"
        mock_result.headers = {}
        mock_result.status_code = 200
        mock_result.content_type = "text/plain"
        mock_result.final_url = "https://example.com"
        mock_result.stderr = ""

        mocker.patch("scurl.cli.execute_curl", return_value=mock_result)

        token = "ghp_" + "a" * 36
        result = run(["--disable", "secret-defender", f"https://example.com?token={token}"])

        assert result == 0


class TestResponseMiddlewareIntegration:
    def test_disable_readability(self, capsys, mocker):
        """--disable readability should return raw HTML."""
        mock_result = mocker.MagicMock()
        mock_result.return_code = 0
        mock_result.body = b"<html><body>Hello</body></html>"
        mock_result.headers = {"content-type": "text/html"}
        mock_result.status_code = 200
        mock_result.content_type = "text/html"
        mock_result.final_url = "https://example.com"
        mock_result.stderr = ""

        mocker.patch("scurl.cli.execute_curl", return_value=mock_result)

        result = run(["--disable", "readability", "https://example.com"])

        assert result == 0
        captured = capsys.readouterr()
        assert "<html>" in captured.out

    def test_raw_skips_all_response_middleware(self, capsys, mocker):
        """--raw should skip all response middleware."""
        mock_result = mocker.MagicMock()
        mock_result.return_code = 0
        mock_result.body = b"<html><body>Hello</body></html>"
        mock_result.headers = {"content-type": "text/html"}
        mock_result.status_code = 200
        mock_result.content_type = "text/html"
        mock_result.final_url = "https://example.com"
        mock_result.stderr = ""

        mocker.patch("scurl.cli.execute_curl", return_value=mock_result)

        result = run(["--raw", "https://example.com"])

        assert result == 0
        captured = capsys.readouterr()
        assert "<html>" in captured.out
