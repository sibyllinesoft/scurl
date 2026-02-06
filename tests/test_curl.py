"""Tests for curl wrapper."""

from scurl.curl import parse_curl_args


class TestParseCurlArgs:
    def test_simple_url(self):
        ctx = parse_curl_args(["https://example.com"])

        assert ctx.url == "https://example.com"
        assert ctx.method == "GET"
        assert ctx.headers == {}
        assert ctx.body is None

    def test_url_with_path(self):
        ctx = parse_curl_args(["https://example.com/api/v1/users"])

        assert ctx.url == "https://example.com/api/v1/users"

    def test_explicit_method(self):
        ctx = parse_curl_args(["-X", "POST", "https://example.com"])

        assert ctx.method == "POST"

    def test_request_method_long_form(self):
        ctx = parse_curl_args(["--request", "DELETE", "https://example.com"])

        assert ctx.method == "DELETE"

    def test_method_case_normalized(self):
        ctx = parse_curl_args(["-X", "post", "https://example.com"])

        assert ctx.method == "POST"

    def test_single_header(self):
        ctx = parse_curl_args(["-H", "Accept: application/json", "https://example.com"])

        assert ctx.headers == {"Accept": "application/json"}

    def test_multiple_headers(self):
        ctx = parse_curl_args([
            "-H", "Accept: application/json",
            "-H", "Authorization: Bearer token",
            "https://example.com",
        ])

        assert ctx.headers == {
            "Accept": "application/json",
            "Authorization": "Bearer token",
        }

    def test_header_long_form(self):
        ctx = parse_curl_args(["--header", "Accept: text/html", "https://example.com"])

        assert ctx.headers == {"Accept": "text/html"}

    def test_header_with_multiple_colons(self):
        ctx = parse_curl_args(["-H", "X-Custom: value:with:colons", "https://example.com"])

        assert ctx.headers == {"X-Custom": "value:with:colons"}

    def test_data_flag(self):
        ctx = parse_curl_args(["-d", '{"key":"value"}', "https://example.com"])

        assert ctx.body == '{"key":"value"}'
        assert ctx.method == "POST"  # -d implies POST

    def test_data_long_form(self):
        ctx = parse_curl_args(["--data", "name=test", "https://example.com"])

        assert ctx.body == "name=test"

    def test_data_raw(self):
        ctx = parse_curl_args(["--data-raw", "raw data", "https://example.com"])

        assert ctx.body == "raw data"

    def test_data_binary(self):
        ctx = parse_curl_args(["--data-binary", "@file.bin", "https://example.com"])

        assert ctx.body == "@file.bin"

    def test_explicit_method_with_data(self):
        """Explicit method should override -d's implied POST."""
        ctx = parse_curl_args(["-X", "PUT", "-d", "data", "https://example.com"])

        assert ctx.method == "PUT"
        assert ctx.body == "data"

    def test_url_flag(self):
        ctx = parse_curl_args(["--url", "https://example.com/path"])

        assert ctx.url == "https://example.com/path"

    def test_http_url(self):
        ctx = parse_curl_args(["http://example.com"])

        assert ctx.url == "http://example.com"

    def test_preserves_curl_args(self):
        args = ["-H", "Accept: text/html", "-v", "https://example.com"]
        ctx = parse_curl_args(args)

        assert ctx.curl_args == args

    def test_url_at_end(self):
        ctx = parse_curl_args(["-X", "POST", "-H", "Accept: */*", "https://example.com/api"])

        assert ctx.url == "https://example.com/api"

    def test_url_at_beginning(self):
        ctx = parse_curl_args(["https://example.com", "-H", "Accept: */*"])

        assert ctx.url == "https://example.com"

    def test_complex_request(self):
        ctx = parse_curl_args([
            "-X", "PATCH",
            "-H", "Content-Type: application/json",
            "-H", "Authorization: Bearer abc123",
            "-d", '{"status":"active"}',
            "https://api.example.com/users/1",
        ])

        assert ctx.url == "https://api.example.com/users/1"
        assert ctx.method == "PATCH"
        assert ctx.headers == {
            "Content-Type": "application/json",
            "Authorization": "Bearer abc123",
        }
        assert ctx.body == '{"status":"active"}'

    def test_empty_args(self):
        ctx = parse_curl_args([])

        assert ctx.url == ""
        assert ctx.method == "GET"
        assert ctx.headers == {}
        assert ctx.body is None

    def test_bare_hostname(self):
        """Bare hostname without scheme should be accepted like curl."""
        ctx = parse_curl_args(["www.google.com"])

        assert ctx.url == "www.google.com"

    def test_bare_hostname_with_path(self):
        ctx = parse_curl_args(["example.com/path/to/page"])

        assert ctx.url == "example.com/path/to/page"

    def test_localhost(self):
        ctx = parse_curl_args(["localhost:8080"])

        assert ctx.url == "localhost:8080"

    def test_unknown_flags_preserved(self):
        """Unknown flags should be preserved in curl_args for pass-through."""
        args = ["-v", "--compressed", "-L", "https://example.com"]
        ctx = parse_curl_args(args)

        assert "-v" in ctx.curl_args
        assert "--compressed" in ctx.curl_args
        assert "-L" in ctx.curl_args
