"""Curl wrapper for executing requests and parsing responses."""

import subprocess
import re
from dataclasses import dataclass
from typing import Optional

from .middleware import RequestContext, ResponseContext


@dataclass
class CurlResult:
    """Result from curl execution."""
    body: bytes
    headers: dict[str, str]
    status_code: int
    content_type: Optional[str]
    final_url: str
    return_code: int
    stderr: str


def parse_curl_args(args: list[str]) -> RequestContext:
    """Parse curl arguments into a RequestContext."""
    url = ""
    method = "GET"
    headers: dict[str, str] = {}
    body: Optional[str] = None

    i = 0
    while i < len(args):
        arg = args[i]

        # URL (positional or after flags)
        # Accept URLs with scheme or bare hostnames (like curl does)
        if arg.startswith("http://") or arg.startswith("https://"):
            url = arg
            i += 1
            continue
        elif not arg.startswith("-") and ("." in arg or arg.startswith("localhost")) and not url:
            # Bare hostname without scheme - curl defaults to http
            url = arg
            i += 1
            continue

        # Method
        if arg in ("-X", "--request"):
            if i + 1 < len(args):
                method = args[i + 1].upper()
                i += 2
                continue

        # Headers
        if arg in ("-H", "--header"):
            if i + 1 < len(args):
                header = args[i + 1]
                if ":" in header:
                    name, value = header.split(":", 1)
                    headers[name.strip()] = value.strip()
                i += 2
                continue

        # Data/body
        if arg in ("-d", "--data", "--data-raw", "--data-binary"):
            if i + 1 < len(args):
                body = args[i + 1]
                if method == "GET":
                    method = "POST"
                i += 2
                continue

        # URL after --url
        if arg == "--url":
            if i + 1 < len(args):
                url = args[i + 1]
                i += 2
                continue

        i += 1

    return RequestContext(
        url=url,
        method=method,
        headers=headers,
        body=body,
        curl_args=args,
    )


def execute_curl(context: RequestContext) -> CurlResult:
    """Execute curl with the given context and return the result."""
    # Build curl command with response header output
    cmd = ["curl", "-sL", "-i", "-w", "\n%{http_code}\n%{url_effective}"]
    cmd.extend(context.curl_args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=300,  # 5 minute timeout
        )
    except subprocess.TimeoutExpired:
        return CurlResult(
            body=b"",
            headers={},
            status_code=0,
            content_type=None,
            final_url=context.url,
            return_code=-1,
            stderr="Request timed out",
        )
    except FileNotFoundError:
        return CurlResult(
            body=b"",
            headers={},
            status_code=0,
            content_type=None,
            final_url=context.url,
            return_code=-1,
            stderr="curl not found in PATH",
        )

    # Parse response
    output = result.stdout
    stderr = result.stderr.decode("utf-8", errors="replace")

    # Split into headers and body
    # curl -i outputs headers, then blank line, then body
    # -w appends status code and effective URL at the end
    headers: dict[str, str] = {}
    body = b""
    status_code = 0
    final_url = context.url

    # Find the header/body separator (double CRLF or double LF)
    # With -L (follow redirects), curl outputs multiple header blocks.
    # We need to find the LAST header block (final response), not the first.
    header_end = -1
    header_section = ""

    # Keep finding header blocks until we find the actual body
    remaining = output
    while True:
        found_sep = False
        for sep in (b"\r\n\r\n", b"\n\n"):
            idx = remaining.find(sep)
            if idx != -1:
                potential_headers = remaining[:idx].decode("utf-8", errors="replace")
                potential_body = remaining[idx + len(sep):]

                # Check if this looks like a header block (starts with HTTP/)
                if potential_headers.strip().startswith("HTTP/"):
                    header_section = potential_headers
                    remaining = potential_body
                    found_sep = True
                    break
                else:
                    # Not a header block, this is the body
                    body = remaining
                    found_sep = False
                    break

        if not found_sep:
            body = remaining
            break

    # Parse headers from the final response
    if header_section:
        for line in header_section.split("\n"):
            line = line.strip()
            if line.startswith("HTTP/"):
                # Status line
                match = re.search(r"HTTP/[\d.]+ (\d+)", line)
                if match:
                    status_code = int(match.group(1))
            elif ":" in line:
                name, value = line.split(":", 1)
                headers[name.strip().lower()] = value.strip()

    # Extract -w output from end of body
    # Format: body + \n + status_code + \n + effective_url
    body_lines = body.rsplit(b"\n", 2)
    if len(body_lines) >= 3:
        body = body_lines[0]
        try:
            status_code = int(body_lines[1].decode("utf-8").strip())
        except ValueError:
            pass
        final_url = body_lines[2].decode("utf-8", errors="replace").strip()

    content_type = headers.get("content-type")

    return CurlResult(
        body=body,
        headers=headers,
        status_code=status_code,
        content_type=content_type,
        final_url=final_url,
        return_code=result.returncode,
        stderr=stderr,
    )


def curl_result_to_response_context(result: CurlResult) -> ResponseContext:
    """Convert CurlResult to ResponseContext for middleware processing."""
    return ResponseContext(
        body=result.body,
        headers=result.headers,
        status_code=result.status_code,
        content_type=result.content_type,
        url=result.final_url,
    )
