"""CLI entry point for scurl."""

import sys
from dataclasses import dataclass, field
from typing import Optional

from .middleware import (
    RequestMiddlewareChain,
    ResponseMiddlewareChain,
    RequestAction,
)
from .request_middleware import SecretDefender
from .response_middleware import ReadabilityExtractor
from .curl import parse_curl_args, execute_curl, curl_result_to_response_context


# Registry of available middleware with their slugs
REQUEST_MIDDLEWARE = {
    "secret-defender": ("SecretDefender", "Detects and blocks requests containing secrets", SecretDefender),
}

RESPONSE_MIDDLEWARE = {
    "readability": ("ReadabilityExtractor", "Extracts clean markdown from HTML", ReadabilityExtractor),
    "prompt-defender": ("PromptInjectionDefender", "Detects prompt injection in web content", None),  # Lazy loaded
}


def print_middleware_list() -> None:
    """Print available middleware."""
    print("Request Middleware:")
    for slug, (name, desc, _) in REQUEST_MIDDLEWARE.items():
        print(f"  {slug:<20} {name} - {desc}")
    print()
    print("Response Middleware:")
    for slug, (name, desc, _) in RESPONSE_MIDDLEWARE.items():
        print(f"  {slug:<20} {name} - {desc}")


@dataclass
class ScurlFlags:
    """Parsed scurl-specific flags."""
    raw: bool = False
    render: bool = False
    readability: bool = False
    disable: set[str] = field(default_factory=set)
    enable: set[str] = field(default_factory=set)
    list_middleware: bool = False
    help: bool = False
    # Prompt injection defender options
    injection_threshold: float = 0.3
    injection_action: str = "redact"  # "warn", "redact", "datamark", "metadata", "silent"


def extract_scurl_flags(args: list[str]) -> tuple[ScurlFlags, list[str]]:
    """Extract scurl-specific flags from args, return (flags, remaining_args)."""
    flags = ScurlFlags()
    remaining = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--raw":
            flags.raw = True
            i += 1
        elif arg == "--render":
            flags.render = True
            i += 1
        elif arg == "--readability":
            flags.readability = True
            i += 1
        elif arg == "--disable":
            if i + 1 < len(args):
                flags.disable.add(args[i + 1])
                i += 2
            else:
                remaining.append(arg)
                i += 1
        elif arg == "--enable":
            if i + 1 < len(args):
                flags.enable.add(args[i + 1])
                i += 2
            else:
                remaining.append(arg)
                i += 1
        elif arg == "--injection-threshold":
            if i + 1 < len(args):
                try:
                    flags.injection_threshold = float(args[i + 1])
                except ValueError:
                    pass  # Keep default
                i += 2
            else:
                i += 1
        elif arg == "--injection-action":
            if i + 1 < len(args):
                action = args[i + 1].lower()
                if action in ("warn", "redact", "datamark", "metadata", "silent"):
                    flags.injection_action = action
                i += 2
            else:
                i += 1
        elif arg == "--list-middleware":
            flags.list_middleware = True
            i += 1
        elif arg in ("--help", "-h") and i == 0:
            # Only treat as scurl help if it's the first arg
            flags.help = True
            i += 1
        else:
            remaining.append(arg)
            i += 1

    return flags, remaining


def print_help() -> None:
    """Print scurl help."""
    print("scurl - A secure curl wrapper with middleware support")
    print()
    print("Usage: scurl [scurl-options] [curl-options] <url>")
    print()
    print("scurl-specific options:")
    print("  --raw                  Disable all response middleware (raw curl output)")
    print("  --render               Use headless browser for JS-rendered pages")
    print("  --readability          Extract article content (strips nav, ads, etc.)")
    print("  --disable <middleware>  Disable a middleware by slug (can be repeated)")
    print("  --enable <middleware>  Enable an opt-in middleware (can be repeated)")
    print("  --list-middleware      List available middleware and their slugs")
    print("  --help, -h             Show this help (use curl --help for curl options)")
    print()
    print("Prompt injection detection (requires --enable prompt-defender):")
    print("  --injection-threshold <0.0-1.0>  Detection threshold (default: 0.5)")
    print("  --injection-action <action>      Action on detection (default: redact):")
    print("                                     warn     - wrap in <suspected-prompt-injection> tag, content unchanged")
    print("                                     redact   - wrap in <suspected-prompt-injection> tag, mask patterns with █")
    print("                                     datamark - wrap in <suspected-prompt-injection> tag, spotlighting mode")
    print("                                     metadata - return JSON analysis")
    print("                                     silent   - pass through unchanged")
    print()
    print("All other options are passed directly to curl.")
    print()
    print("Examples:")
    print("  scurl https://example.com                    # Fetch and extract markdown")
    print("  scurl --raw https://example.com              # Raw HTML output")
    print("  scurl --disable trafilatura https://...      # Disable markdown extraction")
    print("  scurl --disable secret-defender https://...  # Disable secret scanning")
    print("  scurl --enable prompt-defender https://...   # Enable injection detection")
    print("  scurl --render https://github.com/user/repo    # Render JS-heavy pages")
    print("  scurl --enable prompt-defender --injection-threshold 0.5 https://...")
    print("  scurl -H 'Accept: application/json' https://api.example.com/data")


def run(args: Optional[list[str]] = None) -> int:
    """Run scurl with the given arguments. Returns exit code."""
    if args is None:
        args = sys.argv[1:]

    # Extract scurl-specific flags
    flags, curl_args = extract_scurl_flags(args)

    if flags.help:
        print_help()
        return 0

    if flags.list_middleware:
        print_middleware_list()
        return 0

    if not curl_args:
        print("scurl: no URL specified", file=sys.stderr)
        print("Try 'scurl --help' for more information.", file=sys.stderr)
        return 1

    # Parse curl args to get request context
    context = parse_curl_args(curl_args)

    if not context.url:
        print("scurl: no URL specified", file=sys.stderr)
        return 1

    # Build request middleware chain
    request_chain = RequestMiddlewareChain()
    secret_defender_enabled = "secret-defender" not in flags.disable
    secret_defender_override = "secret-defender" in flags.enable

    if secret_defender_enabled and not secret_defender_override:
        request_chain.add(SecretDefender())

    # Execute request middleware
    result = request_chain.execute(context)
    if result.action == RequestAction.BLOCK:
        print(f"scurl: {result.reason}", file=sys.stderr)
        return 1

    # Use potentially modified context
    if result.context:
        context = result.context

    # Execute fetch (curl or browser)
    if flags.render:
        from .browser import execute_browser
        curl_result = execute_browser(context)
    else:
        curl_result = execute_curl(context)

    if curl_result.return_code != 0 and curl_result.return_code != -1:
        # curl failed but not our timeout/not-found
        if curl_result.stderr:
            print(curl_result.stderr, file=sys.stderr)
        return curl_result.return_code

    if curl_result.return_code == -1:
        print(f"scurl: {curl_result.stderr}", file=sys.stderr)
        return 1

    # Build response middleware chain
    response_chain = ResponseMiddlewareChain()
    if not flags.raw:
        if "readability" not in flags.disable:
            response_chain.add(ReadabilityExtractor(use_readability=flags.readability))
        # Prompt defender is opt-in (requires --enable)
        if "prompt-defender" in flags.enable:
            from .prompt_defender import PromptInjectionDefender
            response_chain.add(PromptInjectionDefender(
                threshold=flags.injection_threshold,
                action=flags.injection_action,
            ))

    # Execute response middleware
    response_context = curl_result_to_response_context(curl_result)
    response_result = response_chain.execute(response_context)

    # Output result
    sys.stdout.buffer.write(response_result.body)
    if response_result.body and not response_result.body.endswith(b"\n"):
        sys.stdout.buffer.write(b"\n")

    return 0


def main() -> None:
    """Main entry point."""
    sys.exit(run())


if __name__ == "__main__":
    main()
