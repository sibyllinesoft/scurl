"""Playwright-based browser fetcher for JS-rendered pages."""

from .curl import CurlResult
from .middleware import RequestContext


def execute_browser(context: RequestContext, timeout: int = 60000) -> CurlResult:
    """Fetch a URL using headless Chromium via Playwright.

    Returns a CurlResult with the fully-rendered HTML as the body,
    compatible with the existing response middleware chain.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return CurlResult(
            body=b"",
            headers={},
            status_code=0,
            content_type=None,
            final_url=context.url,
            return_code=-1,
            stderr="playwright not installed. Install with: pip install sibylline-scurl[browser]",
        )

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            response = page.goto(context.url, wait_until="networkidle", timeout=timeout)

            if response is None:
                browser.close()
                return CurlResult(
                    body=b"",
                    headers={},
                    status_code=0,
                    content_type=None,
                    final_url=context.url,
                    return_code=-1,
                    stderr="No response from page",
                )

            status_code = response.status
            final_url = page.url
            html = page.content()

            # Extract response headers
            headers = {}
            for name, value in response.headers.items():
                headers[name.lower()] = value

            # Ensure content-type is text/html so readability middleware processes it
            content_type = headers.get("content-type", "text/html")

            browser.close()

            return CurlResult(
                body=html.encode("utf-8"),
                headers=headers,
                status_code=status_code,
                content_type=content_type,
                final_url=final_url,
                return_code=0,
                stderr="",
            )

    except Exception as e:
        return CurlResult(
            body=b"",
            headers={},
            status_code=0,
            content_type=None,
            final_url=context.url,
            return_code=-1,
            stderr=f"Browser rendering failed: {e}",
        )
