#!/usr/bin/env python3
"""Simple HTTP server that runs scurl commands."""

import json
import subprocess
import sys
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler

from scurl.sanitize import sanitize_text
from scurl.middleware import ResponseContext

TIMEOUT_SECONDS = 90
MAX_CONVERT_SIZE = 256 * 1024  # 256 KB

# Singleton defender — avoids re-loading the ONNX session on every request.
# Safe because HTTPServer.serve_forever() is single-threaded.
_defender = None


def _get_defender(threshold: float = 0.5, action: str = "redact"):
    """Return the singleton PromptInjectionDefender, creating it on first call."""
    global _defender
    if _defender is None:
        from scurl.prompt_defender import PromptInjectionDefender
        _defender = PromptInjectionDefender(
            threshold=threshold,
            action=action,
        )
    else:
        if _defender.threshold != threshold:
            _defender.threshold = threshold
        if _defender.action != action:
            _defender.action = action
    return _defender


def _injection_dict(analysis) -> dict:
    """Convert an InjectionAnalysis into a JSON-safe dict."""
    active_signals = [k for k, v in analysis.pattern_features.items() if v > 0]
    return {
        "score": round(float(analysis.score), 4),
        "flagged": analysis.flagged,
        "threshold": analysis.threshold,
        "action_taken": analysis.action_taken,
        "signals": active_signals,
    }


class ScurlHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if self.path == "/convert":
            self._handle_convert(body)
        else:
            self._handle_fetch(body)

    def _handle_fetch(self, body: bytes):
        try:
            data = json.loads(body)
            url = data.get("url")
            render = data.get("render", True)
            threshold = float(data.get("threshold", 0.5))
            action = data.get("action", "redact")

            if not url:
                self._send_json(400, {"markdown": None, "error": "Missing 'url' field", "injection": None})
                return

            cmd = ["scurl", "--render", url] if render else ["scurl", url]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )

            if result.returncode == 0:
                defender = _get_defender(threshold=threshold, action=action)
                analysis = defender.analyze(result.stdout)
                injection = _injection_dict(analysis)

                ctx = ResponseContext(
                    body=result.stdout.encode("utf-8"),
                    headers={},
                    status_code=200,
                    content_type="text/plain",
                    url=url,
                )
                if analysis.flagged and defender.should_process(ctx):
                    processed = defender.process(ctx)
                    output = processed.body.decode("utf-8", errors="replace")
                else:
                    output = result.stdout

                self._send_json(200, {"markdown": output, "error": None, "injection": injection})
            else:
                error = result.stderr.strip() or f"scurl exited with code {result.returncode}"
                self._send_json(500, {"markdown": None, "error": error, "injection": None})

        except json.JSONDecodeError:
            self._send_json(400, {"markdown": None, "error": "Invalid JSON", "injection": None})
        except subprocess.TimeoutExpired:
            self._send_json(504, {"markdown": None, "error": "Request timed out", "injection": None})
        except Exception as e:
            self._send_json(500, {"markdown": None, "error": str(e), "injection": None})

    def _handle_convert(self, body: bytes):
        try:
            data = json.loads(body)
            text = data.get("html") or data.get("text")
            action = data.get("action", "redact")
            threshold = float(data.get("threshold", 0.5))

            if not text:
                self._send_json(400, {"markdown": None, "error": "Missing 'html' or 'text' field", "injection": None})
                return

            text_bytes = len(text.encode("utf-8"))
            if text_bytes > MAX_CONVERT_SIZE:
                self._send_json(413, {
                    "markdown": None,
                    "error": f"Input too large ({text_bytes // 1024} KB). Maximum is 256 KB.",
                    "injection": None,
                })
                return

            # First sanitize (strip HTML, normalize whitespace)
            cleaned = sanitize_text(text)

            # Reuse singleton defender (avoids re-loading ONNX session)
            defender = _get_defender(threshold=threshold, action=action)
            analysis = defender.analyze(cleaned)
            injection = _injection_dict(analysis)

            ctx = ResponseContext(
                body=cleaned.encode("utf-8"),
                headers={},
                status_code=200,
                content_type="text/plain",
                url="",
            )

            if analysis.flagged and defender.should_process(ctx):
                result = defender.process(ctx)
                output = result.body.decode("utf-8", errors="replace")
            else:
                output = cleaned

            self._send_json(200, {"markdown": output, "error": None, "injection": injection})

        except json.JSONDecodeError:
            self._send_json(400, {"markdown": None, "error": "Invalid JSON", "injection": None})
        except Exception as e:
            self._send_json(500, {"markdown": None, "error": str(e), "injection": None})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "Not found"})

    def _send_json(self, status: int, data: dict):
        response = json.dumps(data)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response.encode())

    def log_message(self, format, *args):
        print(f"[scurl-server] {args[0]}", file=sys.stderr)


if __name__ == "__main__":
    # Eagerly load the defender + ONNX model so the cost is paid at boot,
    # not on the first request.
    print("[scurl-server] Pre-loading prompt injection model…", file=sys.stderr)
    try:
        defender = _get_defender()
        defender._ensure_heavy_components()
        defender._embedder.embed("warmup")
        print("[scurl-server] Model ready", file=sys.stderr)
    except Exception as e:
        print(f"[scurl-server] Model pre-load failed (will retry on first request): {e}", file=sys.stderr)

    server = HTTPServer(("0.0.0.0", 8080), ScurlHandler)
    print("[scurl-server] Listening on port 8080", file=sys.stderr)
    server.serve_forever()
