"""
Dummy OpenAI-compatible server that logs full request payloads.

Saves each request to a JSON file with base64 image data truncated for readability,
but also saves the full raw payload for exact comparison.

Usage:
    python debug/logging_server.py [--port 9999]

The server responds with a valid chat completion response so both
nano-eval and lm-eval think the request succeeded.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

LOG_DIR = Path(__file__).parent / "request_logs"
LOG_DIR.mkdir(exist_ok=True)

REQUEST_COUNT = 0


def _truncate_base64(obj, max_len=80):
    """Recursively truncate base64 strings in a nested dict/list for readability."""
    if isinstance(obj, dict):
        return {k: _truncate_base64(v, max_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_base64(v, max_len) for v in obj]
    if isinstance(obj, str) and obj.startswith("data:image/"):
        prefix = obj[:60]
        return f"{prefix}...[TRUNCATED, total {len(obj)} chars]"
    return obj


class LoggingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global REQUEST_COUNT
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"_raw": body.decode("utf-8", errors="replace")}

        REQUEST_COUNT += 1
        tag = self.headers.get("X-Framework", "unknown")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_name = f"{REQUEST_COUNT:03d}_{tag}_{timestamp}"

        # Save truncated version (readable)
        truncated = _truncate_base64(payload)
        readable_path = LOG_DIR / f"{base_name}_readable.json"
        with open(readable_path, "w") as f:
            json.dump(truncated, f, indent=2)

        # Save full raw payload
        raw_path = LOG_DIR / f"{base_name}_raw.json"
        with open(raw_path, "w") as f:
            json.dump(payload, f, indent=2)

        # Print summary to console
        print(f"\n{'=' * 70}")
        print(f"REQUEST #{REQUEST_COUNT} from [{tag}]  path={self.path}")
        print(f"Headers: Authorization={self.headers.get('Authorization', 'none')}")
        print(f"Saved to: {readable_path}")
        print(f"{'=' * 70}")
        print(json.dumps(truncated, indent=2)[:3000])
        if len(json.dumps(truncated, indent=2)) > 3000:
            print("... [output truncated for console]")
        print(f"{'=' * 70}\n")
        sys.stdout.flush()

        # Return a valid OpenAI chat completion response
        response = {
            "id": f"chatcmpl-debug-{REQUEST_COUNT}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", "debug-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "FINAL ANSWER: 42\nFinal Answer: 42",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
            },
        }

        resp_bytes = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def do_GET(self):
        """Handle GET requests (lm-eval checks /v1/models)."""
        if "/models" in self.path:
            response = {
                "object": "list",
                "data": [
                    {
                        "id": "debug-model",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "debug",
                    }
                ],
            }
        else:
            response = {"status": "ok"}

        resp_bytes = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def log_message(self, format, *args):
        pass  # Suppress default logging


def main():
    port = 9999
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])

    server = HTTPServer(("127.0.0.1", port), LoggingHandler)
    print(f"Logging server running on http://127.0.0.1:{port}")
    print(f"Logs saved to: {LOG_DIR}")
    print("Press Ctrl+C to stop\n")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
