"""Isolated Meta Muse Spark connectivity smoke test.

Run from qapha-ai/backend:
    python app/providers/test_meta_connection.py

Requires MODEL_API_KEY in the current process environment.
This file does not modify the existing scientific core.
"""

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://api.meta.ai/v1/chat/completions"
MODEL = "muse-spark-1.3"


def main() -> int:
    api_key = os.environ.get("MODEL_API_KEY", "").strip()
    if not api_key:
        print("ERROR: MODEL_API_KEY is not set in this terminal.", file=sys.stderr)
        return 1

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
    }
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            data = json.load(response)
    except HTTPError as exc:
        # Avoid printing the complete server response: it might include request details.
        print(f"ERROR: Meta API returned HTTP {exc.code} ({exc.reason}).", file=sys.stderr)
        return 1
    except (URLError, TimeoutError, OSError) as exc:
        print(f"ERROR: Network/connection failure ({type(exc).__name__}).", file=sys.stderr)
        return 1
    except (ValueError, UnicodeError):
        print("ERROR: Meta API returned a response that is not valid JSON.", file=sys.stderr)
        return 1

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        print("ERROR: Successful HTTP response without chat completion choices.", file=sys.stderr)
        return 1

    message = choices[0].get("message", {})
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        print("ERROR: Chat completion contains no text response.", file=sys.stderr)
        return 1

    print("SUCCESS: Meta Model API connectivity verified.")
    print(f"Model: {data.get('model', MODEL)}")
    print(f"Response: {content.strip()}")
    usage = data.get("usage")
    if isinstance(usage, dict):
        print(f"Total tokens: {usage.get('total_tokens', 'not reported')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
