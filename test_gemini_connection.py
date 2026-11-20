import os
import sys

from google import genai
from google.genai import errors


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        print("FAILED: Missing GEMINI_API_KEY", file=sys.stderr)
        return 2

    client = genai.Client(api_key=api_key)
    print("Testing Gemini 2.5 Flash...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Reply exactly: GEMINI_OK",
        )
    except errors.APIError as exc:
        print("FAILED", file=sys.stderr)
        print(f"Exception: {type(exc).__name__}", file=sys.stderr)
        print(f"HTTP code: {exc.code}", file=sys.stderr)
        print(f"Message: {str(exc).replace(api_key, '[REDACTED]')}", file=sys.stderr)
        print(
            f"Details: {str(exc.details).replace(api_key, '[REDACTED]')}",
            file=sys.stderr,
        )
        return 1

    print("SUCCESS")
    print("Response:", response.text)
    print("Usage:", response.usage_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
