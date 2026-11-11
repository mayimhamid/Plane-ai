"""Docker HEALTHCHECK probe: ``python -m app.healthcheck``.

Uses only the standard library (the slim image has no curl). Exit 0 = healthy.
"""

import os
import sys
import urllib.request


def main() -> int:
    port = os.environ.get("PLANE_AI_API_PORT", "8100")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=4) as resp:
            return 0 if resp.status == 200 else 1
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
