"""Entrypoint: ``python -m app``.

Uvicorn installs SIGTERM/SIGINT handlers and drains in-flight requests before
running the FastAPI shutdown hooks, so ``docker stop`` shuts down cleanly.
"""

import uvicorn

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,  # app.core.logging configures logging
        proxy_headers=True,  # trusted proxy IPs: FORWARDED_ALLOW_IPS env (default 127.0.0.1)
        reload=False,
        timeout_graceful_shutdown=20,
    )


if __name__ == "__main__":
    main()
