"""
Admin credentials for the project's own trusted clients.

The Streamlit console, the live end-to-end runner and run_demo.py talk to the
API as its operator: they write to the demo deal and run 41 queries back to
back. With ADMIN_API_KEY set, the API grants that only to requests carrying the
key (api/security.py), so these clients must present it — otherwise setting the
key locally quietly turns the operator into a rate-limited public visitor.

Read from the environment first, then the project's .env, because none of these
clients load .env themselves. Containers never see a key this way unless one is
passed explicitly: .dockerignore keeps .env* out of every image.
"""

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def admin_headers() -> dict[str, str]:
    """
    Returns the X-Admin-Key header when an admin key is configured.

    Returns:
        {"X-Admin-Key": key}, or {} when no key is configured.
    """
    key = os.getenv("ADMIN_API_KEY", "").strip()
    if not key and _ENV_FILE.exists():
        from dotenv import dotenv_values

        key = (dotenv_values(_ENV_FILE).get("ADMIN_API_KEY") or "").strip()
    return {"X-Admin-Key": key} if key else {}
