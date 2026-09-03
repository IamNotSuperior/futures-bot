"""Credential and configuration loading for the data layer.

Secrets live in a ``.env`` file at the project root, which is gitignored.
See ``.env.example`` for the expected keys.
"""

from pathlib import Path

from dotenv import load_dotenv
import os

# Project root is one level above this file (data/loader.py -> project root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Load .env once at import time. Existing environment variables win, so a real
# environment (CI, a shell export) can override the local file without editing it.
load_dotenv(ENV_PATH, override=False)


class MissingCredentialError(RuntimeError):
    """Raised when a required credential is not present in the environment."""


def get_databento_api_key() -> str:
    """Return the Databento API key.

    Raises:
        MissingCredentialError: if DATABENTO_API_KEY is unset or empty. The
            error message never includes the key value itself.
    """
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if not key:
        raise MissingCredentialError(
            "DATABENTO_API_KEY is not set. Add it to "
            f"{ENV_PATH} (see .env.example) or export it in your environment."
        )
    return key
