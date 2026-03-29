"""Configuration — load defaults from .env file and environment variables.

Reads .env from the current working directory (or workspace), then
falls back to environment variables. No external dependencies (no python-dotenv).

Supported variables:
    OPENCOLLAB_MODEL      — default LLM model (e.g., "claude-sonnet-4-20250514")
    OPENCOLLAB_PROVIDER   — LLM provider ("openai", "anthropic")
    OPENCOLLAB_API_KEY    — API key (also reads OPENAI_API_KEY / ANTHROPIC_API_KEY)
    OPENCOLLAB_BASE_URL   — API base URL (also reads OPENAI_BASE_URL)
    OPENCOLLAB_BUDGET     — default token budget
"""

from __future__ import annotations

import os


def load_dotenv(path: str | None = None) -> dict[str, str]:
    """Parse a .env file into a dict. Handles quotes and comments."""
    env_path = path or os.path.join(os.getcwd(), ".env")
    result: dict[str, str] = {}

    if not os.path.isfile(env_path):
        return result

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Remove surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            result[key] = value

    return result


def get_config(workspace: str | None = None) -> dict[str, str | None]:
    """Get resolved configuration from .env + environment variables.

    Priority: environment variable > .env file > built-in default
    """
    # Load .env from workspace directory
    env_path = os.path.join(workspace, ".env") if workspace else None
    dotenv = load_dotenv(env_path)

    def resolve(key: str, *fallback_keys: str, default: str | None = None) -> str | None:
        # Check env vars first (highest priority)
        val = os.environ.get(key)
        if val:
            return val
        for fk in fallback_keys:
            val = os.environ.get(fk)
            if val:
                return val
        # Check .env file
        val = dotenv.get(key)
        if val:
            return val
        for fk in fallback_keys:
            val = dotenv.get(fk)
            if val:
                return val
        return default

    return {
        "model": resolve("OPENCOLLAB_MODEL", default="gpt-4o"),
        "provider": resolve("OPENCOLLAB_PROVIDER", default="openai"),
        "api_key": resolve("OPENCOLLAB_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
        "base_url": resolve("OPENCOLLAB_BASE_URL", "OPENAI_BASE_URL"),
        "budget": resolve("OPENCOLLAB_BUDGET", default="200000"),
    }
