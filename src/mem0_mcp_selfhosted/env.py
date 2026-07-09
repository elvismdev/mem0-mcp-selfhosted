"""Centralized env var readers with whitespace stripping.

Guards against docker-compose .env trailing newlines across all modules.
"""

from __future__ import annotations

import os


def env(key: str, default: str = "") -> str:
    """Read an env var, stripping whitespace."""
    return os.environ.get(key, default).strip()


def opt_env(key: str) -> str | None:
    """Read an optional env var. Returns None if absent, stripped value if present."""
    val = os.environ.get(key)
    return val.strip() if val is not None else None


def bool_env(key: str, default: str = "false") -> bool:
    """Read a boolean env var (true/1/yes)."""
    return env(key, default).lower() in ("true", "1", "yes")


def prompt_env(key: str) -> str | None:
    """Read a prompt from *key*, or from the file named by ``<key>_FILE``.

    Long multi-line prompts are awkward to embed in env blocks (MCP client
    configs, docker-compose), so each prompt var has a ``_FILE`` companion
    that points at a UTF-8 text file instead. The inline var wins when both
    are set. Returns None when neither is set.

    Raises:
        OSError: if ``<key>_FILE`` is set but the file cannot be read.
    """
    val = opt_env(key)
    if val:
        return val
    path = opt_env(f"{key}_FILE")
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return None
