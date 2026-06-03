from __future__ import annotations

import logging
import os
from typing import Any

_CONFIGURED = False

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging() -> None:
    """Configure project logging once, controlled by environment variables.

    Env vars:
    - LOG_LEVEL: level for app loggers (default INFO).
    """

    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))

    app_logger = logging.getLogger("app")
    app_logger.setLevel(level)
    # Avoid duplicate handlers if uvicorn/pytest already attached one.
    if not any(isinstance(h, logging.StreamHandler) for h in app_logger.handlers):
        app_logger.addHandler(handler)
    app_logger.propagate = False

    # Keep noisy provider/client internals out of normal app logs. The LLM planner logs
    # the request and response payloads itself so LiteLLM debug mode is not needed.
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)

    _CONFIGURED = True


def safe_for_log(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(str(key)):
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = safe_for_log(item)
        return sanitized
    if isinstance(value, list):
        return [safe_for_log(item) for item in value]
    if isinstance(value, tuple):
        return tuple(safe_for_log(item) for item in value)
    if hasattr(value, "model_dump"):
        return safe_for_log(value.model_dump(mode="json"))
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return (
        normalized in {"api_key", "apikey", "authorization", "bearer"}
        or normalized.endswith("_api_key")
        or normalized.endswith("_token")
        or "secret" in normalized
        or "password" in normalized
    )
