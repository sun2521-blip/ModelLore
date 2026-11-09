"""Request, upload, and rate limits for the ModelLore API.

Each limit has a safe default and can be raised or lowered with a
``MODELLORE_*`` environment variable. Values are read once, at startup.
"""

import os


def int_from_env(name: str, default: int) -> int:
    """Read a positive whole number from the environment, or use the default."""
    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a whole number") from exc

    if value < 1:
        raise ValueError(f"{name} must be at least 1")

    return value


# Names (CSV headers, feature names, target column) share the database column size.
MAX_NAME_LENGTH = 255

MAX_UPLOAD_ROWS = int_from_env("MODELLORE_MAX_UPLOAD_ROWS", 100_000)
MAX_UPLOAD_COLUMNS = int_from_env("MODELLORE_MAX_UPLOAD_COLUMNS", 200)

MAX_PREDICT_ROWS = int_from_env("MODELLORE_MAX_PREDICT_ROWS", 1_000)
MAX_FEATURE_VALUE_LENGTH = 1_000
