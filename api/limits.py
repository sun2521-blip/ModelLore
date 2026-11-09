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

# Requests per client IP per window, for the endpoints that do real work.
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMITS = {
    "upload": int_from_env("MODELLORE_RATE_LIMIT_UPLOAD", 10),
    "train": int_from_env("MODELLORE_RATE_LIMIT_TRAIN", 5),
    "predict": int_from_env("MODELLORE_RATE_LIMIT_PREDICT", 60),
}


def cors_origins_from_env(raw: str | None) -> list[str]:
    """Parse a comma-separated list of exact browser origins.

    Empty means no cross-origin access at all. A wildcard is refused: the
    allowlist must name each origin that may call the API from a browser.
    """
    origins = [
        origin.strip().rstrip("/")
        for origin in (raw or "").split(",")
        if origin.strip()
    ]

    for origin in origins:
        if origin == "*" or "*" in origin:
            raise ValueError(
                "MODELLORE_CORS_ORIGINS must list exact origins; wildcards are not allowed"
            )
        if not origin.startswith(("http://", "https://")):
            raise ValueError(
                f"MODELLORE_CORS_ORIGINS entries must start with http:// or https://: {origin}"
            )

    return origins


CORS_ORIGINS = cors_origins_from_env(os.getenv("MODELLORE_CORS_ORIGINS"))
