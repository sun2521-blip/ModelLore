import hashlib
import json
import random
from pathlib import Path

import numpy as np


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def hash_file(file_path: str | Path) -> str:
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def hash_config(config: dict) -> str:
    canonical_config = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return hashlib.sha256(
        canonical_config.encode("utf-8")
    ).hexdigest()