import hashlib
import json
import random
from pathlib import Path
import pandas as pd

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
    
def hash_dataframe(X: pd.DataFrame, y: pd.Series) -> str:
    """Create a SHA-256 fingerprint of features and target."""
    digest = hashlib.sha256()

    features_json = X.to_json(
        orient="split",
        date_format="iso",
        double_precision=15,
    )
    target_json = y.to_json(
        orient="split",
        date_format="iso",
        double_precision=15,
    )

    digest.update(features_json.encode("utf-8"))
    digest.update(b"\0")
    digest.update(target_json.encode("utf-8"))

    return digest.hexdigest()
