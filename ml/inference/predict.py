
import hashlib
import hmac
import io
import re
from pathlib import Path

import joblib


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ArtifactIntegrityError(Exception):
    """A model artifact is outside the artifact directory or its hash does not match."""


def load_verified_model(artifact_path, expected_sha256, artifact_dir):
    """Load a joblib model only if it is ours and unchanged.

    Joblib files can run code when they are loaded, so this refuses anything
    outside ``artifact_dir`` and anything whose SHA-256 differs from the hash
    recorded at training time. The file is read once and loaded from those same
    bytes, so it cannot be swapped between the check and the load.
    """
    if not isinstance(expected_sha256, str) or not SHA256_PATTERN.fullmatch(
        expected_sha256
    ):
        raise ArtifactIntegrityError("No valid recorded hash for this artifact")

    artifact_dir = Path(artifact_dir).resolve()
    resolved_path = Path(artifact_path).resolve()

    if not resolved_path.is_relative_to(artifact_dir):
        raise ArtifactIntegrityError("Artifact is outside the artifact directory")

    if not resolved_path.is_file():
        raise FileNotFoundError("Model artifact not found")

    artifact_bytes = resolved_path.read_bytes()
    actual_sha256 = hashlib.sha256(artifact_bytes).hexdigest()

    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ArtifactIntegrityError("Artifact hash does not match the recorded hash")

    return joblib.load(io.BytesIO(artifact_bytes))


def predict(model, data):
    if model is None:
        raise ValueError("Model cannot be None.")

    return model.predict(data)
